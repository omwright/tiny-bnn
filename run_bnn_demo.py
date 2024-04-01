import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse, os, sys, time

import bnn
import plot_utils as pu

def signal_model(x):
    return -(x+0.5)*np.sin(3*np.pi*x)

def noise_model(x, hetero=True):
    if hetero:
        scale = 0.15 + 0.25*(x + 0.5)**2
    else:
        scale = 0.25*np.ones(x.shape)
    return scale

def generate_data(x, rng, hetero=True):
    return signal_model(x) + rng.normal(0.0, noise_model(x, hetero))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    verbose = True if args.verbose else False

    # Assign random seed
    random_seed = 7302519
    rng = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Check cuda
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    num_workers = 8 if cuda else 0
    print(f"Cuda = {cuda} with num_workers = {num_workers}, system version = {sys.version}")
    
    # Generate data
    x = rng.random((256,1)) - 0.5
    x_mean = x.mean()
    x_std = x.var()**0.5
    x_train = torch.tensor((x - x_mean)/x_std, dtype=torch.float32).to(device)

    y = generate_data(x, rng)
    y_mean = y.mean()
    y_std = y.var()**0.5
    y_train = torch.tensor((y - y_mean)/y_std, dtype=torch.float32).to(device)

    x_test = ((torch.linspace(-2, 2, 512) - x_mean)/x_std).unsqueeze(1).to(device)

    filepath = f"out"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Define BNN
    batch_size = len(x_train)
    kl_weight = 0.1
    learning_rate = 1e-2
    n_epochs = 2000
    n_mc_samples = 10

    model = nn.Sequential(
        bnn.BayesLinear(in_features=1, out_features=64),
        nn.ReLU(),
        bnn.BayesLinear(in_features=64, out_features=64),
        nn.ReLU(),
        bnn.BayesLinear(in_features=64, out_features=2) # Outputs mean and log-std
    )

    criterion_nll = nn.GaussianNLLLoss(full=True, reduction='sum').to(device) # Negative log-likelihodd
    criterion_kl = bnn.KLLoss().to(device) # KL divergence
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Train BNN
    nll_loss = np.zeros(n_epochs)
    kl_loss = np.zeros(n_epochs)
    total_loss = np.zeros(n_epochs)
    print("Training BNN...")
    for epoch in range(n_epochs):
        start = time.time()
        optimizer.zero_grad()
        avg_nll = 0.0
        rmse = 0.0
        for i in range(n_mc_samples):
            outputs = model(x_train) # Forward pass
            nll = criterion_nll(outputs[:,:1], y_train, outputs[:,1:].exp())
            avg_nll += nll
            mse = F.mse_loss(outputs[:,:1], y_train)
            rmse += y_std*mse.item()**0.5 # RMSE as a sanity check
        avg_nll /= n_mc_samples
        rmse /= n_mc_samples
        kl = criterion_kl(model)
        loss = (kl_weight*kl + avg_nll)/batch_size # ELBO
        loss.backward()     # Backpropagate
        optimizer.step()    # Update parameters
        end = time.time()
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            if verbose:
                print(
                    f"{epoch}"
                    f"\tLoss: {loss.item():.4f}"
                    f"\tNLL: {avg_nll.item():.4f}"
                    f"\tKL: {kl.item():.4f}"
                    f"\tRMSE: {rmse:.4f}"
                    f"\tTime: {(end - start):.4f}"
                )

            # Get predictions
            out_means, out_stds = [], []
            with torch.no_grad():
                for i in range(100):
                    outputs = model(x_test)
                    out_means.append(outputs[:,0].detach().numpy())
                    out_stds.append(outputs[:,1].exp().detach().numpy())
            out_means = np.array(out_means)*y_std + y_mean
            out_stds = np.array(out_stds)*y_std
            pred = np.mean(out_means, axis=0)                   # Mean prediction
            aleatoric = np.mean(out_stds**2, axis=0)**0.5
            epistemic = np.var(out_means, axis=0)**0.5
            uncertainty = (aleatoric**2 + epistemic**2)**0.5    # 1-std confidence

            # Plot predictions
            outfile = f"{filepath}/demo_{epoch}.png"
            inputs = (np.array(x_test)*x_std + x_mean).squeeze()
            fig = plt.figure(figsize=(3,3))
            ax = fig.add_subplot()
            plt.scatter(x, y, 
                        s=3, lw=0, color=pu.carnegie, alpha=0.75, label='Training samples')
            plt.plot(inputs, pred, 
                    color=pu.gold_thread, linestyle='dashed', linewidth=0.75, label='Prediction')
            plt.fill_between(inputs, pred - uncertainty, pred + uncertainty, 
                            color=pu.gold_thread, alpha=0.2, linewidth=0, label='Uncertainty')
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$', rotation=0)
            plt.xlim([-2, 2])
            plt.ylim([-1.5,2.5])
            plt.gca().yaxis.grid(color=pu.steel_gray, alpha=0.5, linestyle='dotted')
            plt.gca().xaxis.grid(color=pu.steel_gray, alpha=0.5, linestyle='dotted')
            ax.legend(loc='upper left', frameon=True)
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            plt.close()
                
    outfile = f"{filepath}/demo_run.gif"
    pu.create_gif(filepath, outfile, duration=500, loop=1)