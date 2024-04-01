import torch.nn as nn
from torch.nn import functional as F
from .linear import BayesLinear

from .. import functional as BF

class KLLoss(nn.Module):
    """Creates a criterion to measure the KL divergence between model parameters and their prior."""
    def __init__(self):
        super().__init__()

    def forward(self, model):
        kl = 0.0
        for m in model.modules():
            if isinstance(m, BayesLinear):
                kl += BF.kl_div(m.weight_mean, m.weight_logstd, m.prior_mean, m.prior_logstd)
                if m.bias:
                    kl += BF.kl_div(m.bias_mean, m.bias_logstd, m.prior_mean, m.prior_logstd)
        return kl