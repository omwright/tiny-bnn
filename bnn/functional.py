import torch
from torch import Tensor
import math

def kl_div(p_mean: Tensor, p_logstd: Tensor, q_mean: float, q_logstd: float) -> Tensor:
    """Return KL divergence between two diagonal Gaussian distributions."""
    eps = 1e-6 # For numerical stability
    kl = q_logstd - p_logstd + (torch.exp(p_logstd)**2 + (p_mean - q_mean)**2)/(2*math.exp(q_logstd)**2 + eps)
    return torch.sum(kl) - 0.5*p_logstd.numel()