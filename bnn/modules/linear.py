import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import math


class BayesLinear(nn.Module):
    r"""Applies a Bayesian linear transformation to the incoming data: :math:`y = xA^T + b`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        prior_mean: Prior mean of a factorized Gaussian distribution
        prior_var: Prior variance of a factorized Gaussian distribution

    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 prior_mean: float = 0.0, prior_var: float = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.weight_mean = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_logstd = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = True
            self.bias_mean = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_logstd = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.bias = False
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_logstd', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight_mean)
        if self.prior_var is None:
            self.prior_var = 2/(fan_in + fan_out)
        self.prior_logstd = math.log(math.sqrt(self.prior_var))
        nn.init.xavier_uniform_(self.weight_mean)
        if self.bias_mean is not None: # Follows torch.nn.Linear
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mean, -bound, bound)
        
        self.weight_logstd.data.fill_(self.prior_logstd)
        self.bias_logstd.data.fill_(self.prior_logstd)
    
    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight_mean + torch.exp(self.weight_logstd)*torch.randn_like(self.weight_logstd)
        if self.bias_mean is not None:
            bias = self.bias_mean + torch.exp(self.bias_logstd)*torch.randn_like(self.bias_logstd)
        else:
            bias = None
        return F.linear(input, weight, bias)