'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: http://aimsciences.org//article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097
doi: https://dx.doi.org/10.3934/fods.2020019
github: https://github.com/zabaras/deep-turbulence
=====
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActNorm(nn.Module):
    """Activation Normalization layer found in the Glow paper
    This was proposed to be used instead of batch-norm due to
    potentially low batch-sizes. Applies 'normalization' to each
    channel independently.

    :param in_features: Number of input feature channels, this is the dimension of w and b
    :type in_features: int
    :param return_logdet: Return the log determinate, defaults to True
    :type return_logdet: bool, optional
    :param data_init: if weight and bias terms have already been initialized, defaults to False
    :type data_init: bool, optional
    """          
    def __init__(self, in_features, return_logdet=True, data_init=False):
        """Constructor method
        """       
        super(ActNorm, self).__init__()
        # identify transform
        self.weight = nn.Parameter(torch.ones(in_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(in_features, 1, 1))
        self.data_init = data_init
        self.data_initialized = False
        self.return_logdet = return_logdet

    def _init_parameters(self, input):
        """Initializes the weight and bias term given a mini-batch of data

        :param input: [B, C, H, W] mini-batch tensor of data
        :type input: torch.Tensor
        """        
        # mean per channel (B, C, H, W) --> (C, B, H, W) --> (C, BHW)
        input = input.transpose(0, 1).contiguous().view(input.shape[1], -1)
        mean = input.mean(1)
        std = input.std(1) + 1e-6
        self.bias.data = -(mean / std).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = 1. / std.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        """Forward pass of normalization layer :math:`w*x + b`.

        :param x: [B, C, H, W] normed mini-batch tensor of data
        :type x: torch.Tensor
        :returns: 
            - y: [B, C, H, W] un-normed tensor
            - logdet: log determinate of normalization, optional
        :rtype: (torch.Tensor, torch.Tensor)
        """
        if self.data_init and (not self.data_initialized):
            self._init_parameters(x)
            self.data_initialized = True
        if self.return_logdet:
            logdet = self.weight.abs().log().sum() * x.shape[-1] * x.shape[-2]
            return self.weight * x + self.bias, logdet
        else:
            return self.weight * x + self.bias

    def reverse(self, y):
        """Backward pass of normalization layer :math:`(x - b)/w`.

        :param y: [B, C, H, W] un-normed mini-batch tensor of data
        :type x: torch.Tensor
        :returns: 
            - x: [B, C, H, W] normed tensor
            - logdet: log determinate of normalization, optional
        :rtype: (torch.Tensor, torch.Tensor)
        """
        if self.return_logdet:
            logdet = torch.abs(self.weight).log().sum() * y.shape[-1] * y.shape[-2]
            return (y - self.bias) / self.weight, logdet
        else:
            return (y - self.bias) / self.weight