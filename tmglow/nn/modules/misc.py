'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: https://github.com/zabaras/deep-turbulence
=====
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsamplingLinear(nn.Module):
    """Bi-linear upscaling module for upscaling feature maps.

    :param scale_factor: up scale factor, defaults to 2.
    :type scale_factor: int, optional
    """
    def __init__(self, scale_factor=2.):
        
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :returns: 
            - y: [B, in_features, scale_factor*H, scale_factor*W] Output feature tensor
        :rtype: torch.Tensor
        """
        return F.interpolate(x, scale_factor=self.scale_factor, 
            mode='bilinear', align_corners=True)