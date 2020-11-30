'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class Grad1Filter2d(nn.Module):
    """Sobel filter to estimate 1st-order gradient in horizontal & vertical 
    directions for a 2D image

    :param dx: Spatial discretization on x-axis
    :type dx: float
    :param dy: Spatial discretization on y-axis
    :type dy: float
    :param kernel_size: Kernel size for gradient approximation, defaults to 3
    :type kernel_size: int, optional

    :raises ValueError: If kernel_size is not 3 or 5
    """
    def __init__(self, dx, dy, kernel_size=3):
        """Constructor method
        """ 
        super(Grad1Filter2d, self).__init__()
        self.dx = dx
        self.dy = dy

        # smoothed central finite diff
        # https://en.wikipedia.org/wiki/Finite_difference_coefficient
        WEIGHT_H_3x3 = torch.FloatTensor([[[[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]]]) / 8.

        # larger kernel size tends to smooth things out
        WEIGHT_H_5x5 = torch.FloatTensor([[[[1, -8, 0, 8, -1],
                                            [2, -16, 0, 16, -2],
                                            [3, -24, 0, 24, -3],
                                            [2, -16, 0, 16, -2],
                                            [1, -8, 0, 8, -1]]]]) / (9*12.)
                                            
        if kernel_size == 3:
            self.register_buffer("weight_h", WEIGHT_H_3x3)
            self.register_buffer("weight_v", WEIGHT_H_3x3.transpose(-1, -2))
            self.padding = _quadruple(1)

        elif kernel_size == 5:
            self.register_buffer("weight_h", WEIGHT_H_5x5)
            self.register_buffer("weight_v", WEIGHT_H_5x5.transpose(-1, -2))
            self.padding = _quadruple(2)
        else:
            raise ValueError('kernel_size size {:d} is not supported!'.format(kernel_size))

    def xGrad(self, u):
        """ Computes the gradient of the image in the x direction

        :param u: [B, 1, H, W] Input feature
        :type u: torch.Tensor
        :returns:
            - ux: [B, 1, H, W] x-gradient feature
        :rtype: torch.Tensor

        :note: Does not compute edge gradients correctly
        """
        ux = F.conv2d(F.pad(u, self.padding, mode='constant'), self.weight_h, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return ux
    
    def yGrad(self, u):
        """ Computes the gradient of the image in the y direction

        :param u: [B, 1, H, W] Input feature to compute gradient
        :type u: torch.Tensor
        :returns:
            - uy: [B, 1, H, W] y-gradient feature
        :rtype: torch.Tensor

        :note: Does not compute edge gradients correctly
        """
        uy = F.conv2d(F.pad(u, self.padding, mode='constant'), self.weight_v, 
                        stride=1, padding=0, bias=None) / (self.dy)
        return uy
