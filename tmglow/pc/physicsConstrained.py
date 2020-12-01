'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: https://github.com/zabaras/deep-turbulence
=====
'''
from pc.grad1Filter import Grad1Filter2d
from pc.grad2Filter import Grad2Filter2d

import numpy as np
import torch
import torch.nn as nn

class PhysConstrainedLES(nn.Module):
    """Computing residual base losses for the Navier-Stokes equations

    :param dx: Spatial discretization on x-axis
    :type dx: float
    :param dy: Spatial discretization on y-axis
    :type dy: float
    :param rho: Density of fluid, defaults to 1.0
    :type rho: float, optional
    :param grad_kernels: List of kernel sizes for 1st and 2nd order gradients, defaults to [3, 3]
    :type grad_kernels: list, optional
    """
    def __init__(self, dx, dy, rho=1.0, grad_kernels=[3, 3]):
        """Constructor method
        """ 
        super(PhysConstrainedLES, self).__init__()
        self.rho = rho
        self.dx = dx
        self.dy = dy

        # Create gradients
        self.grad1 = Grad1Filter2d(dx, dy, kernel_size=grad_kernels[0])
        self.grad2 = Grad2Filter2d(dx, dy, kernel_size=grad_kernels[1])


    def calcDivergence(self, uPred, scale=True):
        """Calculates the divergence of a velocity field

        :param uPred: [B, 2, H, W] Input velocity field
        :type uPred: torch.Tensor
        :param scale: Scale the residual by dx, defaults to True
        :type scale: bool, optional
        :returns:
            - ustar: [B, 1, H, W] Divergence field
        :rtype: torch.Tensor

        :note: Residual is not calculated on edges where gradients are not correct
        """
        ustar = torch.zeros(uPred[:,0].size()).type(uPred.type())

        uPred = torch.cat((uPred[:,:,:,0].unsqueeze(-1), uPred, uPred[:,:,:,-1].unsqueeze(-1)), dim=-1)
        
        ustar = self.grad1.yGrad(uPred[:,1].unsqueeze(1)) + self.grad1.xGrad(uPred[:,0].unsqueeze(1))

        if(scale):
            ustar = self.dx*ustar
        return torch.clamp(ustar, -1, 1)

    def calcPressurePoisson(self, uPred, pPred, scale=True):
        """Calcs residual of the pressure poisson equation

        :param uPred: [B, 2, H, W] Input velocity field
        :type uPred: torch.Tensor
        :param uPred: [B, 1, H, W] Input pressure field
        :type uPred: torch.Tensor
        :param scale: Scale the residual by cell area (dx*dy), defaults to True
        :type scale: bool, optional
        :returns:
            - pstar: [B, 1, H, W] Divergence field
        :rtype: torch.Tensor

        :note: Residual is not calculated on edges where gradients are not correct

        :note: A good reference for additional information
            http://www.thevisualroom.com/poisson_for_pressure.html
        """
        # Calculate pressure laplacian
        ddp = (1.0/self.rho)*(self.grad2.xGrad(pPred) + self.grad2.yGrad(pPred))

        # Calculate the right-hand side of the pressure poisson equation
        rhs = self.grad1.xGrad(uPred[:,0].unsqueeze(1))**2 + 2*self.grad1.yGrad(uPred[:,0].unsqueeze(1))\
            *self.grad1.xGrad(uPred[:,1].unsqueeze(1)) + self.grad1.yGrad(uPred[:,1].unsqueeze(1))**2

        pstar = ddp+rhs
        if(scale):
            pstar = self.dx*self.dy*pstar

        return  torch.clamp(pstar, -1, 1)
