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
import numpy as np
import scipy.linalg
import time

class InvertibleConv1x1(nn.Module):
    """Invertible 1x1 Conv layer which supports two configuration, the standard
    version, train_sampling = False, requires a matrix inversion on the backwards pass.
    With train_sampling = True, the matrix inversion occurs on the forward pass. This can be
    used to avoid an expensive inversion during training.

    :param in_features: Number of input feature channels
    :type in_features: int
    :param train_sampling: train by sampling (1x1 conv. is inverted for the forward pass), defaults to True
    :type train_sampling: bool, optional

    :note: For additional information see "Glow: Generative Flow with Invertible 1×1 Convolutions" by
        Kingma et al. https://arxiv.org/abs/1807.03039
    """
    def __init__(self, in_features, train_sampling=True):
        """Constructor method
        """
        super(InvertibleConv1x1, self).__init__()
        
        dtype = np.float32
        w_shape = (in_features, in_features)
        # only one copy for both forward and reverse, 
        # depends on `training_sampling`, the inverse happens on less used path
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(dtype)

        self.w_shape = w_shape
        self.train_sampling = train_sampling
        self.weight = nn.Parameter(torch.Tensor(w_init))
    
    def forward(self, x):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :returns: 
            - y: [B, in_features, H, W] Output feature tensor
            - logdet: [B] log determinate of block
        :rtype: (torch.Tensor, torch.Tensor)
        """
        # torch.slogdet() is not stable
        if self.train_sampling:
            W = torch.inverse(self.weight.double()).float()
        else:
            W = self.weight
        logdet = self.log_determinant(x, W)        
        kernel = W.view(*self.w_shape, 1, 1)
        y = F.conv2d(x, kernel)
        return y, logdet

    def reverse(self, y):
        """Backward pass

        :param y:  [B, in_features, H, W] input feature tensor
        :type y: torch.Tensor
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
            - logdet: [B] log determinate of block
        :rtype: (torch.Tensor, torch.Tensor)
        """
        if self.train_sampling:
            W = self.weight
        else:
            W = torch.inverse(self.weight.double()).float()
        logdet = self.log_determinant(y, W)
        kernel = W.view(*self.w_shape, 1, 1)
        # negative logdet, since we are still computing p(x|z)
        x = F.conv2d(y, kernel)
        return x, logdet

    def log_determinant(self, x, W):
        """Calculates the log determinate of convolution

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :param W: [in_features, in_features] 1x1 convolutional weight tensor
        :type W: torch.Tensor
        :returns: 
            - log_det: [B] log determinate
        :rtype: torch.Tensor
        """
        h, w = x.shape[2:]
        det = torch.det(W.to(torch.float64)).to(torch.float32)
        if det.item() == 0:
            det += 1e-6
        log_det = h * w * det.abs().log()
        return log_det


class InvertibleConv1x1LU(nn.Module):
    """Invertible 1x1 convolutional layer with a weight matrix modeled by an LU decompsition.
    The standard version, train_sampling = False, requires a matrix inversion on the backwards pass.
    With train_sampling = True, the matrix inversion occurs on the forward pass. This can be
    used to avoid an expensive inversion during training.

    :param in_features: Number of input feature channels
    :type in_features: int
    :param train_sampling: train by sampling (1x1 conv. is inverted for the forward pass), defaults to True
    :type train_sampling: bool, optional

    :note: For additional information see "Glow: Generative Flow with Invertible 1×1 Convolutions" by
        Kingma et al. https://arxiv.org/abs/1807.03039
    """
    def __init__(self, in_channels, train_sampling=True):
        """Constructor method
        """
        super().__init__()    
        
        dtype = np.float32
        w_shape = (in_channels, in_channels)
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(dtype)

        self.w_shape = w_shape
        self.train_sampling = train_sampling
    
        p_np, l_np, u_np = scipy.linalg.lu(w_init)
        s_np = np.diag(u_np)
        sign_s_np = np.sign(s_np)
        log_s_np = np.log(abs(s_np))
        u_np = np.triu(u_np, k=1)
        l_mask = np.tril(np.ones_like(w_init), -1)
        u_mask = np.triu(np.ones_like(w_init), k=1)
        eye = np.eye(*w_shape, dtype=dtype)

        self.register_buffer('p', torch.Tensor(p_np.astype(dtype)))
        self.l = nn.Parameter(torch.Tensor(l_np.astype(dtype)))
        self.u = nn.Parameter(torch.Tensor(u_np.astype(dtype)))
        self.log_s = nn.Parameter(torch.Tensor(log_s_np.astype(dtype)))

        self.register_buffer('sign_s', torch.Tensor(sign_s_np.astype(dtype)))
        self.register_buffer('l_mask', torch.Tensor(l_mask))
        self.register_buffer('u_mask', torch.Tensor(u_mask))
        self.register_buffer('eye', torch.Tensor(eye))
        self.register_buffer('log_s_old', torch.Tensor(log_s_np.astype(dtype)+1.0))

    def weight(self):
        """Gets the 1x1 convolutional weight matrix from LU representation
        :returns: 
            - W: [in_features, in_features] 1x1 convolutional weight matrix
        :rtype: torch.Tensor
        """
        self.log_s_old = self.log_s.data.clone() 
        l = self.l * self.l_mask + self.eye
        u = self.u * self.u_mask + torch.diag(self.log_s.exp() * self.sign_s) + 0.01*self.eye
        self.W = torch.matmul(self.p, torch.matmul(l, u))
        return self.W

        
    def inv_weight(self):
        """Gets the LU inverse of the 1x1 convolutional weight matrix

        :returns: 
            - inv_W: [in_features, in_features] inverse 1x1 convolutional weight matrix
        :rtype: torch.Tensor
        """
        l = self.l * self.l_mask + self.eye
        u = self.u * self.u_mask + torch.diag(self.log_s.exp() * self.sign_s) + 0.01*self.eye
        inv_W = torch.matmul(u.inverse(), torch.matmul(l.inverse(), self.p.inverse()))
        return inv_W
    
    def forward(self, x):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :returns: 
            - y: [B, in_features, H, W] Output feature tensor
            - logdet: [B] log determinate of block
        :rtype: (torch.Tensor, torch.Tensor)
        """
        logdet = self.log_s.sum() * x.shape[2] * x.shape[3]
        if self.train_sampling:
            # reverse path is used for training, take matrix inverse here
            weight = self.inv_weight()
            logdet = -logdet
        else:
            weight = self.weight()
        kernel = weight.view(*self.w_shape, 1, 1)
        y = F.conv2d(x, kernel)
        return y, logdet

    def reverse(self, y):
        """Backward pass

        :param y:  [B, in_features, H, W] input feature tensor
        :type y: torch.Tensor
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
            - logdet: [B] log determinate of block
        :rtype: (torch.Tensor, torch.Tensor)
        """
        logdet = self.log_s.sum() * y.shape[2] * y.shape[3]
        if self.train_sampling:
            if(torch.all(self.log_s.eq(self.log_s_old))):
                weight = self.W
            else:
                weight = self.weight()
            # reverse path is used for training, do not take inverse here
            # weight = self.weight()
            logdet = -logdet
        else:
            weight = self.inv_weight()

        kernel = weight.view(*self.w_shape, 1, 1)
        x = F.conv2d(y, kernel)

        return x , logdet
