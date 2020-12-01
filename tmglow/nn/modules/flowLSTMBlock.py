'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: https://github.com/zabaras/deep-turbulence
=====
'''
import sys, time
sys.path.append(".")
# sys.path.append("../..")

from nn.modules.actNorm import ActNorm
from nn.modules.glowConv import InvertibleConv1x1, InvertibleConv1x1LU
from nn.modules.flowAffine import LSTMAffineCouplingLayer, AffineCouplingLayer
from nn.modules.flowUtils import Squeeze, CheckerSqueeze, Split

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AffineCouplingBlock(nn.Module):
    """An invertible affine coupling block consisting of an activation
    normalization, conditional affine coupling layer and a 1x1 convolution.
    See Fig. 6 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param cond_features: Number of conditional feature channels
    :type cond_features: int
    :param train_sampling: train by sampling (1x1 conv. is inverted for the forward pass), defaults to True
    :type train_sampling: bool, optional
    :param LUdecompose: Use the LU decomposition for 1x1 convolution, defaults to False
    :type LUdecompose: bool, optional
    """
    def __init__(self, in_features, cond_features, 
        train_sampling=True, LUdecompose=False):
        """Constructor method
        """ 
        super(AffineCouplingBlock, self).__init__()
        self.norm = ActNorm(in_features)
        if LUdecompose:
            self.conv = InvertibleConv1x1LU(in_features, 
                train_sampling=train_sampling)
        else:
            self.conv = InvertibleConv1x1(in_features,
                train_sampling=train_sampling)

        self.coupling = AffineCouplingLayer(in_features, cond_features)

    def forward(self, x, cond):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :param cond: [B, cond_features, H, W] conditional feature tensor
        :type cond: torch.Tensor
        :returns: 
            - y: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine block
        :rtype: (torch.Tensor, torch.Tensor)
        """
        out, logdet1 = self.norm(x)
        out, logdet2 = self.conv(out)
        y, logdet3 = self.coupling(out, cond)

        return y, logdet3 + logdet2 + logdet1

    def reverse(self, y, cond):
        """Backward pass

        :param y:  [B, in_features, H, W] input feature tensor
        :type y: torch.Tensor
        :param cond: [B, cond_features, H, W] conditional feature tensor
        :type cond: torch.Tensor
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine block
        :rtype: (torch.Tensor, torch.Tensor)
        """
        out, logdet1 = self.coupling.reverse(y, cond)
        out, logdet2 = self.conv.reverse(out)
        x, logdet3 = self.norm.reverse(out)
        return x, logdet3 + logdet2 + logdet1

class UnNormedAffineCouplingBlock(nn.Module):
    """An unnormalized invertible affine coupling block consisting 
    of an conditional affine coupling layer and a 1x1 convolution.
    See Fig. 6 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param cond_features: Number of conditional feature channels
    :type cond_features: int
    :param train_sampling: train by sampling (1x1 conv. is inverted for the forward pass), defaults to True
    :type train_sampling: bool, optional
    :param LUdecompose: Use the LU decomposition for 1x1 convolution, defaults to False
    :type LUdecompose: bool, optional
    """
    def __init__(self, in_features, cond_features, 
        train_sampling=True, LUdecompose=False):
        """Constructor method
        """ 
        super(UnNormedAffineCouplingBlock, self).__init__()
        if LUdecompose:
            self.conv = InvertibleConv1x1LU(in_features, 
                train_sampling=train_sampling)
        else:
            self.conv = InvertibleConv1x1(in_features,
                train_sampling=train_sampling)

        self.coupling = AffineCouplingLayer(in_features, cond_features)

    def forward(self, x, cond):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :param cond: [B, cond_features, H, W] conditional feature tensor
        :type cond: torch.Tensor
        :returns: 
            - y: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine block
        :rtype: (torch.Tensor, torch.Tensor)
        """
        out, logdet1 = self.conv(x)
        y, logdet2 = self.coupling(out, cond)
        return y, logdet1 + logdet2

    def reverse(self, y, cond):
        """Backward pass

        :param y:  [B, in_features, H, W] input feature tensor
        :type y: torch.Tensor
        :param cond: [B, cond_features, H, W] conditional feature tensor
        :type cond: torch.Tensor
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine block
        :rtype: (torch.Tensor, torch.Tensor)
        """
        out, logdet1 = self.coupling.reverse(y, cond)
        x, logdet2 = self.conv.reverse(out)
        return x, logdet1 + logdet2

class LSTMCouplingBlock(nn.Module):
    """An LSTM invertible affine coupling block consisting 
    of an activation normalization, lstm affine coupling layer and a 1x1 convolution.
    See Fig. 6 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param cond_features: Number of conditional feature channels
    :type cond_features: int
    :param rec_features: Number of recurrent feature channels in :class:`nn.modules.convLSTM.LSTMAffineCouplingLayer`
    :type rec_features: int
    :param train_sampling: train by sampling (1x1 conv. is inverted for the forward pass), defaults to True
    :type train_sampling: bool, optional
    :param LUdecompose: Use the LU decomposition for 1x1 convolution, defaults to False
    :type LUdecompose: bool, optional
    """
    def __init__(self, in_features, cond_features, rec_features,
        train_sampling=True, LUdecompose=False):
        """Constructor method
        """ 
        super(LSTMCouplingBlock, self).__init__()
        self.norm = ActNorm(in_features)
        self.norm2 = ActNorm(in_features)
        if LUdecompose:
            self.conv = InvertibleConv1x1LU(in_features, 
                train_sampling=train_sampling)
        else:
            self.conv = InvertibleConv1x1(in_features,
                train_sampling=train_sampling)
        
        self.coupling = LSTMAffineCouplingLayer(in_features, cond_features, rec_features)

    def forward(self, x, cond, rec_states=None):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :param cond: [B, cond_features, H, W] conditional feature tensor
        :type cond: torch.Tensor
        :param rec_states: tuple of LSTM states (hidden state, cell state), defaults to None
        :type rec_states: tuple, optional
        :returns: 
            - y: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine block
            - states_out: tuple of LSTM (cell, hidden) states
        :rtype: (torch.Tensor, torch.Tensor, tuple)
        """
        out, logdet1 = self.norm(x)
        out, logdet2 = self.conv(out)
        y, logdet3, out_states = self.coupling(out, cond, rec_states)
        return y, logdet3 + logdet2 + logdet1, out_states

    def reverse(self, y, cond, rec_states=None):
        """Backward pass

        :param y:  [B, in_features, H, W] input feature tensor
        :type y: torch.Tensor
        :param cond: [B, cond_features, H, W] conditional feature tensor
        :type cond: torch.Tensor
        :param rec_states: tuple of LSTM states (hidden state, cell state), defaults to None
        :type rec_states: tuple, optional
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine block
            - states_out: tuple of LSTM (cell, hidden) states
        :rtype: (torch.Tensor, torch.Tensor, tuple)
        """
        out, logdet1, out_states = self.coupling.reverse(y, cond, rec_states)
        out, logdet2 = self.conv.reverse(out)
        x, logdet3 = self.norm.reverse(out)
        return x, logdet3+logdet2+logdet1, out_states

class LSTMFLowBlock(nn.Module):
    """LSTM Conditional Flow block considting of an a stack of
    invertible affine layers.
    See Fig. 6 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param cond_features: Number of conditional feature channels
    :type cond_features: int
    :param rec_features: Number of recurrent feature channels in LSTM
    :type rec_features: int
    :param n_layers: Number of affine layers in block
    :type n_layers: int
    :param factor: Squeeze factor to reduce the dimensionality of the feature map, defaults to 2
    :type factor: int, optional
    :param train_sampling: Train by sampling (1x1 conv. is inverted for the forward pass), defaults to True
    :type train_sampling: bool, optional
    :param LUdecompose: Use the LU decomposition for 1x1 convolution, defaults to False
    :type LUdecompose: bool, optional
    :param do_split: Perform a split at the end of block, defaults to True. Refer to :class:`nn.modules.flowUtils.Split`
    :type do_split: bool, optional
    :param squeeze_type: Type of squeeze method; 0 = :class:`nn.modules.flowUtils.CheckerSqueeze`, 
        1 = :class:`nn.modules.flowUtils.Squeeze`, defaults to 0
    :type squeeze_type: int, optional
    """
    def __init__(self, in_features, cond_features, rec_features, 
        n_layers, factor=2, LUdecompose=True, train_sampling=False, 
        do_split=True, squeeze_type=0):
        """Constructor method
        """ 
        super(LSTMFLowBlock, self).__init__()
        self.do_split = do_split
        self.n_layers = n_layers

        if(squeeze_type == 0):
            self.squeeze = CheckerSqueeze(factor)
        else:
            self.squeeze = Squeeze(factor)
        in_features = in_features * factor ** 2 # Features after squeeze

        self.revlayers = nn.Sequential()
        for i in range(n_layers-1):
            # No actnorm or conv first flow layer
            if(i==0):
                self.revlayers.add_module('affine_layer{}'.format(i+1), 
                    UnNormedAffineCouplingBlock(in_features, cond_features,
                    LUdecompose=LUdecompose, train_sampling=train_sampling))
            else:
                self.revlayers.add_module('affine_layer{}'.format(i+1), 
                    AffineCouplingBlock(in_features, cond_features,
                    LUdecompose=LUdecompose, train_sampling=train_sampling))

        self.revlayers.add_module('affine_layer{}'.format(n_layers), 
                LSTMCouplingBlock(in_features, cond_features, rec_features,
                LUdecompose=LUdecompose, train_sampling=train_sampling))

        # Split off variables into latent params
        if do_split:
            self.split = Split(in_features)

    def forward(self, x, cond, rec_states, return_eps=False):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :param cond: [B, cond_features, H, W] conditional feature tensor
        :type cond: torch.Tensor
        :param rec_states: tuple of LSTM states (hidden state, cell state), defaults to None
        :type rec_states: tuple, optional
        :param return_eps: Return samples from latent densities, defaults to False
        :type return_eps: bool, optional
        :returns: 
            - y: [B, in_features, H, W] Output feature tensor
            - logdet: [B] log determinate of block + the log-likelihood of the latent variables
            - out_states: tuple of LSTM (cell, hidden) states
            - eps: [B, 2*in_features, H//2, W//2] (for factor=2) tensor of sampled latent variables from unit gaussian
        :rtype: (torch.Tensor, torch.Tensor, tuple, torch.Tensor)
        """
        out_states = []
        logdet = 0.
        x = self.squeeze(x)

        # Loop through affine layers
        for i, revlayer in enumerate(self.revlayers._modules.values()):
            # If last flow block in forward its an LSTM layer
            if(i==self.n_layers-1):
                if rec_states is None:
                    x, dlogdet, state_out = revlayer.forward(x, cond)
                else:
                    x, dlogdet, state_out = revlayer.forward(x, cond, rec_states)
                out_states = state_out
            else:
                x, dlogdet = revlayer.forward(x, cond)

            logdet = logdet + dlogdet
        
        if self.do_split:
            x, log_prob_prior, eps = self.split(x, return_eps=return_eps)
            logdet = logdet + log_prob_prior
            return x, logdet, out_states, eps
        else:
            return x, logdet, out_states, None

    def reverse(self, y, cond, rec_states, eps=None):
        """Backward pass

        :param y:  [B, in_features, H, W] input feature tensor
        :type y: torch.Tensor
        :param cond: [B, cond_features, H, W] conditional feature tensor
        :type cond: torch.Tensor
        :param rec_states: tuple of LSTM states (hidden state, cell state), defaults to None
        :type rec_states: tuple, optional
        :param eps: [B, 2*in_features, H//2, W//2] (for factor=2) tensor of sampled latent variables from unit gaussian, defaults to None
        :type eps: torch.Tensor, optional
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate block
            - out_states: tuple of LSTM (cell, hidden) states
        :rtype: (torch.Tensor, torch.Tensor, tuple)
        """
        # eps is not used if this revblock does not split in the end
        # focus on sampling
        out_states = []
        logdet = 0.
        if self.do_split:
            y, log_prob_prior = self.split.reverse(y, eps)
            logdet = logdet + log_prob_prior

        # Loop through affine layers
        for i, revlayer in reversed(list(enumerate(self.revlayers._modules.values()))):
            # If first flow block in reverse it is an LSTM cell
            if(i==self.n_layers-1):
                if rec_states is None:
                    y, dlogdet, state_out = revlayer.reverse(y, cond)
                else:
                    y, dlogdet, state_out = revlayer.reverse(y, cond, rec_states)
                out_states = state_out
            else:
                y, dlogdet = revlayer.reverse(y, cond)
            logdet = logdet + dlogdet
            
        return self.squeeze.reverse(y), logdet, out_states

if __name__ == '__main__':
    import time

    test_flow = LSTMFLowBlock(4, 2, 4, 1, factor=2, do_split=True, squeeze_type=0)
    # couple = AffineCouplingBlock(4, 4, conv_ksize=3)

    x_in =  torch.randn(1, 4, 8, 8)
    h_in =  torch.randn(1, 2, 4, 4)

    tic = time.time()
    y_out, log_det, out_states, eps = test_flow.forward(x_in, h_in, None, return_eps=True)
    print(time.time()-tic)
    print(eps.size())
    tic = time.time()
    x_rev, log_det, out_states = test_flow.reverse(y_out, h_in, None, eps=eps)
    print(time.time()-tic)

    # x_in =  torch.randn(1, 4, 4, 4)
    # y_out, log_det = couple.forward(x_in, h_in)
    # x_rev, log_det = couple.reverse(y_out, h_in)
    print(torch.sum(torch.abs(x_in[0,0]-x_rev[0,0])))
    print(x_rev[0,0])
    print(x_in[0,0])
