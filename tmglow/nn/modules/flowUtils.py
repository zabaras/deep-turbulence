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
import numpy as np
import math
from torch.autograd import Variable
from torch.nn.modules.utils import _pair, _quadruple

class Squeeze(nn.Module):
    """Cqueezes feature map by reducing the dimensions of the feature
    and increasing channel number in chunks.

    :param factor: factor to reduce feature dimensions by, defaults to 2
    :type factor: int, optional

    :note: This is the squeeze approached used in "Glow: Generative flow with invertible 1x1 convolutions" 
        by Kingma et al. https://arxiv.org/abs/1807.03039
    """
    def __init__(self, factor=2):
        """Constructor method
        """ 
        super(Squeeze, self).__init__()
        assert factor >= 1
        if factor == 1:
            Warning('Squeeze factor is 1, this is identity function')
        self.factor = factor

    def forward(self, x):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :returns: 
            - y: [B, factor**2 * in_features, H/factor, W/factor] Squeezed output feature tensor
        :rtype: torch.Tensor
        """
        if self.factor == 1:
            return x
        # n_channels, height, width
        B, C, H, W = x.shape[:]
        assert H % self.factor == 0 and W % self.factor == 0
        x = x.reshape(-1, C, self.factor, H//self.factor, self.factor, W//self.factor)
        x = x.transpose(3, 4)
        y = x.reshape(-1, C * self.factor ** 2, H//self.factor, W//self.factor)

        return y

    def reverse(self, y):
        """Backward pass

        :param y: [B, factor**2 * in_features, H/factor, W/factor] Squeezed input feature tensor
        :type y: torch.Tensor
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
        :rtype: torch.Tensor
        """
        if self.factor == 1:
            return y
        B, C, H, W = y.shape[:]
        assert C >= self.factor ** 2 and C % self.factor ** 2 == 0
        y = y.reshape(-1, C // self.factor ** 2, self.factor, self.factor, H, W)
        y = y.transpose(3, 4)
        x = y.reshape(-1, C // self.factor ** 2, H * self.factor, W * self.factor)

        return x

class CheckerSqueeze(nn.Module):
    """Squeezes feature map by reducing the dimensions of the feature
    and increasing channel number in a checkered pattern.
    See Fig. 8 of paper: https://arxiv.org/abs/2006.04731

    :param factor: factor to reduce feature dimensions by, defaults to 2
    :type factor: int, optional

    :note: This is the squeeze approached used in "Density estimation using real nvp" 
        by Dinh et al. https://arxiv.org/abs/1605.08803
    """
    def __init__(self, factor=2):
        """Constructor method
        """ 
        super(CheckerSqueeze, self).__init__()
        assert factor >= 1
        if factor == 1:
            Warning('Squeeze factor is 1, this is identity function')

        # Not tested for other factor values
        factor = 2 
        self.factor = factor

    def forward(self, x):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :returns: 
            - y: [B, factor**2 * in_features, H/factor, W/factor] Squeezed output feature tensor
        :rtype: torch.Tensor
        """
        if self.factor == 1:
            return x
        # n_channels, height, width
        B, C, H, W = x.shape[:]
        assert H % self.factor == 0 and W % self.factor == 0

        y = torch.zeros(B, C * self.factor ** 2,  H//self.factor, W//self.factor).type(x.type())

        c0 = C
        y[:,:c0,:,:] = x[:,:,::self.factor,::self.factor]
        y[:,c0:2*c0,:,:] = x[:,:,1::self.factor,::self.factor]
        y[:,2*c0:3*c0,:,:] = x[:,:,1::self.factor,1::self.factor]
        y[:,3*c0:,:,:] = x[:,:,::self.factor,1::self.factor]

        return y

    def reverse(self, y):
        """Backward pass

        :param y: [B, factor**2 * in_features, H/factor, W/factor] Squeezed input feature tensor
        :type y: torch.Tensor
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
        :rtype: torch.Tensor
        """
        if self.factor == 1:
            return y
        B, C, H, W = y.shape[:]
        assert C >= self.factor ** 2 and C % self.factor ** 2 == 0
        x = torch.zeros(B, C//self.factor ** 2, H* self.factor, W* self.factor).type(y.type())

        c0 = C//self.factor ** 2
        x[:,:,::self.factor,::self.factor] = y[:,:c0,:,:]
        x[:,:,1::self.factor,::self.factor] = y[:,c0:2*c0,:,:]
        x[:,:,1::self.factor,1::self.factor] = y[:,2*c0:3*c0,:,:]
        x[:,:,::self.factor,1::self.factor] = y[:,3*c0:,:,:]

        return x

class GaussianDiag(object):
    """Multi-variate Gaussian class with diagonal covariance
    for representing the latent variables

    :param mean: [B, in_features, H, W] tensor of mean values
    :type mean: torch.Tensor
    :param log_stddev: [B, in_features, H, W] tensor of log sandard deviations
    :type log_stddev: torch.Tensor
    """
    Log2PI = float(np.log(2 * np.pi))

    def __init__(self, mean, log_stddev):
        """Constructor method
        """ 
        super().__init__()
        self.mean = mean
        self.log_stddev = log_stddev.clamp_(min=-10., max=math.log(5.))
        # self._backward_hook = self.log_stddev.register_hook(
        #     lambda grad: torch.clamp_(grad, -10., 10.))

    def likelihood(self, x):
        """Computes the Gaussian log-likelihood of each element

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :return:
            - like: [B, in_features, H, W] log-likelihood tensor
        :rtype: torch.Tensor
        """
        like =  -0.5 * (GaussianDiag.Log2PI + self.log_stddev * 2. \
            + (x - self.mean) ** 2 / (self.log_stddev * 2.).exp())
        
        return like

    def log_prob(self, x):
        """Computes the log product (sum) of Gaussian likelihoods
        over the entire input feature tensor

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :return: 
            - likelihood: [B] sum log-likelihood over features
        :rtype: torch.Tensor
        """
        likelihood = self.likelihood(x)
        return likelihood.view(x.shape[0], -1).sum(1)

    def sample(self, eps=None):
        """Samples latent variables from learned Gaussian density

        :param eps: [B, in_features, H, W] Latent samples from the unit Gaussian to reconstruct specific latent variables.
        If none are provided latent variables are sampled randomly from learned density, defaults to None
        :type eps: torch.Tensor, optional
        :return: 
            - z: [B, in_features, H, W] sum log-likelihood over features
        :rtype: torch.Tensor
        """
        self.log_stddev.data.clamp_(min=-10., max=math.log(5.))
        if eps is None:
            eps = torch.randn_like(self.log_stddev)
            # print(eps, self.log_stddev.data )
        z = self.mean + self.log_stddev.exp() * eps
        return z

class Conv2dZeros(nn.Module):
    """Convolution with weight and bias initialized to zero followed by channel-wise scaling
    :math:`x*exp(scale * logscale\_factor)`

    :param in_features: Number of input feature channels
    :type in_features: int
    :param out_features: Number of output feature channels
    :type out_features: int
    :param logscale_factor: log factor to scale output tensor by, defaults to 1
    :type logscale_factor: int, optional

    :note: This is proposed in "Glow: Generative flow with invertible 1x1 convolutions" 
        by Kingma et al. https://arxiv.org/abs/1807.03039. Appears to help with stability.
    """
    def __init__(self, in_features, out_features, logscale_factor=1):
        """Constructor method
        """
        super(Conv2dZeros, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, 
            stride=1, padding=0, bias=True)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.logscale_factor = logscale_factor
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass.

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :return: 
            - out: [B, out_features, H, W] output feature tensor
        :rtype: torch.Tensor
        """
        x = self.conv(F.pad(x, _quadruple(1), mode='replicate'))
        return x * torch.exp(torch.clamp(self.scale, -4., np.log(4)) * self.logscale_factor)

class LatentEncoder(nn.Module):
    """Latent encoder used to compute mu and std for Gaussian density
    from split feature map. See NN block in Fig. 8 of paper: 
    https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    """
    def __init__(self, in_features):
        """Constructor method
        """
        super(LatentEncoder, self).__init__()
        self.conv2d = Conv2dZeros(in_features, in_features * 2)
        self.hardtanh = nn.Hardtanh(min_val=-2.0, max_val=np.log(5.0), inplace=False)
        # self.hardtanh = nn.Sigmoid()

    def forward(self, x):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :return: 
            - gauss_diag: Gaussian prior
        :rtype: :class:`nn.modules.flowUtils.GaussianDiag`
        """
        mean, log_stddev =  (self.hardtanh(self.conv2d(x))).chunk(2, 1)
        gauss_diag = GaussianDiag(mean, log_stddev)
        return gauss_diag

class Split(nn.Module):
    """Splits input features into half features that are passed deeper in the model
    and the other half modeled as a Gaussian density.
    See NN block in Fig. 8 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    """
    def __init__(self, in_features):
        """Constructor method
        """
        super(Split, self).__init__()
        self.latent_encoder = LatentEncoder(in_features // 2)

    def forward(self, z, return_eps=False):
        """Forward split

        :param z: [B, in_features, H, W] input feature tensor
        :type z: torch.Tensor
        :param return_eps: Return samples from latent densities, defaults to False
        :type return_eps: bool, optional
        :return: 
            - z1: [B, in_features//2, H, W] output feature tensor
            - log_prob_prior: [B] log-likelihood of split features 
            - eps: [B, in_features//2, H, W] tensor of sampled latent variables from unit gaussian
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        # split out z2, and evalute log prob at z2 which takes the form of 
        # diagonal Gaussian are reparameterized by latent_encoder
        z1, z2 = z.chunk(2, 1)
        prior = self.latent_encoder(z1)
        log_prob_prior = prior.log_prob(z2)
        if return_eps:
            eps = (z2 - prior.mean) / prior.log_stddev.exp()
        else:
            eps = None
        return z1, log_prob_prior, eps

    def reverse(self, z1, eps=None):
        """Backward split

        :param z1: [B, in_features//2, H, W] input split feature tensor
        :type z1: torch.Tensor
        :param eps: [B, in_features//2, H, W] Latent samples from the unit Gaussian to reconstruct specific latent variables.
        If none are provided latent variables are sampled randomly from learned density, defaults to None
        :type eps: torch.Tensor, optional
        :return: 
            - z: [B, in_features, H, W] output reconstructed feature tensor
            - log_prob_prior: [B] log-likelihood of split features 
        :rtype: (torch.Tensor, torch.Tensor)
        """
        # sample z2, then concat with z1
        # intermediate flow, z2 is the split-out latent
        prior = self.latent_encoder(z1)
        z2 = prior.sample(eps)
        z = torch.cat((z1, z2), 1)
        log_prob_prior = prior.log_prob(z2)
        return z, log_prob_prior