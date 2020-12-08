# Distributed by: Notre Dame SCAI Lab (MIT Liscense)
# - Associated publication:
# url: 
# doi: 
# github: https://github.com/zabaras/deep-turbulence

import sys
import time
# sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from nn.modules.convLSTM import ResidLSTMBlock
from nn.modules.denseBlock import DenseBlock
from nn.modules.flowLSTMBlock import LSTMFLowBlock
from nn.modules.flowUtils import GaussianDiag
from nn.modules.misc import UpsamplingLinear

class Encoder(nn.Module):
    """Convolutional dense encoder for conditioning the generative model.
    See Fig. 4 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param enc_block_layers: list of the number of dense layers in each desired dense block
    :type enc_block_layers: list
    :param growth_rate: Growth rate of channels in dense block, defaults to 4
    :type growth_rate: int, optional
    :param init_features: Number of channels after te first convolution, defaults to 48
    :type init_features: int, optional
    :param output_features: Number of output channel features (will be twice this value
        since final value will be turned into mu and stds for latent variables), defaults to 8
    :type output_features: int, optional
    :param cond_features: Number of conditional feature channels, defaults to 8
    :type cond_features: int, optional
    :param cglow_upscale: The factor to upscale the features from the dense encode 
        before passing them to the generative model, defaults to 1
    :type cglow_upscale: int, optional
    :param bn_size: number of features after bottleneck if enabled, defaults to 8
    :type bn_size: int, optional
    :param drop_rate: Dropout rate, defaults to 0.
    :type drop_rate: float, optional
    :param bottleneck: Enable bottle next to reduce the number of feature channels, defaults to False
    :type bottleneck: bool, optional
    """
    def __init__(self, in_features, enc_block_layers, growth_rate=4, init_features=48, 
                output_features=8, cond_features=8, cglow_upscale=1, bn_size=8, drop_rate=0., 
                bottleneck=False):
        """Constructor method
        """ 
        super(Encoder, self).__init__()
        # First encoding module
        self.first_encoder = self.first_encoding(in_features, init_features, drop_rate=0.)

        self.encoding_blocks = []
        self.cond_convs = []

        # Construct dense block layers
        self.num_feat = init_features
        for i, num_layers in enumerate(enc_block_layers):
            block = nn.Sequential()
            # If not the first dense block, encode
            if(i > 0):
                block.add_module('encode_conv{}'.format(i),
                    self.enconding_transition(self.num_feat, self.num_feat//2, drop_rate=drop_rate))
                self.num_feat = self.num_feat//2

            block.add_module('encode_dense_block{}'.format(i),
                    DenseBlock(num_layers=num_layers,
                                in_features=self.num_feat,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck))

            self.encoding_blocks.append(block)
            # Calc. number of features coming out of dense block
            self.num_feat = self.num_feat + num_layers * growth_rate

            
            conv = nn.Sequential(nn.Conv2d(self.num_feat, cond_features, kernel_size=3, stride=1, padding=1, 
                                bias=False, padding_mode='zeros'))
            self.cond_convs.append(conv)

        # Times 2 output features for mu and sigma of the lowest dimensional latent variables
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.num_feat, 2*output_features, kernel_size=3, stride=1, padding=1, 
                                bias=False, padding_mode='zeros'))
        
        # Upscale function for moving features to Glow
        self.enc_up_scale = UpsamplingLinear(scale_factor=cglow_upscale)

        # Convert lists to module lists
        self.encoding_blocks = nn.ModuleList(self.encoding_blocks)
        self.cond_convs = nn.ModuleList(self.cond_convs)

    def forward(self, x):
        """Encoder forward pass

        :param x: [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :returns:
            - out: Final encoder output
            - c_out: List of conditional output states to conditional the generative model
        :rtype: (torch.Tensor, list)
        """
        out = self.first_encoder(x)

        c_out = []
        for i, block in enumerate(self.encoding_blocks):
            # Pass though dense block
            dense_out = block(out)
            # Conv LSTM
            cond_conv = self.cond_convs[i]
            c0 = cond_conv(dense_out)
            c_out.append(self.enc_up_scale(c0))

            out = dense_out

        out = self.enc_up_scale(self.out_conv(out))

        return out, c_out

    def first_encoding(self, in_features, init_features, drop_rate=0.):
        """First transition encodering block, halves feature map size 

        :param in_features: Number of input feature channels
        :type in_features: int
        :param init_features: Number of channels after te first convolution, defaults to 48
        :type init_features: int, optional
        :param drop_rate: Dropout rate, defaults to 0.
        :type drop_rate: float, optional
        :returns:
            - first_encoder: Encoding PyTorch module
        :rtype: nn.Module
        """
        first_encoder = nn.Sequential()
        
        # For even image size: k7s2p3, k5s2p2
        # For odd image size (e.g. 65): k7s2p2, k5s2p1, k13s2p5, k11s2p4, k9s2p3
        first_encoder.add_module('In_conv', nn.Conv2d(in_features, init_features // 2, 
                                kernel_size=3, stride=1, padding=1, 
                                bias=False, padding_mode='zeros'))
        # first_encoder.add_module('In_conv_norm', nn.BatchNorm2d(init_features // 2))
        first_encoder.add_module('In_conv_relu', nn.ReLU(inplace=False))

        first_encoder.add_module('In_conv3', nn.Conv2d(init_features // 2, init_features, 
                                kernel_size=3, stride=2, padding=1, 
                                bias=False, padding_mode='zeros'))
       
        return first_encoder


    def enconding_transition(self, in_features, out_features, drop_rate=0, padding=1):
        """Encoding transition convolution placed between dense blocks, halves feature map size

        :param in_features: Number of input feature channels
        :type in_features: int
        :param output_features: Number of output channels
        :type output_features: int, optional
        :param drop_rate: Dropout rate, defaults to 0.
        :type drop_rate: float, optional
        :param padding: convolutional padding, defaults to 1
        :type padding: int, optional
        :returns:
            - encoder_trans: Transition PyTorch module
        :rtype: nn.Module
        """
        encoder_trans = nn.Sequential()
        # encoder_trans.add_module('norm1', nn.BatchNorm2d(in_features))
        encoder_trans.add_module('relu1', nn.ReLU(inplace=False))

        encoder_trans.add_module('conv1', nn.Conv2d(in_features, out_features,
            kernel_size=(2*padding+1), stride=2, 
            padding=padding, bias=False, padding_mode='zeros'))
        if drop_rate > 0:
            encoder_trans.add_module('dropout1', nn.Dropout3d(p=drop_rate))

        return encoder_trans

class LSTMCFlowDecoder(nn.Module):
    """LSTM Conditional generative normalizing flow decoder model.
    See Fig. 4 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param glow_block_layers: list of the number of affine layers in each flow block
    :type glow_block_layers: list
    :param cond_features: Number of conditional feature channels, defaults to 8
    :type cond_features: int, optional
    :param rec_features: Number of recurrent feature channels in LSTM, defaults to 8
    :type rec_features: int, optional
    :param squeeze_factor: Squeeze factor to reduce the dimensionality of the feature map, defaults to 2
    :type squeeze_factor: int, optional
    :param LUdecompose: Use the LU decomposition for 1x1 convolution, defaults to False
    :type LUdecompose: bool, optional
    :param train_sampling: Train by sampling (1x1 conv. is inverted for the forward pass), defaults to True
    :type train_sampling: bool, optional
    :param squeeze_type: Type of squeeze method, see :class:`nn.modules.flowLSTMBlock.LSTMFLowBlock` for 
        more details, defaults to 0
    :type squeeze_type: int, optional
        """
    def __init__(self, in_features, glow_block_layers, cond_features=8, rec_features=8,
                squeeze_factor=2, conv_ksize=3, LUdecompose=False, 
                train_sampling=True, squeeze_type=0):
        """Constructor method
        """ 
        super(LSTMCFlowDecoder, self).__init__()

        self.flow_blocks = nn.ModuleList()
        self.num_feat = in_features

        for i, num_layers in enumerate(glow_block_layers):
            # in_features, cond_features, n_layers,
            # factor=2, conv_ksize=3, LUdecompose=False, train_sampling=True, do_split=True
            fblock = LSTMFLowBlock(self.num_feat, cond_features, rec_features, num_layers, 
                            LUdecompose=LUdecompose, train_sampling=train_sampling, do_split=True, 
                            squeeze_type=squeeze_type)

            self.flow_blocks.append(fblock)
            # Squeeze then split features
            self.num_feat = (self.num_feat * squeeze_factor ** 2)//2

    def forward(self, x, c_in, h_in, return_eps=False):
        """Forward pass flow model :math:`x \\rightarrow z`

        :param x: [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :param c_in: [len(flow_blocks)] List of conditional features for each flow block
        :type c_in: list
        :param h_in: [len(flow_blocks)] List of recurrent features for each flow block
        :type h_in: list
        :param return_eps: Return samples from latent densities, defaults to False
        :type return_eps: bool, optional
        :returns:
            - z: Final encoded latent variables
            - log_det: [B] log determinate + the log-likelihood of the latent variables
            - s_out: [len(flow_blocks)] List of output recurrent features from each flow block
            - eps: list of sampled latent variables from each split
        :rtype: (torch.Tensor, torch.Tensor, list, list)
        """
        assert(len(c_in) == len(self.flow_blocks)), 'List of conditions need to be same length as flow blocks.'
  
        z = x
        log_det = 0
        eps = []
        s_out = []
        for i, flow_block in enumerate(self.flow_blocks):
            # Forward execute block
            if(h_in is None):
                z, log_det0, s0, eps0 = flow_block.forward(z, c_in[i], None, return_eps)
            else:
                z, log_det0, s0, eps0 = flow_block.forward(z, c_in[i], h_in[i], return_eps)
            # Add log determinate
            log_det = log_det + log_det0
            # Save latent noise terms for reversing if needed
            eps.append(eps0)
            s_out.append(s0)
        
        return z, log_det, s_out, eps

    def reverse(self, z, c_in, h_in, eps):
        """Backward pass flow model :math:`x \leftarrow z`

        :param z: input latent tensor
        :type x: torch.Tensor
        :param c_in: [len(flow_blocks)] List of conditional features for each flow block
        :type c_in: list
        :param h_in: [len(flow_blocks)] List of recurrent features for each flow block
        :type h_in: list
        :param eps: list sampled latent variables, defaults to None
        :type eps: torch.Tensor, optional
        :returns:
            - x: [B, in_features, H, W] input feature tensor
            - log_det: [B] log determinate + the log-likelihood of the latent variables
            - s_out: [len(flow_blocks)] List of output recurrent features from each flow block
        :rtype: (torch.Tensor, torch.Tensor, list)
        """
        assert(len(c_in) == len(self.flow_blocks)), 'List of conditions need to be same length as flow blocks.'

        x = z
        log_det = 0
        s_out = []
        # Go through blocks in reverse!
        for i, flow_block in reversed(list(enumerate(self.flow_blocks))):
            # Reverse execute block
            if(h_in is None):
                x, log_det0, s0 = flow_block.reverse(x, c_in[i], None, eps[i])
            else:
                x, log_det0, s0 = flow_block.reverse(x, c_in[i], h_in[i], eps[i])
            # Add log determinate
            log_det = log_det + log_det0

            s_out.insert(0, s0)
        
        return x, log_det, s_out

class TMGlow(nn.Module):
    """Transient Multi-Fidelity Glow Model. Consists of a dense encoder model: :class:`nn.tmGlow.Encoder`
    and a generative flow decoder: :class:`nn.tmGlow.LSTMCFlowDecoder`.
    See Fig. 4 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param out_features: Number of output feature channels
    :type out_features: int
    :param enc_blocks: list of the number of dense layers in each desired dense block
    :type enc_blocks: list
    :param glow_blocks: list of the number of affine layers in each flow block
    :type glow_blocks: list
    :param cond_features: Number of conditional feature channels, defaults to 8
    :type cond_features: int, optional
    :param cglow_upscale: The factor to upscale the features from the dense encode 
        before passing them to the generative model, defaults to 1
    :type cglow_upscale: int, optional
    :param growth_rate: Growth rate of channels in dense block, defaults to 4
    :type growth_rate: int, optional
    :param init_features: Number of channels after te first convolution, defaults to 48
    :type init_features: int, optional
    :param rec_features: Number of recurrent feature channels in LSTM, defaults to 8
    :type rec_features: int, optional
    :param bn_size: number of features after bottleneck if enabled, defaults to 8
    :type bn_size: int, optional
    :param drop_rate: Dropout rate, defaults to 0.
    :type drop_rate: float, optional
    :param bottleneck: Enable bottle next to reduce the number of feature channels, defaults to False
    :type bottleneck: bool, optional
    """
    def __init__(self, in_features, out_features, enc_blocks, glow_blocks, 
                cond_features=8, cglow_upscale=1, growth_rate=4, init_features=48, rec_features=8, bn_size=8, 
                drop_rate=0, bottleneck=False):
        """Constructor method
        """ 
        super(TMGlow, self).__init__()

        self.glow_blocks = glow_blocks
        self.rec_features = rec_features
        # Number of features output from flow model
        # This assumes each flow block splits data every flow block
        # Each split increases the channel dims by a factor of 2
        enc_out_features = out_features*(2**len(glow_blocks))

        self.encoder = Encoder(in_features, 
                                enc_block_layers=enc_blocks, 
                                growth_rate=growth_rate, 
                                init_features=init_features,
                                output_features=enc_out_features,
                                cond_features=cond_features, 
                                cglow_upscale=cglow_upscale,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                bottleneck=bottleneck)

        self.glow = LSTMCFlowDecoder(out_features,
                                glow_block_layers=glow_blocks, 
                                cond_features=cond_features, 
                                rec_features=rec_features,
                                squeeze_factor=2, 
                                LUdecompose=True, 
                                train_sampling=True,
                                squeeze_type=0)

        # Used for storage of normalization 
        self.register_buffer("in_mu", torch.zeros(3))
        self.register_buffer("in_std", torch.zeros(3))
        self.register_buffer("out_mu", torch.zeros(3))
        self.register_buffer("out_std", torch.zeros(3))

        print('Total number of parameters: {}'.format(self._num_parameters()))

    def forward(self, x, y, h_in=None, return_eps=False):
        """Forward pass of the encoder and flow model

        :param x: [B, in_features, H, W] Input feature tensor
        :type x: torch.Tensor
        :param y: [B, out_features, H2, W2] Output feature tensor
        :type y: torch.Tensor
        :param h_in: [len(glow_blocks)] List of recurrent features for each flow block, defaults to None
        :type h_in: list, optional
        :param return_eps: Return samples from latent densities, defaults to False
        :type return_eps: bool, optional
        :returns:
            - z: [B, C, H, W] latent feature tensor
            - log_det: [B] log determinate + the log-likelihood of the latent variables
            - h_out: [len(flow_blocks)] List of output recurrent features from each flow block
            - eps: list of sampled latent variables from each split
        :rtype: (torch.Tensor, torch.Tensor, list, list)
        """
        # First encoder forward pass
        z_out, c_out = self.encoder.forward(x)
        # Seperate mean and standard deviation
        cmean, clog_stddev = z_out.chunk(2, 1)

        cprior = GaussianDiag(cmean, clog_stddev)
        
        # Now forward pass of conditonal flow
        z, log_det, h_out, eps = self.glow.forward(y, c_out, h_in, return_eps=return_eps)
        
        if(return_eps): # Save deepest latent noise terms for reconstruction
            eps0 = (z - cmean) / clog_stddev.exp()
            eps.append(eps0)
        else:
            eps = None

        log_prior = cprior.log_prob(z)

        return z, log_prior+log_det, h_out, eps


    def sample(self, x, h_in=None):
        """Backward pass; Conditional generation!

        :param x: [B, in_features, H, W] Input feature tensor
        :type x: torch.Tensor
        :param h_in: [len(glow_blocks)] List of recurrent features for each flow block, defaults to None
        :type h_in: list, optional
        :returns:
            - z: [B, C, H, W] latent feature tensor
            - log_det: [B] log determinate + the log-likelihood of the latent variables
            - h_out: [len(flow_blocks)] List of output recurrent features from each flow block
        :rtype: (torch.Tensor, torch.Tensor, list)
        """
        # First encoder forward pass
        z_out, c_out = self.encoder.forward(x)
        # Seperate mean and standard deviation
        cmean, clog_stddev = z_out.chunk(2, 1)
        cprior = GaussianDiag(cmean, clog_stddev)
        z_samp = cprior.sample()

        eps = [None for i in range(len(self.glow_blocks))]
        y_out, log_det, h_out = self.glow.reverse(z_samp, c_out, h_in, eps)
        
        return y_out, log_det, h_out

    def reconstruct(self, x, h_in, eps):
        """Backward pass; Reconstruct specific y

        :param x: [B, in_features, H, W] Input feature tensor
        :type x: torch.Tensor
        :param h_in: [len(glow_blocks)] List of recurrent features for each flow block, defaults to None
        :type h_in: list, optional
        :param eps: list sampled latent variables, defaults to None
        :type eps: torch.Tensor, optional
        :returns:
            - z: [B, C, H, W] latent feature tensor
            - log_det: [B] log determinate + the log-likelihood of the latent variables
            - h_out: [len(flow_blocks)] List of output recurrent features from each flow block
        :rtype: (torch.Tensor, torch.Tensor, list)
        """
        # First encoder forward pass
        z_out, c_out = self.encoder.forward(x)
        # Seperate mean and standard deviation
        cmean, clog_stddev = z_out.chunk(2, 1)
        cprior = GaussianDiag(cmean, clog_stddev)
        # Reconstruct output with the noise samples, eps
        z_samp = cprior.sample(eps[-1])

        y_out, log_det, h_out = self.glow.reverse(z_samp, c_out, h_in, eps[:-1])

        return y_out, log_det, h_out

    def _num_parameters(self):
        """Gets number of paramters in the model

        :return: Number of paramters
        :rtype: int
        """
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count

    def initLSTMStates(self, seeds, input_dim):
        """Intialized the LSTM states using random
        generators with the specified seeds. Generates the 
        LSTM hidden states by a U[-1,1] and cell states by N(0,1).

        :param seeds: Random generator seed
        :type seeds: int
        :param input_dim: input_feature dimensions
        :type input_dim: torch.size
        :returns:
            - a_states: list of initialized recurrent states
        :rtype: list
        """
        a_states = []
        device = next(self.parameters()).device # Get device model currently exists on
        for i in range(len(self.glow_blocks)):
            a0 = []
            c0 = []
            for j in range(seeds.size(0)):
                gen = torch.Generator()
                gen = gen.manual_seed(seeds[j].item())

                dims = [1, self.rec_features, input_dim[0]//(2**(i+1)), input_dim[1]//(2**(i+1))]
                a0.append(2*torch.rand(dims, generator=gen)-1)
                c0.append(torch.randn(dims, generator=gen))

            a_states.append((torch.cat(a0, dim=0).to(device), torch.cat(c0, dim=0).to(device)))
        
        return a_states

# Tests for checking inversion consistency
# Need to uncomment sys.append('..)
# if __name__ == '__main__':

#     # An interability tests to verify the TM-Glow is functioning correctly.
#     glow = TMGlow(1, 1, [4,4], [4,4], cglow_upscale=2, rec_features=2)
    
#     x = torch.randn(1, 1, 16, 16)
#     y = torch.randn(1, 1, 32, 32)

#     h_in = None
#     z_out, logp, h_out, eps = glow.forward(x, y, h_in, return_eps=True)
#     # z_out, logp, h_out2, eps2 = glow.forward(x, y, h_out, return_eps=True)

#     # y0, logp0, h_out = glow.reconstruct(x, h_out, eps2)
#     y0, logp0, h_out = glow.reconstruct(x, None, eps)

#     # y0, logp0, h_out = glow.sample(x, h_out)

#     print(torch.mean(torch.abs(y0-y)))
