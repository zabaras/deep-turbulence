'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: http://aimsciences.org//article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097
doi: https://dx.doi.org/10.3934/fods.2020019
github: https://github.com/zabaras/deep-turbulence
=====
'''
import sys
sys.path.append(".")

from nn.modules.denseBlock import DenseBlock, NoNormDenseBlock
from nn.modules.convLSTM import ResidLSTMBlock
from nn.modules.flowUtils import Conv2dZeros
import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineCouplingLayer(nn.Module):
    """Conditional invertable affine coupling layer.
    See Fig. 7 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param cond_features: Number of conditional feature channels
    :type cond_features: int

    :note: For more information see "NICE: Non-linear Independent 
        Components Estimation" by Dihn et al. https://arxiv.org/abs/1410.8516

    :note: Check Feistel cipher as this functions very similarly.
    """
    def __init__(self, in_features, cond_features):
        """Constructor method
        """ 
        super(AffineCouplingLayer, self).__init__()
        # assert in_features % 2 == 0, '# input features must be evenly split,'\
        #     'but got {} features'.format(in_features)
        if in_features % 2 == 0:
            in_channels = in_features // 2 + cond_features
            out_channels = in_features
        else:
            # chunk is be (2, 1) if in_features==3
            in_channels = in_features // 2 + 1 + cond_features
            out_channels = in_features - 1
        
        # Initialize coupling network (Dense Block)
        num_layers = 2
        growth_rate = 1
        self.coupling_nn = nn.Sequential()
        self.coupling_nn.add_module('dense_block', NoNormDenseBlock(num_layers, in_channels, 
                    growth_rate=growth_rate, drop_rate=0., bottleneck=False))
        self.coupling_nn.add_module('relu1', nn.ReLU(inplace=True))
        self.coupling_nn.add_module('zero_conv', Conv2dZeros(in_channels + growth_rate*num_layers, out_channels))

        self.softsign = nn.Softsign()

    def forward(self, x, cond):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :param cond: [B, cond_features, H, W] input feature tensor
        :type cond: torch.Tensor
        :returns: 
            - y: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine layer
        :rtype: (torch.Tensor, torch.Tensor)
        """
        #  Split in the channel dimension
        # last chunk is smaller if not odd number of channels
        x1, x2 = x.chunk(2, 1)
        h = self.coupling_nn(torch.cat((x1, cond), 1))

        shift = h[:, 0::2]
        scale = (2*self.softsign(h[:, 1::2])).exp()

        x2 = x2 + shift
        x2 = x2 * scale
        logdet = torch.abs(scale).log().view(x.shape[0], -1).sum(1)

        return torch.cat((x1, x2), 1), logdet

    def reverse(self, y, cond):
        """Backward pass

        :param y:  [B, in_features, H, W] input feature tensor
        :type y: torch.Tensor
        :param cond: [B, cond_features, H, W] input feature tensor
        :type cond: torch.Tensor
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine layer
        :rtype: (torch.Tensor, torch.Tensor)
        """
        #  Split in the channel dimension
        y1, y2 = y.chunk(2, 1)

        h = self.coupling_nn(torch.cat((y1, cond), 1))

        shift = h[:, 0::2]
        scale = (2*self.softsign(h[:, 1::2])).exp()

        y2 = y2 / scale
        y2 = y2 - shift
        logdet = torch.abs(scale).log().view(y.shape[0], -1).sum(1)

        return torch.cat((y1, y2), 1), logdet

class LSTMAffineCouplingLayer(nn.Module):
    """Conditional LSTM invertable affine coupling layer.
    See Fig. 7 of paper: https://arxiv.org/abs/2006.04731

    :param in_features: Number of input feature channels
    :type in_features: int
    :param cond_features: Number of conditional feature channels
    :type cond_features: int
    :param rec_features: Number of recurrent feature channels, 
        output from :class:`nn.modules.convLSTM.ResidLSTMBlock`
    :type rec_features: int

    :note: For more information see "NICE: Non-linear Independent 
        Components Estimation" by Dihn et al. https://arxiv.org/abs/1410.8516

    :note: Check Feistel cipher as this functions very similarly.
    """
    def __init__(self, in_features, cond_features, rec_features):
        """Constructor method
        """ 
        super(LSTMAffineCouplingLayer, self).__init__()
        # assert in_features % 2 == 0, '# input features must be evenly split,'\
        #     'but got {} features'.format(in_features)
        if in_features % 2 == 0:
            in_channels = in_features // 2 + cond_features
            out_channels = in_features
        else:
            # chunk is be (2, 1) if in_features==3
            in_channels = in_features // 2 + 1 + cond_features
            out_channels = in_features - 1
        
        # Initialize coupling network (Dense Block)
        num_layers = 2
        growth_rate = 1

        # LSTM block
        self.resid_lstm = ResidLSTMBlock(in_channels, rec_features, in_channels, kernel_size=(3,3))

        self.dense_nn = nn.Sequential()
        self.dense_nn.add_module('dense_block', NoNormDenseBlock(num_layers, in_channels,
                    growth_rate=growth_rate, drop_rate=0., bottleneck=False))
        self.dense_nn.add_module('relu1', nn.ReLU(inplace=True))

        # Output convolution
        num_feat = in_channels + growth_rate*num_layers
        self.out_conv = nn.Sequential()
        self.out_conv.add_module('zero_conv', Conv2dZeros(num_feat, out_channels))

        self.softsign = nn.Softsign()

    def forward(self, x, cond, rec_states=None):
        """Forward pass

        :param x:  [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :param cond: [B, cond_features, H, W] input feature tensor
        :type cond: torch.Tensor
        :param rec_states: tuple of LSTM states (hidden state, cell state), defaults to None
        :type rec_states: tuple, optional
        :returns: 
            - y: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine layer
            - states_out: tuple of LSTM (cell, hidden) states
        :rtype: (torch.Tensor, torch.Tensor, tuple)
        """
        #  Split in the channel dimension
        x1, x2 = x.chunk(2, 1)

        if(rec_states is None):
            out, h_next, c_next = self.resid_lstm(torch.cat((x1, cond), 1), None)
            # Store lstm states for next time-step and flow conditions
            states_out = (h_next, c_next)
        else:
            
            out, h_next, c_next = self.resid_lstm(torch.cat((x1, cond), 1), rec_states)
            # Store lstm states for next time-step and flow conditions
            states_out = (h_next, c_next)

        out = self.dense_nn(out)
        h = self.out_conv(out)
        shift = h[:, 0::2]
        scale = (2*self.softsign(h[:, 1::2])).exp()

        x2 = x2 + shift
        x2 = x2 * scale
        logdet = torch.abs(scale).log().view(x.shape[0], -1).sum(1)

        return torch.cat((x1, x2), 1), logdet, states_out

    def reverse(self, y, cond, rec_states=None):
        """Backward pass

        :param y:  [B, in_features, H, W] input feature tensor
        :type y: torch.Tensor
        :param cond: [B, in_features, H, W] input feature tensor
        :type cond: torch.Tensor
        :param rec_states: tuple of LSTM states (hidden state, cell state), defaults to None
        :type rec_states: tuple, optional
        :returns: 
            - x: [B, in_features, H, W] Output feature tensor
            - logdet: log determinate of affine layer
            - states_out: tuple of LSTM (cell, hidden) states
        :rtype: (torch.Tensor, torch.Tensor, tuple)
        """
        #  Split in the channel dimension
        y1, y2 = y.chunk(2, 1)

        if(rec_states is None):
            out, h_next, c_next = self.resid_lstm(torch.cat((y1, cond), 1), None)
            # Store lstm states for next time-step and flow conditions
            states_out = (h_next, c_next)
        else:
            out, h_next, c_next = self.resid_lstm(torch.cat((y1, cond), 1), rec_states)
            # Store lstm states for next time-step and flow conditions
            states_out = (h_next, c_next)

        out = self.dense_nn(out)
        h = self.out_conv(out)
        shift = h[:, 0::2]
        scale = (2*self.softsign(h[:, 1::2])).exp()

        y2 = y2 / scale
        y2 = y2 - shift
        logdet = torch.abs(scale).log().view(y.shape[0], -1).sum(1)

        return torch.cat((y1, y2), 1), logdet, states_out