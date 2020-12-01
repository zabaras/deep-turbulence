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
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM from the paper "Convolutional LSTM Network: 
    A Machine Learning Approach for Precipitation Nowcasting" by Shi et al.

    :param input_dim: Number of channels of input tensor
    :type input_dim: int
    :param hidden_dim: Number of channels of hidden state
    :type hidden_dim: int
    :param kernel_size: Size of the convolutional kernel, e.g. (3, 3)
    :type kernel_size: tuple
    :param bias: Use a bias term, defaults to True
    :type bias: bool, optional

    :note: Implementation base is based on https://github.com/ndrplz/ConvLSTM_pytorch
    """  

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """Constructor method
        """        
        super(ConvLSTMCell, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4*self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.h0_in = None
        self.c0_in = None
        

    def forward(self, input_tensor, cur_state):
        """Forward pass of LSTM cell

        :param input_tensor: [B, C, H, W] input feature tensor to the LSTM
        :type input_tensor: torch.Tensor
        :param cur_state: tuple of LSTM states (hidden state, cell state)
        :type cur_state: tuple
        :returns: 
            - h_next: [B, hidden_dim, H, W] Output hidden state of the LSTM cell
            - c_next: [B, hidden_dim, H, W] Output cell state of the LSTM cell
        :rtype: (torch.Tensor, torch.Tensor)
        """
        if(cur_state is None):
            # If no recurrent states are passed 
             h_cur, c_cur = self.init_hidden(input_tensor)
        else:
            h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i*g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, input_tensor):
        """Default initialization for lstm cell and hidden state
        if none are provide. Both are initialized to zero arrays.

        :param input_tensor: input feature to the LSTM
        :type input_tensor: torch.tensor
        :returns: 
            - h_cur: Hidden state of the LSTM cell
            - c_cur: Cell state of the LSTM cell
        :rtype: (torch.Tensor, torch.Tensor)
        """
        dims = list(input_tensor.size())
        dims[1] = self.hidden_dim

        h_cur = Variable(torch.zeros(dims).type(input_tensor.type()))
        c_cur = Variable(torch.zeros(dims).type(input_tensor.type()))

        return h_cur, c_cur


class ResidLSTMBlock(nn.Module):
    """Convolutional LSTM Cell, :class:`nn.modules.convLSTM.ConvLSTMCell`, with a residual connection

    :param input_dim: Number of channels of input tensor
    :type input_dim: int
    :param hidden_dim: Number of channels of hidden state
    :type hidden_dim: int
    :param output_dim: Number of channels of the output of the residual LSTM block
    :type output_dim: int
    :param kernel_size: Size of the convolutional kernel, e.g. (3, 3)
    :type kernel_size: tuple
    :param bias: Use a bias term, defaults to True
    :type bias: bool, optional
    """
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, bias=True):
        """Constructor method
        """  
        super(ResidLSTMBlock, self).__init__()

        self.convLSTM = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

        self.out_seq = nn.Sequential()
        self.out_seq.add_module('LSTM_out_conv', nn.Conv2d(in_channels=hidden_dim+input_dim,
                              out_channels=output_dim,
                              kernel_size=3,
                              padding=1,
                              stride=1))
        self.out_seq.add_module('LSTM_relu2', nn.ReLU())
        # self.out_seq.add_module('dropout', nn.Dropout2d(p=0.05))

    def forward(self, input_tensor, cur_state=None):
        """Forward pass

        :param input_tensor: [B, C, H, W] input feature tensor to the LSTM
        :type input_tensor: torch.Tensor
        :param cur_state: tuple of LSTM states (hidden state, cell state), defaults to None
        :type cur_state: tuple, optional
        :returns: 
            - out: [B, output_dim, H, W] Output of residual connection
            - h_next: [B, hidden_dim, H, W] Output hidden state of the LSTM cell
            - c_next: [B, hidden_dim, H, W] Output cell state of the LSTM cell
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        h_next, c_next = self.convLSTM(input_tensor, cur_state)
        out = torch.cat([input_tensor, h_next], dim=1) # Residual connection around LSTM cell
        return self.out_seq(out), h_next, c_next
