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

class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck 
    design

    :param in_features: Number of input feature channels
    :type in_features: int
    :param growth_rate: Growth rate of channels in dense block
    :type growth_rate: int
    :param drop_rate: dropout rate, defaults to 0.
    :type drop_rate: float, optional
    :param bn_size: number of features after bottleneck if enabled, defaults to 8
    :type bn_size: int, optional
    :param bottleneck: enable bottle next to reduce the number of feature channels, defaults to False
    :type bottleneck: bool, optional
    :param padding: convolutional padding, defaults to 1
    :type padding: int, optional
    """  
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False, padding=1):
        """Constructor method
        """ 
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=False))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))

            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=False))
            self.add_module('conv2', nn.Conv2d(in_features, growth_rate,
                            kernel_size=(2*padding+1), stride=1, padding=padding, 
                            bias=False, padding_mode='zeros'))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=False))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=(2*padding+1), stride=1, padding=padding, 
                            bias=False, padding_mode='zeros'))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout3d(p=drop_rate))
        
    def forward(self, x):
        """Forward pass

        :param x: [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :returns: 
            - out: Output tensor of dense layer 
        :rtype: torch.Tensor
        """
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)

class DenseBlock(nn.Sequential):
    """Dense block with bottleneck design
    See Fig. 9 of paper: https://arxiv.org/abs/2006.04731

    :param num_layers: Number of dense layers in block
    :type num_layers: int
    :param in_features: Number of input feature channels
    :type in_features: int
    :param growth_rate: Growth rate of channels in dense block
    :type growth_rate: int
    :param drop_rate: Dropout rate, defaults to 0.
    :type drop_rate: float, optional
    :param bn_size: Number of features after bottleneck if enabled, defaults to 8
    :type bn_size: int, optional
    :param bottleneck: Enable bottle next to reduce the number of feature channels, defaults to False
    :type bottleneck: bool, optional
    :param padding: Convolutional padding, defaults to 1
    :type padding: int, optional

    :note: For additional information on dense blocks see "Densely Connected Convolutional Networks"
        by Huang et al. https://arxiv.org/abs/1608.06993v5
    """  
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        """Constructor method
        """ 
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)

class _DenseLayerNoNorm(nn.Sequential):
    """One dense layer within dense block, with bottleneck 
    design with no normalization

    :param in_features: Number of input feature channels
    :type in_features: int
    :param growth_rate: Growth rate of channels in dense block
    :type growth_rate: int
    :param drop_rate: dropout rate, defaults to 0.
    :type drop_rate: float, optional
    :param bn_size: number of features after bottleneck if enabled, defaults to 8
    :type bn_size: int, optional
    :param bottleneck: enable bottle next to reduce the number of feature channels, defaults to False
    :type bottleneck: bool, optional
    :param padding: convolutional padding, defaults to 1
    :type padding: int, optional
    """ 
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False, padding=1):
        """Constructor method
        """ 
        super(_DenseLayerNoNorm, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('relu1', nn.ReLU(inplace=False))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                            growth_rate, kernel_size=(2*padding+1), stride=1, bias=False))

            # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=False))
            self.add_module('conv2', nn.Conv2d(in_features, growth_rate,
                            kernel_size=(2*padding+1), stride=1, padding=padding, 
                            bias=False, padding_mode='zeros'))
        else:
            self.add_module('relu1', nn.ReLU(inplace=False))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=(2*padding+1), stride=1, padding=padding, 
                            bias=False, padding_mode='zeros'))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout3d(p=drop_rate))
        
    def forward(self, x):
        """Forward pass

        :param x: [B, in_features, H, W] input feature tensor
        :type x: torch.Tensor
        :returns:
            - out: Output tensor of dense layer 
        :rtype: torch.Tensor
        """
        y = super(_DenseLayerNoNorm, self).forward(x)
        return torch.cat([x, y], 1)

class NoNormDenseBlock(nn.Sequential):
    """Dense block with bottleneck design and no batch-normalization
    See Fig. 9 of paper: https://arxiv.org/abs/2006.04731

    :param num_layers: Number of dense layers in block
    :type num_layers: int
    :param in_features: Number of input feature channels
    :type in_features: int
    :param growth_rate: Growth rate of channels in dense block
    :type growth_rate: int
    :param drop_rate: dropout rate, defaults to 0.
    :type drop_rate: float, optional
    :param bn_size: number of features after bottleneck if enabled, defaults to 8
    :type bn_size: int, optional
    :param bottleneck: enable bottle next to reduce the number of feature channels, defaults to False
    :type bottleneck: bool, optional
    :param padding: convolutional padding, defaults to 1
    :type padding: int, optional

    :note: For additional information on dense blocks see "Densely Connected Convolutional Networks"
        by Huang et al. https://arxiv.org/abs/1608.06993v5
    """ 
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        """Constructor method
        """ 
        super(NoNormDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayerNoNorm(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)