import torch
import torch.nn as nn

from lib.utils import set_norm_layer, set_activate_layer, InstanceNorm, AdaIN

#------------------------------------------------------------------------------------------
# ConvBlock
#   1. Upsample / Conv(padding)
#       - padding options : 'zeros'(default), 'reflect', 'replicate' or 'circular'
#       - if you choose upsample option, you have to set stride==1
#   2. Norm
#       - Norm options : 'bn', 'in', 'adain', 'none'
#   3. activation
#       - activation options : 'relu', 'tanh', 'sig', 'none'
#------------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride=1, upsample=False,
                 pad_type='zeros', conv_padding=0, norm_type='none', activation_type='relu'):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        
        # check upsample
        self.up = nn.Upsample(scale_factor=stride) if upsample else False
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=conv_padding, padding_mode=pad_type,bias=self.use_bias)
        
        # initialize normalization
        self.norm = set_norm_layer(norm_type, output_dim)

        # initialize activation
        self.activation = set_activate_layer(activation_type)

    def forward(self, x):
        if self.up:
            x = self.up(x)

        x = self.conv(x)

        if self.norm:
            x = self.norm(x)

        if self.activation:
            x = self.activation(x)
        return x


#------------------------------------------------------
# ResnetBlock
#------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim, pad_type, norm_type):
        super(ResnetBlock, self).__init__()
        self.convblock1 = ConvBlock(dim ,dim, 3, 1, pad_type=pad_type, conv_padding=1, norm_type=norm_type, activation_type='relu')
        self.convblock2 = ConvBlock(dim ,dim, 3, 1, pad_type=pad_type, conv_padding=1, norm_type=norm_type, activation_type='none')

    def forward(self, x):
        residual_x = self.convblock1(x)
        residual_x = self.convblock2(residual_x)
        return x + residual_x


#------------------------------------------------------
# ResnetBlock_Adain
#   - padding options : 'reflect', 'replicate', 'zero'
#------------------------------------------------------
class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, padding_type = 'reflect', activation='relu', style_dim=256):
        super(ResnetBlock_Adain, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding= 1,\
                                padding_mode=padding_type)
        self.norm1 = InstanceNorm()
        self.style1 = AdaIN(style_dim, dim)
        self.act1 = set_activate_layer(activation)
        
        # Block 2
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding= 1,\
                                padding_mode=padding_type)
        self.norm2 = InstanceNorm()
        self.style2 = AdaIN(style_dim, dim)

    def forward(self, x, dlatents_in_slice):
        # Block 1
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)

        # Block 2
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.style2(y, dlatents_in_slice)

        out = x + y
        return out
