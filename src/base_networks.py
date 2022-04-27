'''
# This file is provided by our supervisor, Dr Iman Yi Liao.
'''

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision
#from src import const


class CustomUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, last_act='sigmoid'):
        super(CustomUnetGenerator, self).__init__()

        # construct unet structure
        innermost_nc = 2 ** num_downs
        unet_block = UnetSkipConnectionBlock(ngf * innermost_nc, ngf * innermost_nc, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True, keep_size=True)
        for i in range(num_downs):
            k = num_downs - i
            unet_block = UnetSkipConnectionBlock(ngf * (2 ** (k - 1)), ngf * (2 ** k), input_nc=None,
                                                 submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, last_act=last_act, keep_size=True)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 last_act='sigmoid', keep_size=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        if keep_size:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=1,
                             stride=1, padding=0, bias=use_bias)
        else:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if keep_size:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=1, stride=1,
                                            padding=0)
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            down = [downconv]
            if last_act == 'tanh':
                up = [uprelu, upconv, nn.Tanh()]
            elif last_act == 'sigmoid':
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                raise NotImplementedError
            model = down + [submodule] + up
        elif innermost:
            if keep_size:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=1, stride=1,
                                            padding=0)
            else:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if keep_size:
                raise Exception("can not keep size")
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            #print(x.shape,self.model(x).shape)
            return torch.cat([x, self.model(x)], 1)