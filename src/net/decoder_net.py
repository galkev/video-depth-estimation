import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from tools.tools import json_support

# <class 'torch.nn.modules.conv.Conv3d'> in: torch.Size([1, 3, 16, 224, 224]) out: torch.Size([1, 64, 16, 112, 112]) -> 64/3*c 0.5wh
# <class 'torch.nn.modules.batchnorm.BatchNorm3d'> in: torch.Size([1, 64, 16, 112, 112]) out: torch.Size([1, 64, 16, 112, 112])
# <class 'torch.nn.modules.activation.ReLU'> in: torch.Size([1, 64, 16, 112, 112]) out: torch.Size([1, 64, 16, 112, 112])
# <class 'torch.nn.modules.pooling.MaxPool3d'> in: torch.Size([1, 64, 16, 112, 112]) out: torch.Size([1, 64, 8, 56, 56]) -> 0.5d

# <class 'net.cnn3d.densenet._DenseBlock'> in: torch.Size([1, 64, 8, 56, 56]) out: torch.Size([1, 256, 8, 56, 56]) -> 4c
# <class 'net.cnn3d.densenet._Transition'> in: torch.Size([1, 256, 8, 56, 56]) out: torch.Size([1, 128, 4, 28, 28]) -> 0.5c 0.5d
# <class 'net.cnn3d.densenet._DenseBlock'> in: torch.Size([1, 128, 4, 28, 28]) out: torch.Size([1, 512, 4, 28, 28]) ->4c
# <class 'net.cnn3d.densenet._Transition'> in: torch.Size([1, 512, 4, 28, 28]) out: torch.Size([1, 256, 2, 14, 14]) ->0.5c 0.5d
# <class 'net.cnn3d.densenet._DenseBlock'> in: torch.Size([1, 256, 2, 14, 14]) out: torch.Size([1, 1024, 2, 14, 14]) -> 4c
# <class 'net.cnn3d.densenet._Transition'> in: torch.Size([1, 1024, 2, 14, 14]) out: torch.Size([1, 512, 1, 7, 7]) -> 0.5c 0.5d
# <class 'net.cnn3d.densenet._DenseBlock'> in: torch.Size([1, 512, 1, 7, 7]) out: torch.Size([1, 1024, 1, 7, 7]) -> 2c


@json_support
class DecoderBlock3D(nn.Module):
    # inshape [C T H W]
    def __init__(self, in_channels, out_channels, use_concat=False, use_transp_conv=False, res_increase=2,
                 kernel_sizes=None, use_2d=False):
        super().__init__()

        self.use_concat = use_concat
        self.use_transp_conv = use_transp_conv
        self.res_increase = res_increase
        self.use_2d = use_2d

        if self.use_transp_conv:
            self.deconv = self.create_conv_transp(in_channels, in_channels, kernel_size=2 * self.res_increase,
                                                  stride=self.res_increase, padding=1, bias=False)

        if kernel_sizes is None:
            kernel_sizes = [1, 3]

        # TODO: Decide if 1 or 3
        self.conv1 = self.create_conv(in_channels * (2 if self.use_concat else 1), in_channels,
                                      kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, bias=False)
        self.norm1 = self.create_norm(in_channels, in_channels)

        self.conv2 = self.create_conv(in_channels, out_channels,
                                      kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2, bias=False)
        self.norm2 = self.create_norm(out_channels, out_channels)

    def create_conv_transp(self, *args, **kwargs):
        return nn.ConvTranspose3d(*args, **kwargs) if not self.use_2d else nn.ConvTranspose2d(*args, **kwargs)

    def create_conv(self, *args, **kwargs):
        return nn.Conv3d(*args, **kwargs) if not self.use_2d else nn.Conv2d(*args, **kwargs)

    def create_norm(self, *args, **kwargs):
        return nn.BatchNorm3d(*args, **kwargs) if not self.use_2d else nn.BatchNorm2d(*args, **kwargs)

    def upscale(self, x):
        if self.use_transp_conv:
            x = self.deconv(x)
        else:
            if not self.use_2d:
                x = F.interpolate(x, scale_factor=self.res_increase, mode='trilinear', align_corners=True)
            else:
                x = F.interpolate(x, scale_factor=self.res_increase, mode='bilinear', align_corners=True)

        return x

    def forward(self, x, ccx=None):
        x = self.upscale(x)

        if self.use_concat:
            if self.use_2d:
                # TODO: improve
                ccx = ccx[:, :, ccx.shape[2] // 2]
            x = torch.cat([x, ccx], 1)

        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))

        return x


@json_support
class DecoderNet3D(nn.Module):
    def __init__(self, channels, use_concat, use_transp_conv, decoder_conv_kernel_sizes=None, use_2d=False):
        super().__init__()

        self.use_transp_conv = use_transp_conv
        self.use_2d = use_2d

        num_blocks = len(channels) - 1

        self.decoder_blocks = nn.Sequential(*[
            DecoderBlock3D(in_channels=channels[i],
                           out_channels=channels[i+1],
                           use_concat=use_concat[i],
                           use_transp_conv=use_transp_conv,
                           kernel_sizes=decoder_conv_kernel_sizes,
                           use_2d=use_2d)
            for i in range(num_blocks)
        ])

        if self.use_transp_conv:
            if not self.use_2d:
                self.deconv = nn.ConvTranspose3d(
                    channels[-1], channels[-1], kernel_size=(1, 4, 4),
                    stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
            else:
                self.deconv = nn.ConvTranspose2d(
                    channels[-1], channels[-1], kernel_size=4,
                    stride=2, padding=1, bias=False
                )

    def upscale(self, x):
        if self.use_transp_conv:
            x = self.deconv(x)
        else:
            if not self.use_2d:
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            else:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        return x

    def forward(self, x, ccx):
        #x = self.decoder_blocks(x)

        if self.use_2d:
            # remove time dimension
            x = x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4])

        print(x.size())
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, ccx[-(i+1)] if i < len(ccx) else None)
            #print(x.size())

        x = self.upscale(x)

        return x
