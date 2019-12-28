import torch
from torch import nn
from net.cnn3d.model import make_densenet121
from net.decoder_net import DecoderNet3D

from tools.tools import json_support


@json_support
class DenseNet3DAutoEncoder(nn.Module):
    def __init__(self, pretrained_root, use_concat=None, use_transp_conv=False, decoder_conv_kernel_sizes=None,
                 use_2d_dec=False, depth_for_all=False, load_pretrained=True):
        super().__init__()
        decoder_channels = [1024, 512, 256, 64, 1]  # last 1 is 3 instead of 1
        num_blocks = len(decoder_channels) - 1
        out_seq_length = 2 ** num_blocks

        if use_concat is None:
            use_concat = [False, False, True, False]

        self.depth_for_all = depth_for_all

        if depth_for_all and use_2d_dec:
            raise Exception("Error cant combine depth for all and 2D mode")

        self.encoder = make_densenet121(pretrained_root, load_pretrained=load_pretrained)
        self.decoder = DecoderNet3D(decoder_channels, use_concat, use_transp_conv,
                                    decoder_conv_kernel_sizes=decoder_conv_kernel_sizes, use_2d=use_2d_dec)

        if not self.decoder.use_2d:
            self.scoring = nn.Conv2d(out_seq_length * decoder_channels[-1], 1, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # encoder expects channels first and then frames

        x, ccx = self.encoder(x)
        x = self.decoder(x, ccx)

        if not self.decoder.use_2d:
            if not self.depth_for_all:
                x = x.view(x.shape[0], -1, *x.shape[3:])
                x = self.scoring(x)
            else:
                x = x.permute(0, 2, 1, 3, 4)

        return x
