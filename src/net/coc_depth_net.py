import torch
from torch import nn

from net.pool_net import PoolNet, PoolNetEncoder, PoolNetDecoder, PoolNetConvNet
from net.modules import MultiDecoderNet, LayerSplitModule
from tools.tools import module_flat_str


class FgbgCocMultiDecoderPoolNet(MultiDecoderNet):
    def __init__(self, enc_sizes=None, dec_sizes=None, bn_eps=None, last_act=False, bias=False, act_func=nn.ReLU(),
                 enc_pool_layers=None, dec_pool_layers=None, use_coc_cc=None, use_depth_cc=None,
                 in_channels=3, out_channels=1, final_conv_sizes=None):
        if final_conv_sizes is None:
            final_conv_sizes = [enc_sizes[0]]

        encoder = PoolNetEncoder(in_channels, enc_sizes, bn_eps, bias, act_func, pool_layers=enc_pool_layers)

        decoders = [nn.Sequential(
            PoolNetDecoder(enc_sizes[-1], dec_sizes, enc_sizes, bn_eps, bias, act_func,
                           pool_layers=dec_pool_layers, use_enc_cc=[use_coc_cc, use_depth_cc][i]),
            PoolNetConvNet(dec_sizes[-1], out_channels, final_conv_sizes, bn_eps, bias, act_func, last_act and not i == 0))
            for i in range(2)
        ]

        super().__init__(encoder, decoders)


class CoCFgbgLayeredPoolNet(nn.Module):
    def __init__(self, last_act=False, **kwargs):
        super().__init__()

        self.model = LayerSplitModule(PoolNet(out_channels=2, last_act=False, **kwargs))

        # self.fgbg_act = nn.Sigmoid()
        self.coc_act = nn.ReLU() if last_act else None

    def get_output_mode(self, i=None):
        return self.model.model.get_output_mode(i)

    def forward(self, x):
        fgbg, coc = self.model(x)

        if self.coc_act is not None:
            coc = self.coc_act(coc)

        return coc, fgbg


class CoCDepthLayeredNet(PoolNet):
    def __init__(self, **kwargs):
        super().__init__(out_channels=2, **kwargs)

    def forward(self, x):
        x = super().forward(x)

        if len(x.shape) == 5:
            return x[:, :, 0:1], x[:, :, 1:2]
        else:
            return x[:, 0:1], x[:, 1:2]


class CoCDepthEncShareNet(nn.Module):
    def __init__(self, enc_sizes=None, dec_sizes=None, final_conv_sizes=None, bn_eps=None, last_act=False, bias=False,
                 act_func=nn.ReLU(),
                 enc_pool_layers=None, dec_pool_layers=None, use_coc_cc=None, use_depth_cc=None,
                 in_channels=3, out_channels=1):
        super().__init__()

        if final_conv_sizes is None:
            final_conv_sizes = [enc_sizes[0]]

        self.encoder = PoolNetEncoder(in_channels, enc_sizes, bn_eps, bias, act_func, pool_layers=enc_pool_layers)

        self.decoders = nn.ModuleList(nn.Sequential(
            PoolNetDecoder(enc_sizes[-1], dec_sizes, enc_sizes, bn_eps, bias, act_func,
                           pool_layers=dec_pool_layers, use_enc_cc=[use_coc_cc, use_depth_cc][i]),
            PoolNetConvNet(dec_sizes[-1], out_channels, final_conv_sizes, bn_eps, bias, act_func, last_act))
            for i in range(2)
        )

        self.apply(PoolNet.init_weights)

    def get_dec_pool_layers(self, i=0):
        return self.decoders[i][0].pool_layers

    def get_output_mode(self, i=0):
        return "all" if not self.get_dec_pool_layers(i)[-1] else "middle"

    def forward(self, x):
        encoding = self.encoder(x)

        x_coc = self.decoders[0](encoding)
        x_depth = self.decoders[1](encoding)

        return x_coc, x_depth

    def flat_str(self, module_filter=None):
        text = ""

        text += "Encoder:\n"
        text += module_flat_str(self.encoder, module_filter)

        text += "\n\nDecoder CoC:\n"
        text += module_flat_str(self.decoders[0], module_filter)

        text += "\n\nDecoder Depth:\n"
        text += module_flat_str(self.decoders[1], module_filter)

        return text
