import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from tools import json_support
from tools.tools import module_flat_str
from net.modules import MaxReduce


def create_layer(in_channels, out_channels, kernel_size, bn_eps, act_func, bias, stride=1, padding=0, use_act=True,
                 use_transp_conv=False):
    modules = []

    modules.append(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if not use_transp_conv else
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=False)
    )

    if bn_eps is not None:
        modules.append(nn.BatchNorm2d(out_channels, eps=bn_eps))

    if use_act:
        modules.append(act_func)

    return nn.Sequential(*modules)


class ConvCat(nn.Module):
    def __init__(self, in_channels_list, out_channels, bn_eps, bias, act_func):
        super().__init__()

        self.layer = create_layer(np.sum(in_channels_list), out_channels, kernel_size=1, bn_eps=bn_eps, bias=bias,
                                  act_func=act_func)

    def _conv_cat(self, tensors, dim=1):
        return self.layer(torch.cat([x for x in tensors if x is not None], dim=dim))

    def forward(self, x_conv, x_pool, x_enc=None):
        num_frames = x_conv.shape[1]
        x_conv_res = torch.stack([
            self._conv_cat([x_conv[:, f], x_pool] + ([x_enc[:, f]] if x_enc is not None else []))
            for f in range(num_frames)],
            dim=1
        )

        return x_conv_res


@json_support
class PoolNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_eps, concat_dim, bias, pool_layer, act_func,
                 conv_kernel=4, conv_stride=2, conv_pad=1, reduce=MaxReduce()):
        super().__init__()

        self.convcat = ConvCat([in_channels] + concat_dim, out_channels, bn_eps=bn_eps, bias=bias, act_func=act_func) \
            if concat_dim is not None else None

        self.layer = create_layer(out_channels if concat_dim is not None else in_channels, out_channels, bias=bias, bn_eps=bn_eps,
                                  kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad, act_func=act_func)

        self.reduce = reduce if pool_layer else None

    # x         [B, F, C, H, W]
    # x_pool    [B, C, H, W]
    def forward(self, x_conv, x_pool):
        batch_size, num_frames = x_conv.shape[:2]

        if self.convcat is not None:
            x_conv = self.convcat(x_conv, x_pool)

        x_conv = x_conv.view(batch_size * num_frames, *x_conv.shape[2:])
        x_conv = self.layer(x_conv)
        x_conv = x_conv.view(batch_size, num_frames, *x_conv.shape[1:])

        # global max pool
        x_pool = self.reduce(x_conv) if self.reduce else None

        return x_conv, x_pool


@json_support
class PoolNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_eps, concat_dim, bias, use_transp_conv, act_func, pool_layer=True,
                 reduce=MaxReduce()):
        super().__init__()

        self.convcat = ConvCat([in_channels] + concat_dim, in_channels, bn_eps=bn_eps, bias=bias, act_func=act_func) \
            if concat_dim is not None else None

        self.conv_layer = create_layer(in_channels, in_channels if use_transp_conv else out_channels,
                                       kernel_size=3, padding=1, bn_eps=bn_eps, bias=bias,
                                       act_func=act_func)

        self.deconv_layer = create_layer(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                                         use_transp_conv=True, bn_eps=bn_eps, bias=bias,
                                         act_func=act_func) if use_transp_conv else None

        self.reduce = reduce if pool_layer else None

    # x         [B, F, C, H, W]
    # x_pool    [B, C, H, W]
    def forward(self, x_conv, x_pool, x_enc):
        batch_size, num_frames = x_conv.shape[:2]

        if self.convcat is not None:
            x_conv = self.convcat(x_conv, x_pool, x_enc)

        x_conv = x_conv.view(batch_size * num_frames, *x_conv.shape[2:])
        x_conv = self.conv_layer(x_conv)
        if self.deconv_layer is not None:
            x_conv = self.deconv_layer(x_conv)
        x_conv = x_conv.view(batch_size, num_frames, *x_conv.shape[1:])

        x_pool = self.reduce(x_conv) if self.reduce is not None else None

        return x_conv, x_pool


def default_seq_prop(pool_layers, n, val=True):
    if pool_layers is None:
        pool_layers = [val] * n
    elif isinstance(pool_layers, bool):
        pool_layers = [pool_layers] * n

    return pool_layers


@json_support
class PoolNetEncoder(nn.Module):
    dft_sizes = np.array([64, 96, 128, 256, 384])

    def __init__(self, in_channels, enc_sizes, bn_eps, bias, act_func, pool_layers=None, reduce=MaxReduce()):
        super().__init__()

        self.in_channels = in_channels

        block_sizes = [in_channels] + list(enc_sizes)

        pool_layers = default_seq_prop(pool_layers, len(enc_sizes) - 1)
        pool_layers.append(False)

        self.encoder_blocks = nn.ModuleList(*[[
            PoolNetEncoderBlock(block_sizes[0], block_sizes[1], bn_eps=bn_eps,
                                concat_dim=None, bias=bias,
                                conv_kernel=3, conv_stride=1, conv_pad=1,
                                pool_layer=pool_layers[0], act_func=act_func, reduce=reduce)] + [
            PoolNetEncoderBlock(block_sizes[i], block_sizes[i + 1],
                                concat_dim=[block_sizes[i]] if pool_layers[i-1] else None,
                                bn_eps=bn_eps, bias=bias,
                                pool_layer=pool_layers[i],  # no pooling for last encoder block
                                act_func=act_func, reduce=reduce)
            for i in range(1, len(block_sizes) - 1)
        ]])

    def get_in_channels_count(self):
        return self.in_channels

    def get_output_mode(self):
        return None

    def forward(self, x):
        x_conv, x_pool = x, None
        x_enc_cc = []

        for enc_block in self.encoder_blocks:
            x_conv, x_pool = enc_block(x_conv, x_pool)
            x_enc_cc.append(x_conv)

        return x_conv, x_pool, x_enc_cc


@json_support
class PoolNetDecoder(nn.Module):
    dft_sizes = np.array([384, 256, 192, 96, 96])

    def __init__(self, in_channels, dec_sizes, enc_sizes, bn_eps, bias, act_func, use_enc_cc=None, pool_layers=None,
                 reduce=MaxReduce()):
        super().__init__()

        block_sizes = [in_channels] + list(dec_sizes)

        pool_layers = default_seq_prop(pool_layers, len(dec_sizes))
        self.use_enc_cc = default_seq_prop(use_enc_cc, len(dec_sizes))

        concat_dims = [
            None if i == 0 or not self.use_enc_cc[i] else (
                [enc_sizes[-(i + 1)]] if not pool_layers[i-1]
                else [block_sizes[i], enc_sizes[-(i + 1)]]
            )
            for i in range(len(block_sizes) - 1)
        ]

        self.decoder_blocks = nn.ModuleList(
            PoolNetDecoderBlock(block_sizes[i], block_sizes[i + 1], bn_eps=bn_eps, bias=bias,
                                concat_dim=concat_dims[i], use_transp_conv=i < len(block_sizes) - 2,
                                act_func=act_func, pool_layer=pool_layers[i], reduce=reduce)
            for i in range(len(block_sizes) - 1)
        )

        self.pool_layers = pool_layers

    def get_output_mode(self):
        return "all" if not self.pool_layers[-1] else "last"

    def get_skip_conn_usage(self):
        return self.use_enc_cc

    def forward(self, encoding):
        x_conv, x_pool, x_enc_cc = encoding

        for dec_block, x_enc in zip(self.decoder_blocks, reversed(x_enc_cc)):
            x_conv, x_pool = dec_block(x_conv, x_pool, x_enc)

        x = x_pool if x_pool is not None else x_conv

        return x


class PoolNetConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, bn_eps, bias, act_func, last_act):
        super().__init__()

        self.out_channels = out_channels

        self.conv_net = nn.Sequential(*[
            create_layer(cin, cout, kernel_size=3, padding=1, bn_eps=bn_eps, bias=bias,
                         act_func=act_func)
            for cin, cout in zip([in_channels] + inter_channels[:-1], inter_channels)],
            create_layer(inter_channels[-1], out_channels, kernel_size=3, padding=1, use_act=last_act, bn_eps=bn_eps,
                         bias=bias, act_func=act_func)
        )

    def get_out_channels_count(self):
        return self.out_channels

    def forward(self, x):
        org_dim = None

        multi_out = len(x.shape) == 5

        if multi_out:
            org_dim = x.shape[:2]
            x = x.view(np.prod(org_dim), *x.shape[2:])

        #print(x.shape)
        x = self.conv_net(x)

        if multi_out:
            x = x.view(*org_dim, *x.shape[1:])

        return x


@json_support
class PoolNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.encoder = PoolNet.create_component("encoder", *args, **kwargs)
        self.decoder = PoolNet.create_component("decoder_only", *args, **kwargs)
        self.conv_net = PoolNet.create_component("final", *args, **kwargs)

        self.apply(PoolNet.init_weights)

    @staticmethod
    def create_component(comp_type, enc_sizes=None, dec_sizes=None, final_conv_sizes=None, bn_eps=1e-4, last_act=False,
                         bias=False, act_func=nn.ReLU(), enc_pool_layers=None, dec_pool_layers=False, use_enc_cc=None,
                         in_channels=3, out_channels=1, reduce=MaxReduce()):
        if comp_type == "decoder":
            return nn.Sequential(
                PoolNet.create_component(
                    "decoder_only", enc_sizes, dec_sizes, final_conv_sizes, bn_eps, last_act,
                    bias, act_func, enc_pool_layers, dec_pool_layers, use_enc_cc,
                    in_channels, out_channels, reduce
                ),
                PoolNet.create_component(
                    "final", enc_sizes, dec_sizes, final_conv_sizes, bn_eps, last_act,
                    bias, act_func, enc_pool_layers, dec_pool_layers, use_enc_cc,
                    in_channels, out_channels, reduce
                )
            )
        else:
            if comp_type == "encoder":
                comp = PoolNetEncoder(in_channels, enc_sizes, bn_eps, bias, act_func, pool_layers=enc_pool_layers, reduce=reduce)
            elif comp_type == "decoder_only":
                comp = PoolNetDecoder(enc_sizes[-1], dec_sizes, enc_sizes, bn_eps, bias, act_func,
                                      pool_layers=dec_pool_layers, use_enc_cc=use_enc_cc, reduce=reduce)
            elif comp_type == "final":
                if final_conv_sizes is None:
                    final_conv_sizes = [enc_sizes[0]]

                comp = PoolNetConvNet(dec_sizes[-1], out_channels, final_conv_sizes, bn_eps, bias, act_func,
                                      last_act)
            else:
                raise Exception()

            return comp

    def get_in_channels_count(self):
        return self.encoder.get_in_channels_count()

    def get_out_channels_count(self):
        return self.conv_net.get_out_channels_count()

    def get_output_mode(self, i=None):
        return self.decoder.get_output_mode()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(0, 1.0)
            m.running_var.normal_(0, 1.0)
            m.running_mean.fill_(0)
            m.bias.data.fill_(0)

    def forward(self, x):
        encoding = self.encoder(x)
        x = self.decoder(encoding)

        x = self.conv_net(x)

        return x

    def flat_str(self, module_filter=None):
        text = ""

        text += "Encoder:\n"
        text += module_flat_str(self.encoder, module_filter)

        text += "\n\nDecoder:\n"
        text += module_flat_str(self.decoder, module_filter)

        text += "\n\nConvNet:\n"
        text += module_flat_str(self.conv_net, module_filter)

        return text
