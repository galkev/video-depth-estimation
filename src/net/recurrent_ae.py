import torch
from torch import nn
import torch.nn.functional as F
from tools.tools import alternating_range, json_support
from net.modules import ConvReduce
from data import VideoDepthFocusData
from tools.tools import type_adv


def create_layer(in_channels, out_channels, bn_eps=1e-4, bias=False, act=nn.ReLU(inplace=True)):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
    ]

    if bn_eps is not None:
        layers.append(nn.BatchNorm2d(out_channels, eps=bn_eps))

    if act is not None:
        layers.append(act)

    return nn.Sequential(*layers)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=True, *args, **kwargs):
        super().__init__()

        self.pool = nn.MaxPool2d(2) if use_pool else None

        self.layer1 = create_layer(in_channels, out_channels, *args, **kwargs)
        self.layer2 = create_layer(2*out_channels, out_channels, *args, **kwargs)
        self.layer3 = create_layer(out_channels, out_channels, *args, **kwargs)

    def forward(self, x, hidden):
        if self.pool is not None:
            x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(torch.cat([x, hidden], dim=1))
        x = self.layer3(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, concat_enc_channels, use_transp_conv=True, bias=False,
                 bn_eps=1e-4, act=nn.ReLU(inplace=True), last_act=True, last_bn=True, *args, **kwargs):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, concat_enc_channels, kernel_size=2, stride=2, bias=bias) \
            if use_transp_conv else None

        self.layers = nn.Sequential(
            create_layer(2 * concat_enc_channels if use_transp_conv else in_channels, out_channels, bias=bias,
                         act=act, bn_eps=bn_eps, *args, **kwargs),
            create_layer(out_channels, out_channels, bias=bias,
                         act=act if last_act else None,
                         bn_eps=bn_eps if last_bn else None,
                         *args, **kwargs)
        )

    def forward(self, x, x_cat):
        if self.up is not None:
            x = self.up(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = self.layers(torch.cat([x, x_cat], dim=1))

        return x


class RecurrentAEEncoder(nn.Module):
    dft_sizes = [32, 43, 57, 76, 101]

    def __init__(self, feat_sizes=None, in_channels=3, alternate=False, output_all=False, *args, **kwargs):
        super().__init__()

        if feat_sizes is None:
            feat_sizes = RecurrentAEEncoder.dft_sizes

        self.enc_sizes = feat_sizes + [feat_sizes[-1]]

        conv_sizes = [in_channels] + self.enc_sizes

        blocks = [
            EncoderBlock(conv_sizes[i], conv_sizes[i + 1], use_pool=i > 0, *args, **kwargs)
            for i in range(len(conv_sizes) - 1)
        ]

        self.blocks = nn.Sequential(*blocks)

        self.alternate = alternate
        self.output_all = output_all

    def get_channel_sizes(self):
        return self.enc_sizes

    def get_feature_map_sizes(self, i, in_dim=None):
        if in_dim is None:
            in_dim = VideoDepthFocusData.crop_size

            if isinstance(in_dim, int):
                in_dim = [in_dim, in_dim]
            else:
                in_dim = [in_dim[1], in_dim[0]]

        return [self.enc_sizes[i]] + [d // 2 ** i for d in in_dim]

    def get_out_dim(self):
        return self.get_feature_map_sizes(len(self.blocks)-1)

    def create_hidden(self, batch_size, in_dim=None):
        dev = next(self.parameters()).device
        return [torch.zeros([batch_size] + self.get_feature_map_sizes(i, in_dim=in_dim), device=dev)
                for i in range(len(self.blocks))]

    def _encode(self, x, hidden):
        new_hidden = []

        for i, block in enumerate(self.blocks):
            x = block(x, hidden[i])
            new_hidden.append(x)

        return x, new_hidden

    def _get_fwd_order(self, stop, reverse):
        if not self.alternate:
            indices = range(stop)
        else:
            assert stop % 2 == 1
            indices = alternating_range(stop)

        if reverse:
            indices = reversed(indices)

        return indices

    def set_attr(self, k, v):
        if k == "output_all":
            self.output_all = v
        else:
            raise Exception()

    def get_output_mode(self):
        if self.output_all:
            return "all"
        elif self.alternate:
            return "middle"
        else:
            return "last"

    def forward(self, x, hidden=None, reverse=False):
        if hidden is None:
            hidden = self.create_hidden(x.shape[0], x.shape[-2:])  # x.shape[3] if len(x.shape) == 5 else x.shape[2]

        if len(x.shape) == 5:
            out_list = []

            for i in self._get_fwd_order(x.shape[1], reverse):
                x_enc, hidden = self._encode(x[:, i], hidden)

                if self.output_all:
                    out_list.append([x_enc, hidden])

            if self.output_all:
                num_blocks = len(out_list[0][1])

                x_enc = [o[0] for o in out_list]
                hidden_blocks = [[o[1][block_idx] for o in out_list] for block_idx in range(num_blocks)]

                # reverse back to get the last frame last
                # TODO: overthink if really doing it here
                if reverse:
                    x_enc = list(reversed(x_enc))
                    hidden_blocks = [list(reversed(hidden_block)) for hidden_block in hidden_blocks]

                x_enc = torch.stack(x_enc, dim=1)
                hidden = [torch.stack(hidden_block, dim=1) for hidden_block in hidden_blocks]
        else:
            x_enc, hidden = self._encode(x, hidden)

        return x_enc, hidden


class RecurrentAEDecoder(nn.Module):
    dft_sizes = list(reversed(RecurrentAEEncoder.dft_sizes))

    def __init__(self, feat_sizes=None, enc_feat_sizes=None, out_channels=1, last_act=False, last_bn=False,
                 use_transp_conv=True, dec_add_input=None, *args, **kwargs):
        super().__init__()

        if feat_sizes is None:
            feat_sizes = RecurrentAEDecoder.dft_sizes

        if enc_feat_sizes is None:
            enc_feat_sizes = RecurrentAEEncoder.dft_sizes

        enc_feat_sizes = enc_feat_sizes + [enc_feat_sizes[-1]]
        dec_sizes = list(feat_sizes) + [out_channels]

        if dec_add_input is not None:
            dec_sizes[0] += dec_add_input

        self.decoder_blocks = nn.Sequential(*[
            DecoderBlock(dec_sizes[i], dec_sizes[i + 1], enc_feat_sizes[-(i+2)],
                         last_act=True if i < len(dec_sizes) - 2 else last_act,
                         last_bn=True if i < len(dec_sizes) - 2 else last_bn,
                         use_transp_conv=use_transp_conv,
                         *args, **kwargs)
            for i in range(len(dec_sizes) - 1)
        ])

    def get_skip_conn_usage(self):
        return [True] * len(self.decoder_blocks)

    def forward(self, x, enc_hidden):
        x_cat_list = enc_hidden[:-1]

        multi_out = len(x.shape) == 5

        if multi_out:
            batch_size, seq_length = x.shape[:2]
            x = x.view(batch_size * seq_length, *x.shape[2:])
            x_cat_list = [x_cat.view(batch_size * seq_length, *x_cat.shape[2:]) for x_cat in x_cat_list]

        x_cat_list = list(reversed(x_cat_list))

        for block, x_cat in zip(self.decoder_blocks, x_cat_list):
            x = block(x, x_cat)

        if multi_out:
            x = x.view(batch_size, seq_length, *x.shape[1:])

        return x


# alternate -> predict middle frame by giving frames alternatingly from start and end
@json_support
class RecurrentAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.encoder = RecurrentAE.create_component("encoder", *args, **kwargs)
        self.decoder = RecurrentAE.create_component("decoder", *args, **kwargs)

    def get_output_mode(self):
        return self.encoder.get_output_mode()

    @staticmethod
    def create_component(comp_type, enc_sizes=None, dec_sizes=None, in_channels=3, out_channels=1,
                         bn_eps=1e-4, bias=False, last_act=False, last_bn=False, act=nn.ReLU(inplace=True),
                         use_transp_conv=True, alternate=False, output_all=False, dec_add_input=0):
        if enc_sizes is not None and dec_sizes is None:
            dec_sizes = list(reversed(enc_sizes))

        if comp_type == "encoder":
            comp = RecurrentAEEncoder(enc_sizes, in_channels, bn_eps=bn_eps, bias=bias, act=act, alternate=alternate,
                                      output_all=output_all)
        elif comp_type == "decoder":
            comp = RecurrentAEDecoder(dec_sizes, enc_sizes, out_channels=out_channels, bn_eps=bn_eps, bias=bias,
                                      last_act=last_act, last_bn=last_bn, act=act, use_transp_conv=use_transp_conv,
                                      dec_add_input=dec_add_input)
        else:
            raise Exception()

        return comp

    def set_attr(self, k, v):
        if k == "output_all":
            self.encoder.output_all = v
        else:
            raise Exception()

    # B F C H W
    def forward(self, x, hidden=None, output_hidden=False, reverse=False, encoding_modifier=None):
        x_enc, hidden = self.encoder(x, hidden=hidden, reverse=reverse)

        if encoding_modifier is not None:
            x_enc = encoding_modifier(x_enc)

        x = self.decoder(x_enc, hidden)

        if output_hidden:
            return x, hidden
        else:
            return x


@json_support
class BidirRecurrentComposeBase(nn.Module):
    def __init__(self, net_fwd, net_bwd=None, reduce=ConvReduce(seq_length=2, num_channels=1), use_hidden=False):
        super().__init__()

        self.net_fwd = net_fwd
        self.net_bwd = net_bwd
        self.reduce = reduce
        self.use_hidden = use_hidden

    def _foward_seq(self, x, *args, **kwargs):
        return self.net_fwd(x, reverse=False, output_hidden=True, *args, **kwargs)

    def _reverse_seq(self, x, hidden, *args, **kwargs):
        net = self.net_bwd if self.net_bwd is not None else self.net_fwd
        return net(x, reverse=True, hidden=hidden if self.use_hidden else None, *args, **kwargs)

    def _bidirectinal_seq(self, x_fwd, x_bwd=None, **kwargs):
        x_fwd_new, hidden = self._foward_seq(x_fwd, **kwargs)
        x_bwd_new = self._reverse_seq(x_bwd if x_bwd is not None else x_fwd, hidden=hidden, **kwargs)

        return x_fwd_new, x_bwd_new


@json_support
class BidirRecurrentComposeAll(BidirRecurrentComposeBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """
    def get_output_mode(self):
        return "all"

    def forward(self, x):
        x_fwd, x_bwd = self._bidirectinal_seq(x, output_all=True)

        seq_length = x.shape[1]

        x = torch.stack([
            self.reduce(torch.stack([x_fwd[:, i], x_bwd[:, i]], dim=1))
            for i in range(seq_length)
        ], dim=1)

        return x
    """


@json_support
class BidirRecurrentComposeFirstLast(BidirRecurrentComposeBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_output_mode(self):
        return "middle" if self.reduce is not None else "firstlast"

    def forward(self, x):
        x_fwd, x_bwd = self._bidirectinal_seq(x)

        x = torch.stack([x_fwd, x_bwd], dim=1)

        if self.reduce is not None:
            x = self.reduce(x)

        return x


@json_support
class BidirRecurrentComposeCenter(BidirRecurrentComposeBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_output_mode(self):
        return "middle"

    def forward(self, x):
        center_idx = x.shape[1] // 2

        x_fwd, x_bwd = self._bidirectinal_seq(x[:, :center_idx], x[:, center_idx:])

        x = torch.stack([x_fwd, x_bwd], dim=1)
        x = self.reduce(x)

        return x
