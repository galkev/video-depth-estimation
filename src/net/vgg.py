import torch
from torch import nn
from data import VideoDepthFocusData


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


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_eps=None, bias=False, act=True):
        super().__init__()

        seq = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)]

        if bn_eps is not None:
            seq.append(nn.BatchNorm2d(out_channels, bn_eps))

        if act:
            seq.append(nn.ReLU(inplace=True))

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, use_pool=True, bn_eps=None, bias=False):
        super().__init__()

        layers = []

        if use_pool:
            layers.append(nn.MaxPool2d(2))

        layers += [
            Layer(in_channels if i == 0 else out_channels, out_channels, bn_eps=bn_eps, bias=bias)
            for i in range(num_layers)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, skip_channels=None, bn_eps=None, bias=False,
                 last_act=True):
        super().__init__()

        if skip_channels is None:
            skip_channels = 0

        self.concat = skip_channels > 0
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, padding=1, stride=2, bias=bias)

        layers = [
            Layer(in_channels + (skip_channels if i == 0 else 0),
                  out_channels if i == num_layers - 1 else in_channels,
                  bn_eps=bn_eps, bias=bias, act=i < num_layers-1 or last_act)
            for i in range(num_layers)
        ]

        self.layers = nn.Sequential(*layers)

    def uses_skip(self):
        return self.concat

    def forward(self, x, x_cat):
        x = self.up_conv(x)

        if self.concat:
            x = torch.cat([x, x_cat], dim=1)

        x = self.layers(x)

        return x


class VGGEncoder(nn.Module):
    dft_sizes = [64, 128, 256, 512, 512]
    dft_num_layers = [2, 2, 3, 3, 3]

    def __init__(self, in_channels=3, enc_sizes=None, num_layers=None, bn_eps=None, bias=False, final_pool=True):
        super().__init__()

        if enc_sizes is None:
            enc_sizes = VGGEncoder.dft_sizes

        if num_layers is None:
            num_layers = VGGEncoder.dft_num_layers

        blocks = [
            EncoderBlock(in_size, out_size, layers, use_pool=i > 0, bn_eps=bn_eps, bias=bias)
            for i, (in_size, out_size, layers) in enumerate(zip([in_channels] + enc_sizes[:-1], enc_sizes, num_layers))
        ]

        self.enc_sizes = enc_sizes
        self.blocks = nn.ModuleList(blocks)

        self.enc_final_pool = nn.MaxPool2d(2) if final_pool else None

        init_weights(self)

    def get_channel_sizes(self):
        return self.enc_sizes

    def get_out_dim(self):
        in_dim = VideoDepthFocusData.crop_size
        return [self.enc_sizes[-1]] + [in_dim // 2**len(self.blocks)] * 2

    def forward(self, x):
        x_cat_list = []

        for block in self.blocks:
            x = block(x)
            x_cat_list.append(x)

        if self.enc_final_pool:
            x = self.enc_final_pool(x)

        return x, x_cat_list


class VGGDecoder(nn.Module):
    dft_sizes = list(reversed(VGGEncoder.dft_sizes))
    dft_num_layers = list(reversed(VGGEncoder.dft_num_layers))

    def __init__(self, out_channels=1, dec_sizes=None, enc_sizes=None, num_layers=None, use_skip=None,
                 bn_eps=None, bias=False,
                 last_act=False):
        super().__init__()

        if dec_sizes is None:
            dec_sizes = VGGDecoder.dft_sizes

        if enc_sizes is None:
            enc_sizes = list(reversed(dec_sizes))

        if num_layers is None:
            num_layers = VGGDecoder.dft_num_layers

        if use_skip is None:
            use_skip = [True] * len(dec_sizes)

        blocks = [
            DecoderBlock(in_size, out_size, layers, skip_channels=cat_size if skip else None,
                         bn_eps=bn_eps, bias=bias, last_act=i < len(dec_sizes)-1 or last_act)
            for i, (in_size, out_size, cat_size, layers, skip) in enumerate(zip(
                dec_sizes,
                dec_sizes[1:] + [out_channels],
                reversed(enc_sizes),
                num_layers,
                use_skip
            ))
        ]

        self.blocks = nn.ModuleList(blocks)

        init_weights(self)

    def get_skip_conn_usage(self):
        return list(reversed([block.uses_skip() for block in self.blocks]))

    def forward(self, x, x_cat_list):
        for block, x_cat in zip(self.blocks, reversed(x_cat_list)):
            x = block(x, x_cat)

        return x


class VGGAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.encoder, self.decoder = VGGAE.create_encoder_decoder(*args, **kwargs)

        init_weights(self)

    @staticmethod
    def create_encoder_decoder(enc_sizes=None, dec_sizes=None, in_channels=3, out_channels=1, num_layers=None,
                               use_skip=None, bn_eps=1e-4, bias=False, last_act=False):
        if enc_sizes is not None and dec_sizes is None:
            dec_sizes = list(reversed(enc_sizes))

        encoder = VGGEncoder(
            in_channels=in_channels,
            enc_sizes=enc_sizes,
            num_layers=num_layers,
            bn_eps=bn_eps,
            bias=bias,
            final_pool=True
        )

        decoder = VGGDecoder(
            out_channels=out_channels,
            dec_sizes=dec_sizes,
            enc_sizes=enc_sizes,
            num_layers=num_layers,
            use_skip=use_skip,
            bn_eps=bn_eps,
            bias=bias,
            last_act=last_act
        )

        return encoder, decoder

    def forward(self, x):
        x, x_cat_list = self.encoder(x)
        return self.decoder(x, x_cat_list)
