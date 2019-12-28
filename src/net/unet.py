import torch
from torch import nn
import torch.nn.functional as F
from data import VideoDepthFocusData


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=True):
        super().__init__()

        self.layers = nn.Sequential(*(([
            nn.MaxPool2d(2)] if use_pool else []) + [
            Layer(in_channels, out_channels),
            Layer(out_channels, out_channels)]
        ))

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, concat_enc_channels, use_transp_conv=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, concat_enc_channels, kernel_size=2, stride=2) \
            if use_transp_conv else None

        self.layers = nn.Sequential(
            Layer(2 * concat_enc_channels if use_transp_conv else in_channels, out_channels),
            Layer(out_channels, out_channels)
        )

    def forward(self, x, x_cat):
        if self.up is not None:
            x = self.up(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = self.layers(torch.cat([x, x_cat], dim=1))

        return x


class UNetEncoder(nn.Module):
    def __init__(self, feat_sizes, in_channels=3):
        super().__init__()

        self.enc_sizes = feat_sizes

        conv_sizes = [in_channels] + feat_sizes

        self.encoder = nn.Sequential(*[
            EncoderBlock(conv_sizes[i], conv_sizes[i + 1], use_pool=i > 0)
            for i in range(len(conv_sizes) - 1)
        ])

    def get_channel_sizes(self):
        return self.enc_sizes

    def get_out_dim(self):
        in_dim = VideoDepthFocusData.crop_size
        return [self.enc_sizes[-1]] + [in_dim // 2**(len(self.encoder) - 1)] * 2

    def forward(self, x):
        x_list = []

        for block in self.encoder:
            x = block(x)
            x_list.append(x)

        return x_list[-1], x_list[:-1]


class UNetDecoder(nn.Module):
    def __init__(self, feat_sizes, enc_feat_sizes, out_channels=1):
        super().__init__()

        dec_sizes = list(feat_sizes)
        enc_feat_sizes = enc_feat_sizes

        self.decoder_blocks = nn.Sequential(*[
            DecoderBlock(dec_sizes[i], dec_sizes[i + 1], enc_feat_sizes[-(i+2)])
            for i in range(len(dec_sizes) - 1)
        ])

        self.last_conv = nn.Conv2d(dec_sizes[-1], out_channels, kernel_size=1)

    def get_skip_conn_usage(self):
        return [True] * len(self.decoder_blocks)

    def forward(self, x, x_cat_list):
        x_cat_list = list(reversed(x_cat_list))

        for block, x_cat in zip(self.decoder_blocks, x_cat_list):
            x = block(x, x_cat)

        return self.last_conv(x)


class UNet(nn.Module):
    def __init__(self, feat_sizes=None, in_channels=3, out_channels=1):
        super().__init__()

        self.encoder, self.decoder = UNet.create_encoder_decoder(feat_sizes,
                                                                 in_channels=in_channels,
                                                                 out_channels=out_channels)

    @staticmethod
    def create_encoder_decoder(feat_sizes=None, in_channels=3, out_channels=1):
        if feat_sizes is None:
            feat_sizes = [64, 128, 256, 512, 1024]

        encoder = UNetEncoder(feat_sizes, in_channels)
        decoder = UNetDecoder(list(reversed(feat_sizes)), feat_sizes, out_channels=out_channels)
        return encoder, decoder

    def forward(self, x):
        x, x_cat_list = self.encoder(x)
        x = self.decoder(x, x_cat_list)
        return x

