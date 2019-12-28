from torch import nn
import torch.nn.functional as F


class DDFFNetEncoderDummy(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 512, 1),
            nn.MaxPool2d(2, stride=32)
        )

    def get_out_dim(self):
        return [512, 8, 8]

    def get_channel_sizes(self):
        return [64, 128, 256, 512, 512]

    def forward(self, x):
        res = self.seq(x)
        return res, [res] * 5


class DDFFNetDecoderDummy(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(512, 1, 1),
        )

    # for compatibility
    def get_skip_conn_usage(self):
        return [False] * 5


    def forward(self, x, x_cat):
        x = nn.functional.interpolate(x, 256)
        return self.seq(x)


class DDFFDummyTwoDec(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec1 = DDFFNetDecoderDummy()
        self.dec2 = DDFFNetDecoderDummy()

    def forward(self, *args):
        out1 = self.dec1(*args)
        out2 = self.dec2(*args)

        return out1, out2

    # for compatibility
    def get_skip_conn_usage(self):
        return [False] * 5


class DDFFNetDummy(nn.Module):
    def __init__(self, focal_stack_size=10, input_dim=3, output_dim=1, dropout=0.5, bias=False, load_pretrained=True,
                 use_scoring=True):
        super().__init__()

        kernel_size = 3

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size, padding=1)
        self.scoring = nn.Conv2d(focal_stack_size * output_dim, output_dim, 1, bias=False) if use_scoring else None

    def get_output_mode(self):
        return "all" if self.scoring is None else "middle"

    @staticmethod
    def create_encoder_decoder(two_dec=False, *args, **kwargs):
        enc = DDFFNetEncoderDummy()
        dec = DDFFNetDecoderDummy() if not two_dec else DDFFDummyTwoDec()

        return enc, dec

    def forward(self, x):
        batch_size, fs_size, num_channels, img_size = x.shape[0], x.shape[1], x.shape[2], x.shape[3:]

        x = x.view(batch_size * fs_size, num_channels, *img_size)

        x = self.conv1(x)

        if self.scoring:
            x = x.view(batch_size, fs_size, *img_size)
            x = self.scoring(x)
        else:
            x = x.view(batch_size, fs_size, -1, *img_size)

        return x


class DDFFNetCoCDummy(DDFFNetDummy):
    def __init__(self, **kwargs):
        super().__init__(output_dim=2, **kwargs)

    def forward(self, x):
        x = super().forward(x)

        if len(x.shape) == 5:
            return x[:, :, 0:1], x[:, :, 1:2]
        else:
            return x[:, 0:1], x[:, 1:2]


class DDFFNetFgbgCoCDummy(DDFFNetCoCDummy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        fgbg, coc = super().forward(x)

        fgbg = F.sigmoid(fgbg)

        return fgbg, coc
