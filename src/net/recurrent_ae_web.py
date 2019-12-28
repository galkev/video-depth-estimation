import torch.nn as nn
from torch.nn import functional as F
import torch


class RecurrentBlock(nn.Module):
    def __init__(self, input_nc, output_nc, downsampling=False, bottleneck=False, upsampling=False, use_pool=False):
        super(RecurrentBlock, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        self.downsampling = downsampling
        self.upsampling = upsampling
        self.bottleneck = bottleneck

        self.hidden = None
        self.hidden_size = None

        if self.downsampling:
            self.l1 = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            )
            self.l2 = nn.Sequential(
                nn.Conv2d(2 * output_nc, output_nc, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(output_nc, output_nc, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            )
        elif self.upsampling:
            self.l1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(2 * input_nc, output_nc, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(output_nc, output_nc, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            )
        elif self.bottleneck:
            self.l1 = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            )
            self.l2 = nn.Sequential(
                nn.Conv2d(2 * output_nc, output_nc, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(output_nc, output_nc, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            )

        self.pool = nn.MaxPool2d(kernel_size=2) if use_pool else None

    def forward(self, inp):
        out = None

        if self.downsampling:
            op1 = self.l1(inp)
            op2 = self.l2(torch.cat((op1, self.hidden), dim=1))

            self.hidden = op2

            out = op2
        elif self.upsampling:
            op1 = self.l1(inp)

            out = op1
        elif self.bottleneck:
            op1 = self.l1(inp)
            op2 = self.l2(torch.cat((op1, self.hidden), dim=1))

            self.hidden = op2

            out = op2

        if self.pool is not None:
            out = self.pool(out)

        return out

    def reset_hidden(self, inp, dfac):
        size = list(inp.size())
        size[1] = self.output_nc
        size[2] //= dfac
        size[3] //= dfac

        # print(size)

        dev = next(self.parameters()).device

        self.hidden_size = size
        self.hidden = torch.zeros(size).to(dev)


class RecurrentAEWeb(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(RecurrentAEWeb, self).__init__()

        """
        encoder_sizes = [32, 43, 57, 76, 101]
        decoder_sizes = [101, 76, 57, 43, 32]

        self.encoder = nn.ModuleList([
            RecurrentBlock(input_nc=in_size, output_nc=out_size, downsampling=True, use_pool=True)
            for in_size, out_size in zip([in_channels] + encoder_sizes[:-1], encoder_sizes)
        ])

        self.bottleneck = RecurrentBlock(input_nc=encoder_sizes[-1], output_nc=decoder_sizes[0], bottleneck=True)

        self.decoder = nn.ModuleList([
            RecurrentBlock(input_nc=in_size, output_nc=out_size, upsampling=True)
            for in_size, out_size in zip(decoder_sizes, decoder_sizes[1:] + [out_channels])
        ])
        """

        self.d1 = RecurrentBlock(input_nc=in_channels, output_nc=32, downsampling=True, use_pool=True)
        self.d2 = RecurrentBlock(input_nc=32, output_nc=43, downsampling=True, use_pool=True)
        self.d3 = RecurrentBlock(input_nc=43, output_nc=57, downsampling=True, use_pool=True)
        self.d4 = RecurrentBlock(input_nc=57, output_nc=76, downsampling=True, use_pool=True)
        self.d5 = RecurrentBlock(input_nc=76, output_nc=101, downsampling=True, use_pool=True)

        self.bottleneck = RecurrentBlock(input_nc=101, output_nc=101, bottleneck=True)

        self.u5 = RecurrentBlock(input_nc=101, output_nc=76, upsampling=True)
        self.u4 = RecurrentBlock(input_nc=76, output_nc=57, upsampling=True)
        self.u3 = RecurrentBlock(input_nc=57, output_nc=43, upsampling=True)
        self.u2 = RecurrentBlock(input_nc=43, output_nc=32, upsampling=True)
        self.u1 = RecurrentBlock(input_nc=32, output_nc=out_channels, upsampling=True)

    def process_frame(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        b = self.bottleneck(d5)

        u5 = self.u5(torch.cat((b, d5), dim=1))
        u4 = self.u4(torch.cat((u5, d4), dim=1))
        u3 = self.u3(torch.cat((u4, d3), dim=1))
        u2 = self.u2(torch.cat((u3, d2), dim=1))
        u1 = self.u1(torch.cat((u2, d1), dim=1))

        return u1

    def reset_hidden(self, x):
        x_slice = x[:, 0]

        self.d1.reset_hidden(x_slice, dfac=1)
        self.d2.reset_hidden(x_slice, dfac=2)
        self.d3.reset_hidden(x_slice, dfac=4)
        self.d4.reset_hidden(x_slice, dfac=8)
        self.d5.reset_hidden(x_slice, dfac=16)

        self.bottleneck.reset_hidden(x_slice, dfac=32)

        self.u4.reset_hidden(x_slice, dfac=16)
        self.u3.reset_hidden(x_slice, dfac=8)
        self.u5.reset_hidden(x_slice, dfac=4)
        self.u2.reset_hidden(x_slice, dfac=2)
        self.u1.reset_hidden(x_slice, dfac=1)

    def forward(self, x):
        self.reset_hidden(x)

        out = torch.stack([
            self.process_frame(x[:, i]) for i in range(x.shape[1])
        ], dim=1)

        return out