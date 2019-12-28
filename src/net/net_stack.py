import torch
from torch import nn


class NetStack(nn.Module):
    def __init__(self, net_gen, n, multi_in):
        super().__init__()

        self.nets = nn.ModuleList([
            net_gen() for _ in range(n)
        ])

        self.multi_in = multi_in

    def forward(self, net_inputs):
        return [net(*net_input) if self.multi_in else net(net_input) for net, net_input in zip(self.nets, net_inputs)]


class ConcatNetStackSandwitch(nn.Module):
    def __init__(self, enc_net_gen, dec_net_gen, inner_net, num_frames, num_decoder=1):
        super().__init__()

        self.num_decoder = num_decoder

        self.enc_stack = NetStack(enc_net_gen, num_frames, False)
        self.inner_net = inner_net

        if num_decoder == 1:
            self.dec_stack = NetStack(dec_net_gen, num_frames, True)
        else:
            self.dec_stack = nn.ModuleList([NetStack(dec_net_gen, num_frames, True) for _ in range(num_decoder)])

    def get_output_mode(self, i=0):
        return "all"

    # [B, F, C, H, W]
    def forward(self, x):
        num_frames = x.shape[1]

        enc_stack_out = self.enc_stack([x[:, i] for i in range(num_frames)])

        x = torch.stack([x[0] for x in enc_stack_out], dim=1)
        x = self.inner_net(x)

        if self.num_decoder == 1:
            enc_stack_out = [(x[:, i], enc_stack_out[i][1]) for i in range(num_frames)]
            dec_stack_out = self.dec_stack(enc_stack_out)

            return torch.stack(dec_stack_out, dim=1)
        else:
            return tuple(
                torch.stack(dec(
                    [(x_part[:, i], enc_stack_out[i][1]) for i in range(num_frames)]
                    ), dim=1)
                for dec, x_part in zip(self.dec_stack, x)
            )
