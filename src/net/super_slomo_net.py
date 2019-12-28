import torch
from torch import nn

from .util import same_padding


# https://github.com/avinashpaliwal/Super-SloMo/blob/master/model.py
class DownModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_input=True):
        super().__init__()

        self.pool = nn.AvgPool2d(2, stride=2) if pool_input else None

        leaky_relu_alpha = 0.1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=same_padding(kernel_size))
        self.act1 = nn.LeakyReLU(leaky_relu_alpha)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=same_padding(kernel_size))
        self.act2 = nn.LeakyReLU(leaky_relu_alpha)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)

        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))

        return x


class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        leaky_relu_alpha = 0.1

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=same_padding(3))
        self.act1 = nn.LeakyReLU(leaky_relu_alpha)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, 3, padding=same_padding(3))
        self.act2 = nn.LeakyReLU(leaky_relu_alpha)

    def forward(self, x, x_skip):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.act1(self.conv1(x))
        x = torch.cat((x, x_skip), 1)
        x = self.act2(self.conv2(x))

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        leaky_relu_alpha = 0.1

        self.down1 = DownModule(in_channels, 32, kernel_size=7, pool_input=False)
        self.down2 = DownModule(32, 64, kernel_size=5)
        self.down3 = DownModule(64, 128)
        self.down4 = DownModule(128, 256)
        self.down5 = DownModule(256, 512)
        self.down6 = DownModule(512, 512)

        self.up1 = UpModule(512, 512)
        self.up2 = UpModule(512, 256)
        self.up3 = UpModule(256, 128)
        self.up4 = UpModule(128, 64)
        self.up5 = UpModule(64, 32)

        self.conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=same_padding(3))
        self.act = nn.LeakyReLU(leaky_relu_alpha)

    def forward(self, x):
        s1 = self.down1(x)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        s5 = self.down5(s4)
        x  = self.down6(s5)

        x  = self.up1(x, s5)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)

        x  = self.act(self.conv(x))

        return x


class BackWarp(nn.Module):
    def __init__(self, w, h, device):
        super().__init__()

        self.width = w
        self.height = h

        self.grid_y, self.grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])

        self.grid_x.required_grad = False
        self.grid_y.required_grad = False

        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)

    def forward(self, img, flow):
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        x = self.grid_x.unsqueeze(0).expand_as(u).float() + u
        y = self.grid_y.unsqueeze(0).expand_as(v).float() + v

        x = 2 * x / self.width - 1
        y = 2 * y / self.height - 1

        grid = torch.stack((x, y), dim=3)

        img_warped = nn.functional.grid_sample(img, grid)

        return img_warped


class SuperSlomoNet(nn.Module):
    def __init__(self, w, h, device,
                 inter_frame_count=7,
                 flow_comp_net=UNet(6, 4),
                 flow_interp_net=UNet(20, 5)
                 ):
        super().__init__()

        self.flow_comp_net = flow_comp_net.to(device)
        self.flow_interp_net = flow_interp_net.to(device)
        self.back_warp = BackWarp(w, h, device).to(device)

        self.inter_frame_count = inter_frame_count
        self.timesteps = torch.linspace(
            1 / (inter_frame_count+1),
            1 - 1 / (inter_frame_count+1),
            inter_frame_count).to(device)

    # compute ft0 + ft1
    def compute_flow_t(self, time_idx, flow_01, flow_10):
        t = self.timesteps[time_idx][:, None, None, None]

        flow_t0 = -(1 - t) * t * flow_01 + t**2 * flow_10
        flow_t1 = (1 - t)**2 * flow_01 - t * (1 - t) * flow_10

        return flow_t0, flow_t1

    def compute_inter_img(self, time_idx, vis_t0, vis_t1, g_img_0_flow_t0_f, g_img_1_flow_t1_f):
        t = self.timesteps[time_idx][:, None, None, None]

        inter_frame = ((1 - t) * vis_t0 * g_img_0_flow_t0_f + t * vis_t1 * g_img_1_flow_t1_f) / \
                      ((1 - t) * vis_t0 + t * vis_t1)

        return inter_frame

    def forward(self, input_data):
        frames, time_idx = input_data  # I0, I1

        img_0, img_1 = frames[:, :3, :, :], frames[:, 3:, :, :]

        flow_comp_out = self.flow_comp_net(frames)  # F01, F10

        flow_01, flow_10 = flow_comp_out[:, :2, :, :], flow_comp_out[:, 2:, :, :]

        flow_t0, flow_t1 = self.compute_flow_t(time_idx, flow_01, flow_10)

        g_img_0_flow_t0 = self.back_warp(img_0, flow_t0)
        g_img_1_flow_t1 = self.back_warp(img_1, flow_t1)

        interp_out = self.flow_interp_net(torch.cat([img_0, img_1, flow_01, flow_10, flow_t1, flow_t0,
                                                     g_img_1_flow_t1, g_img_0_flow_t0], dim=1))

        vis_t0 = torch.sigmoid(interp_out[:, 4:5, :, :])
        vis_t1 = 1 - vis_t0

        delta_flow_t0 = interp_out[:, :2, :, :]
        delta_flow_t1 = interp_out[:, 2:4, :, :]

        flow_t0_final = flow_t0 + delta_flow_t0
        flow_t1_final = flow_t1 + delta_flow_t1

        g_img_0_flow_t0_final = self.back_warp(img_0, flow_t0_final)
        g_img_1_flow_t1_final = self.back_warp(img_1, flow_t1_final)

        inter_img_pred = self.compute_inter_img(time_idx, vis_t0, vis_t1, g_img_0_flow_t0_final, g_img_1_flow_t1_final)

        # for loss function only
        # img_0, img_1
        g_img_0_flow_10 = self.back_warp(img_0, flow_10)
        g_img_1_flow_01 = self.back_warp(img_1, flow_01)

        return {
            "inter_img_pred": inter_img_pred,
            "flow_01": flow_01,
            "flow_10": flow_10,
            "g_img_0_flow_t0": g_img_0_flow_t0,
            "g_img_1_flow_t1": g_img_1_flow_t1,
            "img_0": img_0,
            "img_1": img_1,
            "g_img_0_flow_10": g_img_0_flow_10,
            "g_img_1_flow_01": g_img_1_flow_01
        }


class UNetDummy(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=same_padding(1))

    def forward(self, x):
        return self.conv(x)


class SuperSlomoNetDummy(SuperSlomoNet):
    def __init__(self, w, h, device,
                 inter_frame_count=7,):
        super().__init__(w, h, device,
                         inter_frame_count=inter_frame_count,
                         flow_comp_net=UNetDummy(6, 4),
                         flow_interp_net=UNetDummy(20, 5))
