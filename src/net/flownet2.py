import torch
from torch import nn
from types import SimpleNamespace
import net.flownet2_nvidia.models
import net.flownet2_nvidia.networks.resample2d_package.resample2d


def load_flownet_nvidia(model):
    from tools.project import proj_dir
    path = proj_dir("pretrained", "FlowNet2_checkpoint.pth.tar")
    ckeckpoint = torch.load(path)
    model.load_state_dict(ckeckpoint["state_dict"])


def create_grids(dim, device):
    h, w = dim

    grid_y, grid_x = torch.meshgrid([torch.arange(h, device=device), torch.arange(w, device=device)])

    grid_x.required_grad = False
    grid_y.required_grad = False

    grid_x = grid_x.float().unsqueeze(0)
    grid_y = grid_y.float().unsqueeze(0)

    return grid_x, grid_y


def sample_grid(tensor, x, y):
    height, width = x.shape[-2:]

    x = 2 * x / width - 1
    y = 2 * y / height - 1

    grid = torch.stack((x, y), dim=-1)

    if len(tensor.shape) == 5:
        reshape = True
        batch_size, frame_count = tensor.shape[:2]
        tensor = tensor.view(batch_size, frame_count, *tensor.shape[2:])
    else:
        reshape = False

    sample = nn.functional.grid_sample(tensor, grid)

    if reshape:
        sample = sample.view(batch_size, frame_count, sample.shape[1:])

    return sample


# Input [x0, x1] [B, F, C, H, W]
# Output F0->1
class FlowNet2(net.flownet2_nvidia.models.FlowNet2):
    def __init__(self, flow_to_first=False):
        super().__init__(SimpleNamespace(rgb_max=1, fp16=False))

        self.flow_to_first = flow_to_first

    def forward(self, x):
        assert x.is_cuda

        if x.shape[1] > 2:
            return torch.stack([
                # self.forward(x[:, i:i+2])
                self.forward(x[:, [i if not self.flow_to_first else 0, i+1]])
                for i in range(x.shape[1] - 1)
            ], dim=1)
        else:
            assert x.shape[1] == 2

            if x.shape[2] == 1:
                x = torch.cat([x] * 3, dim=2)

            assert x.shape[2] == 3

            x = x.permute(0, 2, 1, 3, 4)
            return super().forward(x)


class BackWarpGridSample(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, img, flow):
        # print(img.shape, flow.shape)
        u, v = flow[:, 0, :, :], flow[:, 1, :, :]

        x, y = create_grids(u.shape[-2:], img.device)

        x += u
        y += v

        return sample_grid(img, x, y)


# Input x1, F0->1
# Output x0'
class Warp(nn.Module):
    def __init__(self, use_nvidia=False):
        super().__init__()

        self.use_nvidia = use_nvidia

        self.warp = net.flownet2_nvidia.networks.resample2d_package.resample2d.Resample2d() \
            if use_nvidia else BackWarpGridSample()

    def forward(self, img1, flow01):
        if len(img1.shape) == 5:
            return torch.stack([self.forward(img1[:, f], flow01[:, f]) for f in range(img1.shape[1])], dim=1)
        else:
            assert len(img1.shape) == 4

            if self.use_nvidia:
                assert img1.is_cuda
                assert flow01.is_cuda

                if not img1.is_contiguous():
                    img1 = img1.contiguous()

                if not flow01.is_contiguous():
                    flow01 = flow01.contiguous()

            return self.warp(img1, flow01)


class FlowComposite(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_flow(self, flows):
        h, w = flows.shape[-2:]

        if len(flows.shape) == 5:
            n = flows.shape[0] * flows.shape[1]
        elif len(flows.shape) == 4:
            n = flows.shape[0]
        else:
            raise Exception("Not supported")

        grid_x, grid_y = create_grids([h, w], flows.device)
        u, v = torch.zeros([n, h, w], device=flows.device), torch.zeros([n, h, w], device=flows.device)

        grid_x, grid_y = grid_x.expand_as(u), grid_y.expand_as(v)

        return u, v, grid_x, grid_y

    def _trace_flow(self, u, v, flow_map, grid_x, grid_y):
        new_flow = sample_grid(flow_map, grid_x + u, grid_y + v)
        u_new, v_new = new_flow[:, 0, :, :], new_flow[:, 1, :, :]

        u = u + u_new
        v = v + v_new

        return u, v, u_new, v_new

    # B F 2 H W
    def forward(self, flows):
        u, v, grid_x, grid_y = self._init_flow(flows[:, 0])

        for i in range(flows.shape[1]):
            u, v, _, _ = self._trace_flow(u, v, flows[:, i], grid_x, grid_y)

        return torch.stack([u, v], dim=1)


class FlowTrajectory(FlowComposite):
    def __init__(self):
        super().__init__()

    def forward(self, flows):
        u, v, grid_x, grid_y = self._init_flow(flows[:, 0])

        traj_flows = []

        # B F 2 H W
        for i in range(flows.shape[1]):
            u, v, u_new, v_new = self._trace_flow(u, v, flows[:, i], grid_x, grid_y)
            traj_flows.append(torch.stack([u_new, v_new], dim=1))

        traj_flows = torch.stack(traj_flows, dim=1)

        return traj_flows

"""
if flows.shape[1] == 1:
    return flows[:, 0]
else:
    # assert flows.shape[1] == 2
    # return flows[:, 0] + self.warp(flows[:, 1], flows[:, 0]) - flows[:, 1]
    flow_prev = self.forward(flows[:, :-1])
    return flow_prev + self.warp(flows[:, -1], flow_prev) - flows[:, -1]
"""


class MultiWarp(nn.Module):
    def __init__(self, use_composite_flow=False):
        super().__init__()

        self.warp = Warp()
        self.flow_composite = FlowComposite() if use_composite_flow else None

    # B F 2 H W
    def forward(self, img, flows):
        if self.flow_composite is not None:
            total_flow = self.flow_composite(flows)
            return self.warp(img, total_flow)
        else:
            for i in reversed(range(flows.shape[1])):
                img = self.warp(img, flows[:, i])

            return img
