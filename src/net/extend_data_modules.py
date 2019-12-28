import torch
from torch import nn
import torchvision
import numpy as np

from tools.vis_tools import flow_to_vis
import uuid

from tools import dump_tensor


class ExtendDataModuleBase(nn.Module):
    def __init__(self):
        super().__init__()

    def get_layers(self):
        return None

    def _add_channel(self, x, y):
        return torch.cat([x, y], dim=2)


class ExtendDataMultiple(ExtendDataModuleBase):
    def __init__(self, extenders):
        super().__init__()

        self.extenders = nn.Sequential(*extenders)

    def forward(self, data):
        data = self.extenders(data)
        return data


# flow_to_first == [None, "composite", "trajectory", "direct"]
class ExtendDataFlow(ExtendDataModuleBase):
    def __init__(self, flow_source="flownet2", flow_to_first=None, warp_images=False):
        super().__init__()

        from net.flownet2 import FlowNet2, Warp, FlowComposite, FlowTrajectory, load_flownet_nvidia

        assert not (flow_to_first == "direct" and flow_source == "disk")

        self.flow_module = FlowNet2(flow_to_first=flow_to_first == "direct") if flow_source == "flownet2" else None

        if self.flow_module is not None:
            self.flow_module.eval()
            load_flownet_nvidia(self.flow_module)

        self.flow_source = flow_source
        self.flow_to_first = flow_to_first
        self.warp_images = warp_images

        self.warp = Warp(use_nvidia=True) if self.warp_images else None
        self.flow_composite = FlowComposite() if self.flow_to_first == "composite" else None
        self.flow_trajectory = FlowTrajectory() if self.flow_to_first == "trajectory" else None

    def get_layers(self):
        return [3, 2] if not self.warp_images else None

    def _get_flow(self, data):
        if self.flow_source == "flownet2":
            return self.flow_module(data["color"])
        elif self.flow_source == "disk":
            return data["flow"]
        else:
            raise Exception(str(self.flow_source) + " not recognized")

    def forward(self, data):
        with torch.no_grad():
            B, F, C, H, W = data["color"].size()

            flow = self._get_flow(data)
            flow = torch.cat([torch.zeros([B, 1, 2, H, W], device=data["color"].device), flow], dim=1)

            if self.flow_to_first == "composite":
                for i in range(1, F):
                    flow[:, i] = self.flow_composite(flow[:, i-1:i+1])
            elif self.flow_to_first == "trajectory":
                flow = self.flow_trajectory(flow)

            # torchvision.utils.save_image(flow_to_vis(flow.view(B*F, 2, H, W)),
            #                             "/home/kevin/Documents/master-thesis/logs/dump/{}.tif".format(uuid.uuid1()))

            if self.warp_images:
                data["color"] = self.warp(data["color"], flow)
            else:
                data["color"] = self._add_channel(data["color"], flow)

            return data


class ExtendEncoding(ExtendDataModuleBase):
    def __init__(self, extend_data, data):
        super().__init__()

        self.extend_data = extend_data
        self.data = data

    def forward(self, x):
        return self.extend_data(self.data, x)


class ExtendDataFov(ExtendDataModuleBase):
    def __init__(self, use_fov_x, use_fov_y):
        super().__init__()

        self.use_fov_x = use_fov_x
        self.use_fov_y = use_fov_y

    @staticmethod
    def _calc_fov(grid, total_size, focal_length, num_frames):
        focal_length = focal_length[:, None, None, None, None]
        total_size = total_size[:, None, None, None, None]

        # print(total_size.shape, grid.shape)

        fov = torch.atan(total_size * grid / focal_length)
        fov = fov.repeat(1, num_frames, 1, 1, 1)

        return fov

    @staticmethod
    def _create_grids(p1, p2, num_px):
        batch_size = p1.shape[0]
        dev = p1.device

        if not isinstance(num_px, torch.Tensor):
            num_px = torch.tensor(num_px, device=dev)

        num_px = num_px[None, :].repeat(batch_size, 1)

        px_size2 = 0.5 * ((p2 - p1) / num_px.float())

        grid_y, grid_x = [], []

        for b in range(batch_size):
            y, x = torch.meshgrid([
                torch.linspace(p1[b, 1] + px_size2[b, 1], p2[b, 1] - px_size2[b, 1], num_px[b, 1], device=dev),
                torch.linspace(p1[b, 0] + px_size2[b, 0], p2[b, 0] - px_size2[b, 0], num_px[b, 0], device=dev)
            ])

            grid_x.append(x)
            grid_y.append(y)

        grid_x = torch.stack(grid_x, dim=0)
        grid_y = torch.stack(grid_y, dim=0)

        grid_x = grid_x[:, None, None, :, :]
        grid_y = grid_y[:, None, None, :, :]

        # print(grid_x)

        return grid_x, grid_y

    @staticmethod
    def _calc_crop_points(crop_res, total_res, crop_offset_center):
        crop_rel_size2 = 0.5 * crop_res.float() / total_res
        crop_rel_offset = crop_offset_center / total_res

        crop_p1 = -crop_rel_size2 + crop_rel_offset
        crop_p2 = crop_rel_size2 + crop_rel_offset

        return crop_p1, crop_p2

    @staticmethod
    def _get_tensor_res(tensor):
        dev = tensor.device
        batch_size = tensor.shape[0]

        crop_res = torch.tensor([tensor.shape[-1], tensor.shape[-2]],
                                device=dev)[None, :].repeat(batch_size, 1)

        return crop_res

    def _dump_test(self, x):
        dump_tensor(x[0, 0, 3:4], normalize=True)
        dump_tensor(x[0, 0, 4:5], normalize=True)
        raise Exception()

    # B, F, C, H, W
    def forward(self, data, tensor_to_cat=None):
        num_frames = data["color"].shape[1]
        crop_res = ExtendDataFov._get_tensor_res(data["color"])
        total_res = data["lens"]["resolution"].float()
        crop_offset_center = data["crop_offset_center"].float()

        num_pixels = ExtendDataFov._get_tensor_res(tensor_to_cat)[0] if tensor_to_cat is not None else crop_res[0]

        crop_p1, crop_p2 = ExtendDataFov._calc_crop_points(crop_res, total_res, crop_offset_center)
        grid_x, grid_y = ExtendDataFov._create_grids(crop_p1, crop_p2, num_pixels)

        channels = []

        if self.use_fov_x:
            fov_x = ExtendDataFov._calc_fov(
                grid_x,
                total_res[:, 0],
                data["lens"]["focal_length_pixel"][:, 0],
                num_frames
            )

            # print(crop_offset_center[0])
            # print(data["lens"]["fov"][:, 0] / 2, data["lens"]["fov"][:, 1] / 2)
            # print(crop_p1, crop_p2)
            # print(fov_x)

            channels.append(fov_x)

        if self.use_fov_y:
            fov_y = ExtendDataFov._calc_fov(
                grid_y,
                total_res[:, 1],
                data["lens"]["focal_length_pixel"][:, 1],
                num_frames
            )

            channels.append(fov_y)

        for c in channels:
            if tensor_to_cat is not None:
                if len(tensor_to_cat.shape) == 4:
                    tensor_to_cat = torch.cat([tensor_to_cat, c[:, -1]], dim=1)
                elif len(tensor_to_cat.shape) == 5:
                    tensor_to_cat = torch.cat([tensor_to_cat, c], dim=2)
                else:
                    assert False
            else:
                data["color"] = torch.cat([data["color"], c], dim=2)

        # print(crop_offset_center[0])
        # print(data["lens"]["fov"][0] / 2, data["lens"]["fov"][1] / 2)
        # print(data["color"][0, :1, 3:5])
        # raise Exception()

        # self._dump_test(tensor_to_cat if tensor_to_cat is not None else data["color"])

        if tensor_to_cat is not None:
            # print(tensor_to_cat.shape)
            # print(data["lens"]["fov"][0, 0] / 2, data["lens"]["fov"][0, 1] / 2)
            # print(tensor_to_cat[0, -2:])
            return tensor_to_cat
        else:
            return data


class ExtendDataConstant(ExtendDataModuleBase):
    def __init__(self, params):
        super().__init__()

        self.params = params

    @staticmethod
    def _val_per_frame(F, val):
        if len(val.shape) == 1:
            val = val[:, None]

        if len(val.shape) == 2:
            val = val[:, None, :].repeat(1, F, 1)

        return val

    def _get_params_rec(self, data, params):
        if isinstance(params, list):
            return torch.cat([self._get_params_rec(data, p) for p in params], dim=2)
        else:
            B, F = data["color"].shape[:2]

            if params == "focal_length":
                val = data["lens"]["focal_length"] * 1000
            elif params == "sensor_size_x":
                val = data["lens"]["sensor_size"][:, 0] * 1000
            elif params == "sensor_size_y":
                val = data["lens"]["sensor_size"][:, 1] * 1000
            elif params == "fov_x":
                val = data["lens"]["fov"][:, 0]
            elif params == "fov_y":
                val = data["lens"]["fov"][:, 1]
            elif params == "foc_dist":
                val = data["focus_dist_all"][:, :, None]
            elif params == "aperture":
                val = data["lens"]["aperture"] * 1000
            elif params == "f_number":
                val = data["lens"]["f_number"]
            else:
                raise Exception(params + " not recognized")

            val = ExtendDataConstant._val_per_frame(F, val)

            # print(val.shape)

            assert val.shape[0] == B
            assert val.shape[1] == F
            assert len(val.shape) == 3

            return val

    def _get_params(self, data):
        return self._get_params_rec(data, self.params)

    # B, F, C, H, W
    def forward(self, data):
        B, F, C, H, W = data["color"].shape

        add_data = self._get_params(data)

        add_data = add_data.repeat(1, 1, int(np.ceil((H * W) / add_data.shape[2])))
        add_data = add_data[:, :, :(H * W)]
        add_data = add_data.view(B, F, 1, H, W)

        data["color"] = torch.cat([data["color"], add_data], dim=2)

        # print(data["color"].shape)
        # print(data["color"][0, :3, 3])

        return data


class ExtendDataFocusDist(ExtendDataModuleBase):
    def __init__(self):
        super().__init__()
        # self.focus_dist_normalize(focus_dist)

    def forward(self, data):
        in_size = data["color"].size()

        batches = []

        for focus_dist_batch in data["focus_dist_all"]:
            batches.append(
                torch.stack([
                    torch.full([1, *in_size[3:]], focus_dist, device=data["color"].device)
                    for focus_dist in focus_dist_batch
                ])
            )

        channel = torch.stack(batches)

        data["color"] = self._add_channel(data["color"], channel)

        return data
