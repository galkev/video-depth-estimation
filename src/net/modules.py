import torch
from torch import nn
from tools.tools import module_flat_str, is_size_equal, transpose_lists_2d, type_adv
import numpy as np
from data.video_depth_focus_data import VideoDepthFocusData
from data.data_transforms import crop_tensor, pad_to_multiple, pad_to_size

import torchvision.utils


def _output_mode_to_indices(output_mode, length):
    if isinstance(output_mode, list):
        indices = [_output_mode_to_indices(o, length) for o in output_mode]
        indices = [idx + i * length for i, idx_list in enumerate(indices) for idx in idx_list]
        return indices
    else:
        if output_mode == "all":
            return list(range(length))
        elif output_mode == "middle":
            return [length // 2]
        elif output_mode == "last":
            return [-1 + length]

    raise Exception(str(output_mode) + " not recognized")


class MaxReduce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, dim=1)[0]


class AvgReduce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)


class ConvReduce(nn.Module):
    def __init__(self, seq_length, num_channels):
        super().__init__()

        self.conv = nn.Conv2d(seq_length * num_channels, num_channels, kernel_size=1)

    def forward(self, x):
        batch_size, seq_length, num_channels, h, w = x.shape

        x = x.view(batch_size, seq_length * num_channels, h, w)
        x = self.conv(x)

        return x


def _apply_to_output(out, func):
    if isinstance(out, tuple):
        return tuple(func(o) for o in out)
    else:
        return func(out)


class PadCropModule(nn.Module):
    def __init__(self, model, multiple, center=True):
        super().__init__()

        self.model = model
        self.multiple = multiple
        self.center = center

    def get_output_mode(self):
        return self.model.get_output_mode()

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)

    def forward(self, x):
        x_pad, pad_amt = pad_to_multiple(x, self.multiple, self.center)
        out = self.model(x_pad)

        if isinstance(out, tuple):
            out_crop = tuple(crop_tensor(o, pad_amt) for o in out)
        else:
            out_crop = crop_tensor(out, pad_amt)

        return out_crop


class DownUpScaleModule(nn.Module):
    def __init__(self, model, interp_mode="nearest"):
        super().__init__()

        self.model = model
        self.target_size = [256, 256]
        self.interp_mode = interp_mode

    def get_output_mode(self):
        return self.model.get_output_mode()

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)

    def forward(self, x):
        B, F, C, H, W = x.shape

        assert B == 1

        target_aspect = self.target_size[0] / self.target_size[1]

        assert target_aspect >= 1

        pad_size = max(H, W) * self.target_size[0] // self.target_size[1]
        pad_size = [pad_size] * 2

        x_pad, pad_amt = pad_to_size(x, pad_size)

        assert x_pad.shape[-1] / x_pad.shape[-2] == target_aspect

        x_pad_down = torch.stack([
            nn.functional.interpolate(x_pad[:, f], size=self.target_size, mode=self.interp_mode)
            for f in range(F)], dim=1)

        assert is_size_equal([x_pad_down.shape[-1], x_pad_down.shape[-2]], self.target_size)

        out = self.model(x_pad_down)

        out_up_crop = _apply_to_output(
            out,
            lambda o: crop_tensor(nn.functional.interpolate(o, size=pad_size, mode=self.interp_mode), pad_amt)
        )

        return out_up_crop


class CropCatModule(nn.Module):
    def __init__(self, model, crop_border=False):
        super().__init__()

        self.crop_size = np.array([256, 256])
        self.model = model
        self.crop_border = crop_border

    def get_output_mode(self):
        return self.model.get_output_mode()

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)

    @staticmethod
    def _make_tensor_grid(tensors, grid_dim):
        crop_size = list(tensors[0].shape[1:])
        shape = list(tensors[0].size())

        shape[-2] *= grid_dim[0]
        shape[-1] *= grid_dim[1]

        x = torch.zeros(shape, device=tensors[0].device)

        for i in range(grid_dim[0]):
            for j in range(grid_dim[1]):
                y1, y2 = i * crop_size[0], (i + 1) * crop_size[0]
                x1, x2 = j * crop_size[1], (j + 1) * crop_size[1]

                idx = i * grid_dim[1] + j

                x[..., y1:y2, x1:x2] = tensors[idx]

        return x

    @staticmethod
    def _make_grid(tensors, grid_dim):
        assert isinstance(tensors, list)
        assert len(tensors) == grid_dim[0] * grid_dim[1]

        batch_size = tensors[0].shape[0]

        assert batch_size == 1

        batches = []

        for b in range(batch_size):
            batches.append(CropCatModule._make_tensor_grid([t[b] for t in tensors], grid_dim=grid_dim))

        out = torch.stack(batches)

        return out

    def forward(self, x):
        B, F, C, H, W = x.shape

        assert B == 1

        x_size = np.array([H, W])

        if (x_size % self.crop_size).any():
            raise Exception(f"{x_size} % {self.crop_size} != 0")

        num_crops = x_size // self.crop_size

        # out = torch.zeros([B, 1, H, W], device=x.device)
        outputs = []
        multi_out = False

        for i in range(num_crops[0]):
            for j in range(num_crops[1]):
                y1, y2 = i * self.crop_size[0], (i + 1) * self.crop_size[0]
                x1, x2 = j * self.crop_size[1], (j + 1) * self.crop_size[1]

                part = self.model(x[..., y1:y2, x1:x2])

                multi_out = isinstance(part, tuple)

                outputs.append(part)

        if multi_out:
            # outputs = transpose_lists_2d(outputs)
            outputs = tuple(map(list, zip(*outputs)))

        out = _apply_to_output(
            outputs,
            lambda o: CropCatModule._make_grid(o, num_crops)
        )

        assert not self.crop_border

        if self.crop_border:
            for b in range(B):
                min_val = torch.min(out[b])

                for i in range(num_crops[0]):
                    for j in range(num_crops[1]):
                        y1, y2 = i * self.crop_size[0], (i + 1) * self.crop_size[0]
                        x1, x2 = j * self.crop_size[1], (j + 1) * self.crop_size[1]

                        if i > 0:
                            out[..., y1, :] = min_val

                        if j > 0:
                            out[..., :, x1] = min_val

        return out


class AE2Decoder(nn.Module):
    def __init__(self, ae_class, *args, **kwargs):
        super().__init__()

        self.encoder = ae_class.create_component("encoder", *args, **kwargs)
        self.decoders = nn.ModuleList([
            ae_class.create_component("decoder", *args, **kwargs) for _ in range(2)
        ])

    def set_attr(self, k, v):
        if k == "output_all":
            self.encoder.output_all = v
        else:
            raise Exception()

    def get_output_mode(self):
        mode = self.encoder.get_output_mode()

        if mode is None:
            if isinstance(self.decoders[0], nn.Sequential):
                mode = self.decoders[0][0].get_output_mode()
            else:
                mode = self.decoders[0].get_output_mode()

        return mode

    def forward(self, x, encoding_modifier=None):
        # x_enc, hidden = self.encoder(x)
        encoding = self.encoder(x)

        if encoding_modifier is not None:
            encoding[0] = encoding_modifier(encoding[0])

        if isinstance(self.decoders[0], nn.Sequential):
            return tuple(dec(encoding) for dec in self.decoders)
        else:
            return tuple(dec(*encoding) for dec in self.decoders)


class OutputSplitModule(nn.Module):
    def __init__(self, model_class, *args, **kwargs):
        super().__init__()

        self.model = model_class(out_channels=2, *args, **kwargs)

    def set_attr(self, k, v):
        self.model.set_attr(k, v)

    def get_output_mode(self):
        return self.model.get_output_mode()

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out[:, 0:1], out[:, 1:2]


class InputSubsetModule(nn.Module):
    def __init__(self, model, indices):
        super().__init__()
        self.model = model
        self.indices = indices

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)

    def get_output_mode(self):
        return self.model.get_output_mode()

    def forward(self, x):
        return self.model(x[:, self.indices])


class InputSplitModule(nn.Module):
    def __init__(self, model, num_splits, pad_output=False):
        super().__init__()
        self.model = model
        self.num_splits = num_splits
        self.pad_output = pad_output

    def set_attr(self, k, v):
        self.model.set_attr(k, v)

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)

    def get_output_mode(self):
        return [self.model.get_output_mode()] * self.num_splits

    def _pad_outputs(self, outputs, input_per_model):
        outputs_padded = []

        fill_img = torch.zeros_like(outputs[0])
        for output, output_mode in zip(outputs, self.get_output_mode()):
            output_mode_indices = _output_mode_to_indices(output_mode, input_per_model)

            output_padded = [fill_img for _ in range(input_per_model)]

            assert len(output_mode_indices) == 1

            output_padded[output_mode_indices[0]] = output

            outputs_padded += output_padded

        return outputs_padded

    def forward(self, x):
        num_frames = x.shape[1]
        input_per_model = num_frames // self.num_splits

        assert num_frames == self.num_splits * input_per_model

        outputs = [
            self.model(x[:, i:i+input_per_model])
            for i in range(0, num_frames, input_per_model)
        ]

        if isinstance(outputs[0], tuple):
            num_outputs = len(outputs[0])

            x = []

            for i in range(num_outputs):
                x_part = torch.stack([o[i] for o in outputs], dim=1)

                if len(x_part.shape) == 6:
                    x_part = x_part.view(x_part.shape[0], x_part.shape[1] * x_part.shape[2], *x_part.shape[3:])

                x.append(x_part)

            x = tuple(x)
        else:
            x = torch.stack(outputs, dim=1)

            if len(x.shape) == 6:
                x = x.view(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])

        return x


class ParallelNet(nn.Module):
    def __init__(self, models):
        super().__init__()

        self.models = nn.ModuleList(models)

    def get_output_mode(self, i=0):
        return self.models[0].get_output_mode(i)

    def forward(self, x):
        out = tuple(m(x) for m in self.models)
        return out


class MultiDecoderNet(nn.Module):
    def __init__(self, encoder, decoders):
        super().__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)

    def get_dec_pool_layers(self, i=0):
        return self.decoders[i][0].pool_layers

    def get_output_mode(self, i=0):
        return "all" if not self.get_dec_pool_layers(i)[-1] else "middle"

    def forward(self, x):
        encoding = self.encoder(x)

        return tuple(dec(encoding) for dec in self.decoders)

    def flat_str(self, module_filter=None):
        text = ""

        text += "Encoder:\n"
        text += module_flat_str(self.encoder, module_filter)

        text += "\n\nDecoder CoC:\n"
        text += module_flat_str(self.decoders[0], module_filter)

        text += "\n\nDecoder Depth:\n"
        text += module_flat_str(self.decoders[1], module_filter)

        return text


class LayerSplitModule(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        x = self.model(x)

        if len(x.shape) == 5:
            return x[:, :, 0:1], x[:, :, 1:2]
        else:
            return x[:, 0:1], x[:, 1:2]
