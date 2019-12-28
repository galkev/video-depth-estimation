import numpy as np
import random
import torch
from torchvision import transforms
from torch.nn import functional as F
from torchvision.transforms import functional as VF
from PIL import Image, ImageOps


class DataTransform:
    def __repr__(self):
        return self.__class__.__name__


class Identity(DataTransform):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class Pad(DataTransform):
    def __init__(self, pad_width, mode, **kwargs):
        self.pad_width = pad_width
        self.mode = mode
        self.kwargs = kwargs

    def __call__(self, img):
        return np.pad(img, self.pad_width, self.mode, **self.kwargs)


class MultiArgTransform(DataTransform):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, input_list):
        return [transform(x) for x, transform in zip(input_list, self.transform_list)]


class TensorStackTransform(DataTransform):
    def __init__(self, transform, combine_mode="stack"):
        self.transform = transform
        self.combine_mode = combine_mode

    def __call__(self, pic_list):
        transformed = [self.transform(pic) for pic in pic_list]

        if self.combine_mode == "stack":
            transformed = torch.stack(transformed)
        elif self.combine_mode == "cat":
            transformed = torch.cat(transformed)

        return transformed


class Normalize(DataTransform):
    def __init__(self, min_val, max_val, eps_min=0, eps_max=0, clamp_max=False):
        self.mean = min_val-eps_min
        self.std = max_val + eps_max - (min_val-eps_min)
        self.min_val = min_val
        self.max_val = max_val
        self.clamp_max = clamp_max

    def __call__(self, tensor):
        if self.clamp_max:
            tensor.clamp_max_(self.max_val)

        tensor.sub_(self.mean).div_(self.std)
        return tensor

    def rev(self, tensor):
        tensor.mul_(self.std).add_(self.mean)
        return tensor

    def __repr__(self):
        return "Normalize({}, {}, clamp_max={})".format(
            self.min_val, self.max_val, self.clamp_max
        )


class ToFloatTensor(transforms.ToTensor):
    def __init__(self, div_by=None):
        super().__init__()
        self.div_by = div_by

    def __call__(self, pic):
        if isinstance(pic, Image.Image):
            if pic.mode == "I" or pic.mode == "I;16":
                tensor = super().__call__(np.array(pic, dtype=float)[:, :, None])
                tensor = tensor.float().div(0xffff)
            else:
                tensor = super().__call__(pic)
        elif isinstance(pic, np.ndarray):
            if len(pic.shape) == 2:
                pic = np.expand_dims(pic, axis=2)
            if pic.dtype == np.float64:
                pic = pic.astype(np.float32)
            tensor = super().__call__(pic)
        else:
            raise Exception("Error ToFloatTensor: no matching datatype")

        if self.div_by is not None:
            tensor.div_(self.div_by)

        return tensor


class RandomHFlipTensor(DataTransform):
    def __init__(self, p=0.5):
        self.apply_flip = random.random() < p

    def __call__(self, x):
        if self.apply_flip:
            return x.flip(2)
        else:
            return x


class RandomHFlip(DataTransform):
    def __init__(self, p=0.5):
        self.apply_flip = random.random() < p

    def __call__(self, x):
        if self.apply_flip:
            return VF.hflip(x)
        else:
            return x


def center_crop(tensor, size):
    h, w = size
    y = (tensor.shape[1] - h) // 2
    x = (tensor.shape[2] - w) // 2

    # print(x, y, w, h)

    return tensor[:, y:y + h, x:x + w]


class CenterCrop(DataTransform):
    def __init__(self, size):
        if isinstance(size, int):
            size = [size, size]

        self.size = size

    def __call__(self, tensor):
        return center_crop(tensor, self.size)


class FiveCrop(DataTransform):
    def __init__(self, size, crop_loc_idx):
        self.size = [size, size]
        self.crop_loc_idx = crop_loc_idx

    def _crop(self, tensor):
        th, tw = tensor.shape[1:3]
        h, w = self.size

        if self.crop_loc_idx == 0:  # tl
            return tensor[:, :h, :w]
        elif self.crop_loc_idx == 1:  # tr
            return tensor[:, :h, tw-w:]
        elif self.crop_loc_idx == 2:  # bl
            return tensor[:, th-h:, :w]
        elif self.crop_loc_idx == 3:  # br
            return tensor[:, th-h:, tw-w:]
        elif self.crop_loc_idx == 4:  # center
            return center_crop(tensor, self.size)
        else:
            raise Exception("crop_loc_idx " + str(self.crop_loc_idx) + " invalid")

    def __call__(self, tensor):
        return self._crop(tensor)


# crops tensor
class RandomCrop(DataTransform):
    def __init__(self, output_size, img_size, valid_crops=None):
        nw, nh = output_size if type(output_size) is list else [output_size, output_size]

        if valid_crops is None or len(valid_crops) == 0:
            w, h = img_size
            if w == nw and h == nh:
                i, j = 0, 0
            else:
                i = random.randint(0, h - nh)
                j = random.randint(0, w - nw)
        else:
            i, j = random.choice(valid_crops)

        self.crop_params = i, j, nh, nw
        # self.crop_params = h - nh, w - nw, nh, nw
        # print("FIX DATA TRANSFORMED")
        # self.crop_params = 0, 0, nh, nw

        self.img_size = img_size

    def get_crop_offset_center(self):
        i, j, nh, nw = self.crop_params
        offset = [
            (j + nw // 2) - self.img_size[0] // 2,
            (i + nh // 2) - self.img_size[1] // 2
        ]

        # print(self.img_size, (self.crop_params[1], self.crop_params[0]), offset)
        # raise Exception()

        return offset

    def __call__(self, tensor):
        i, j, nh, nw = self.crop_params
        crop = tensor[:, i:i+nh, j:j+nw]

        assert crop.shape[-2:] == (nh, nw)

        return crop

    @staticmethod
    def get_all_valid_crop_pos(tensor, size, valid_pixel_cond=lambda x: x > 0.0, thresh=0.8):
        h, w = tensor.size(1), tensor.size(2)
        nh, nw = size if type(size) is list else [size, size]
        all_crop_pos = np.asarray([(y, x) for y in range(h - nh) for x in range(w - nw)])
        valid_crop_pos = []

        def is_valid_crop(img_crop):
            return (valid_pixel_cond(img_crop.numpy()).sum() / (
                        img_crop.size(1) * img_crop.size(2))) >= thresh

        for (y, x) in all_crop_pos:
            crop = tensor[:, y:y+nh, x:x+nw]
            if is_valid_crop(crop):
                valid_crop_pos.append((y, x))

        # print("Found {}".format(len(valid_crop_pos)))
        return valid_crop_pos


def crop_tensor(x, crop_amt):
    # assert all(abs(a) > 0 for a in crop_amt)

    h, w = x.shape[-2:]

    w1, w2, h1, h2 = crop_amt
    w2 = w - w2
    h2 = h - h2

    x = x[..., h1:h2, w1:w2]

    return x


def crop_to_size(x, size):
    h, w = x.shape[-2:]

    h_new_diff = h - size[1]
    w_new_diff = w - size[0]

    crop_h1, crop_w1 = h_new_diff // 2, w_new_diff // 2
    crop_h2, crop_w2 = (h_new_diff + 1) // 2, (w_new_diff + 1) // 2

    crop_amt = (crop_w1, crop_w2, crop_h1, crop_h2)

    x = crop_tensor(x, crop_amt)

    assert list(x.shape[-2:]) == [size[1], size[0]]

    return x


def crop_to_aspect(x, size):
    h, w = x.shape[-2:]

    if w / h >= size[0] / size[1]:
        # crop width
        crop_size = [h * size[0] // size[1], h]
    else:
        # crop height
        crop_size = [w, w * size[1] // size[0]]

    return crop_to_size(x, crop_size)


def pad_to_size(x, size, center=True):
    h, w = x.shape[-2:]

    h_new_diff = size[1] - h
    w_new_diff = size[0] - w

    if center:
        pad_h1, pad_w1 = h_new_diff // 2, w_new_diff // 2
        pad_h2, pad_w2 = (h_new_diff + 1) // 2, (w_new_diff + 1) // 2

        pad = (pad_w1, pad_w2, pad_h1, pad_h2)
    else:
        pad = (0, w_new_diff, 0, h_new_diff)

    x = F.pad(x, pad)

    assert list(x.shape[-2:]) == [size[1], size[0]]

    return x, pad


def pad_to_multiple(x, multiple, center=True):
    h, w = x.shape[-2:]

    return pad_to_size(
        x,
        (
            int(np.ceil(w / multiple) * multiple),
            int(np.ceil(h / multiple) * multiple)
        ),
        center=center
    )


class PadToSize(DataTransform):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, x):
        return pad_to_size(x, self.target_size)[0]


class PadToMultiple(DataTransform):
    def __init__(self, multiple, center=True):
        self.multiple = multiple
        self.center = center

    def __call__(self, x):
        return pad_to_multiple(x, self.multiple, self.center)[0]


class RandomNoise(DataTransform):
    def __init__(self, mean=0, stddev=0.01):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, x):
        noise = x.new(x.size()).normal_(self.mean, self.stddev)
        return torch.clamp(x + noise, 0, 1)
