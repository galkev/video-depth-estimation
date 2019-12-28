import numpy as np
import torch
from tools.tools import torch_expand_back_as


def _abs_val(x):
    if isinstance(x, np.ndarray) or isinstance(x, float) or isinstance(x, int):
        return np.abs(x)
    else:
        return x.abs()


# sensor_size = [width, height]
# fov = [width, height]
class CameraLens:
    def __init__(self, focal_length, sensor_size_full=(0, 0), resolution=(1, 1), aperture_diameter=None, f_number=None, depth_scale=1):
        self.focal_length = focal_length
        self.depth_scale = depth_scale
        self.sensor_size_full = sensor_size_full

        if aperture_diameter is not None:
            # raise Exception("Check again if giving diameter and not radius")
            self.aperture_diameter = aperture_diameter
            self.f_number = (focal_length / aperture_diameter) if aperture_diameter != 0 else 0
        else:
            self.f_number = f_number
            self.aperture_diameter = focal_length / f_number

        if self.sensor_size_full is not None:
            self.resolution = resolution
            self.aspect_ratio = resolution[0] / resolution[1]
            self.sensor_size = [self.sensor_size_full[0], self.sensor_size_full[0] / self.aspect_ratio]
            self.fov = [CameraLens.calc_fov(self.focal_length, s) for s in self.sensor_size]
            self.focal_length_pixel = [s / (2 * np.tan(fov / 2)) for s, fov in zip(self.resolution, self.fov)]
        else:
            self.resolution = None
            self.aspect_ratio = None
            self.sensor_size = None
            self.fov = None
            self.focal_length_pixel = None

        # print("FOV:", self.fov)
        # print("Lens:", self.__dict__)

    def _to_tensor(self, x, dft=0.0):
        return torch.tensor(x if x is not None else dft)

    def to_torch(self):
        return {
            "focal_length": torch.tensor(self.focal_length),
            "sensor_size": self._to_tensor(self.sensor_size),
            "fov": self._to_tensor(self.fov),
            "aperture": torch.tensor(self.aperture_diameter),
            "f_number": torch.tensor(self.f_number),
            "resolution": torch.tensor(self.resolution),
            "focal_length_pixel": torch.tensor(self.focal_length_pixel)
        }

    @staticmethod
    def calc_fov(focal_length, sensor_length):
        return 2 * np.arctan(0.5 * sensor_length / focal_length)

    def _get_indep_fac(self, focus_distance):
        return (self.aperture_diameter * self.focal_length) / (focus_distance - self.focal_length)

    def get_coc(self, focus_distance, depth):
        focus_distance = torch_expand_back_as(focus_distance, depth)
        return (_abs_val(depth - focus_distance) / depth) * self._get_indep_fac(focus_distance)

    def get_signed_coc(self, focus_distance, depth):
        focus_distance = torch_expand_back_as(focus_distance, depth)
        return ((depth - focus_distance) / depth) * self._get_indep_fac(focus_distance)

    def get_depth_from_signed_coc(self, focus_distance, signed_coc, signed_coc_normalize=None, depth_normalize=None):
        signed_coc = signed_coc if signed_coc_normalize is None else signed_coc_normalize.rev(signed_coc.clone())

        focus_distance = torch_expand_back_as(focus_distance, signed_coc)
        depth = focus_distance / (1 - signed_coc / self._get_indep_fac(focus_distance))

        if depth_normalize is not None:
            depth = depth_normalize(depth)

        return depth

    def get_depth_from_fgbg_coc(self, focus_distance, fgbg, coc, coc_normalize=None, depth_normalize=None):
        signed_coc = -(coc.clone() if coc_normalize is None else coc_normalize.rev(coc.clone()))
        signed_coc[fgbg] *= -1

        return self.get_depth_from_signed_coc(
            focus_distance=focus_distance,
            signed_coc=signed_coc,
            signed_coc_normalize=None,
            depth_normalize=depth_normalize)
