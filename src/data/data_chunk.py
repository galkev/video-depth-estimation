import os
import numpy as np
from PIL import Image
from tools.tools import load_exr, load_blender_flow_exr, GetItemBase


def create_placeholder_img(dim, val=0, dtype=None):
    return np.full(dim, val, dtype=dtype)


class ImageFileBase:
    def __init__(self, path, data_type):
        self.path = path
        self.data_type = data_type
        self.resolution = None
        self.data = None

    def _load(self):
        raise NotImplementedError()

    def get(self):
        if self.data is None:
            self.data = self._load()

        return self.data


class ImageFile(ImageFileBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self):
        return Image.open(self.path)


class DepthFile(ImageFileBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self):
        if os.path.isfile(self.path):
            return load_exr(self.path, ["R"])
        elif self.data_type == "test":
            return create_placeholder_img([self.resolution[1], self.resolution[0], 1], np.nan)
        else:
            raise Exception("Depth not found: " + self.path)


class FlowFile(ImageFileBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self):
        return load_blender_flow_exr(self.path)


class FrameTypeSeqChunk(GetItemBase):
    def __init__(self, base_path, data_class, indices, data_type):
        self.indices = indices

        self.data_files = [
            data_class(base_path.format(idx), data_type)
            for idx in indices
        ]

    def set_resolution(self, resolution):
        for f in self.data_files:
            f.resolution = resolution

    def get(self, idx):
        return self.data_files[idx].get()

    def __len__(self):
        return len(self.indices)


class FrameTypeSeqChunkConst(GetItemBase):
    def __init__(self, data):
        self.data = data

    def get(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class FrameSeqChunk:
    def __init__(self, base_path, indices, params, data_type, img_ext, lens):
        data_type_config = [
            ("color", ImageFile, "color{:0>4d}." + img_ext),
            ("depth", DepthFile, "depth{:0>4d}.exr"),
            ("flow", FlowFile, "flow{:0>4d}.exr"),
            ("allinfocus", ImageFile, "allinfocus{:0>4d}.tif")
        ]

        self.data = {
            data_label: FrameTypeSeqChunk(os.path.join(base_path, base_name), data_class, indices, data_type)
            for data_label, data_class, base_name in data_type_config
        }

        for chunk in self.data.values():
            chunk.set_resolution(self["color"][0].size)

        self.data["focus_dist"] = self._get_focus_dists(params, indices)
        self.data["lens"] = lens

    def _get_all_focus_dists(self, params):
        return [f["focDist"] for f in params["frames"]]

    def _get_focus_dists(self, params, frame_indices):
        focus_dists = self._get_all_focus_dists(params)
        return [focus_dists[i] for i in frame_indices]

    def __getitem__(self, data_label):
        return self.data[data_label]

    def __len__(self):
        return len(next(iter(self.data.values())))
