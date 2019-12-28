from torchvision import transforms
import numpy as np

import data.dataset
from data import data_transforms


class DDFFData(data.dataset.H5Dataset):
    ddff_h5files = {
        "train": "ddff-dataset-trainval.h5",
        "val": "ddff-dataset-trainval.h5",
        "test": "ddff-dataset-test.h5"
    }

    ddff_h5keys = {
        "train": ["stack_train", "disp_train"],
        "val": ["stack_val", "disp_val"],
        "test": ["stack_test", ""]
    }

    # missing depth at pixels with value 0
    depth_range = [0.0202, 0.28212]

    def __init__(self,
                 root_dir,
                 data_type,
                 data_folder_name="ddff",
                 preload_mode=None):
        super().__init__(
            root_dir,
            data_type,
            data_folder_name,
            self.ddff_h5files,
            self.ddff_h5keys,
        )

        self.depth_normalize = data_transforms.Normalize(self.depth_range[0], self.depth_range[1], eps_min=1e-7)

        self.preload_data(preload_mode)

    def get_transform(self, key, input_data, target_data):
        transforms_input = [
            data_transforms.ToFloatTensor()
        ]

        transforms_target = [
            data_transforms.ToFloatTensor(),
            self.depth_normalize
        ]

        if self.data_type == "test":
            # pad to multiple of 32 for DDFFNet input
            size = np.array(input_data.shape[1:3])
            pad = np.ceil(size / 32).astype(int) * 32 - size
            pad_transform = data_transforms.Pad(((0, pad[0]), (0, pad[1]), (0, 0)), "reflect")

            transforms_input = [pad_transform] + transforms_input
            transforms_target = [pad_transform] + transforms_target

        return data_transforms.TensorStackTransform(transforms.Compose(transforms_input)), \
               transforms.Compose(transforms_target)

    def rev_transform(self, input_data, target_data, output_data):
        rev_normalize = False
        if rev_normalize:
            return [
                input_data,
                self.depth_normalize.rev(target_data),
                self.depth_normalize.rev(output_data)
            ]
        else:
            return [
                input_data,
                target_data,
                output_data
            ]

    def load_target(self, key):
        if self.data_type != "test":
            return super().load_target(key)
        else:
            # there is no test target data in ddff
            return np.full([383, 552, 1], self.depth_range[0])
