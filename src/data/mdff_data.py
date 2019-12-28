import os
import pickle
from PIL import Image
from torchvision import transforms
import numpy as np

import data
import data.dataset
from tools import tools
from data import data_transforms


def MdffH5Data(root_dir,
               data_type,
               data_folder_name="mDFFDataset",
               preload_mode=None):
    mdff_data = MdffH5DataFull(root_dir, data_type, data_folder_name, preload_mode)
    # leave out first image if not in test mode since its empty/black
    return data.DatasetSubset(mdff_data, range(1, len(mdff_data))) if data_type != "test" \
        else mdff_data


class MdffH5DataFull(data.dataset.H5Dataset):
    mdff_h5files = {
        "train": "mDFF-dataset_cropped_clipped.h5",
        "val": "mDFF-dataset_cropped_clipped.h5",
        "test": "mDFF-dataset_cropped_clipped.h5"
    }

    mdff_h5keys = {
        "train": ["stack_train", "disp_train"],
        "val": ["stack_val", "disp_val"],
        "test": ["stack_test", "disp_test"]
    }

    depth_range = [0.1002, 4.0]

    def __init__(self,
                 root_dir,
                 data_type,
                 data_folder_name="mDFFDataset",
                 preload_mode=None):
        super().__init__(
            root_dir,
            data_type,
            data_folder_name,
            self.mdff_h5files,
            self.mdff_h5keys
        )

        self.depth_normalize = data_transforms.Normalize(self.depth_range[0], self.depth_range[1], eps_min=1e-7)

        self.preload_data(preload_mode)

    def get_transform(self, key, input_data, target_data):
        transforms_input = [
            data_transforms.ToFloatTensor(255)
        ]

        transforms_target = [
            data_transforms.ToFloatTensor(),
            self.depth_normalize
        ]

        if self.data_type == "test":
            # pad to multiple of 32 for DDFFNet input
            size = np.array(input_data.shape[1:3])
            pad = np.ceil(size / 32).astype(int) * 32 - size
            pad_transform_input = data_transforms.Pad(((0, pad[0]), (0, pad[1]), (0, 0)), "reflect")
            pad_transform_target = data_transforms.Pad(((0, pad[0]), (0, pad[1])), "constant")

            transforms_input = [pad_transform_input] + transforms_input
            transforms_target = [pad_transform_target] + transforms_target

        return [
            data_transforms.TensorStackTransform(transforms.Compose(transforms_input)),
            transforms.Compose(transforms_target)
        ]

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


class MdffData(data.dataset.Dataset):
    focalstack_folder = "focalstacks"
    depth_folder = "depth"
    depthmap_name = "depth.png"
    focalstack_size = 10

    def __init__(self,
                 root_dir,
                 data_type,
                 data_folder_name="mDFFDataset",
                 preload_mode=None):
        super().__init__(root_dir, data_type, data_folder_name)

        self.focalstack_paths = []
        self.depth_paths = []

        image_id_file = "{}_data.csv".format(self.data_type)

        path_sets = tools.load_file(os.path.join(self.data_path, image_id_file))

        for path_set in path_sets:
            self.focalstack_paths.append(path_set[:10])
            self.depth_paths.append(path_set[-1])

        crops_file = "{}_data_valid_crops.pkl".format(data_type)
        with open(os.path.join(self.data_path, crops_file), "rb") as f:
            self.valid_crops = pickle.load(f)

        self.preload_data(preload_mode)

    def get_transform(self, key, input_data, target_data):
        transform = transforms.Compose([
            data_transforms.ToFloatTensor(),
            data_transforms.RandomCrop(224, target_data.size, self.valid_crops[key])
        ])

        return data_transforms.TensorStackTransform(transform), transform

    def load_input(self, key):
        fs_images = []

        for fs_file in self.focalstack_paths[key]:
            path = os.path.join(self.data_path, fs_file)
            fs_img = Image.open(path).convert('RGB')

            fs_images.append(fs_img)

        return fs_images

    def load_target(self, key):
        path = os.path.join(self.data_path, self.depth_paths[key])
        depth_image = Image.open(path)

        return depth_image

    def __len__(self):
        return len(self.focalstack_paths)
