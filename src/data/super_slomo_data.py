import glob
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms

import data.dataset
from data import data_transforms


class SuperSlomoData(data.dataset.Dataset):
    data_dirs = {
        "train": "train",
        "val": "validation",
        "test": "test"
    }

    frames_per_sample = 12
    inter_frames_count = 7
    frame_range_size = inter_frames_count + 2

    crop_size = [352, 352]

    def __init__(self,
                 root_dir,
                 data_type,
                 data_folder_name="adobe240fps",
                 preload_mode=None,
                 transform_data=True):
        super().__init__(root_dir, data_type, data_folder_name)

        self.filepaths = self.get_all_filepaths()
        self.transform_data = transform_data

        self.preload_data(preload_mode)

    def get_all_filepaths(self):
        video_dirs = sorted(glob.glob(os.path.join(self.data_path, self.data_dirs[self.data_type], "*")))

        filepaths = [sorted(glob.glob(os.path.join(video_dir, "*"))) for video_dir in video_dirs]

        return filepaths

    def get_transform(self, key, input_data, target_data):
        mean = [0.429, 0.431, 0.397]
        std = [1, 1, 1]

        if self.transform_data:
            transform = transforms.Compose([
                data_transforms.RandomHFlip(),
                data_transforms.ToFloatTensor(),
                data_transforms.RandomCrop(self.crop_size, target_data.size),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            transform = data_transforms.ToFloatTensor()

        return data_transforms.MultiArgTransform([
                data_transforms.TensorStackTransform(transform, combine_mode="cat"),
                data_transforms.Identity()
            ]),\
            transform

    def load_item(self, key):
        start_frame = np.random.randint(0, self.frames_per_sample - (self.frame_range_size - 1))
        end_frame = start_frame + (self.frame_range_size - 1)
        inter_frame_idx = np.random.randint(0, self.inter_frames_count)
        inter_frame_idx_abs = inter_frame_idx + start_frame + 1

        # reverse randomly
        if random.randint(0, 1):
            start_frame, end_frame = end_frame, start_frame
            inter_frame_idx = (self.inter_frames_count-1) - inter_frame_idx

        input_data = [
            [
                Image.open(self.filepaths[key][start_frame]),
                Image.open(self.filepaths[key][end_frame])
            ],
            inter_frame_idx
        ]

        target_data = Image.open(self.filepaths[key][inter_frame_idx_abs])

        # print(start_frame, end_frame, inter_frame_idx_abs, inter_frame_idx)

        return input_data, target_data

    def load_input(self, key):
        raise Exception("Seperate loading not supported")

    def load_target(self, key):
        raise Exception("Seperate loading not supported")

    def __len__(self):
        return len(self.filepaths)
