import torch
from data import Dataset
import glob
import os
from PIL import Image
from torchvision import transforms
from tools.tools import natural_sort


class SimpleData(Dataset):
    def load_input(self, key):
        pass

    def load_target(self, key):
        pass

    def rev_transform(self, input_data, target_data, output_data):
        pass

    def __init__(self,
                 root_dir,
                 data_type,
                 data_folder_name,
                 ramp_length,
                 ramps_per_clip):
        super().__init__(root_dir, data_type, data_folder_name)

        self.clip_dirs = natural_sort(glob.glob(os.path.join(root_dir, data_folder_name, data_type, "*")))
        self.ramp_length = ramp_length
        self.ramps_per_clip = ramps_per_clip

        self.color_basename = "color{:0>4d}.jpg"

    def __len__(self):
        return len(self.clip_dirs) * self.ramps_per_clip

    def load_item(self, key):
        clip_idx = key // self.ramps_per_clip
        ramp_idx = key % self.ramps_per_clip

        files = [os.path.join(self.clip_dirs[clip_idx], self.color_basename.format(ramp_idx * self.ramp_length + i))
                 for i in range(self.ramp_length)]

        return [Image.open(f) for f in files]

    def transform(self, key, data):
        return torch.stack([transforms.ToTensor()(d) for d in data])

    def get_num_clips(self):
        return len(self.clip_dirs)