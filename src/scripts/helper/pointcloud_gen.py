import re
import glob
import os
import numpy as np
from PIL import Image

# noinspection PyUnresolvedReferences
import pathmagic
from dataset_tools.depth_registration import DepthRegistration


def natural_sorted(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    root_dir = "/home/kevin/Pictures/calib/calib1"

    color_dir = os.path.join(root_dir, "android")
    depth_dir = os.path.join(root_dir, "depth")
    pc_dir = os.path.join(root_dir, "pointclouds")

    color_basename = "hd{}.jpg"
    depth_basename = "depth{}.tif"
    pc_basename = "pointcloud{}.ply"

    depth_regist = DepthRegistration.from_matlab_calib(
        os.path.join(root_dir, "stereo_calib/Calib_Results_stereo.mat"),
        undistort=False
    )

    num_frames = 32

    for i in range(num_frames):
        color_file = os.path.join(color_dir, color_basename.format(i + 1))
        depth_file = os.path.join(depth_dir, depth_basename.format(i + 1))
        pc_file = os.path.join(pc_dir, pc_basename.format(i + 1))

        rgb_img = np.array(Image.open(color_file).convert('RGB'))
        depth_img = np.array(Image.open(depth_file))

        depth_regist.save_pointcloud(pc_file, rgb_img, depth_img)
        #depth_regist.save_pointcloud(pc_file, rgb_img, depth_regist(depth_img))


if __name__ == "__main__":
    main()
