import argparse
import os
import glob
import numpy as np
import cv2
import time

from dataset_tools.depth_registration import DepthRegistration
from dataset_tools.frame_data import FrameDataVideo, FrameDataImage, FrameSync


def main():
    depth_img_type = "depth"

    parser = argparse.ArgumentParser()

    root_dir = "/home/kevin/Documents/master-thesis/datasets/andasus_dataset"
    parser.add_argument("--dir", default=root_dir)
    args = parser.parse_args()
    root_dir = args.dir

    clip_dirs = glob.glob(os.path.join(root_dir, "toproc", "*"))

    """if perform_regist:
        depth_regist = DepthRegistration.from_matlab_calib(
            os.path.join(root_dir, "calib/Calib_Results_stereo.mat"),
            undistort=False,
            rot_depth=1
        )

        transform = lambda hd, depth: [hd, depth_regist(depth)]
    elif rot_img:
        #transform = lambda hd, depth: [np.rot90(hd, k=-1), np.rot90(depth, k=-1)]
        transform = lambda hd, depth: [np.rot90(hd, k=2-1), cv2.cvtColor(np.rot90(depth, k=0-1), cv2.COLOR_RGB2BGR)]
    else:
        transform = None"""

    #mode = "calib"
    # mode = "regist"
    mode = None

    if mode == "calib":
        transform = lambda hd, depth: [
            np.rot90(cv2.resize(hd, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST), k=-1),
            np.rot90(depth, k=-1)
        ]
    elif mode == "regist":
        depth_regist = DepthRegistration.from_matlab_calib(
            os.path.join(root_dir, "calib/Calib_Results_stereo.mat"),
            undistort=False,
            rot_depth=1
        )

        transform = lambda hd, depth: [hd, depth_regist(depth[:,:,0])]
    else:
        transform = None


    print("Start")
    print(clip_dirs)

    for clip_dir in clip_dirs:
        start_time = time.time()

        frame_data_android = FrameDataVideo(os.path.join(clip_dir, "hd"))
        frame_data_rgbd = FrameDataImage(os.path.join(clip_dir, "depth"), img_type=depth_img_type)

        sync = FrameSync(frame_data_android, frame_data_rgbd)
        sync.sync()

        print(sync)
        sync.save(os.path.join(root_dir, "synced", os.path.split(clip_dir)[1]), transform=transform, frame_limit=None)

        print("{} processed ({:.2f}s)".format(clip_dir, time.time() - start_time))


if __name__ == "__main__":
    main()
