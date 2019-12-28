import os
import cv2
import numpy as np

# noinspection PyUnresolvedReferences
import pathmagic
from tools.project import proj_dir
from tools.vis_tools import flow_to_vis
from data.video_depth_focus_data import VideoDepthFocusData


def main():
    print(os.getcwd())
    datasets = [VideoDepthFocusData("/home/kevin/Documents/master-thesis/datasets", t, "dining_room")
                for t in ["train", "val", "test"]]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    video_writer = cv2.VideoWriter('/home/kevin/Documents/master-thesis/datasets/vid.avi', fourcc, 60.0, (512*2, 512*2))

    for dataset in datasets:
        for c in range(len(dataset)):
            for f in range(dataset.frames_per_clip):
                color_img = np.array(dataset._load_color(c, f))
                depth_img = np.array(dataset._load_depth(c, f))
                allinfocus = np.array(dataset._load_allinfocus(c, f))
                flow = flow_to_vis(dataset._load_flow(c, f, [512, 512]))

                all_img = np.concatenate([
                    np.concatenate([allinfocus, color_img], axis=1),
                    np.concatenate([np.stack([(depth_img * (0xff/0xffff)).astype(np.uint8)]*3, axis=2), flow], axis=1)
                    ], axis=0)

                img_cv2 = cv2.cvtColor(all_img, cv2.COLOR_RGB2BGR)
                video_writer.write(img_cv2)

                #img_to_show = np.stack([depth_img, depth_img, depth_img], axis=2)
                #cv2.imshow("test", img_to_show)
                #cv2.waitKey(1)

    video_writer.release()


if __name__ == "__main__":
    main()
