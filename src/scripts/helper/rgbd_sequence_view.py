import cv2
import os
import numpy as np
import argparse

# noinspection PyUnresolvedReferences
import pathmagic
from dataset_tools.rgbd_record import RgbdView
from dataset_tools.frame_data import FrameDataImage, FrameDataVideo, FrameSync


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--regview", action='store_true')
    parser.add_argument("--allview", action='store_true')
    args = parser.parse_args()

    #root_dir = "/home/kevin/Pictures/frame_test/synced/clip1_regist"
    root_dir = "/home/kevin/Documents/master-thesis/datasets/andasus_dataset/synced/clip1"

    depth_cmap, depth_invalid_color = None, np.array([0x8fff, 0, 0])
    normalize_depth = True

    view = RgbdView(depth_cmap=depth_cmap, depth_invalid_color=depth_invalid_color, normalize_depth=normalize_depth)

    #vid1 = FrameDataVideo(os.path.join(root_dir, "hd"))
    #vid1 = FrameDataImage(root_dir, "color")
    #vid2 = FrameDataImage(root_dir, "depth")
    vid1 = FrameDataImage(os.path.join(root_dir, "color"), "color")
    vid2 = FrameDataImage(os.path.join(root_dir, "depth"), "depth")

    #frame_sync = FrameSync(vid1, vid2)
    #frame_sync.sync()

    #print(frame_sync)

    num_frames = len(vid1)
    print(num_frames)

    while True:
        for i in range(num_frames):
            #frame1 = vid1.get_frame_by_idx(i)
            #frame2 = vid2.get_frame_by_idx(i)

            #frame1, frame2 = frame_sync.get_sync_frame_pair(i)
            frame1, frame2 = vid1.get_frame_by_idx(i), vid2.get_frame_by_idx(i)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

            if frame1.size != frame2.size:
                frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]), interpolation=cv2.INTER_NEAREST)

            #args.regview = True
            #if args.regview or args.allview:
                #color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            view.show_registration(frame1, frame2, weight=0.75)

            #if not args.regview or args.allview:
                #view.show(depth=depth_frame, is_bgr=True)
                #view.show(frame1, frame2, is_bgr=True)

            if frame1.size != frame2.size:
                frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]), interpolation=cv2.INTER_NEAREST)

            #view.show(np.concatenate([frame1, cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)], axis=1))
            #view.show((frame1 / 2 + cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR) / 2).astype(np.uint8))

            key = cv2.waitKey(1) & 0xff

            if key == ord('q'):
                exit(0)


if __name__ == "__main__":
    main()
