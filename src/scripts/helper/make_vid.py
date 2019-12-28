import os
import glob
import torch
import torchvision

from tools.tools import load_tensor_image
from trainer.train_logger import TrainLogger


def empty_like(tensor):
    out = torch.zeros_like(tensor)
    out = TrainLogger._apply_colormap(out, "viridis")
    return out


def main():
    root_dir = "/home/kevin/Documents/master-thesis/other_data/consist_vid2"

    vid_files_color = sorted(glob.glob(os.path.join(root_dir, "color*.jpg")))

    vid_files_depth = [
        sorted(glob.glob(os.path.join(root_dir, "1_*.jpg"))),
        sorted(glob.glob(os.path.join(root_dir, "3_*.jpg")))
    ]

    # print(vid_files_color)
    print(vid_files_depth)

    depth_freq = 4

    grids = []

    cur_depths = None

    for i in range(len(vid_files_color)):
        cur_color = load_tensor_image(vid_files_color[i])

        if i % depth_freq == depth_freq - 1:
            cur_depths = [load_tensor_image(col[i // depth_freq]) for col in vid_files_depth]
        elif cur_depths is None:
            cur_depths = [empty_like(cur_color) for _ in vid_files_depth]

        imgs = [cur_color] + cur_depths

        grid = torchvision.utils.make_grid(imgs, padding=3, pad_value=21/255)
        print(grid.shape)

        grids.append(grid)

    # for i, img in enumerate(grids):
    #    torchvision.utils.save_image(img, os.path.join(root_dir, f"img-{i:02d}.png"))

    grids.append(grids[-1])

    TrainLogger._save_tensor_video(grids, os.path.join(root_dir, "vid.avi"), fps=1.5)


if __name__ == "__main__":
    main()
