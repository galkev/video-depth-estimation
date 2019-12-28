import os


def move_files():
    color_base = "color{:0>4d}.tif"
    depth_base = "depth{:0>4d}.tif"
    allinfocus_base = "allinfocus{:0>4d}.tif"
    flow_base = "flow{:0>4d}.exr"

    num_frames = 5100
    clip_size = 300
    root_dir = "/home/kevin/Documents/master-thesis/datasets/dining_room/raw"

    for i in range(num_frames):
        clip_dir = os.path.join(root_dir, "clip{}".format(i//clip_size))

        if not os.path.isdir(clip_dir):
            os.makedirs(clip_dir)

            os.makedirs(os.path.join(clip_dir, "color"))
            os.makedirs(os.path.join(clip_dir, "depth"))
            os.makedirs(os.path.join(clip_dir, "allinfocus"))
            os.makedirs(os.path.join(clip_dir, "flow"))

        idx_src = i+1
        idx_dest = i % clip_size - 1  # leave first out since unaivable flow at frame 0

        if idx_dest >= 0:
            os.rename(
                os.path.join(root_dir, color_base.format(idx_src)),
                os.path.join(clip_dir, "color", color_base.format(idx_dest)))

            os.rename(
                os.path.join(root_dir, depth_base.format(idx_src)),
                os.path.join(clip_dir, "depth", depth_base.format(idx_dest)))

            os.rename(
                os.path.join(root_dir, allinfocus_base.format(idx_src)),
                os.path.join(clip_dir, "allinfocus", allinfocus_base.format(idx_dest)))

            os.rename(
                os.path.join(root_dir, flow_base.format(idx_src)),
                os.path.join(clip_dir, "flow", flow_base.format(idx_dest)))


def main():
    move_files()


if __name__ == "__main__":
    main()
