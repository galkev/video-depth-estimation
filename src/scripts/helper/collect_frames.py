import glob
import os
import shutil


def main():
    path = "/home/kevin/Documents/master-thesis/logs/maxim/01"
    num_frames = 50
    tensor_idx = 0
    seq_idx = 0

    dest = path + "_collect"

    if not os.path.isdir(dest):
        os.mkdir(dest)

    for f in range(num_frames):
        folder = os.path.join(glob.glob(os.path.join(path, f"{f}*"))[0], "img")

        filename = f"seq{seq_idx:03}_depth_tensor{tensor_idx}.jpg"
        dest_filename = f"depth{f:04}.jpg"

        shutil.copyfile(os.path.join(folder, filename), os.path.join(dest, dest_filename))


if __name__ == "__main__":
    main()
