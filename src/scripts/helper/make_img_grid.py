import os
import glob
import PIL
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import re
import numpy as np
import cv2


def natural_sorted(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_filenames():
    return natural_sorted(glob.glob("/home/kevin/Documents/master-thesis/datasets/andasus_dataset/calib/depth/*.tif"))


def get_images2():
    # tensors = [transforms.ToTensor()(img) for img in imgs]
    return [Image.open(filename) for filename in get_filenames()]


def read_vid_frame(filename, idx=0):
    cap = cv2.VideoCapture(filename)

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()

    if not ret:
        print("Error cap.read")
        frame = None
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms.ToTensor()(
            Image.fromarray(frame).resize((frame.shape[1] // 4, frame.shape[0] // 4), resample=Image.ANTIALIAS))

    cap.release()

    return frame


def get_images():
    files = natural_sorted(glob.glob("/home/kevin/Documents/master-thesis/datasets/andasus_dataset/raw/clips/clip*/hd/*.mp4"))

    return [read_vid_frame(file, idx=0) for file in files]


def main():
    tensors = get_images()

    assert len(tensors) > 0

    # Image.resize(, interpolation=PIL.Image.ANTIALIAS)

    nrow = 3
    pad = 2

    name = "andasus_grid_and"

    directory = "/home/kevin/Documents/master-thesis/thesis/figures/pictures"

    torchvision.utils.save_image(tensors, os.path.join(directory, name + ".png"), nrow=nrow, padding=pad, pad_value=1)


if __name__ == "__main__":
    main()
