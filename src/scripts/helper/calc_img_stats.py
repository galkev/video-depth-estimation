import glob
from PIL import Image
import numpy as np


def main():
    #image_files = glob.glob("/home/kevin/Documents/master-thesis/datasets/andasus_dataset/synced/clip1/depth/*.tif")
    image_files = glob.glob("/home/kevin/Documents/master-thesis/datasets/dining_room/*/clip*/depth/*.tif")

    min_val = np.inf
    max_val = 0

    for file in image_files:
        img = np.array(Image.open(file))
        img = img[np.nonzero(img)]

        min_val = min(np.min(img), min_val)
        max_val = max(np.max(img), max_val)

    print("Min val:", min_val)
    print("Max val:", max_val)


if __name__ == "__main__":
    main()
