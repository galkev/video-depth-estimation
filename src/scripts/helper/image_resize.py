from PIL import Image
import sys
import argparse
import glob
import os


def main():
    #parser = argparse.ArgumentParser()

    #parser.add_argument("--dir")
    #parser.add_argument("--perc")

    #args = parser.parse_args()

    #if args.dir is None or args.perc is None:
        #print("Not enough args")
        #exit(1)

    #files = glob.glob(os.path.join(args.dir, "*"))
    #factor = int(args.perc) / 100

    files = glob.glob(os.path.join("/home/kevin/Pictures/frame_test_data/synced/color*.tif"))

    for file in files:
        img = Image.open(file)
        #new_size = (int(img.size[0] * factor), int(img.size[1] * factor))
        #new_size = (480,640)
        new_size = (640, 480)
        img = img.resize(new_size, Image.ANTIALIAS)
        img.save(file)


if __name__ == "__main__":
    main()

