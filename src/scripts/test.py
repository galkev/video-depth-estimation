import os
import re
import glob


def main():
    seqs = glob.glob("/home/kevin/Documents/master-thesis/datasets/my_book_real/test/*")

    for seq in seqs:
        mp4 = glob.glob(os.path.join(seq, "*.mp4"))[0]
        jpg = os.path.join(seq, "color%04d.jpg")

        cmd = f"ffmpeg -i {mp4} -start_number 0  {jpg}"
        print(cmd)

        os.system(cmd)


if __name__ == "__main__":
    main()

