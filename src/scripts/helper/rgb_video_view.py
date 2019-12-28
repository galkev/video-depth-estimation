import cv2
import os
import glob

# noinspection PyUnresolvedReferences
import pathmagic


def main():
    # file = sys.argv[1]
    # file = "/home/kevin/Documents/master-thesis/render/s7_test/test/seq0/flow0000.exr"
    seq_dir = "/home/kevin/Documents/master-thesis/datasets/s7_4ramp_real/test/seq7"
    file_base = "color{:0>4d}.jpg"
    count = 20

    while True:
        for i in range(count):
            img = cv2.imread(os.path.join(seq_dir, file_base.format(i)))
            cv2.imshow("Video", img)

            key = cv2.waitKey(100) & 0xff

            if key == ord('q'):
                exit(0)


if __name__ == "__main__":
    main()
