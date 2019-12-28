import cv2
import os
import glob

# noinspection PyUnresolvedReferences
import pathmagic


def main():
    # file = sys.argv[1]
    # file = "/home/kevin/Documents/master-thesis/render/s7_test/test/seq0/flow0000.exr"
    seq_dirs = glob.glob("/home/kevin/Documents/master-thesis/datasets/s7_4ramp_real/test/seq*")
    file_base = "color{:0>4d}.jpg"
    count = 20
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    for seq_dir in seq_dirs:
        writer = None

        for i in range(count):
            img = cv2.imread(os.path.join(seq_dir, file_base.format(i)))

            if writer is None:
                writer = cv2.VideoWriter(os.path.join(seq_dir, "vid.avi"), fourcc, fps,
                                         (img.shape[1], img.shape[0]))

            writer.write(img)

        writer.release()
        print(seq_dir)

    print("END")


if __name__ == "__main__":
    main()
