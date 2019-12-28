import glob
import os


def main():
    path = "/home/kevin/Documents/master-thesis/datasets/s7_real_mp4/test"

    files = sorted([os.path.splitext(f)[0] for f in glob.glob(os.path.join(path, "*.mp4"))])

    for i, file in enumerate(files):
        clip_dir = os.path.join(path, "seq" + str(i))
        os.mkdir(clip_dir)

        for ext in ["mp4", "json"]:
            filename = file + "." + ext
            os.rename(os.path.join(path, filename), os.path.join(clip_dir, "color." + ext))

        with open(os.path.join(clip_dir, "params.json"), "w") as f:
            f.write("{}")


if __name__ == "__main__":
    main()
