import os
import argparse
import glob
import re


def natural_sorted(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default=".")
    parser.add_argument("--prefix")

    args = parser.parse_args()

    if args.prefix is None:
        print("Not enough args")
        exit(1)

    root_dir = args.dir

    files = natural_sorted(glob.glob(os.path.join(root_dir, "*")))

    print("\n".join(files))

    for i, file in enumerate(files):
        ext = os.path.splitext(file)[1]
        os.rename(file, os.path.join(root_dir, "".join([args.prefix, str(i+1), ext])))


if __name__ == "__main__":
    main()
