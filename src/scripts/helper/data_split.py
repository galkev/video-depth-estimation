import random
import glob
import os
import argparse
import json
import numpy as np

# noinspection PyUnresolvedReferences
import pathmagic
from tools.project import proj_dir, use_slurm_system


def split_data(data_paths, ratios, labels, rel_path=None):
    split_rel_idx = list(np.cumsum(ratios) / np.sum(ratios))
    # print(split_rel_idx)

    split_dict = {l: {} for l in labels}

    random.shuffle(data_paths)
    print(len(data_paths))
    for i, (start, end) in enumerate(zip([0] + split_rel_idx[:-1], split_rel_idx)):
        # print(start, end)
        start_abs, end_abs = int(round(start * len(data_paths))), int(round(end * len(data_paths)))

        part = data_paths[start_abs:end_abs]
        split_dict[labels[i]] = [os.path.relpath(p, rel_path) for p in part] if rel_path is not None else part

    return split_dict


def main():
    ratio = [80, 20]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="art_scene")
    args = parser.parse_args()

    use_slurm_system()
    data_dir = proj_dir("datasets", args.dataset)

    assert os.path.exists(proj_dir("datasets"))

    data_paths = glob.glob(os.path.join(data_dir, "train", "*"))
    data_paths = [os.path.basename(p) for p in data_paths]

    split_dict = split_data(data_paths, ratio, ["train", "val"])

    with open(os.path.join(data_dir, "split.json"), "w") as f:
        json.dump(split_dict, f, indent=4)


if __name__ == "__main__":
    main()
