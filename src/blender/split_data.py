import random
import numpy as np
import glob
import os
import json
import argparse


def split_data(path_dict, ratios, rel_path, labels, split_dict=None):
    split_rel_idx = list(np.cumsum(ratios) / np.sum(ratios))
    # print(split_rel_idx)

    if split_dict is None:
        split_dict = {l: {} for l in labels}

    for data_type, data_paths in path_dict.items():
        data_paths = [os.path.relpath(p, rel_path) for p in data_paths]

        if data_type in split_dict[labels[0]]:
            old_size = len(data_paths)
            print(data_paths)
            old_data = sum([split_dict[k][data_type] for k in split_dict.keys()], [])
            data_paths = [d for d in data_paths if d not in old_data]

            print(len(data_paths), old_size)

        random.shuffle(data_paths)
        print(data_type, len(data_paths))
        for i, (start, end) in enumerate(zip([0] + split_rel_idx[:-1], split_rel_idx)):
            # print(start, end)
            start_abs, end_abs = int(round(start * len(data_paths))), int(round(end * len(data_paths)))

            part = data_paths[start_abs:end_abs]

            if data_type not in split_dict[labels[i]]:
                split_dict[labels[i]][data_type] = part
                print("Created", labels[i], data_type, len(part))
            else:
                split_dict[labels[i]][data_type].extend(part)
                print("Appended", labels[i], data_type, len(part))

            print(labels[i], data_type, len(part), np.sum([os.stat(os.path.join(rel_path, p)).st_size for p in part]) // 1024**2, "mb")

    #print(json.dumps(split_dict, indent=4))

    # return
    with open(os.path.join(rel_path, "split.json"), "w") as f:
        json.dump(split_dict, f, indent=4)


def get_data_paths(scene_pool_path):
    tex_paths = \
        glob.glob(os.path.join(scene_pool_path, "TextureHaven", "*")) + \
        glob.glob(os.path.join(scene_pool_path, "grsites", "*", "*.*")) + \
        glob.glob(os.path.join(scene_pool_path, "texninja", "*", "*.*"))

    obj_paths = glob.glob(os.path.join(scene_pool_path, "Thingi10K/raw_meshes/*.stl"))

    env_paths = glob.glob(os.path.join(scene_pool_path, "TextureHavenEnv", "*"))

    max_obj_file_size = 1024 ** 2
    obj_paths = [f for f in obj_paths if os.stat(f).st_size <= max_obj_file_size]

    return {"tex": tex_paths, "obj": obj_paths, "env": env_paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/kevin/Documents/master-thesis/scene_pool")
    args = parser.parse_args()

    scene_pool_path = args.path

    if not os.path.isdir(scene_pool_path):
        print("Error", scene_pool_path, "not found")
    else:
        with open(os.path.join(scene_pool_path, "split.json"), "r") as f:
            split_dict = json.load(f)

        split_data(get_data_paths(scene_pool_path), [80, 20], scene_pool_path, ["train", "test"], split_dict)


if __name__ == "__main__":
    main()
