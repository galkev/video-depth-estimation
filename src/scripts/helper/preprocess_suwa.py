import glob
import os
import shutil
import json
from PIL import Image


def collect_seq_files(seq_name, path):
    files = []

    for i in range(9999999):
        filename = path.format(i)

        if not os.path.exists(filename):
            break

        files.append({"file": filename})

    if seq_name == "metals":
        print("Reverse file metals")
        files = list(reversed(files))

    return files


def append_focus_dist(seq_dict):
    seq_dict["has_config"] = seq_dict["name"] not in ["bucket", "kitchen"]

    if not seq_dict["has_config"]:
        print(seq_dict["name"], "no config")
        return

    calib_name_trans = {
        "metals": "metal",
        "zeromotion": "GT",
        "largemotion": "GTLarge",
        "smallmotion": "GTSmall",
    }

    cal_dir = "/home/kevin/Documents/master-thesis/other_data/depth_from_focus_data2/calibration"
    config_path = os.path.join(cal_dir, calib_name_trans.get(seq_dict["name"], seq_dict["name"]), "calibrated.txt")

    with open(config_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line_parts = line.strip().split()

            if i == len(lines) - 1:
                assert len(line_parts) == 1
                focal_length = line_parts[0]
                seq_dict["focal_length"] = float(focal_length)
            else:
                assert len(line_parts) == 2
                focus_dist_cm, aperture = line_parts
                seq_dict["files"][i]["focus_dist"] = float(focus_dist_cm) / 100
                seq_dict["files"][i]["aperture"] = float(aperture)


def sort_focus_dists(seq_dict):
    if seq_dict["has_config"]:
        if seq_dict["files"][0]["focus_dist"] > seq_dict["files"][-1]["focus_dist"]:
            seq_dict["files"] = list(reversed(seq_dict["files"]))

        remove_outliers(seq_dict)

        check_order(seq_dict)
    elif seq_dict["name"] == "bucket":
        seq_dict["files"] = list(reversed(seq_dict["files"]))
        print("Reverse bucket")


def check_order(seq_dict):
    focus_dists = [f["focus_dist"] for f in seq_dict["files"]]

    if not focus_dists == sorted(focus_dists):
        print(focus_dists)
        print(sorted(focus_dists))
        raise Exception()


def remove_outliers(seq_dict):
    files_filtered = []

    cur_foc = 0
    for file in seq_dict["files"]:
        if cur_foc < file["focus_dist"]:
            files_filtered.append(file)
            cur_foc = file["focus_dist"]
        else:
            print("not in order")

    assert len(files_filtered) > 3

    seq_dict["files"] = files_filtered


camera = {
    "d80": [5, 6, 9, 12],
    "s3": [0, 1, 2, 3, 4, 7, 8, 10, 11]
}

camera_config = {
    "d80": {
        "focal_length": 22.0,
        "sensor_dim": [23.6, 15.8],
        "f_number": 3.5
    },
    "s3": {
        "focal_length": 3.7,
        "sensor_dim": [4.54, 3.42],
        "f_number": 2.6
    }
}


def get_camera_config(idx):
    cam_label = None

    for l, v in camera.items():
        if idx in v:
            cam_label = l
            break

    if cam_label is None:
        raise Exception("Error id not found")

    return camera_config[cam_label]


def main():
    print("START")

    seq_dirs = glob.glob("/home/kevin/Documents/master-thesis/other_data/depth_from_focus_data2/Aligned/*")
    source_name = "a{:0>2d}.jpg"
    target_name = "color{:0>4d}.jpg"
    target_dir = "/home/kevin/Documents/master-thesis/datasets/suwajanakorn/test/seq{}"

    seq_dicts = []

    for seq_id, seq_dir in enumerate(seq_dirs):
        print(seq_dir)
        seq_dict = {
            "name": os.path.basename(seq_dir),
            "files": collect_seq_files(os.path.basename(seq_dir), os.path.join(seq_dir, source_name))
        }

        # print(seq_dict)

        append_focus_dist(seq_dict)
        sort_focus_dists(seq_dict)

        seq_dicts.append(seq_dict)

    seq_dicts.sort(key=lambda s: s["name"])

    # print(json.dumps(seq_dicts, indent=4))

    for seq_id, seq_dict in enumerate(seq_dicts):
        new_path = target_dir.format(seq_id)
        os.makedirs(new_path)

        if "aperture" in seq_dict["files"][0]:
            assert len(set(file["aperture"] for file in seq_dict["files"])) == 1
            print(seq_dict["files"][0]["aperture"])

        img_res = Image.open(seq_dict["files"][0]["file"]).size

        params = {
            "focusRangeStart": seq_dict["files"][0]["focus_dist"] if seq_dict["has_config"] else 0,
            "focusRangeEnd": seq_dict["files"][-1]["focus_dist"] if seq_dict["has_config"] else 0,
            "name": seq_dict["name"],
            "frames": [
                {"idx": i,
                 "focDist": file.get("focus_dist", 0)}
                for i, file in enumerate(seq_dict["files"])
            ],
            "resolution": img_res,
            **get_camera_config(seq_id)
        }

        with open(os.path.join(new_path, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        for file_id, file in enumerate(seq_dict["files"]):
            target_file = os.path.join(target_dir.format(seq_id), target_name.format(file_id))

            # print(file["file"], ">", target_file)

            shutil.copy(file["file"], target_file)

    """
    for i in to_rotate:
        seq_dir_img = os.path.join(target_dir.format(i), "*.jpg")
        os.system(f"mogrify -rotate \"90\" {seq_dir_img}")
        os.system(f"mogrify -auto-orient {seq_dir_img}")
    """

    print("END")


if __name__ == "__main__":
    main()

