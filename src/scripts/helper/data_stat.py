import random
import glob
import os
import argparse
import numpy as np
import json

# noinspection PyUnresolvedReferences
import pathmagic
from data.video_depth_focus_data import VideoDepthFocusData
from tools.project import proj_dir


def dict_stat_compose(dicts, par_key=""):
    res_dic = {}

    for k in dicts[0].keys():
        idx_range = range(len(dicts)) if par_key != "flow" else range(1, len(dicts)-1)
        val = [dicts[i][k] for i in idx_range]

        if isinstance(val[0], dict):
            res_dic[k] = dict_stat_compose(val, par_key=k)
        else:
            if k == "min":
                res_dic[k] = np.min(val)
            elif k == "avg":
                res_dic[k] = np.mean(val)
            elif k == "max":
                res_dic[k] = np.max(val)
            elif k == "nonzero":
                res_dic[k] = np.sum(val)
                res_dic["nonzero_per"] = round(res_dic["nonzero"] / (len(val) * 512 * 512 * 4) * 100, 2)

    return res_dic


def default(o):
    return float(o)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    VideoDepthFocusData.use_config = False
    data = VideoDepthFocusData(proj_dir("datasets"), "train", "dining_room")

    stats_per_frame = []
    stats_per_clip = []

    params = None

    #args.update = True
    if not args.update:
        for clip in range(len(data)):
            clip_stats = []
            params = data._load_params(clip)
            for idx in range(25):
                depth = np.array(data._load_depth(clip, idx, params))
                flow = np.array(data._load_flow(clip, idx, params, [512, 512]))
                depth = depth[depth < 0xffff]

                clip_stats.append({
                    "idx": idx,
                    "depth": {
                        "min": depth.min(),
                        "avg": depth.mean(),
                        "max": depth.max()
                    },
                    "flow": {
                        "min": flow.min(),
                        "avg": flow.mean(),
                        "max": flow.max(),
                        "nonzero": np.count_nonzero(flow)
                    }
                })

            clip_name = os.path.basename(data.clip_dirs[clip])

            stats_per_frame.append({"clip": clip_name, "frames": clip_stats})

            print("Clip", clip, "done")

        with open(os.path.join(data.data_path, "stats_per_frame.json"), "w") as f:
            json.dump(stats_per_frame, f, default=default, indent=4)
    else:
        with open(os.path.join(data.data_path, "stats_per_frame.json"), "r") as f:
            stats_per_frame = json.load(f)

    for clip in range(len(data)):
        params = data._load_params(clip)
        clip_name = os.path.basename(data.clip_dirs[clip])
        stats_per_clip.append({"clip:": clip_name, "frames": dict_stat_compose(stats_per_frame[clip]["frames"])})

    stats_total = {
        "focus_min": params["frames"][0]["focDist"],
        "focus_max": params["frames"][-1]["focDist"],
        **dict_stat_compose(stats_per_clip)
    }

    #print(stats_per_frame)
    print(stats_per_clip)
    print(stats_total)

    with open(os.path.join(data.data_path, "stats_total.json"), "w") as f:
        json.dump(stats_total, f, default=default, indent=4)

    with open(os.path.join(data.data_path, "stats_per_clip.json"), "w") as f:
        json.dump(stats_per_clip, f, default=default, indent=4)


if __name__ == "__main__":
    main()
