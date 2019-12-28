import torch
import numpy as np
import matplotlib.pyplot as plt
from tools import latex
from tools.camera_lens import CameraLens
import json


metric_labels = {
    "val_acc": "Accuracy (\\%)",
    "val_mse": "MSE"
}


def plot_coc(lens, focus_dists, coc_mm=True, rel_coc=False, max_depth=1):
    depth = torch.linspace(0, max_depth, 1000)
    plt.xlim(0, max_depth)

    for focus_dist in focus_dists:
        coc = lens.get_coc(focus_dist, depth).numpy()

        if rel_coc:
            coc /= lens.sensor_size[0]
            plt.gca().set_yticklabels([r"{:.0f}\%".format(x * 100) for x in plt.gca().get_yticks()])
        elif coc_mm:
            coc *= 1000

        plt.plot(depth.numpy(),
                 coc,
                 label=f"Focus Distance = {focus_dist}")


def load_train_data():
    with open("/home/kevin/Documents/master-thesis/logs/final/tblogs.json", "r") as f:
        train_data = json.load(f)

    tags = [
        "train_train_loss",
        "val_acc",
        "val_mse"
    ]

    metric_stats = {m: {"min": np.Inf, "max": 0} for m in tags}

    for model in train_data.keys():
        for metric in train_data[model].keys():
            data = train_data[model][metric]

            values = [d["value"] for d in data]

            if metric != "val_acc":
                values = [v for v in values if v < 1]

            metric_stats[metric]["min"] = min(metric_stats[metric]["min"], np.min(values))
            metric_stats[metric]["max"] = max(metric_stats[metric]["max"], np.max(values))

    metric_stats["val_acc"]["min"] = 0
    metric_stats["val_acc"]["max"] = 100

    train_data["metric_stats"] = metric_stats

    print(metric_stats)

    return train_data


def plot_training(train_data, metric, models, labels=None, clamp=None):
    # max time: 19h 49m 33s
    max_time = 19 + 49 / 60

    # plt.xlim(right=max_time)
    plt.xlim(right=200)
    # plt.xticks(range(0, int(max_time), 5))

    for i, model in enumerate(models):
        data = train_data[model][metric]

        # x = np.array([(d["wall_time"] - data[0]["wall_time"]) / 3600 for d in data])
        x = np.array([d["step"] for d in data])
        y = np.array([d["value"] for d in data])

        if clamp is not None:
            bad_indices = np.where(y > clamp)[0]

            x = np.delete(x, bad_indices)
            y = np.delete(y, bad_indices)

        label_prefix = "" #  chr(97 + i) + ") "
        label = label_prefix + (labels[i] if labels is not None else model.replace("_", "\\_"))

        plt.plot(x, y, label=label)

    # latex.plot_labels("Time (h)", metric_labels[metric])
    latex.plot_labels("Epochs", metric_labels[metric])

    plt.legend()
