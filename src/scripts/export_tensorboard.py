import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import json
import argparse


def export_tensorboard(check_only=False):
    root_dir = os.path.expanduser("~/Documents/master-thesis")

    tb_dir = os.path.join(root_dir, "logs/tblogs")

    log_dirs = sorted(glob.glob(os.path.join(tb_dir, "*")))

    tags = ["train_train_loss", "val_acc", "val_mse"]

    train_data = {"tags": tags}

    for tb_output_folder in log_dirs:
        x = EventAccumulator(path=tb_output_folder)
        x.Reload()
        x.FirstEventTimestamp()

        if tags[0] in x.scalars.Keys():
            name = os.path.basename(tb_output_folder)
            train_data[name] = {}

            for tag in tags:
                train_data[name][tag] = []

                for e in x.Scalars(tag):
                    train_data[name][tag].append({
                        "step": e.step,
                        "value": e.value,
                        "wall_time": e.wall_time,
                    })

                if check_only:
                    break

            print(tb_output_folder, "-> OK")
        else:
            print(tb_output_folder, "-> NO")

    if not check_only:
        with open(os.path.join(root_dir, "logs/tblogs.json"), "w") as f:
            json.dump(train_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    export_tensorboard(check_only=args.check)


if __name__ == "__main__":
    main()
