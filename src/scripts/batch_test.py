import argparse
import os
import csv

# noinspection PyUnresolvedReferences
import pathmagic
from model_setup.train_setup_records import train_setups


def get_train_setup(name, model_id, model_desc, use_slurm):
    setup = train_setups[name]
    setup.set_model_info(model_id, model_desc, use_slurm)

    return setup


def main():
    deterministic_val = 1
    test_set = "val"

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default="test_ids.txt")

    args = parser.parse_args()

    with open(args.file, "r") as f:
        reader = csv.reader(f)

        model_info = list(reader)

    #print(model_info)

    cmd_prefix = "python train_model.py --test --deterministic {} --test_set {} ".format(
        deterministic_val,
        test_set
    )

    cmd_args_fmt = "--model_id {} --setup {}"

    cmd = " && ".join(cmd_prefix + cmd_args_fmt.format(*m) for m in model_info)

    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
