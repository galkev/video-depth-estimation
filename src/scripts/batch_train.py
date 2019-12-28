import argparse
import os
import csv

# noinspection PyUnresolvedReferences
import pathmagic


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default="train_ids.txt")

    args = parser.parse_args()

    with open(args.file, "r") as f:
        reader = csv.reader(f)
        model_info = list(reader)

    cmd_prefix = "python run_slurm.py --sbatch sbatch/p6000.sbatch -- train_model.py "

    cmd_args_fmt = "--setup {}"

    cmd = " && ".join(cmd_prefix + cmd_args_fmt.format(*m) for m in model_info)

    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
