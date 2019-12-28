import os
import json
import argparse

# noinspection PyUnresolvedReferences
import pathmagic
from tools.project import proj_dir
from trainer import BatchTrain


def train_model(params_file, model_name):
    with open(params_file) as json_data:
        params = json.load(json_data)

    params = BatchTrain.set_param_paths(
        params,
        dataset_path=proj_dir("datasets"),
        model_path=proj_dir("models", "ddff_mdff"))

    if model_name is not None:
        params["save"]["name"] = model_name

    batch_train = BatchTrain()

    batch_train.train(params)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_id", default=None)
    parser.add_argument("--params", default="tmp_params.json")

    args = parser.parse_args()

    if not os.path.isfile(args.params):
        print("Error params file \"{}\" not found".format(args.params))
        exit(1)

    if args.job_id is not None:
        model_name = "model_{}".format(args.job_id)
    else:
        model_name = None

    train_model(params_file=args.params, model_name=model_name)


if __name__ == "__main__":
    main()
