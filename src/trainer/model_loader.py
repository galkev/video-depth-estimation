import os
import glob
import re

from tools.tools import load_file, save_file


class ModelLoader(object):
    model_files = {
        "model": "model.pth",
        "desc": "desc.txt",
        "train_stats": "train_stats.pkl",
        "trainer_state": "trainer_state.pth",
        "params": "params.json",
        "checkpoint": os.path.join("checkpoints", "checkpoint_epoch_{epoch:04}.pth"),
        "eval": os.path.join("eval", "eval_{epoch}_{data_name}_{data_type}.pth")
    }

    def __init__(self, model_dir, model_name, require_exist=False):
        self.model_dir = model_dir
        self.model_name = model_name

        if not os.path.exists(os.path.join(self.model_dir, self.model_name)):
            if require_exist:
                raise Exception("Error " + os.path.join(self.model_dir, self.model_name) + " not found")
            else:
                os.makedirs(os.path.join(self.model_dir, self.model_name))

    def model(self, new_model_name):
        return ModelLoader(self.model_dir, new_model_name)

    def model_filepath(self, key, **kwargs):
        return os.path.join(self.model_dir, self.model_name, self.model_files[key].format(**kwargs))

    def get_latest_epoch(self):
        checkpoint_dir = os.path.join(self.model_dir, self.model_name, "checkpoints")

        if not os.path.exists(checkpoint_dir):
            raise Exception(checkpoint_dir + " doesnt exist")

        paths = glob.glob(os.path.join(checkpoint_dir, "*"))
        epoch = max(int(re.findall(r"\d+", os.path.split(path)[1])[0]) for path in paths)
        return epoch

    def model_eval_filepath(self, epoch, data_name, data_type):
        return self.model_filepath("eval").format(epoch, data_name, data_type)

    def save(self, key, value, **kwargs):
        save_file(self.model_filepath(key, **kwargs), value, create_dirs=True)

    def save_all(self, data_dict):
        for k, v in data_dict.items():
            self.save(k, v)

    def load(self, key, value=None, **kwargs):
        if key == "checkpoint" and ("epoch" not in kwargs or kwargs["epoch"] is None):
            kwargs["epoch"] = self.get_latest_epoch()
            print("Latest epoch =", kwargs["epoch"])

        return load_file(self.model_filepath(key, **kwargs), value)

    def load_all(self, data_dict):
        for k, v in data_dict.items():
            data_dict[k] = self.load(k, v)
        return data_dict

    def __repr__(self):
        return "ModelLoader"
