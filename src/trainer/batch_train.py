import torch
import time
import os
import numpy as np
from datetime import timedelta

from trainer import Trainer, TrainLogger
import tools.tools as misc_tools
from trainer.model_loader import ModelLoader
from tools import project


class UniversialTrainer(object):
    @staticmethod
    def get_default_params():
        return {
            "name": "Trainer",
            "optimizer": {
                "name": None,
                # args
            },
            "batch_size": 1,
            "num_epochs": 5,
            "max_gradient": None,
            "scheduler_args": None,
            "load_checkpoint": {
                "model": None,
                "epoch": None
            }
        }

    def __init__(self, device):
        self.device = device

    def init_params(self, ps):
        params = self.get_default_params()
        params = misc_tools.dict_update_rec(params, ps)

        return params

    def train(self, params, model, train_data, val_data,
              log_it_freq, log_val_freq, log_prefix, model_loader, checkpoint_freq):
        params = self.init_params(params)

        scheduler_gen = torch.optim.lr_scheduler.StepLR if params["scheduler_args"] is not None else None

        optimizer = project.create_component("torch.optim", **params["optimizer"], params=model.parameters())
        scheduler = scheduler_gen(optimizer, **params["scheduler_args"]) if scheduler_gen is not None else None

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"],
                                                   shuffle=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=params["batch_size"],
                                                 shuffle=False, num_workers=1)

        logger = TrainLogger(log_it_freq, log_val_freq, log_prefix=log_prefix)

        trainer_gen = project.get_class("trainer", params["name"])

        trainer = trainer_gen(model=model,
                              device=self.device,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              max_gradient=params["max_gradient"],
                              logger=logger)

        if params["load_checkpoint"]["model"] is not None:
            checkpoint = model_loader.model(params["load_checkpoint"]["model"])\
                .load("checkpoint", epoch=params["load_checkpoint"]["epoch"])
            trainer.load_checkpoint(checkpoint)

        score = trainer.train(train_loader,
                              val_loader,
                              params["num_epochs"],
                              model_loader=model_loader,
                              checkpoint_freq=checkpoint_freq)

        return score, trainer


class BatchTrain(object):
    @staticmethod
    def get_default_param_sets():
        return {
            "train": {},
            "save": {
                "mode": "all",
                "checkpoint_freq": 0,
                "name": None
            },
            "net": {
                "name": None
                # args
            },
            "dataset": {
                "name": None,
                # args
            },
            "logging": {
                "log_it_freq": 0,
                "log_val_freq": 1
            },
            "paths": {}
        }

    @staticmethod
    def set_param_paths(params, dataset_path, model_path):
        params["paths"] = {
            "model": model_path,
            "data": dataset_path
        }

        return params

    def __init__(self, device=None):
        self.device = device if device is not None \
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.best_pass = {
            "score": np.inf,
            "model": None,
            "params": None,
            "trainer": None
        }

        self.start_time = None

    def init_param_sets(self, ps):
        param_sets = self.get_default_param_sets()
        param_sets = misc_tools.dict_update_rec(param_sets, ps)

        if param_sets["save"]["name"] is None:
            param_sets["save"]["name"] = self.generate_model_name()

        return param_sets

    def generate_model_name(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        return "model-{}".format(timestr)

    def create_model_loader(self, model_path, model_name, model_id=None):
        if model_id is not None:
            model_name = os.path.join(model_name, "model_{}".format(model_id))

        return ModelLoader(model_path, model_name)

    def create_datasets(self, params, data_type):
        return project.create_component(
            "data", **params["dataset"], root_dir=params["paths"]["data"], data_type=data_type)

    def model_log_entry(self, params):
        return "Params = {}".format(misc_tools.dict_str_desc(params))

    def get_model_desc(self, params):
        return "{}\n{}".format(
            project.get_class("net", params["net"]["name"]).__name__,
            self.model_log_entry(params))

    def save_model_info(self, model_loader, params):
        model_loader.save_all({
            # "desc": self.get_model_desc(params),
            "params": params
        })

    def save_model_data(self, model_loader, train_pass):
        model_loader.save_all({
            "model": train_pass["model"],
            "train_stats": train_pass["trainer"].train_stats,
            # "trainer_state": train_pass["trainer"]
        })

    def print_device_info(self):
        print("TRAIN ON", self.device)
        if self.device.type == "cuda":
            print("{} {}GB".format(torch.cuda.get_device_name(self.device),
                                   round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3)))

    def log_progess(self, state, param_sets=None):
        if state == "start":
            print("START GRID TRAIN ({})".format(time.ctime()))
            # print("Model: {}".format(self.model_name(params)))

            print("PARAM SETTING")
            print(misc_tools.dict_str_desc(param_sets))

            self.print_device_info()
        elif state == "end":
            print("DONE GRID TRAIN")

            print("Final best")
            print(self.model_log_entry(self.best_pass["params"]))

            print("Time elapsed: {}".format(timedelta(seconds=time.time() - self.start_time)))
            print("END GRID TRAIN ({})".format(time.ctime()))

    def get_log_prefix(self, i, num_passes):
        return "[{:>{}}/{}]".format(i + 1, len(str(num_passes)), num_passes) if num_passes > 1 else ""

    def train_instance(self, i, num_passes, train_data, val_data, params, save_model):
        print("Train model")
        print(self.get_model_desc(params))

        model_loader = self.create_model_loader(params["paths"]["model"],
                                                params["save"]["name"],
                                                i + 1 if num_passes > 1 else None)

        self.save_model_info(model_loader, params)

        model = project.create_component("net", **params["net"])

        uni_trainer = UniversialTrainer(self.device)
        score, trainer = uni_trainer.train(params["train"],
                                           model,
                                           train_data,
                                           val_data,
                                           log_it_freq=params["logging"]["log_it_freq"],
                                           log_val_freq=params["logging"]["log_val_freq"],
                                           log_prefix=self.get_log_prefix(i, num_passes),
                                           model_loader=model_loader,
                                           checkpoint_freq=params["save"]["checkpoint_freq"])
        train_pass = {
            "score": score,
            "model": model,
            "params": params,
            "trainer": trainer
        }

        if save_model:
            self.save_model_data(model_loader, train_pass)

        return train_pass

    # def train_continue(self, params):

    def eval(self, params_sets):
        pass

    def train(self, param_sets):
        self.start_time = time.time()

        param_sets = self.init_param_sets(param_sets)

        self.log_progess("start", param_sets)

        train_data = self.create_datasets(param_sets, data_type="train")
        val_data = self.create_datasets(param_sets, data_type="val")

        param_prod = misc_tools.dict_cross_prod(param_sets, ignore_keys=["logging"])
        num_passes = len(param_prod)

        for i, params in enumerate(param_prod):
            train_pass = self.train_instance(i, num_passes, train_data, val_data, params,
                                             save_model=param_sets["save"]["mode"] == "all")

            if train_pass["score"] < self.best_pass["score"]:
                # new best score
                self.best_pass = train_pass
                if param_sets["save"]["mode"] != "best":
                    self.best_pass["model"] = None

        if param_sets["save"]["mode"] == "best":
            model_loader = self.create_model_loader(param_sets["paths"]["model"],
                                                    param_sets["save"]["name"])
            self.save_model_data(model_loader, self.best_pass)

        self.log_progess("end")

        return self.best_pass
