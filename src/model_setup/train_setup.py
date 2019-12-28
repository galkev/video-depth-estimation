from tools.project import proj_dir, use_slurm_system
from trainer.model_loader import ModelLoader
import torch
from tools import tools
from trainer import TrainLogger
from tools.tools import json_support
from data.simple_data import SimpleData
from trainer.simple_tester import SimpleTester


@json_support
class TrainTestSetupBase(object):
    @staticmethod
    def create_data_loader_helper(data, data_type, batch_size, num_threads=4):
        loader_dataset = data
        shuffle = data_type == "train"

        if data_type == "test":
            batch_size = 1

        return torch.utils.data.DataLoader(loader_dataset, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_threads)

    def __init__(self):
        self.model_id = None
        self.model_desc = None

        self.device = tools.default_device()

        self.model = None

        self.trainer = None
        self.model_loader = None
        self.logger = None

        self.device_desc = tools.device_info(self.device)

    def create_components(self, mode):
        font = proj_dir("fonts", "Roboto-Regular.ttf")

        self.model = self.create_model(None, mode)

        self.logger = self.create_logger(proj_dir("logs"), font)
        self.trainer = self.create_trainer(self.device, self.model, None, self.logger)

        self.model_loader = self.create_model_loader(self.model_id)

    def set_model_info(self, model_id, model_desc):
        self.model_id = model_id
        self.model_desc = model_desc

    def load_checkpoint_from_disk(self, model_id, epoch=None):
        return TrainTestSetupBase.create_model_loader(model_id).model(
            self.get_model_name(model_id)).load("checkpoint", epoch=epoch)

    def load_checkpoint(self, epoch, model_id=None, mode="train"):
        if model_id is None:
            model_id = self.model_id

        checkpoint = self.model_loader.model(self.get_model_name(model_id)).load("checkpoint", epoch=epoch)
        self.trainer.load_checkpoint(checkpoint, mode=mode)

    def create_logger(self, log_dir, font):
        return TrainLogger()

    @staticmethod
    def create_model_loader(model_id):
        return ModelLoader(TrainTestSetupBase.get_model_path(), TrainTestSetupBase.get_model_name(model_id))

    @staticmethod
    def get_model_name(model_id=None):
        return "model_{}".format(model_id)

    def get_dataset_path(self):
        return proj_dir("datasets")

    @staticmethod
    def get_model_path():
        return proj_dir("models")

    def create_trainer(self, device, model, optimizer, logger):
        raise NotImplementedError

    def create_model(self, num_in_channels, mode):
        raise NotImplementedError

    def create_dataset(self, dataset_path, data_type):
        raise NotImplementedError

    def create_data_loader(self, data, data_type):
        raise NotImplementedError


@json_support
class TrainSetup(TrainTestSetupBase):
    def __init__(self):
        super().__init__()

        self.train_data = None
        self.val_data = None
        self.optimizer = None
        self.scheduler = None

        self.test_data = None

    def finalize_components(self):
        pass

    def create_components(self, load_pretrained=True, mode=None):
        super().create_components(mode)

        print(mode)

        if mode is None or mode == "train":
            self.create_components_train(load_pretrained=load_pretrained)

        if mode is None or mode == "test":
            self.create_components_test()

        self.finalize_components()

    def load_pretrained_modules(self, model):
        pass

    def create_components_train(self, load_pretrained=True):
        if load_pretrained:
            self.load_pretrained_modules(self.model)

        self.train_data = self.create_dataset(self.get_dataset_path(), "train")
        self.val_data = self.create_dataset(self.get_dataset_path(), "val")

        self.optimizer = self.create_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.trainer.set_optimizer(self.optimizer)

        self.scheduler = self.create_scheduler(self.optimizer)
        self.trainer.set_scheduler(self.scheduler)

    def create_components_test(self):
        self.test_data = self.create_dataset(self.get_dataset_path(), "test")

    def create_scheduler(self, optimizer):
        raise NotImplementedError

    def create_optimizer(self, parameters):
        raise NotImplementedError

    def train_model(self, train_loader, val_loader, model_loader):
        raise NotImplementedError

    def test_model(self, test_loader):
        raise NotImplementedError

    def train(self, create_comp=True):
        if create_comp:
            self.create_components(mode="train")

        train_loader = self.create_data_loader(self.train_data, "train")
        val_loader = self.create_data_loader(self.val_data, "val")

        self.train_model(train_loader, val_loader, self.model_loader)

    def test(self, data_type="test", create_comp=True):
        if create_comp:
            self.create_components()

        test_loader = None

        if data_type == "train":
            test_loader = self.create_data_loader(self.train_data, "test")
        elif data_type == "val":
            test_loader = self.create_data_loader(self.val_data, "test")
        elif data_type == "test":
            test_loader = self.create_data_loader(self.test_data, "test")

        self.test_model(test_loader)

    def train_continue(self, epoch, model_id=None):
        self.create_components()
        self.load_checkpoint(epoch, model_id, mode="train")
        self.train(create_comp=False)

    def test_epoch(self, epoch, model_id, data_type="test"):
        self.create_components(load_pretrained=False, mode="test")
        self.load_checkpoint(epoch, model_id, mode="test")
        self.test(data_type=data_type, create_comp=False)

    def simple_test(self, epoch, model_id, dataset, ramp_length, ramps_per_clip, model_out_idx):
        self.model = self.create_model(None, "test")
        self.trainer = SimpleTester(self.model, self.device)
        self.model_loader = self.create_model_loader(self.model_id)
        self.test_data = SimpleData(self.get_dataset_path(), "test", dataset, ramp_length, ramps_per_clip)

        self.load_checkpoint(epoch, model_id, mode="test")

        test_loader = self.create_data_loader(self.test_data, "test")

        return self.trainer.test(test_loader, out_idx=model_out_idx)
