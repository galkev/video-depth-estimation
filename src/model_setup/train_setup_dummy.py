from net import DDFFNetDummy
from data import MdffH5DataDbgSmall
from model_setup.train_setup_ddff import TrainSetupDDFF, TrainSetupDDFFBlender


class TrainSetupDDFFDummy(TrainSetupDDFF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_model(self, train_loader, val_loader, model_loader):
        super().train_model(train_loader, val_loader, model_loader)

    def create_model(self):
        return DDFFNetDummy(focal_stack_size=self.sample_size, dropout=self.dropout)

    def create_dataset(self, dataset_path, data_type):
        return MdffH5DataDbgSmall(root_dir=dataset_path, data_type=data_type)


class TrainSetupDDFFBlenderDummy(TrainSetupDDFFBlender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_model(self):
        return DDFFNetDummy(focal_stack_size=self.sample_size, dropout=self.dropout)

    """
    def create_dataset(self, dataset_path, data_type):
        data = VideoDepthFocusData(dataset_path, data_type, "dining_room")
        data.configure_dining_room_ddff()
        return data
    """
