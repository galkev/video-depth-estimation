import torch

from trainer import TrainLogger, TrainLoggerTensorboardX
from net import DDFFNet
from data import DatasetExpand, DatasetJoin, DDFFData, VideoDepthFocusData, VideoDepthFocusDataMp4, VideoDepthFocusDataFiveCrop
from trainer import TrainerDepth
from trainer.loss import AccuracyLoss, MaskedLoss, MultiMaskedLoss

from .train_setup import TrainSetup


class TrainSetupDDFF(TrainSetup):
    def __init__(self,
                 lr=1e-4,
                 batch_size=4,
                 num_epochs=10000000,
                 sample_size=10,
                 dropout=0.0,
                 checkpoint_freq=5,
                 log_img_freq=5,
                 log_val_freq=5,
                 optim_type="adam",
                 mask_loss_type=True,
                 sgd_mom=0.9,
                 weight_decay=0,  # 0.0005
                 max_gradient=None,
                 ddff_scoring="ccx_last",
                 load_pretrained=True,
                 save_imgs=False,
                 select_focus_dists=None,
                 num_data_threads=4,
                 extend_data_module=None,
                 extend_encoding=False):
        super().__init__()

        if mask_loss_type is not None:
            self.loss_mask = lambda x: x < 1.0
        else:
            self.loss_mask = None

        self.lr = lr

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.sample_size = sample_size if select_focus_dists is None else len(select_focus_dists)
        self.dropout = dropout
        self.ddff_scoring = ddff_scoring

        self.optim_type = optim_type
        self.sgd_mom = sgd_mom
        self.weigth_decay = weight_decay
        self.max_gradient = max_gradient

        self.checkpoint_freq = checkpoint_freq
        self.log_img_freq = log_img_freq
        self.log_val_freq = log_val_freq

        self.load_pretrained = load_pretrained

        self.save_imgs = save_imgs
        self.vis_whole_seq = False

        self.test_crop = None

        self.use_tensorboard = True
        self.num_data_threads = num_data_threads

        self.select_focus_dists = select_focus_dists
        self.extend_data_module = extend_data_module
        self.extend_encoding = extend_encoding

    def create_model(self, num_in_channels, mode):
        assert num_in_channels == 3
        return DDFFNet(focal_stack_size=self.sample_size, dropout=self.dropout, scoring_mode=self.ddff_scoring,
                       load_pretrained=self.load_pretrained)

    def create_dataset(self, dataset_path, data_type):
        return DDFFData(root_dir=dataset_path, data_type=data_type)

    def create_optimizer(self, parameters):
        if self.optim_type == "adam":
            return torch.optim.Adam(params=parameters, lr=self.lr, weight_decay=self.weigth_decay)
        elif self.optim_type == "sgd":
            return torch.optim.SGD(params=parameters,
                                   lr=self.lr,
                                   momentum=self.sgd_mom,
                                   weight_decay=self.weigth_decay)

    def create_scheduler(self, optimizer):
        return None

    def create_trainer(self, device, model, optimizer, logger):
        return TrainerDepth(model=model,
                            device=device,
                            optimizer=optimizer,
                            scheduler=None,
                            max_gradient=self.max_gradient,
                            logger=logger,
                            loss_mask=self.loss_mask,
                            vis_whole_seq=self.vis_whole_seq,
                            extend_data_module=self.extend_data_module,
                            extend_encoding=self.extend_encoding
                            )

    def create_data_loader(self, data, data_type):
        return TrainSetup.create_data_loader_helper(data, data_type, self.batch_size, num_threads=self.num_data_threads)

    def _create_logger(self, use_tb, **kwargs):
        if use_tb:
            return TrainLoggerTensorboardX(**kwargs)
        else:
            return TrainLogger(**kwargs)

    def create_logger(self, log_dir, font):
        return self._create_logger(
            use_tb=self.use_tensorboard,
            model_id=self.model_id,
            model_desc=self.model_desc,
            log_dir=log_dir,
            log_it_freq=5,
            log_img_freq=self.log_img_freq,
            log_val_freq=self.log_val_freq,
            save_imgs=self.save_imgs,
            val_loss_func=self.get_val_loss_func(),
            img_font_path=font
            )

    def get_val_loss_func(self):
        val_loss_func = [torch.nn.MSELoss(), AccuracyLoss()]

        if self.loss_mask is not None:
            val_loss_func = [MaskedLoss(l, self.loss_mask) for l in val_loss_func]

        return val_loss_func

    def train_model(self, train_loader, val_loader, model_loader):
        self.logger.log_setup(self)
        #self.logger.log_graph(self.model, torch.zeros([1, self.focal_stack_size, 3, 224, 224], device=self.device))

        self.trainer.train(train_loader,
                           val_loader,
                           self.num_epochs,
                           model_loader=model_loader,
                           checkpoint_freq=self.checkpoint_freq)

    def test_model(self, test_loader):
        self.logger.log_setup(self)

        self.trainer.test(test_loader)

    def __repr__(self):
        attr = vars(self)
        return "{}(\n{}\n)".format(
            self.__class__.__name__,
            "\n".join("{}: {}".format(k, v) for k, v in attr.items())
        )


class TrainSetupDDFFBlender(TrainSetupDDFF):
    def __init__(self,
                 dataset,
                 checkpoint_freq=60,
                 sample_skip=0,
                 use_allinfocus=False,
                 color_noise_stddev=None,
                 depth_noise_stddev=None,
                 data_expand=None,
                 depth_output_indices=None,
                 include_coc=False,
                 include_fgbg=False,
                 target_indices=None,
                 include_flow=False,
                 ramp_sample_count=1,
                 fixed_ramp_idx=None,
                 fixed_frame_indices=None,
                 relative_fixed_frame_indices=False,
                 select_rel_indices=None,
                 include_all_coc=False,
                 **kwargs
                 ):
        super().__init__(**kwargs, checkpoint_freq=checkpoint_freq)

        self.dataset = dataset

        self.sample_skip = sample_skip
        self.use_allinfocus = use_allinfocus

        self.color_noise_stddev = color_noise_stddev
        self.depth_noise_stddev = depth_noise_stddev

        self.data_expand = data_expand

        self.depth_output_indices = depth_output_indices

        self.include_coc = include_coc
        self.include_fgbg = include_fgbg

        self.test_target_frame = None
        self.limit_data = None

        self.test_single_img_seq = None

        self.five_crop_dataset = False

        self.target_indices = target_indices

        self.include_flow = include_flow
        self.ramp_sample_count = ramp_sample_count

        self.fixed_ramp_idx = fixed_ramp_idx
        self.fixed_frame_indices = fixed_frame_indices
        self.relative_fixed_frame_indices = relative_fixed_frame_indices

        self.select_rel_indices = select_rel_indices
        self.include_all_coc = include_all_coc

        self.pad_to_multiple = 32
        self.pad_center = True

    def create_dataset(self, *args, **kwargs):
        if self.five_crop_dataset:
            return VideoDepthFocusDataFiveCrop(lambda: self._create_dataset(*args, **kwargs))
        else:
            return self._create_dataset(*args, **kwargs)

    def _create_dataset_instance(self, dataset_path, data_type, dataset_name):
        if dataset_name.endswith("_mp4"):
            data = VideoDepthFocusDataMp4(dataset_path, data_type, dataset_name)
        else:
            data = VideoDepthFocusData(dataset_path, data_type, dataset_name)

        print("Data class", type(data))

        data.configure(sample_count=self.sample_size,
                       sample_skip=self.sample_skip,
                       depth_output_indices=self.depth_output_indices,
                       use_allinfocus=self.use_allinfocus,
                       color_noise_stddev=self.color_noise_stddev,
                       depth_noise_stddev=self.depth_noise_stddev,
                       include_coc=self.include_coc,
                       include_fgbg=self.include_fgbg,
                       test_crop=self.test_crop,
                       test_target_frame=self.test_target_frame,
                       limit_data=self.limit_data,
                       test_single_img_seq=self.test_single_img_seq,
                       include_flow=self.include_flow,
                       target_indices=self.target_indices,
                       select_focus_dists=self.select_focus_dists,
                       ramp_sample_count=self.ramp_sample_count,
                       fixed_ramp_idx=self.fixed_ramp_idx,
                       fixed_frame_indices=self.fixed_frame_indices,
                       relative_fixed_frame_indices=self.relative_fixed_frame_indices,
                       select_rel_indices=self.select_rel_indices,
                       include_all_coc=self.include_all_coc,
                       pad_to_multiple=self.pad_to_multiple,
                       pad_center=self.pad_center
                       )

        if self.data_expand is not None:
            data = DatasetExpand(data, self.data_expand)

        return data

    def _create_dataset(self, dataset_path, data_type):
        if isinstance(self.dataset, list):
            data = DatasetJoin([self._create_dataset_instance(dataset_path, data_type, d) for d in self.dataset])
        else:
            data = self._create_dataset_instance(dataset_path, data_type, self.dataset)

        return data
