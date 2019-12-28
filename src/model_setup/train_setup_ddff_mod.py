import torch

from model_setup.train_setup_ddff import TrainSetupDDFFBlender
from net import AERNN
#from net.ddff_net import DDFFEncoderNet, DDFFDecoderNet
from net.auto_encoder import DenseNet3DAutoEncoder
from tools.project import proj_dir
from tools.tools import model_freeze_state_repr
from trainer import TrainerDepth, TrainerCoCDepth, TrainerFgbgCoc, TrainerSignedCoC, TrainerCoC
from model_setup.pretrained_pool import load_pretrained
from trainer.loss import MultiMaskedLoss, AccuracyLoss
from net.modules import _output_mode_to_indices
from net.consec_net import ConsecNet
from data.video_depth_focus_data import VideoDepthFocusData
from tools.tools import is_size_equal
from net.modules import CropCatModule, PadCropModule, DownUpScaleModule


class TrainSetupDepth(TrainSetupDDFFBlender):
    def __init__(self, model_gen, model_modifier_dict=None, **kwargs):
        super().__init__(**kwargs)

        self.model_gen = model_gen
        self.model_modifier_dict = model_modifier_dict
        self.aernn_resize_mode = None

    def _create_model(self, num_in_channels, mode):
        if self.model_gen.__code__.co_argcount == 0:
            print("WARNING: model_gen lambda has no input param for num_in_channels")
            model = self.model_gen()
        else:
            model = self.model_gen(in_channels=num_in_channels)

        modifier_dict = self.model_modifier_dict

        if modifier_dict is not None:
            print("Apply modifier to model", modifier_dict)
            for k, v in modifier_dict.items():
                model.set_attr(k, v)

        if mode == "test":
            aernn_size = [256, 256]

            if isinstance(model, AERNN):
                if not isinstance(VideoDepthFocusData.crop_size, int):
                    if not is_size_equal(aernn_size, VideoDepthFocusData.crop_size):
                        if self.aernn_resize_mode == "crops":
                            model = CropCatModule(model)
                            model = PadCropModule(model, aernn_size[0], False)
                            print("model crops")
                        elif self.aernn_resize_mode == "downscale":
                            model = DownUpScaleModule(model)
                            print("model downscale")
                        else:
                            assert False

                        # print("AERNN CropCatModule used")
            else:
                model = PadCropModule(model, 32, True)

        return model

    def create_model(self, num_in_channels, mode):
        model = self._create_model(num_in_channels=num_in_channels, mode=mode)

        print("Output mode:", model.get_output_mode())

        # raise Exception("Refactor")
        self.depth_output_indices = _output_mode_to_indices(
            model.get_output_mode(),
            len(self.fixed_frame_indices) if self.fixed_frame_indices is not None
            else (len(self.select_rel_indices) if self.select_rel_indices is not None
                  else (len(self.select_focus_dists) if self.select_focus_dists is not None else self.sample_size))
        )

        self.include_all_coc = isinstance(model, ConsecNet)
        print("include_all_coc", self.include_all_coc)

        return model


class TrainSetupCoC(TrainSetupDepth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.include_coc = True

    def get_val_loss_func(self):
        val_loss_func = [torch.nn.MSELoss(), AccuracyLoss()]

        if self.loss_mask is not None:
            val_loss_func = [MultiMaskedLoss(l, self.loss_mask) for l in val_loss_func]

        return val_loss_func

    def create_trainer(self, device, model, optimizer, logger):
        return TrainerCoC(model=model,
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


class TrainSetupSignedCoC(TrainSetupDepth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.include_coc = "signed"

    def create_trainer(self, device, model, optimizer, logger):
        return TrainerSignedCoC(model=model,
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


class TrainSetupCoCDepthNet(TrainSetupDepth):
    def __init__(self, vis_coc=True, is_predepth_net=False, pretrained_coc=False, **kwargs):
        super().__init__(**kwargs)

        self.include_coc = True
        self.vis_coc = vis_coc
        self.is_predepth_net = is_predepth_net
        self.pretrained_coc = pretrained_coc

    def load_pretrained_modules(self, model):
        if self.pretrained_coc:
            load_pretrained(model.get_net(0), "coc", self)

        print(model_freeze_state_repr(self.model))

    def create_trainer(self, device, model, optimizer, logger):
        return TrainerCoCDepth(model=model,
                               device=device,
                               optimizer=optimizer,
                               logger=logger,
                               loss_mask=self.loss_mask,
                               vis_whole_seq=self.vis_whole_seq,
                               vis_coc=self.vis_coc,
                               is_predepth_net=self.is_predepth_net,
                               ingore_coc_loss=self.pretrained_coc,
                               extend_data_module=self.extend_data_module,
                               extend_encoding=self.extend_encoding
                               )


class TrainSetupCoCFgbgNet(TrainSetupDepth):
    def __init__(self, vis_coc=True, vis_fgbg=True, use_depth_loss=False, use_coc_bce_weight=False,
                 pretrained_coc=False, **kwargs):
        super().__init__(**kwargs)

        self.include_coc = True
        self.include_fgbg = True
        self.vis_coc = vis_coc
        self.vis_fgbg = vis_fgbg
        self.use_depth_loss = use_depth_loss
        self.use_coc_bce_weight = use_coc_bce_weight
        self.pretrained_coc = pretrained_coc

    def load_pretrained_modules(self, model):
        if self.pretrained_coc:
            load_pretrained(model.get_net(0), "coc", self)

        print(model_freeze_state_repr(self.model))

    def create_trainer(self, device, model, optimizer, logger):
        return TrainerFgbgCoc(model=model,
                              device=device,
                              optimizer=optimizer,
                              logger=logger,
                              loss_mask=self.loss_mask,
                              vis_whole_seq=self.vis_whole_seq,
                              vis_coc=self.vis_coc,
                              vis_fgbg=self.vis_fgbg,
                              use_depth_loss=self.use_depth_loss,
                              use_coc_bce_weight=self.use_coc_bce_weight,
                              ingore_coc_loss=self.pretrained_coc,
                              extend_data_module=self.extend_data_module,
                              extend_encoding=self.extend_encoding
                              )


# TODO: fix encoder layers and only fine tune last
# TODO: conv layer custom weight init
# TODO: fix first layers (dont train them)
class TrainSetupDenseNet3DAutoEncoder(TrainSetupDDFFBlender):
    def __init__(self,
                 use_concat=None,
                 use_transp_conv=False,
                 decoder_conv_kernel_sizes=None,
                 use_2d_dec=False,
                 depth_for_all=False,
                 **kwargs):
        super().__init__(sample_size=16, depth_output_indices=8, **kwargs)

        self.use_concat = use_concat
        self.use_transp_conv = use_transp_conv
        self.decoder_conv_kernel_sizes = decoder_conv_kernel_sizes
        self.use_2d_dec = use_2d_dec
        self.depth_for_all = depth_for_all

        if self.depth_for_all:
            self.depth_output_indices = None
        else:
            self.depth_output_indices = self.sample_size // 2

    def create_model(self, num_in_channels, mode):
        assert num_in_channels == 3
        return DenseNet3DAutoEncoder(proj_dir("pretrained"),
                                     use_concat=self.use_concat,
                                     use_transp_conv=self.use_transp_conv,
                                     decoder_conv_kernel_sizes=self.decoder_conv_kernel_sizes,
                                     use_2d_dec=self.use_2d_dec,
                                     depth_for_all=self.depth_for_all,
                                     load_pretrained=self.load_pretrained)

    """
    def create_logger(self, log_dir):
        return TrainLoggerTensorboardX(
            model_id=self.model_id,
            model_desc=self.model_desc,
            log_dir=log_dir,
            log_it_freq=5,
            log_img_freq=self.log_img_freq // self.data_expand if self.data_expand is not None else self.log_img_freq,
            log_val_freq=self.log_val_freq,
            val_loss_func=[AccuracyLoss()],) # loss wrong
    """

    def create_trainer(self, device, model, optimizer, logger):
        return TrainerDepth(model=model,
                            device=device,
                            optimizer=optimizer,
                            loss_mask=self.loss_mask,
                            scheduler=None,
                            max_gradient=self.max_gradient,
                            logger=logger,
                            extend_data_module=self.extend_data_module,
                            extend_encoding=self.extend_encoding
                            )



class TrainSetupDDFFRNN(TrainSetupDDFFBlender):
    def __init__(self,
                 rnn_dim=2048,
                 rnn_type="lstm",
                 rnn_embedding="fc",
                 rnn_num_layers=1,
                 batch_first=True,
                 rnn_dropout=0,
                 use_ccx=None,
                 ccx_reduce_mode="ccx_last",
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.rnn_dim = rnn_dim
        self.rnn_type = rnn_type
        self.rnn_embedding = rnn_embedding
        self.rnn_num_layers = rnn_num_layers
        self.batch_first = batch_first
        self.rnn_dropout = rnn_dropout
        self.use_ccx = use_ccx
        self.ccx_reduce_mode = ccx_reduce_mode

    def create_model(self, num_in_channels, mode):
        assert num_in_channels == 3
        """
        return AERNN(encoder=DDFFEncoderNet(dropout=self.dropout, load_pretrained=self.load_pretrained),
                     decoder=DDFFDecoderNet(output_dim=1, dropout=self.dropout, use_cc=self.use_ccx),
                     fs_size=self.sample_size, rnn_dim=self.rnn_dim,
                     rnn_type=self.rnn_type,
                     rnn_num_layers=self.rnn_num_layers, rnn_embedding_type=self.rnn_embedding,
                     batch_first=self.batch_first, rnn_dropout=self.rnn_dropout,
                     ccx_reduce_mode=self.ccx_reduce_mode)
        """
