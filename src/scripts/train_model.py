import argparse
import json
import os
import time
import torch
import torchvision

# noinspection PyUnresolvedReferences
import pathmagic
from model_setup.train_setup_records import create_train_setup
from tools.tools import dict_cross_prod
from tools.project import proj_dir, use_slurm_system
from trainer.train_logger import TrainLogger
from net.sliding_window_net import SlidingWindowNet, SlidingWindowNoReduceNet
from net.modules import InputSplitModule
from data.video_depth_focus_data import VideoDepthFocusData
from tools.tools import normalize_tensor

from trainer import loss


class SlidingWindowTestSetup:
    def __init__(self, setup, no_reduce=False, stride=1):
        self.setup = setup
        self.no_reduce = no_reduce

        self.wnd_size = self.setup.sample_size if (self.setup.depth_output_indices is None or self.no_reduce) else 1
        self.stride = stride

        num_frames = 25
        self.setup.test_target_frame = 12
        self.setup.sample_size = num_frames
        # self.setup.target_indices = range(self.wnd_size-1, num_frames - (self.wnd_size-1))

    def test_epoch(self, epoch, model_id, data_type="test"):
        self.setup.create_components(load_pretrained=False)
        self.setup.load_checkpoint(epoch, model_id, mode=data_type)

        if self.no_reduce:
            self.setup.trainer.model = SlidingWindowNoReduceNet(self.wnd_size, self.setup.trainer.model, stride=self.stride)
        else:
            self.setup.trainer.model = SlidingWindowNet(self.wnd_size, self.setup.trainer.model, stride=self.stride)

        self.setup.depth_output_indices = None
        self.setup.train_data.depth_output_indices = None
        self.setup.val_data.depth_output_indices = None
        self.setup.test_data.depth_output_indices = None

        self.setup.test(data_type=data_type, create_comp=False)

    @property
    def logger(self):
        return self.setup.logger

    @property
    def trainer(self):
        return self.setup.trainer


def get_train_setup(name, model_id, model_desc):
    setup = create_train_setup(name)
    setup.set_model_info(model_id, model_desc)

    return setup


def train_model(model_id, setup_name, epoch=None):
    model_desc = setup_name

    setup = get_train_setup(setup_name, model_id, model_desc)

    if epoch is not None:
        setup.train_continue(epoch, model_id=model_id)
    else:
        setup.train()


def test_model(model_id, setup_name, data_type="test", dataset_name=None, epoch=None,
               log_img_freq=1,
               test_crop=False, use_allinfocus=False, test_target_frame=None, limit_data=None, sample_skip=None,
               test_single_img_seq=None, color_noise_stddev=None, five_crop_dataset=False, whole_seq_out=None,
               slide_wnd_test=None, slide_wnd_test_stride=1, select_focus_dists=None, ramp_sample_count=1,
               fixed_frame_indices=None, relative_fixed_frame_indices=False, model_modifier=None, select_rel_indices=None,
               masked_only=False, aernn_resize_mode=None, fixed_ramp_idx=None):
    model_desc = setup_name + "_testset"

    setup = get_train_setup(setup_name, model_id, model_desc)

    setup.use_tensorboard = False
    # setup.vis_depth_only = True
    setup.log_img_freq = log_img_freq
    setup.test_crop = test_crop
    setup.use_allinfocus = use_allinfocus
    setup.test_target_frame = test_target_frame
    setup.limit_data = limit_data
    setup.test_single_img_seq = test_single_img_seq
    setup.color_noise_stddev = color_noise_stddev
    setup.five_crop_dataset = five_crop_dataset
    setup.vis_whole_seq = whole_seq_out is not None
    setup.select_focus_dists = select_focus_dists
    setup.ramp_sample_count = ramp_sample_count
    setup.fixed_frame_indices = fixed_frame_indices
    setup.relative_fixed_frame_indices = relative_fixed_frame_indices
    setup.model_modifier_dict = model_modifier
    setup.select_rel_indices = select_rel_indices
    setup.aernn_resize_mode = aernn_resize_mode
    setup.fixed_ramp_idx = fixed_ramp_idx

    if masked_only:
        loss.copy_unmasked_target_to_input = True

    if sample_skip is not None:
        setup.sample_skip = sample_skip

    if dataset_name is not None:
        setup.dataset = dataset_name

    if slide_wnd_test is not None:
        if slide_wnd_test == "input_split":
            inner_model = setup._create_model(num_in_channels=None, mode="test")
            setup.model_gen = lambda: InputSplitModule(model=inner_model, num_splits=ramp_sample_count, pad_output=True)
        else:
            if slide_wnd_test == "reduce":
                no_reduce = False
            elif slide_wnd_test == "no_reduce":
                no_reduce = True
            else:
                raise Exception("slide_wnd_test invalid value: " + str(slide_wnd_test))

            setup = SlidingWindowTestSetup(setup, no_reduce=no_reduce, stride=slide_wnd_test_stride)

    setup.test_epoch(epoch, model_id, data_type=data_type)

    return setup


def parse_setup(setup):
    if isinstance(setup, str):
        parts = setup.split("_", 1)

        mid = parts[0][1:]
        setup_name = parts[1]

        return mid, setup_name
    else:
        return setup


def test_model_set(test_name, model_setup, dataset_name, loss="mse", mask_img=False, colormap=None, model_labels=None,
                   append_cmap=False, whole_seq_out=None, img_format="jpg", img_grid_nrow=8,
                   fixed_frame_indices=None, select_focus_dists=None, select_rel_indices=None, normalize_tensors=False,
                   single_tensor_out=False, **kwargs):
    loggers = []
    epochs = []

    for i, setup_cfg in enumerate(model_setup):
        (mid, setup_name) = parse_setup(setup_cfg)

        if fixed_frame_indices is not None and isinstance(fixed_frame_indices[0], list):
            ffi = fixed_frame_indices[i]
        else:
            ffi = fixed_frame_indices

        if select_focus_dists is not None and isinstance(select_focus_dists[0], list):
            sfd = select_focus_dists[i]
        else:
            sfd = select_focus_dists

        setup = test_model(model_id=mid, setup_name=setup_name, dataset_name=dataset_name,
                           whole_seq_out=whole_seq_out, fixed_frame_indices=ffi,
                           select_rel_indices=select_rel_indices, select_focus_dists=sfd,
                           **kwargs)

        loggers.append(setup.logger)
        epochs.append(setup.trainer.epoch - 1)

    mask_img_func = (lambda x: x <= 1) if mask_img else None

    # path = proj_dir("logs", "test", time.strftime("%Y%m%d-%H%M%S") + "_" + dataset_name)
    path = proj_dir("logs", "test", test_name + "_" + time.strftime("%Y%m%d-%H%M%S"))

    TrainLogger.save_all_test_images(loggers, mask_img_func=mask_img_func, loss_label=loss, model_labels=model_labels,
                                     path=os.path.join(path, "img"), colormap=colormap, append_cmap=append_cmap,
                                     whole_seq_out=whole_seq_out, img_format=img_format, epochs=epochs,
                                     nrow=img_grid_nrow, normalize_tensors=normalize_tensors,
                                     single_tensor_out=single_tensor_out, rotate_suwa=dataset_name == "suwajanakorn")
    TrainLogger.save_all_test_stats(loggers, path=path)

    return path


def save_image_batch(batch, path, normalize=None, colormap=None):
    if not os.path.exists(path):
        os.makedirs(path)

    if normalize == "clip":
        batch = normalize_tensor(batch)

    for i, img in enumerate(batch):
        img = img.cpu()

        if normalize == "single":
            img = normalize_tensor(img)

        if colormap is not None:
            img = TrainLogger._apply_colormap(img, colormap)
        else:
            img = torch.cat([img] * 3)

        torchvision.utils.save_image(img, os.path.join(path, f"out{i}.jpg"), nrow=4)


def simple_test_models(test_name, model_setup, model_id, dataset_name, ramps_per_clip, ramp_length=4,
                       epoch=None, colormap=None, model_out_idx=-1):
    model_desc = model_setup + "_testset"

    setup = get_train_setup(model_setup, model_id, model_desc)

    imgs = setup.simple_test(epoch, model_id, dataset_name, ramp_length, ramps_per_clip, model_out_idx=model_out_idx)
    path = proj_dir("logs", "simple_test", test_name)  # + "_" + time.strftime("%Y%m%d-%H%M%S"))

    num_clips = imgs.shape[0] // ramps_per_clip

    for i in range(num_clips):
        save_image_batch(imgs[i*ramps_per_clip:(i+1)*ramps_per_clip],
                         os.path.join(path, f"seq{i}"),
                         normalize="clip", colormap=colormap)


def test_models(test_name, model_setup, model_labels, whole_seq_out=None, img_format="jpg", select_focus_dists=None,
                resolution=None, fixed_frame_indices=None, select_rel_indices=None,
                single_tensor_out=False, fixed_ramp_idx=None, **kwargs):
    if isinstance(fixed_ramp_idx, list):
        for idx in fixed_ramp_idx:
            test_name_sub = os.path.join(test_name, str(idx))
            path = test_models(test_name_sub, model_setup, model_labels, whole_seq_out, img_format,
                               select_focus_dists, resolution, fixed_frame_indices, select_rel_indices,
                               single_tensor_out, fixed_ramp_idx=idx, **kwargs)

        path = os.path.dirname(path)
    else:
        if resolution is not None:
            VideoDepthFocusData.crop_size = resolution

        test_args_list = dict_cross_prod(kwargs)
        assert len(test_args_list) == 1

        for test_args in test_args_list:
            path = test_model_set(test_name, model_setup, model_labels=model_labels, whole_seq_out=whole_seq_out,
                                  img_format=img_format, select_focus_dists=select_focus_dists,
                                  fixed_frame_indices=fixed_frame_indices, select_rel_indices=select_rel_indices,
                                  single_tensor_out=single_tensor_out, fixed_ramp_idx=fixed_ramp_idx, **test_args)
            with open(os.path.join(path, "test_params.json"), "w") as f:
                json.dump(test_args, f)

    print("END")

    return path


def run_model_operation(args):
    use_slurm = not args.userfs

    if use_slurm:
        use_slurm_system()

    if args.test or args.simple_test:
        print("torch.multiprocessing.set_sharing_strategy('file_system')")
        torch.multiprocessing.set_sharing_strategy('file_system')

        if args.test_cfg is not None:
            with open(args.test_cfg, "r") as f:
                test_args = json.load(f)

            test_name = os.path.basename(args.test_cfg).split('.')[0]

            if args.simple_test:
                simple_test_models(test_name=test_name, **test_args)
            else:
                path = test_models(test_name=test_name, **test_args)

                with open(os.path.join(path, "test_cfg.json"), "w") as f:
                    json.dump(test_args, f, indent=4)
        else:
            test_model(args.model_id, args.setup, args.test_set, epoch=args.load)
    else:
        train_model(args.model_id, args.setup, epoch=args.load)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_id", default="unnamed")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--job_name", default=None)  # unused
    parser.add_argument("--setup")
    parser.add_argument("--load", type=int, default=None)
    parser.add_argument("--userfs", action="store_true")

    # test
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test_set", default="test")
    parser.add_argument("--test_cfg", default=None)
    parser.add_argument("--simple_test", action="store_true")

    args = parser.parse_args()

    if args.model_id is None:
        args.model_id = args.job_id

    run_model_operation(args)


if __name__ == "__main__":
    main()
