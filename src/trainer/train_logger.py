import torch
import numpy as np
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import OrderedDict
# from tools.logger import logger
# from tools.list_tools import *
from tools.tools import NumpyJSONEncoder
import json
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch.nn.functional as F
from matplotlib import cm
import cv2
from tools.vis_tools import flow_to_vis
import uuid


from trainer.loss import *
from tools.tools import to_json_dict, type_adv, normalize_tensor, valid_filename

loss_tag = {
    type(None): "train_loss",
    torch.nn.MSELoss: "mse",
    RMSELoss: "rmse",
    AccuracyLoss: "acc",
    PSNRLoss: "psnr"
}

loss_format = {
    "train_loss": "{:.7f}",
    "mse": "{:.7f}",
    "psnr": "{:.7f}",
    "rmse": "{:.7f}",
    "acc": "{:.2f}%"
}


def get_loss_tag(loss_instance):
    if isinstance(loss_instance, MaskedLoss) or isinstance(loss_instance, MultiMaskedLoss):
        return loss_tag[type(loss_instance.loss)]
    else:
        return loss_tag[type(loss_instance)]


def get_loss_format(tag):
    return loss_format[tag]


class LossCollection(object):
    def __init__(self, loss_func, save_inter_losses=False):
        self.loss_func = loss_func
        self.save_inter_losses = save_inter_losses

        self.losses = None
        self.running_losses = None
        self.reset()

    def get_all_tags(self):
        return [get_loss_tag(loss_func) for loss_func in self.loss_func]

    def todict(self, use_running_loss=False):
        return {
            get_loss_tag(self.loss_func[i]): self.losses[i] if not use_running_loss else self.running_losses[i][-1]
            for i in range(len(self))
        }

    def reset(self):
        self.losses = [0] * len(self)
        self.running_losses = [0 if not self.save_inter_losses else [] for _ in range(len(self))]

    def _update_loss_value(self, i, value):
        if not self.save_inter_losses:
            self.running_losses[i] += value
        else:
            self.running_losses[i].append(value)

    def _update(self, indices, *args, **kwargs):
        with torch.no_grad():
            for i in indices:
                self._update_loss_value(i, self.loss_func[i](*args, **kwargs).data.cpu().numpy())

    def add_train_loss(self, loss):
        self._update_loss_value(0, loss)

    def add_val_losses(self, *args, **kwargs):
        self._update(range(1, len(self)), *args, **kwargs)

    def compute(self, data_size):
        self.losses = [
            (loss if not self.save_inter_losses else np.sum(loss)) / data_size
            for loss in self.running_losses
        ]

    def __len__(self):
        return len(self.loss_func)


class TrainLogger(object):
    img_font_path = "Roboto-Regular.ttf"

    def __init__(self,
                 log_it_freq=0,
                 log_val_freq=1,
                 val_loss_func=None,
                 log_timestamp=True,
                 log_prefix="",
                 log_img_freq=1,
                 mode="train",
                 log_dir=None,
                 model_id=None,
                 model_desc=None,
                 save_imgs=False,
                 img_font_path=None):
        self.num_epochs = None
        self.iter_per_epoch = None
        self.log_it_freq = log_it_freq
        self.log_val_freq = log_val_freq
        self.log_img_freq = log_img_freq
        self.log_timestamp = log_timestamp
        self.log_prefix = log_prefix

        self.log_iter_text = False

        self.train_stats = {"train": None, "val": None, "test": None}

        self.train_loss_func = [None]
        self.val_loss_func = [None, *val_loss_func] if val_loss_func is not None else [None]
        # [MaskedLoss(loss_func) for loss_func in [torch.nn.MSELoss(), RMSELoss(), AccuracyLoss()]]
        if len(self.val_loss_func) > 1:
            print("Val losses", self.val_loss_func[1:])

        self.model_name = "m{}_{}".format(model_id, model_desc)
        self.log_dir = log_dir

        self.save_imgs = save_imgs

        self.test_stats = []
        self.image_cache = {}

        self.img_out_labels = None

        if img_font_path is not None:
            TrainLogger.img_font_path = img_font_path

    def start(self, num_epochs, iter_per_epoch, img_out_labels):
        self.num_epochs = num_epochs
        self.iter_per_epoch = iter_per_epoch
        self.img_out_labels = img_out_labels

    def end(self):
        pass

    def init_train_stats(self, mode, loss_coll):
        self.train_stats[mode] = OrderedDict([(label, []) for label in loss_coll.get_all_tags()])

        # import json
        # print(json.dumps(self.train_stats, indent=4))

    def create_loss_coll(self, mode, save_inter_losses=False):
        coll = self.train_loss_func if mode == "train" else self.val_loss_func
        loss_coll = LossCollection(coll, save_inter_losses=save_inter_losses)
        return loss_coll

    def append_losses(self, mode, loss_coll):
        if self.train_stats[mode] is None:
            self.init_train_stats(mode, loss_coll)

        for k, v in loss_coll.todict().items():
            self.train_stats[mode][k].append(v)

    def log_text(self, msg):
        text = "{}{}".format(self.log_prefix, msg)

        if self.log_timestamp:
            timestr = datetime.now().strftime("%H:%M:%S")
            text = "{} {}".format(timestr, text)

        self.write_text(text)

    def write_text(self, text):
        print(text)

    def total_iteration(self, it, epoch):
        return it + (epoch - 1) * self.iter_per_epoch

    def log_setup(self, setup):
        self.log_text(str(setup))

    def log_iter(self, it, epoch, train_loss):
        total_num_it = self.iter_per_epoch * self.num_epochs

        if self.log_iter_text:
            self.log_text('[Iteration {:>{}}/{}] TRAIN loss: {}'.format(
                self.total_iteration(it, epoch),
                len(str(total_num_it)),
                total_num_it,
                train_loss))

    def _log_epoch_prefix(self, epoch):
        if self.num_epochs is not None:
            return '[Epoch {:>{}}/{}]'.format(
                epoch,
                len(str(self.num_epochs)),
                self.num_epochs)
        else:
            return '[Epoch {}]'.format(
                epoch)

    def _get_loss_log_entry(self, mode, loss_dict):
        return " ".join([
            ": ".join([
                l_tag,
                get_loss_format(l_tag).format(loss_val)
            ])
            for l_tag, loss_val in loss_dict.items()
        ])

    def log_test(self, idx, loss_coll):
        # print(loss_coll.running_losses)
        loss_dict = loss_coll.todict(use_running_loss=True)
        self.test_stats.append(loss_dict)

        self.log_text("[Test {:>5}] {}".format(
            idx,
            self._get_loss_log_entry("test", loss_dict)
        ))

    def _write_epoch(self, epoch, mode, loss_dict):
        self.log_text('{} {:<5} {}'.format(
            self._log_epoch_prefix(epoch),
            mode.upper(),
            self._get_loss_log_entry(mode, loss_dict)))

    def log_epoch(self, epoch, mode, loss_coll=None):
        if loss_coll is not None:
            self.append_losses(mode, loss_coll)
            self._write_epoch(epoch, mode, loss_coll.todict())

    def log_images(self, mode, idx, epoch, img_vis):
        if idx % self.log_img_freq == 0:
            if mode == "test":
                self.image_cache[idx] = img_vis

            if self.save_imgs:
                img_grid = TrainLogger._make_one_line_grid(img_vis)
                self._save_log_img(img_grid, idx, epoch=epoch, mode=mode)

    def log_create_checkpoint(self, epoch):
        self.log_text("{} {}".format(
            self._log_epoch_prefix(epoch),
            "Created checkpoint"
        ))

    def log_load_checkpoint(self, epoch):
        #print(self.train_stats)

        self.log_text("Loaded checkpoint")

        for mode in ["train", "val"]:
            last_epoch_stats = OrderedDict((k, v[-1]) for k, v in self.train_stats[mode].items())
            self._write_epoch(epoch, mode, last_epoch_stats)

    def get_last_loss(self):
        return next(iter(self.train_stats["val"].items()))[1][-1]

    def __repr__(self):
        return "TrainLogger"

    def log_subdir(self, name, raise_on_exist=False):
        subdir = os.path.join(self.log_dir, name, self.model_name)

        if not os.path.exists(subdir):
            os.makedirs(subdir)
        elif raise_on_exist:
            raise Exception("[TrainLoggerTensorboardX] Error: '{}' already exists".format(subdir))

        return subdir

    def image_this_epoch(self, mode, epoch):
        return epoch % self.log_img_freq == 0 or mode == "test"

    @staticmethod
    def _add_img_title(img, title):
        if title is not None and title != "":
            # "Roboto-Regular.ttf"
            font = ImageFont.truetype(TrainLogger.img_font_path, int(30 * (img.shape[1] / 512)))
            text_color = (255, 127, 0, 255)
            x, y = 0, 0
            w = img.shape[2]
            h = np.sum(font.getmetrics())

            title_img = Image.new('RGB', (w, h), color="black")
            draw = ImageDraw.Draw(title_img)
            draw.text([x, y], title, font=font, fill=text_color)
            title_img = transforms.ToTensor()(title_img)

            img = torch.cat([title_img, img], dim=1)

        return img

    @staticmethod
    def _apply_colormap(tensor, colormap):
        if colormap is not None:
            return torch.Tensor(cm.get_cmap(colormap)(tensor[0])).permute(2, 0, 1)[:3]
        else:
            return torch.cat([tensor] * 3)

    @staticmethod
    def _image_to_vis(x, colormap, normalize_tensors=False):
        assert len(x.size()) == 3

        if x.shape[0] >= 3:
            return torch.clamp(x[:3], 0, 1)
        elif x.shape[0] == 2:
            return flow_to_vis(x)
        elif x.shape[0] == 1:
            if normalize_tensors:
                x = normalize_tensor(x)
            else:
                x = torch.clamp(x, 0, 1)

            return TrainLogger._apply_colormap(x, colormap)
        else:
            raise Exception(str(x.size()) + " unsupported")

    # seq_single_frame_idx for video making select idx from input sequence
    @staticmethod
    def _make_img_grid(tensors, titles=None, colormap=None, append_cmap=False, seq_single_frame_idx=None, nrow=8,
                       normalize_tensors=False, single_tensor_out=False, *args, **kwargs):
        imgs = [x.cpu() for x in tensors]
        imgs = [x.float() if isinstance(x, torch.ByteTensor) else x for x in imgs]

        imgs_flat = []
        titles_flat = []

        for i, x in enumerate(imgs):
            if len(x.shape) == 4:
                if seq_single_frame_idx is not None:
                    imgs_flat.append(x[seq_single_frame_idx])
                    titles_flat.append(titles[i])
                else:
                    nrow = x.shape[0]
                    imgs_flat.extend([x_slice for x_slice in x])

                    if titles is not None:
                        titles_flat.extend(["{} (frame: {})".format(titles[i], f) for f in range(x.shape[0])])
            else:
                imgs_flat.append(x)

                if titles is not None:
                    titles_flat.append(titles[i])

        imgs = [TrainLogger._image_to_vis(x, colormap, normalize_tensors=normalize_tensors) for x in imgs_flat]

        titles = titles_flat

        if single_tensor_out:
            return imgs, titles
        else:
            if len(titles_flat) > 0:
                assert len(imgs) == len(titles)
                imgs = [TrainLogger._add_img_title(x, title) for x, title in zip(imgs, titles)]

            grid = torchvision.utils.make_grid(imgs, nrow=nrow, *args, **kwargs)

            if append_cmap:
                cmap_strip = torch.linspace(1, 0, grid.shape[1]).repeat(1, 10, 1).permute(0, 2, 1)
                cmap_strip = TrainLogger._apply_colormap(cmap_strip, colormap)

                grid = torch.cat([grid, cmap_strip], dim=2)

            return grid

    @staticmethod
    def _make_one_line_grid(imgs):
        img_list = imgs["input"] if isinstance(imgs["input"], list) else [imgs["input"]]

        for a, b in zip(imgs["target"], imgs["output"]):
            img_list += [a, b]

        # img_list = [torch.clamp(img, 0, 1) for img in img_list]

        return TrainLogger._make_img_grid(img_list)

    def _save_log_img(self, img, idx, epoch=0, out_label=None, mode="test"):
        file_name = "{}_{}_{}".format(mode, epoch, idx)

        if out_label is not None:
            file_name += "_{}".format(out_label)

        file_name += ".tif"

        torchvision.utils.save_image(img, os.path.join(self.log_subdir("images"), file_name))

    @staticmethod
    def _save_img(tensor, path, make_dirs=False):
        if make_dirs and not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        Image.fromarray((tensor * 0xff).permute(1, 2, 0).numpy().astype(np.uint8)).save(
            path, subsampling=0, quality=100)

    def _save_stats_file(self, obj, file="test_stats.json", path=None):
        if path is None:
            path = self.log_subdir("test_stats")

        with open(os.path.join(path, file), "w") as f:
            json.dump(obj, f, cls=NumpyJSONEncoder, indent=4)

    def save_stats(self):
        self._save_stats_file(self.test_stats)

    @staticmethod
    def save_all_test_stats(loggers, path=None):
        all_stats = {str(i) + "_" + l.model_name: l.test_stats for i, l in enumerate(loggers)}
        loggers[0]._save_stats_file(all_stats, "all_test_stats.json", path=path)

    @staticmethod
    def _save_tensor_video(tensor_list, path, fps=1.0):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(path, fourcc, fps, (tensor_list[0].shape[2], tensor_list[0].shape[1]))

        for tensor in tensor_list:
            out.write(cv2.cvtColor((tensor * 0xff).permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))

        out.release()

    @staticmethod
    def _save_img_grid(imgs, img_titles, colormap, append_cmap, nrow, normalize_tensors, path, img_id, out_label,
                       img_format, single_tensor_out):
        grid = TrainLogger._make_img_grid(imgs, img_titles, padding=10, colormap=colormap,
                                          append_cmap=append_cmap, nrow=nrow,
                                          normalize_tensors=normalize_tensors,
                                          single_tensor_out=single_tensor_out)

        if single_tensor_out:
            imgs, titles = grid

            for i, (img, title) in enumerate(zip(imgs, titles)):
                img_name = f"seq{img_id:03d}_{out_label}_tensor{i}.{img_format}"

                TrainLogger._save_img(img, os.path.join(path, img_name), make_dirs=True)

                """
                TrainLogger._save_img(img, os.path.join(path,
                                                        str(img_id),
                                                        str(out_label),
                                                        "{}_{}_{}.{}".format(
                                                            img_id,
                                                            out_label,
                                                            valid_filename(title),
                                                            img_format)),
                                      make_dirs=True)
                """
        else:
            TrainLogger._save_img(grid, os.path.join(path, "{}_{}.{}".format(
                img_id, out_label, img_format)))

    @staticmethod
    def save_all_test_images(loggers, mask_img_func=None, loss_label="train_loss", path=None, colormap=None,
                             model_labels=None, append_cmap=False, whole_seq_out=None, leave_out_missing_target=True,
                             img_format="jpg", epochs=None, nrow=8, normalize_tensors=False, single_tensor_out=False,
                             rotate_suwa=False):
        # model_test_img_vis = [l.image_cache for l in loggers]

        print("Saving all test images:", path)
        img_out_labels = list(set(sum((l.img_out_labels for l in loggers), [])))

        print("Out labels", img_out_labels)

        image_ids = list(loggers[0].image_cache.keys())
        num_outputs = len(loggers[0].image_cache[image_ids[0]]["target"])
        print("image_ids", image_ids)
        print("num out", num_outputs)

        print(type_adv(loggers[0].image_cache))

        if path is None:
            # loggers[0]._save_img(img_grid, img_id, out_label=out_label)
            raise Exception("Path is None")

        if not os.path.isdir(path):
            os.makedirs(path)

        # print(loggers[0].image_cache)
        # raise Exception("Break")

        for out_label in img_out_labels:
            for img_id in image_ids:

                imgs = []
                mask = None

                img_titles = []

                seq_length = None

                input_imgs, target_imgs = None, None

                for mid, l in enumerate(loggers):
                    if input_imgs is None:
                        input_imgs = l.image_cache[img_id]["input"]
                        if len(input_imgs[0].shape) == 4:
                            seq_length = input_imgs[0].shape[0]

                    if out_label in l.img_out_labels:
                        out_idx = l.img_out_labels.index(out_label)

                        if target_imgs is None:
                            # set input and target
                            # print(type_adv(l.image_cache))
                            target_imgs = l.image_cache[img_id]["target"][out_idx]

                            mask = ~mask_img_func(target_imgs) if mask_img_func is not None else None

                        output_img = l.image_cache[img_id]["output"][out_idx].cpu()

                        if mask is not None:
                            output_img[mask] = target_imgs[1][mask]

                        loss = l.test_stats[img_id][loss_label]

                        imgs.append(output_img)
                        if model_labels is not None:
                            img_title = model_labels[mid]
                        else:
                            img_title = "[{}] {:.5f}".format(mid, loss)

                        if epochs is not None:
                            img_title = "[{}] {}".format(epochs[mid], img_title)

                        img_titles.append(img_title)
                    else:
                        imgs.append(None)
                        img_titles.append("[{}] No out".format(mid))

                # print([x.shape for x in imgs])
                imgs = [img if img is not None else torch.zeros_like(target_imgs) for img in imgs]

                if not (leave_out_missing_target and (~torch.isnan(target_imgs)).sum() == 0):
                    imgs = [target_imgs] + imgs
                    img_titles = ["Target"] + img_titles

                imgs = input_imgs + imgs
                img_titles = [f"Input {i}" for i in range(len(input_imgs))] + img_titles

                if rotate_suwa:
                    if img_id in [1, 3, 7, 8, 10, 11]:
                        imgs = [img.transpose(-2, -1).flip(-1) for img in imgs]

                if whole_seq_out is None or "line" in whole_seq_out:
                    TrainLogger._save_img_grid(
                        imgs, img_titles, colormap, append_cmap, nrow, normalize_tensors, path, img_id, out_label,
                        img_format, single_tensor_out=single_tensor_out)

                if whole_seq_out is not None and ("single" in whole_seq_out or "video" in whole_seq_out):
                    img_grids = [
                        TrainLogger._make_img_grid(imgs, img_titles, padding=10, colormap=colormap,
                                                   append_cmap=append_cmap, nrow=nrow, seq_single_frame_idx=frame_idx,
                                                   normalize_tensors=normalize_tensors)
                        for frame_idx in range(seq_length)
                    ]

                    if "video" in whole_seq_out:
                        TrainLogger._save_tensor_video(
                            img_grids,
                            os.path.join(path, "video_{}_{}.avi".format(img_id, out_label))
                        )

                    if "single" in whole_seq_out:
                        for frame_idx, img_grid in enumerate(img_grids):
                            TrainLogger._save_img(
                                img_grid,
                                os.path.join(path, "single_id{}_{}_frame{}.{}".format(
                                    img_id, out_label, frame_idx, img_format))
                            )


class TrainLoggerTensorboardX(TrainLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tb_log_dir = self.log_subdir("tblogs", raise_on_exist=False)

        # logger.debug(self.tb_log_dir)
        print(self.tb_log_dir)

        if os.path.isdir(self.tb_log_dir):
            shutil.rmtree(self.tb_log_dir)

        self.writer = None
        self.start_writer()

    def start_writer(self):
        if self.writer is None:
            print("CREATED SummaryWriter")
            self.writer = SummaryWriter(self.tb_log_dir)

    def start(self, *args, **kwargs):
        super().start(*args, **kwargs)
        self.start_writer()

    def end(self):
        super().end()
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def write_text_tensorboard(self, tag, text, step=None):
        self.writer.add_text(tag, text.replace("\n", "  \n"), step)

    def log_setup(self, setup):
        super().log_setup(setup)
        self.write_text_tensorboard("Setup", json.dumps(to_json_dict(setup)))

    def log_iter(self, it, epoch, train_loss):
        super().log_iter(it, epoch, train_loss)
        # self.writer.add_scalar("iter_train_loss", train_loss, self.total_iteration(it, epoch))

    def log_epoch(self, epoch, mode, loss_coll=None):
        super().log_epoch(epoch, mode, loss_coll)
        if loss_coll is not None:
            for k, v in loss_coll.todict().items():
                self.writer.add_scalar(mode + "_" + k, v, epoch)

    def log_images(self, mode, idx, epoch, imgs):
        super().log_images(mode, idx, epoch, imgs)
        if mode != "test" and self.image_this_epoch(mode, epoch):
            img_grid = TrainLogger._make_one_line_grid(imgs)
            self.writer.add_image(mode, img_grid, epoch)

    def log_graph(self, net, sample_in):
        self.writer.add_graph(net, sample_in, True)

    def log_test(self, idx, loss_coll):
        super().log_test(idx, loss_coll)

        for k, v in loss_coll.todict(use_running_loss=True).items():
            self.writer.add_scalar("test" + "_" + k, v, idx)
