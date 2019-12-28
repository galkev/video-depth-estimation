import torch
import torch.nn.functional as F
import random
import torchvision
import uuid

from tools.vis_tools import flow_to_vis

from trainer.train_logger import TrainLogger
from trainer.loss import MaskedLoss, MultiMaskedLoss, FgbgCocLoss, CoCDepthLoss, FgbgLoss
from tools.tools import is_size_greater, is_size_less, type_adv
from data.data_transforms import crop_to_size

from net.extend_data_modules import ExtendEncoding


class Trainer(object):
    def __init__(self,
                 model,
                 device,
                 optimizer=None,
                 loss_func=torch.nn.MSELoss(),
                 loss_mask=None,
                 scheduler=None,
                 max_gradient=None,
                 logger=TrainLogger(),
                 extend_data_module=None,
                 extend_encoding=False,
                 vis_whole_seq=False):
        self.device = device

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_func = loss_func if loss_mask is None else MaskedLoss(loss_func, loss_mask)
        self.scheduler = scheduler
        self.max_gradient = max_gradient

        self.epoch = 1
        self.num_epochs = -1

        self.logger = logger

        self.vis_count_per_epoch = {"train": 3, "val": 3, "test": 999999999}

        self.mode = None
        self.extend_data_module = extend_data_module.to(device) if extend_data_module is not None else None
        self.extend_encoding = extend_encoding

        print("Train loss", self.loss_func)

        self.vis_idx = None
        self.vis_whole_seq = vis_whole_seq

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def data_to_dev(self, data, dev=None):
        dev = self.device if dev is None else dev

        if isinstance(data, torch.Tensor):
            return data.to(dev)
        elif isinstance(data, dict):
            return {k: self.data_to_dev(v, dev) for k, v in data.items()}
        else:
            raise Exception(str(data) + " not recognized")

    def update_loss_coll(self, loss_coll, loss_value, *val_loss_args, **val_loss_kwargs):
        loss_coll.add_train_loss(loss_value)
        loss_coll.add_val_losses(*val_loss_args, **val_loss_kwargs)

    def compute_loss(self, mode, data, data_loader, loss_coll, backward_loss):
        raise NotImplementedError

    def _visualize_prog(self, mode, dataloader, it, vis_ids, data, outputs):
        if self.logger.image_this_epoch(mode, self.epoch):
            cur_batch_size = next(iter(data.values())).shape[0]

            for i in range(cur_batch_size):
                abs_idx = (it - 1) * dataloader.batch_size + i

                if abs_idx in vis_ids:
                    imgs_vis = self.get_output_vis_images(i, data, outputs)

                    # TODO: rev transform
                    imgs_vis = self.rev_transform_vis(mode, data, imgs_vis)

                    #torchvision.utils.save_image(flow_to_vis(imgs_vis["input"][1].view(2, 256, 256)),
                    #                             "/home/kevin/Documents/master-thesis/logs/dump/{}.tif".format(
                    #                                 uuid.uuid1()))

                    #if mode == "test":
                    #    print(f"self.logger.log_images({mode}, {abs_idx}, {self.epoch}, ...)")

                    self.logger.log_images(mode, abs_idx, self.epoch, imgs_vis)

    def rev_transform_vis(self, mode, data, imgs_vis):
        if mode == "test":
            for k, tensor_list in imgs_vis.items():
                for i, tensor in enumerate(tensor_list):
                    if len(tensor.shape) >= 3 and \
                            is_size_greater([tensor.shape[-1], tensor.shape[-2]], data["org_res"][0]):
                        print("Copping tensor")
                        imgs_vis[k][i] = crop_to_size(tensor, data["org_res"][0])

        return imgs_vis

    def get_vis_ids(self, mode, data_loader):
        vis_count = self.vis_count_per_epoch[mode]
        data_size = len(data_loader.dataset)
        sample_ids = range(data_size)

        if vis_count < data_size:
            return random.sample(sample_ids, vis_count)
        else:
            return sample_ids

    def create_loss_coll(self, mode, save_inter_losses=False):
        return self.logger.create_loss_coll(mode, save_inter_losses)

    def compute_test_loss(self, test_loader):
        self.model.eval()

        with torch.no_grad():
            vis_ids = self.get_vis_ids("test", test_loader)
            loss_coll = self.create_loss_coll("test", save_inter_losses=True)

            for test_it, data in enumerate(test_loader, 1):
                loss, outputs, data = self.compute_loss("test", data, test_loader, loss_coll, backward_loss=False)

                self.logger.log_test(test_it, loss_coll)
                self._visualize_prog("test", test_loader, test_it, vis_ids, data, outputs)

            loss_coll.compute(len(test_loader))

            return loss_coll

    def compute_val_loss(self, val_loader):
        self.model.eval()

        with torch.no_grad():
            vis_ids = self.get_vis_ids("val", val_loader)
            loss_coll = self.create_loss_coll("val")

            for val_it, data in enumerate(val_loader, 1):
                loss, outputs, data = self.compute_loss("val", data, val_loader, loss_coll, backward_loss=False)
                # visualize output (if needed, decided later)
                self._visualize_prog("val", val_loader, val_it, vis_ids, data, outputs)

            loss_coll.compute(len(val_loader))

        return loss_coll

    def train_epoch(self, train_loader):
        self.model.train()

        vis_ids = self.get_vis_ids("train", train_loader)
        loss_coll = self.create_loss_coll("train")

        for train_it, data in enumerate(train_loader, 1):
            loss, outputs, data = self.compute_loss("train", data, train_loader, loss_coll, backward_loss=True)

            if self.max_gradient is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient, norm_type=2)

            self.optimizer.step()

            # visualize output (if needed, decided later)
            self._visualize_prog("train", train_loader, train_it, vis_ids, data, outputs)

        loss_coll.compute(len(train_loader))

        if self.scheduler is not None:
            self.scheduler.step()

        return loss_coll

    def create_checkpoint(self):
        return {
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "model": self.model.state_dict(),
            "stats": self.logger.train_stats
        }

    def load_checkpoint(self, checkpoint, mode="train"):
        print("Type:", type(checkpoint))
        self.epoch = checkpoint["epoch"] + 1
        self.logger.train_stats = checkpoint["stats"]
        self.model.load_state_dict(checkpoint["model"])

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            error_msg = "Error loading optimizer state dict"
            if mode == "test":
                print(error_msg, "-> OK since in test mode")
            else:
                raise Exception(error_msg)

        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.logger.log_load_checkpoint(checkpoint["epoch"])

    def train(self, train_loader, val_loader, num_epochs=10,
              model_loader=None, checkpoint_freq=0):
        self.mode = "train"

        self.num_epochs = num_epochs

        self._get_dataset_info(train_loader.dataset)

        self.logger.start(self.num_epochs, len(train_loader), self.get_output_vis_image_labels())
        self.logger.log_text("START")

        for self.epoch in range(self.epoch, self.num_epochs+1):
            loss_coll = self.train_epoch(train_loader)

            self.logger.log_epoch(self.epoch, "train", loss_coll)

            if self.logger.log_val_freq and self.epoch % self.logger.log_val_freq == 0:
                loss_coll = self.compute_val_loss(val_loader)

                self.logger.log_epoch(self.epoch, "val", loss_coll)

            if checkpoint_freq and self.epoch % checkpoint_freq == 0:
                model_loader.save("checkpoint", self.create_checkpoint(), epoch=self.epoch)
                self.logger.log_create_checkpoint(self.epoch)

        self.logger.log_text('FINISH')

        self.logger.end()

        self.mode = None

        # print(self.logger.train_stats)
        # return self.logger.get_last_loss()

    def _get_dataset_info(self, dataset):
        #self.vis_idx = dataset.depth_output_indices[len(dataset.depth_output_indices) // 2]
        # TODO: Caution if correct
        # self.vis_idx = len(dataset.depth_output_indices) // 2
        self.vis_idx = dataset.depth_output_indices[0]
        self.depth_output_indices = dataset.depth_output_indices

        print("Vis idx", self.vis_idx)

    def test(self, test_loader):
        self.mode = "test"

        # TODO: Assure loader has batch_size 1 and shuffle False
        self._get_dataset_info(test_loader.dataset)

        self.logger.start(1, len(test_loader), self.get_output_vis_image_labels())
        self.logger.log_text("START TEST")

        self.compute_test_loss(test_loader)

        self.logger.log_text('FINISH TEST')
        self.logger.end()

        self.mode = None

    def forward_model(self, data):
        encoding_modifier = None

        if self.extend_data_module is not None:
            if not self.extend_encoding:
                data = self.extend_data_module(data)
            else:
                encoding_modifier = ExtendEncoding(self.extend_data_module, data)

        if encoding_modifier is None:
            return self.model(data["color"]), data
        else:
            return self.model(data["color"], encoding_modifier=encoding_modifier), data

    def get_output_vis_images(self, idx, data, outputs):
        return None

    def get_output_vis_image_labels(self):
        return None

    def __repr__(self):
        return "Trainer"

    def _select_img(self, img_type, idx, img_list, layers=None):
        if not isinstance(img_list, list):
            img_list = [img_list]

        img_selected = []

        for img in img_list:
            if len(img[idx].shape) == 4:
                if not self.vis_whole_seq:
                    # print(img_type)
                    x = img[idx][self.vis_idx]
                elif img_type == "input":
                    x = img[idx][self.depth_output_indices]
                else:
                    x = img[idx]
            else:
                x = img[idx]

            if layers is not None:
                assert len(layers) == 2 and layers[0] == 3 and layers[1] == 2

                if len(x.shape) == 4:
                    if sum(layers) != x.shape[1]:
                        raise Exception("sum(layers) != x.shape[1] " + str(layers) + " != " + str(x.shape))

                    x = [x[:, :3], x[:, 3:]]
                elif len(x.shape) == 3:
                    if sum(layers) != x.shape[0]:
                        raise Exception("sum(layers) != x.shape[0] " + str(layers) + " != " + str(x.shape))

                    x = [x[:3], x[3:]]
                else:
                    raise Exception("Shape error")
            else:
                x = [x]

            img_selected += x

        return img_selected

    def _get_input_layer_desc(self):
        if self.extend_data_module is None or self.extend_encoding:
            return None
        else:
            return self.extend_data_module.get_layers()

    def _batch_to_vis(self, idx, inputs, target, output):
        return {
            "input": self._select_img("input", idx, inputs, self._get_input_layer_desc()),
            "target": self._select_img("target", idx, target),
            "output": self._select_img("output", idx, output)
        }


class TrainerDepth(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, mode, data, data_loader, loss_coll, backward_loss=False):
        data = self.data_to_dev(data)
        outputs, data = self.forward_model(data)
        # print(outputs.shape, data["depth"].shape)
        loss = self.loss_func(outputs, data["depth"])

        if backward_loss:
            self.optimizer.zero_grad()
            loss.backward()

        loss_value = loss.detach().cpu().numpy()
        self.update_loss_coll(loss_coll, loss_value, outputs, data["depth"])

        return loss_value, outputs, data

    def get_output_vis_images(self, idx, data, outputs):
        return self._batch_to_vis(idx, data["color"], data["depth"], outputs)

    def get_output_vis_image_labels(self):
        return ["depth"]


class TrainerCoC(Trainer):
    def __init__(self, loss_func=torch.nn.MSELoss(), loss_mask=None, **kwargs):
        assert loss_func is not None

        loss_func = MultiMaskedLoss(loss_func, loss_mask)
        super().__init__(loss_func=loss_func, **kwargs)

    def compute_loss(self, mode, data, data_loader, loss_coll, backward_loss=False):
        data = self.data_to_dev(data)
        outputs, data = self.forward_model(data)
        loss = self.loss_func(data["depth"], outputs, data["coc"])

        if backward_loss:
            self.optimizer.zero_grad()
            loss.backward()

        loss_value = loss.detach().cpu().numpy()
        self.update_loss_coll(loss_coll, loss_value, data["depth"], outputs, data["coc"])

        return loss_value, outputs, self.data_to_dev(data, "cpu")

    def get_output_vis_images(self, idx, data, outputs):
        return self._batch_to_vis(idx, data["color"], data["coc"], outputs)

    def get_output_vis_image_labels(self):
        return ["coc"]


class TrainerSignedCoC(TrainerDepth):
    def __init__(self, compute_coc_abs=False, **kwargs):
        super().__init__(loss_func=torch.nn.MSELoss(), **kwargs)

        self.clamp_computed_depth = True

    @property
    def compute_coc_abs(self):
        return self.mode == "test"

    def _compute_depth(self, data_loader, signed_coc, focus_dist):
        dataset = data_loader.dataset
        depth_out = dataset.lens.get_depth_from_signed_coc(
            focus_distance=focus_dist,
            signed_coc=signed_coc,
            signed_coc_normalize=dataset.signed_coc_normalize,
            depth_normalize=dataset.depth_normalize
        )

        return depth_out

    def compute_loss(self, mode, data, data_loader, loss_coll, backward_loss=False):
        data = self.data_to_dev(data)
        signed_coc_out, data = self.forward_model(data)
        loss = self.loss_func(signed_coc_out, data["signed_coc"])

        with torch.no_grad():
            depth_out = self._compute_depth(data_loader, signed_coc_out, data["focus_dist"])

        if backward_loss:
            self.optimizer.zero_grad()
            loss.backward()

        loss_value = loss.detach().cpu().numpy()

        if self.clamp_computed_depth:
            with torch.no_grad():
                depth_out.clamp_(0, 1)

        self.update_loss_coll(loss_coll, loss_value, depth_out, data["depth"])

        out = {"signed_coc": signed_coc_out, "depth": depth_out}

        if self.compute_coc_abs:
            with torch.no_grad():
                out["coc"] = signed_coc_out.abs()

        return loss_value, out, data

    def get_output_vis_images(self, idx, data, outputs):
        target_vis, out_vis = [data["signed_coc"], data["depth"]], [outputs["signed_coc"], outputs["depth"]]

        if self.compute_coc_abs:
            target_vis, out_vis = [data["signed_coc"].abs()] + target_vis, [outputs["coc"]] + out_vis

        return self._batch_to_vis(
            idx,
            data["color"],
            target_vis,
            out_vis
        )

    def get_output_vis_image_labels(self):
        labels = ["signed_coc", "depth"]

        if self.compute_coc_abs:
            labels = ["coc"] + labels

        return labels


class TrainerFgbgCoc(TrainerDepth):
    def __init__(self, vis_fgbg=True, vis_coc=True, loss_mask=None, use_depth_loss=False, use_coc_bce_weight=False,
                 ingore_coc_loss=False, **kwargs):
        if ingore_coc_loss:
            if use_depth_loss:
                loss_func = torch.nn.MSELoss()
            else:
                loss_func = FgbgLoss()
        else:
            loss_func = CoCDepthLoss() if use_depth_loss else FgbgCocLoss(
                coc_ratio=1 if not use_coc_bce_weight else 0.5,
                use_coc_bce_weight=use_coc_bce_weight)

        if loss_mask is not None:
            loss_func = MultiMaskedLoss(loss_func, loss_mask)

        super().__init__(loss_func=loss_func, **kwargs)

        self.use_depth_loss = use_depth_loss
        self.vis_fgbg = vis_fgbg
        self.vis_coc = vis_coc
        self.ingore_coc_loss = ingore_coc_loss
        self.clamp_computed_depth = True

    def _eval_loss_func(self, data, fgbg_out, coc_out, depth_out):
        if self.ingore_coc_loss:
            if self.use_depth_loss:
                loss_args = [depth_out, data["depth"]]
            else:
                loss_args = [fgbg_out, data["fgbg"], data["coc"]]
        else:
            if self.use_depth_loss:
                loss_args = [coc_out, depth_out, data["coc"], data["depth"]]
            else:
                loss_args = [fgbg_out, coc_out, data["fgbg"], data["coc"]]

        if isinstance(self.loss_func, MultiMaskedLoss):
            loss_args = [data["depth"]] + loss_args

        return self.loss_func(*loss_args)

    def compute_loss(self, mode, data, data_loader, loss_coll, backward_loss=False):
        data = self.data_to_dev(data)
        (coc_out, fgbg_out), data = self.forward_model(data)

        # TEST
        # fgbg_out = data["fgbg"]
        # coc_out = data["coc"]

        if self.use_depth_loss:
            fgbg_out_sig = torch.sigmoid(fgbg_out)
            depth_out = self._compute_depth(data_loader, fgbg_out_sig, coc_out, data["focus_dist"])
        else:
            with torch.no_grad():
                fgbg_out_sig = torch.sigmoid(fgbg_out)
                depth_out = self._compute_depth(data_loader, fgbg_out_sig, coc_out, data["focus_dist"])

        loss = self._eval_loss_func(data, fgbg_out, coc_out, depth_out)

        if backward_loss:
            self.optimizer.zero_grad()
            loss.backward()

        loss_value = loss.detach().cpu().numpy()

        if self.clamp_computed_depth and not self.use_depth_loss:
            with torch.no_grad():
                depth_out.clamp_(0, 1)

        self.update_loss_coll(loss_coll, loss_value, depth_out, data["depth"])

        return loss_value, {"fgbg": fgbg_out_sig, "coc": coc_out, "depth": depth_out}, data

    def _compute_depth(self, data_loader, fgbg, coc, focus_dist):
        dataset = data_loader.dataset

        if len(coc.shape) == 5 and len(fgbg.shape) == 4:
            coc = coc[:, dataset.depth_output_indices]
            # focus_dist = focus_dist[:, dataset.depth_output_indices]

        fgbg_out_thresh = fgbg > 0.5

        depth_out = dataset.lens.get_depth_from_fgbg_coc(
            focus_distance=focus_dist,
            fgbg=fgbg_out_thresh,
            coc=coc,
            coc_normalize=dataset.coc_normalize,
            depth_normalize=dataset.depth_normalize
        )

        return depth_out

    def get_output_vis_images(self, idx, data, outputs):
        target_vis, out_vis = [data["depth"]], [outputs["depth"]]

        if self.vis_coc:
            target_vis, out_vis = [data["coc"]] + target_vis, [outputs["coc"]] + out_vis

        if self.vis_fgbg:
            target_vis, out_vis = [data["fgbg"]] + target_vis, [outputs["fgbg"]] + out_vis

        return self._batch_to_vis(
            idx,
            data["color"],
            target_vis,
            out_vis
        )

    def get_output_vis_image_labels(self):
        labels = ["depth"]

        if self.vis_coc:
            labels = ["coc"] + labels

        if self.vis_fgbg:
            labels = ["fgbg"] + labels

        return labels


class TrainerCoCDepth(TrainerDepth):
    def __init__(self, vis_coc=True, loss_mask=None, is_predepth_net=False, ingore_coc_loss=False, **kwargs):
        loss_func = CoCDepthLoss() if not ingore_coc_loss else torch.nn.MSELoss()

        if loss_mask is not None:
            loss_func = MultiMaskedLoss(loss_func, loss_mask)

        super().__init__(loss_func=loss_func,
                         **kwargs)

        self.vis_coc = vis_coc
        self.is_predepth_net = is_predepth_net
        self.ingore_coc_loss = ingore_coc_loss

        assert not self.is_predepth_net

    def _get_dataset_info(self, dataset):
        super()._get_dataset_info(dataset)

        if self.is_predepth_net:
            self.model.set_dataset(dataset)

    def _eval_loss_func(self, data, coc_out, depth_out):
        if self.ingore_coc_loss:
            loss_args = [depth_out, data["depth"]]
        else:
            loss_args = [coc_out, depth_out, data["coc"], data["depth"]]

        if isinstance(self.loss_func, MultiMaskedLoss):
            if "depth_mask" in data:
                loss_args = [data["depth_mask"]] + loss_args
            else:
                loss_args = [data["depth"]] + loss_args

        return self.loss_func(*loss_args)

    def compute_loss(self, mode, data, data_loader, loss_coll, backward_loss=False):
        data = self.data_to_dev(data)

        # model_args = [data["color"]]
        # if self.is_predepth_net:
        #    model_args.append(data["focus_dist"])

        (coc_out, depth_out), data = self.forward_model(data)

        loss = self._eval_loss_func(data, coc_out, depth_out)

        if backward_loss:
            self.optimizer.zero_grad()
            loss.backward()

        loss_value = loss.detach().cpu().numpy()

        self.update_loss_coll(loss_coll, loss_value, depth_out, data["depth"])

        return loss_value, {"coc": coc_out, "depth": depth_out}, data

    def get_output_vis_images(self, idx, data, outputs):
        target_vis, out_vis = [data["depth"]], [outputs["depth"]]

        if self.vis_coc:
            target_vis, out_vis = [data["coc"]] + target_vis, [outputs["coc"]] + out_vis

        return self._batch_to_vis(
            idx,
            data["color"],
            target_vis,
            out_vis
        )

    def get_output_vis_image_labels(self):
        labels = ["depth"]

        if self.vis_coc:
            labels = ["coc"] + labels

        return labels
