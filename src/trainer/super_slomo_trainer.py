import torch
from torch import nn
from torchvision.models import vgg16
from torch.autograd import Variable

from . import Trainer, TrainLogger
from .loss import PSNRLoss


class SuperSlomoLoss(nn.Module):
    def __init__(self, device, inter_frame_count=7):
        super().__init__()

        self.inter_frame_count = inter_frame_count

        self.l1loss = nn.L1Loss()  # no reduce ?
        self.mseloss = nn.MSELoss()
        self.vgg16_conv_4_3 = vgg16(pretrained=True).features[:22].to(device)

        self.lambda_rec = 0.8
        self.lambda_perc = 0.005
        self.lambda_warp = 0.4
        self.lambda_smooth = 1

        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets):
        inter_img_target = targets

        loss_rec = 255 * self.l1loss(inputs["inter_img_pred"], inter_img_target)

        loss_perc = self.mseloss(
            self.vgg16_conv_4_3(inputs["inter_img_pred"]),
            self.vgg16_conv_4_3(inter_img_target)
        )

        loss_warp = 255 * (
                self.l1loss(inputs["g_img_1_flow_01"], inputs["img_0"]) +
                self.l1loss(inputs["g_img_0_flow_10"], inputs["img_1"]) +
                self.l1loss(inputs["g_img_0_flow_t0"], inputs["inter_img_pred"]) +
                self.l1loss(inputs["g_img_1_flow_t1"], inputs["inter_img_pred"])
        )

        loss_smooth = \
            self.l1loss(inputs["flow_10"][:, :, :, 1:], inputs["flow_10"][:, :, :, :-1]) + \
            self.l1loss(inputs["flow_10"][:, :, 1:, :], inputs["flow_10"][:, :, :-1, :]) + \
            self.l1loss(inputs["flow_01"][:, :, :, 1:], inputs["flow_01"][:, :, :, :-1]) + \
            self.l1loss(inputs["flow_01"][:, :, 1:, :], inputs["flow_01"][:, :, :-1, :])

        return \
            self.lambda_rec * loss_rec + \
            self.lambda_perc * loss_perc + \
            self.lambda_warp * loss_warp + \
            self.lambda_smooth * loss_smooth


class SuperSlomoTrainer(Trainer):
    def __init__(self,
                 model,
                 device,
                 optimizer,
                 scheduler=None,
                 max_gradient=None,
                 logger=TrainLogger()):
        logger.loss_func = [None, PSNRLoss()]
        super().__init__(model,
                         device,
                         optimizer,
                         SuperSlomoLoss(device),
                         scheduler,
                         max_gradient,
                         logger)

    def compute_loss(self, inputs, targets, backward_loss=False):
        frames = Variable(inputs[0]).to(self.device)
        inter_img_idx = inputs[1]
        inter_img_target = Variable(targets).to(self.device)

        outputs = self.model([frames, inter_img_idx])

        loss = self.loss_func(outputs, inter_img_target)

        if backward_loss:
            self.optimizer.zero_grad()
            loss.backward()

        self.loss_coll.update(outputs["inter_img_pred"], inter_img_target, range(1, len(self.loss_coll)))

        return loss.detach().cpu().numpy()
