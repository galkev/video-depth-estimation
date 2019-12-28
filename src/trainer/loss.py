import torch
from torch import nn
import torch.nn.functional as F


copy_unmasked_target_to_input = False


class MaskedLoss(nn.Module):
    def __init__(self, loss, cond):
        super().__init__()
        self.loss = loss
        self.cond = cond

    def forward(self, inputs, targets):
        mask = self.cond(targets)
        # print("Mask ratio", mask.sum().float() / targets.numel())

        if copy_unmasked_target_to_input:
            inputs[~mask] = targets[~mask]

        return self.loss(inputs[mask], targets[mask])

    def __repr__(self):
        self_name = "Masked"  # self._get_name()
        return "{}({})".format(self_name, self.loss.__repr__())


def _apply_mask(x, mask):
    if len(mask.shape) == 5 and len(x.shape) == 4:
        mask = mask[:, -1]

    return x[mask]


class MultiMaskedLoss(nn.Module):
    def __init__(self, loss, mask_cond):
        super().__init__()
        self.loss = loss
        self.mask_cond = mask_cond

    def forward(self, mask_input, *args):
        # print(mask_input.numel(), mask_input.shape)
        # print(mask_input.shape, [a.shape for a in args])
        if self.mask_cond is not None:
            mask = self.mask_cond(mask_input)

            if copy_unmasked_target_to_input:
                # assert len(kwargs) == 0

                for i in range(len(args) // 2):
                    args[i][~mask] = _apply_mask(args[i + len(args) // 2], ~mask)

            return self.loss(*[_apply_mask(a, mask) for a in args])
        else:
            return self.loss(*args)

    def __repr__(self):
        self_name = "MultiMaskedLoss"  # self._get_name()
        return "{}({})".format(self_name, self.loss.__repr__())


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights):
        return F.binary_cross_entropy_with_logits(inputs, targets, weights)


class FgbgLoss(nn.Module):
    def __init__(self, use_coc_bce_weight=False):
        super().__init__()
        self.bce = WeightedBCEWithLogitsLoss()  # nn.BCELoss()
        self.use_coc_bce_weight = use_coc_bce_weight

    def forward(self, fgbg_out, fgbg_target, coc_target=None):
        assert not (self.use_coc_bce_weight and coc_target is None)
        return self.bce(fgbg_out, fgbg_target, coc_target if self.use_coc_bce_weight else None)


class FgbgCocLoss(nn.Module):
    def __init__(self, coc_ratio=1, use_coc_bce_weight=False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = WeightedBCEWithLogitsLoss()  # nn.BCELoss()
        self.coc_ratio = coc_ratio
        self.use_coc_bce_weight = use_coc_bce_weight

    # depth target for mask
    def forward(self, fgbg_out, coc_out, fgbg_target, coc_target):
        fgbg_loss = self.bce(fgbg_out, fgbg_target, coc_target if self.use_coc_bce_weight else None)
        coc_loss = self.mse(coc_out, coc_target) if coc_out.requires_grad else 0

        # print(fgbg_loss, coc_loss)

        return fgbg_loss + self.coc_ratio * coc_loss


class CoCDepthLoss(nn.Module):
    def __init__(self, coc_weight=1):
        super().__init__()
        self.mse_loss = nn.MSELoss()

        self.coc_weight = coc_weight

    def forward(self, coc_out, depth_out, coc_target, depth_target):
        depth_loss = self.mse_loss(depth_out, depth_target)
        coc_loss = self.mse_loss(coc_out, coc_target) if coc_out.requires_grad else 0

        return depth_loss + self.coc_weight * coc_loss


class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return 10 * torch.log10(1 / self.mse(inputs, targets))


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.mse(inputs, targets).sqrt()

    def __repr__(self):
        return self._get_name()


class LogRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.mse(inputs.log_text(), targets.log_text()).sqrt()

    def __repr__(self):
        return self._get_name()


class AbsRelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return ((inputs - targets).abs() / targets).sum() / inputs.numel()


class SquaredRelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return ((inputs - targets).pow(2) / targets).sum() / inputs.numel()


class AccuracyLoss(nn.Module):
    def __init__(self, thresh=1.25):
        super().__init__()
        self.thresh = thresh

    def forward(self, inputs, targets):
        delta = torch.max(inputs/targets, targets/inputs)
        return ((delta < self.thresh).sum().float() / inputs.numel()) * 100


class BadPixLoss(nn.Module):
    def __init__(self, thresh=0.05):
        super().__init__()
        self.thresh = thresh

    def forward(self, inputs, targets):
        delta = (inputs - targets).abs()
        return ((delta > self.thresh).sum().float() / inputs.numel()) * 100
