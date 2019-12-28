from net.extend_data_modules import ExtendDataFocusDist
from tools.project import proj_dir
from net.pool_net import PoolNet
from trainer.model_loader import ModelLoader
import torch


def get_saved_model_name(model, model_type, train_setup):
    tags = [model_type]

    if type(model) == PoolNet:
        tags.append("poolnet")

    if model.get_in_channels_count() == 4:
        if isinstance(train_setup.extend_data_module, ExtendDataFocusDist):
            tags.append("add_channel_" + "focusdist")

    if train_setup.select_focus_dists == [0.1, 0.2167, 0.33, 0.45]:
        tags.append("sel_focus_dist_nonlin4")
    elif train_setup.select_focus_dists == [0.1, 0.13, 0.2167, 0.45]:
        tags.append("sel_focus_dist_lin4")
    else:
        tags.append(f"sample_size{train_setup.sample_size}")
        tags.append(f"sample_skip{train_setup.sample_skip}")

    model_name = "-".join(tags)

    print("Pretrained model name", model_name)

    return model_name


def load_pretrained(model, model_type, train_setup=None, freeze=True):
    if model_type == "flow_nvidia":
        from net.flownet2 import load_flownet_nvidia
        load_flownet_nvidia(model)
    else:
        name = get_saved_model_name(model, model_type, train_setup)
        checkpoint = ModelLoader(proj_dir("pretrained"), name, require_exist=True).load("checkpoint")
        model.load_state_dict(checkpoint["model"])

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
