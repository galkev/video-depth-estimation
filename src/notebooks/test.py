

import os
import sys

import pathmagic
from tools.project import proj_dir

device = "cuda"

from data.video_depth_focus_data import VideoDepthFocusData

def create_dataset(data_type):
    data = VideoDepthFocusData(proj_dir("datasets"), data_type, "dining_room")

    data.configure(sample_count=2,
                   sample_skip=1,
                   depth_output_indices=1,
                   use_allinfocus=False)

    return data

train_data = create_dataset("train")
val_data = create_dataset("val")

from trainer import TrainLoggerTensorboardX

train_logger = TrainLoggerTensorboardX(
                model_id="43",
                model_desc="no_desc",
                log_dir="/home/kevin/Documents/master-thesis/logs/test_log",
                log_it_freq=1,
                log_img_freq=1,
                log_val_freq=1,
                val_loss_func=[],
                save_imgs=True)

from torch import nn
from net.pool_net import PoolNet, PoolNetEncoder, PoolNetDecoder

model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
              dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
              bn_eps=1e-4,
              bias=False,
              act_func=nn.ReLU()).to(device)

import torch

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)

from trainer import TrainerDepth

trainer = TrainerDepth(model=model,
                       device=device,
                       optimizer=optimizer,
                       loss_func=nn.MSELoss(),
                       logger=train_logger
                       )

trainer.train(train_loader,
              val_loader,
              5,
              checkpoint_freq=9999999)

val_data[0]
