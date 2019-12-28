#! /usr/bin/python3

import torch.nn as nn
import torchvision
import torch
import numpy as np

from net.ddff_net_har_map_state import *


class DDFFTwoDec(nn.Module):
    def __init__(self, dec1, dec2):
        super().__init__()

        self.dec1 = dec1
        self.dec2 = dec2

    def forward(self, *args):
        out1 = self.dec1(*args)
        out2 = self.dec2(*args)

        return out1, out2

    # for compatibility
    def get_skip_conn_usage(self):
        return [self.dec1.cc1_enabled, self.dec1.cc2_enabled, self.dec1.cc3_enabled,
                self.dec1.cc4_enabled, self.dec1.cc5_enabled]


class DDFFNet(nn.Module):
    def __init__(self, focal_stack_size, pred_middle=False, *args, **kwargs):
        super(DDFFNet, self).__init__()
        self.encoder, self.decoder, self.scoring = self.create_encoder_decoder(focal_stack_size, *args, **kwargs)

        self.pred_middle = pred_middle

        # self.init_weights(pretrained, bias)

    def get_output_mode(self):
        return "last" if not self.pred_middle else "middle"

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(-1, *x.shape[2:])

        x, ccx = self.encoder(x)
        x = self.decoder(x, ccx)

        x = x.view(batch_size, -1, *x.shape[2:])

        x = self.scoring(x)

        return x

    @staticmethod
    def create_encoder_decoder(focal_stack_size=None, dropout=0.5,
                               output_dims=1, cc1_enabled=False, cc2_enabled=False, cc3_enabled=True, cc4_enabled=False,
                               cc5_enabled=False, bias=False, pretrained='no_bn', two_dec=False):

        encoder = DDFFEncoder(dropout=dropout, bias=bias)

        decoder = DDFFDecoder(output_dims, cc1_enabled, cc2_enabled, cc3_enabled, cc4_enabled, cc5_enabled,
                                   dropout=dropout, bias=bias)

        if focal_stack_size is not None:
            scoring = nn.Conv2d(focal_stack_size * output_dims, output_dims, 1, bias=False)

        encoder.apply(DDFFNet.weights_default)
        decoder.apply(DDFFNet.weights_default)

        if focal_stack_size is not None:
            scoring.apply(DDFFNet.weights_default)

        encoder.load_pretrained(pretrained, bias)

        if two_dec:
            decoder2 = DDFFDecoder(output_dims, cc1_enabled, cc2_enabled, cc3_enabled, cc4_enabled, cc5_enabled,
                                   dropout=dropout, bias=bias)

            decoder2.apply(DDFFNet.weights_default)

            decoder = DDFFTwoDec(decoder, decoder2)

        if focal_stack_size is not None:
            return encoder, decoder, scoring
        else:
            return encoder, decoder

    """
    def init_weights(self, pretrained, bias):
        #self.apply(DDFFNet.weights_default)

        self.encoder.apply(DDFFNet.weights_default)
        self.decoder.apply(DDFFNet.weights_default)
        self.scoring.apply(DDFFNet.weights_default)

        self.encoder.load_pretrained(pretrained, bias)
    """

    @staticmethod
    def weights_default(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0, 1.0)
            m.running_var.normal_(0, 1.0)
            m.running_mean.fill_(0)
            m.bias.data.fill_(0)


class DDFFEncoder(nn.Module):
    def __init__(self, dropout, bias=False):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1, bias=bias)
        self.conv1_1_bn = nn.BatchNorm2d(64, eps=0.001)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, bias=bias)
        self.conv1_2_bn = nn.BatchNorm2d(64, eps=0.001)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, bias=bias)
        self.conv2_1_bn = nn.BatchNorm2d(128, eps=0.001)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, bias=bias)
        self.conv2_2_bn = nn.BatchNorm2d(128, eps=0.001)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, bias=bias)
        self.conv3_1_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_2_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_3_bn = nn.BatchNorm2d(256, eps=0.001)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.encdrop3 = nn.Dropout(p=dropout)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, bias=bias)
        self.conv4_1_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_3_bn = nn.BatchNorm2d(512, eps=0.001)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.encdrop4 = nn.Dropout(p=dropout)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_1_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_3_bn = nn.BatchNorm2d(512, eps=0.001)
        self.pool5 = nn.MaxPool2d(2, stride=2)
        self.encdrop5 = nn.Dropout(p=dropout)

    def load_pretrained(self, pretrained, bias):
        if pretrained == 'no_bn':
            autoencoder_state_dict = self.state_dict()
            pretrained_dict = torchvision.models.vgg16(pretrained=True).features.state_dict()
            pretrained_dict = map_state_dict(pretrained_dict, bias=bias)
            autoencoder_state_dict.update(pretrained_dict)
            self.load_state_dict(autoencoder_state_dict)

            print("Loaded pretrained")
        elif pretrained == 'bn':
            autoencoder_state_dict = self.state_dict()
            pretrained_dict = torchvision.models.vgg16_bn(pretrained=True).features.state_dict()
            pretrained_dict = map_state_dict_bn(pretrained_dict, bias=bias)
            autoencoder_state_dict.update(pretrained_dict)
            self.load_state_dict(autoencoder_state_dict)
        elif pretrained is not None:
            autoencoder_state_dict = self.state_dict()
            pretrained_weights = np.load(pretrained, encoding="latin1").item()
            pretrained_dict = map_state_dict_tf(pretrained_weights, bias=bias)
            autoencoder_state_dict.update(pretrained_dict)
            self.load_state_dict(autoencoder_state_dict)

    def forward(self, x):
        x = nn.functional.relu(self.conv1_1_bn(self.conv1_1(x)))
        cc1 = nn.functional.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(cc1)
        x = nn.functional.relu(self.conv2_1_bn(self.conv2_1(x)))
        cc2 = nn.functional.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(cc2)
        x = nn.functional.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = nn.functional.relu(self.conv3_2_bn(self.conv3_2(x)))
        cc3 = nn.functional.relu(self.conv3_3_bn(self.conv3_3(x)))
        x = self.pool3(cc3)
        x = self.encdrop3(x)
        x = nn.functional.relu(self.conv4_1_bn(self.conv4_1(x)))
        x = nn.functional.relu(self.conv4_2_bn(self.conv4_2(x)))
        cc4 = nn.functional.relu(self.conv4_3_bn(self.conv4_3(x)))
        x = self.pool4(cc4)
        x = self.encdrop4(x)
        x = nn.functional.relu(self.conv5_1_bn(self.conv5_1(x)))
        x = nn.functional.relu(self.conv5_2_bn(self.conv5_2(x)))
        cc5 = nn.functional.relu(self.conv5_3_bn(self.conv5_3(x)))
        x = self.pool5(cc5)
        x = self.encdrop5(x)

        return x, [cc1, cc2, cc3, cc4, cc5]


    # for compatibility

    def get_out_dim(self):
        return [512, 8, 8]

    def get_channel_sizes(self):
        return [64, 128, 256, 512, 512]


class DDFFDecoder(nn.Module):
    def __init__(self, output_dims, cc1_enabled, cc2_enabled, cc3_enabled, cc4_enabled, cc5_enabled,
                 dropout, bias=False):
        super().__init__()

        self.output_dims = output_dims
        self.cc1_enabled = cc1_enabled
        self.cc2_enabled = cc2_enabled
        self.cc3_enabled = cc3_enabled
        self.cc4_enabled = cc4_enabled
        self.cc5_enabled = cc5_enabled

        self.upconv5 = nn.ConvTranspose2d(512, 512, 4, padding=1, stride=2, bias=False)
        if self.cc5_enabled:
            self.conv5_3_D = nn.Conv2d(1024, 512, 3, padding=1, bias=bias)
        else:
            self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_3_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_2_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_1_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.decdrop5 = nn.Dropout(p=dropout)

        self.upconv4 = nn.ConvTranspose2d(512, 512, 4, padding=1, stride=2, bias=False)
        if self.cc4_enabled:
            self.conv4_3_D = nn.Conv2d(1024, 512, 3, padding=1, bias=bias)
        else:
            self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_3_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_2_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1, bias=bias)
        self.conv4_1_D_bn = nn.BatchNorm2d(256, eps=0.001)
        self.decdrop4 = nn.Dropout(p=dropout)

        self.upconv3 = nn.ConvTranspose2d(256, 256, 4, padding=1, stride=2, bias=False)
        if self.cc3_enabled:
            self.conv3_3_D = nn.Conv2d(512, 256, 3, padding=1, bias=bias)
        else:
            self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_3_D_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_2_D_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1, bias=bias)
        self.conv3_1_D_bn = nn.BatchNorm2d(128, eps=0.001)
        self.decdrop3 = nn.Dropout(p=dropout)

        self.upconv2 = nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2, bias=False)
        if self.cc2_enabled:
            self.conv2_2_D = nn.Conv2d(256, 128, 3, padding=1, bias=bias)
        else:
            self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1, bias=bias)
        self.conv2_2_D_bn = nn.BatchNorm2d(128, eps=0.001)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1, bias=bias)
        self.conv2_1_D_bn = nn.BatchNorm2d(64, eps=0.001)

        self.upconv1 = nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2, bias=False)
        if self.cc1_enabled:
            self.conv1_2_D = nn.Conv2d(128, 64, 3, padding=1, bias=bias)
        else:
            self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1, bias=bias)
        self.conv1_2_D_bn = nn.BatchNorm2d(64, eps=0.001)
        self.conv1_1_D = nn.Conv2d(64, self.output_dims, 3, padding=1, bias=bias)
        self.conv1_1_D_bn = nn.BatchNorm2d(self.output_dims, eps=0.001)

    def forward(self, x, ccx):
        cc1, cc2, cc3, cc4, cc5 = ccx

        x = self.upconv5(x)
        if self.cc5_enabled:
            x = torch.cat([x, cc5], 1)
        x = nn.functional.relu(self.conv5_3_D_bn(self.conv5_3_D(x)))
        x = nn.functional.relu(self.conv5_2_D_bn(self.conv5_2_D(x)))
        x = nn.functional.relu(self.conv5_1_D_bn(self.conv5_1_D(x)))
        x = self.decdrop5(x)
        x = self.upconv4(x)
        if self.cc4_enabled:
            x = torch.cat([x, cc4], 1)
        x = nn.functional.relu(self.conv4_3_D_bn(self.conv4_3_D(x)))
        x = nn.functional.relu(self.conv4_2_D_bn(self.conv4_2_D(x)))
        x = nn.functional.relu(self.conv4_1_D_bn(self.conv4_1_D(x)))
        x = self.decdrop4(x)
        x = self.upconv3(x)
        if self.cc3_enabled:
            x = torch.cat([x, cc3], 1)
        x = nn.functional.relu(self.conv3_3_D_bn(self.conv3_3_D(x)))
        x = nn.functional.relu(self.conv3_2_D_bn(self.conv3_2_D(x)))
        x = nn.functional.relu(self.conv3_1_D_bn(self.conv3_1_D(x)))
        x = self.decdrop3(x)
        x = self.upconv2(x)
        if self.cc2_enabled:
            x = torch.cat([x, cc2], 1)
        x = nn.functional.relu(self.conv2_2_D_bn(self.conv2_2_D(x)))
        x = nn.functional.relu(self.conv2_1_D_bn(self.conv2_1_D(x)))
        x = self.upconv1(x)
        if self.cc1_enabled:
            x = torch.cat([x, cc1], 1)
        x = nn.functional.relu(self.conv1_2_D_bn(self.conv1_2_D(x)))
        x = nn.functional.relu(self.conv1_1_D_bn(self.conv1_1_D(x)))

        return x

    # for compatibility
    def get_skip_conn_usage(self):
        return [self.cc1_enabled, self.cc2_enabled, self.cc3_enabled, self.cc4_enabled, self.cc5_enabled]
