import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg

from tools.tools import json_support


"""
@json_support
class DDFFEncoderNet(vgg.VGG):
    def __init__(self, dropout=0, load_pretrained=True):
        super().__init__(vgg.make_layers(vgg.cfgs['D'], batch_norm=True), init_weights=False)

        if load_pretrained:
            vgg_model = 'vgg16_bn'
            self.load_state_dict(vgg.load_state_dict_from_url(vgg.model_urls[vgg_model]))
            print("Load pretrained VGG")
        else:
            print("Not loading pretrained weights")

        if dropout > 0.0:
            mod_feat = []
            # add dropout and batchnorm layers
            # bn_eps = 0.001
            pool_id = 0
            for feat in self.features:
                mod_feat.append(feat)
                if type(feat) == nn.MaxPool2d:
                    if pool_id >= 2:
                        mod_feat.append(nn.Dropout(p=dropout))
                    pool_id += 1

            self.features = torch.nn.Sequential(*mod_feat)

        self.cc_idx = []
        # identify skip connections (3 before every pool layer (Conv, BN, RELU)
        for i, feat in enumerate(self.features):
            if type(feat) == nn.MaxPool2d:
                self.cc_idx.append(i - 3)

        # print(self.cc_idx)

    def get_out_dim(self):
        return [512, 7, 7]

    def forward(self, x):
        ccx = []

        for i, feature in enumerate(self.features):
            # print(i)
            x = feature(x)

            if i in self.cc_idx:
                ccx.append(x)

        return x, ccx

    def print_vgg_info(self):
        for i, feature in enumerate(self.features):
            print(i, feature)


@json_support
class DDFFDecoderNet(nn.Module):
    def __init__(self, output_dim=1, dropout=0, bias=False, use_cc=None):
        super(DDFFDecoderNet, self).__init__()

        self.in_channels = 512
        self.in_dim = [7, 7]

        if use_cc is None:
            self.use_cc = {"1": False, "2": False, "3": True, "4": False, "5": False}
        else:
            self.use_cc = {"1": use_cc[0], "2": use_cc[1], "3": use_cc[2], "4": use_cc[3], "5": use_cc[4]}

        bn_eps = 0.001

        self.upconv5 = nn.ConvTranspose2d(512, 512, 4, padding=1, stride=2, bias=False)
        self.conv5_3_D = nn.Conv2d(512 if not self.use_cc["5"] else 1024, 512, 3, padding=1, bias=bias)
        self.conv5_3_D_bn = nn.BatchNorm2d(512, eps=bn_eps)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_2_D_bn = nn.BatchNorm2d(512, eps=bn_eps)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_1_D_bn = nn.BatchNorm2d(512, eps=bn_eps)
        self.decdrop5 = nn.Dropout(p=dropout)

        self.upconv4 = nn.ConvTranspose2d(512, 512, 4, padding=1, stride=2, bias=False)
        self.conv4_3_D = nn.Conv2d(512 if not self.use_cc["4"] else 1024, 512, 3, padding=1, bias=bias)
        self.conv4_3_D_bn = nn.BatchNorm2d(512, eps=bn_eps)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_2_D_bn = nn.BatchNorm2d(512, eps=bn_eps)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1, bias=bias)
        self.conv4_1_D_bn = nn.BatchNorm2d(256, eps=bn_eps)
        self.decdrop4 = nn.Dropout(p=dropout)

        self.upconv3 = nn.ConvTranspose2d(256, 256, 4, padding=1, stride=2, bias=False)
        self.conv3_3_D = nn.Conv2d(256 if not self.use_cc["3"] else 512, 256, 3, padding=1, bias=bias)
        self.conv3_3_D_bn = nn.BatchNorm2d(256, eps=bn_eps)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_2_D_bn = nn.BatchNorm2d(256, eps=bn_eps)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1, bias=bias)
        self.conv3_1_D_bn = nn.BatchNorm2d(128, eps=bn_eps)
        self.decdrop3 = nn.Dropout(p=dropout)

        self.upconv2 = nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2, bias=False)
        self.conv2_2_D = nn.Conv2d(128 if not self.use_cc["2"] else 256, 128, 3, padding=1, bias=bias)
        self.conv2_2_D_bn = nn.BatchNorm2d(128, eps=bn_eps)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1, bias=bias)
        self.conv2_1_D_bn = nn.BatchNorm2d(64, eps=bn_eps)

        self.upconv1 = nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2, bias=False)
        self.conv1_2_D = nn.Conv2d(64 if not self.use_cc["1"] else 128, 64, 3, padding=1, bias=bias)
        self.conv1_2_D_bn = nn.BatchNorm2d(64, eps=bn_eps)
        self.conv1_1_D = nn.Conv2d(64, output_dim, 3, padding=1, bias=bias)
        self.conv1_1_D_bn = nn.BatchNorm2d(output_dim, eps=bn_eps)

        self.apply(DDFFDecoderNet.init_weights)

    def get_skip_conn_usage(self):
        return [self.use_cc[str(i+1)] for i in range(5)]

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(0, 1.0)
            m.running_var.normal_(0, 1.0)
            m.running_mean.fill_(0)
            m.bias.data.fill_(0)

    # i start at 0
    def ccx_idx_activated(self, i):
        return self.use_cc[str(i + 1)]

    def ccx_activated(self):
        return [self.ccx_idx_activated(i) for i in range(5)]

    def forward(self, x, ccx):
        cc1, cc2, cc3, cc4, cc5 = ccx

        x = self.upconv5(x)
        if self.use_cc["5"]:
            x = torch.cat([x, cc5], 1)
        x = F.relu(self.conv5_3_D_bn(self.conv5_3_D(x)))
        x = F.relu(self.conv5_2_D_bn(self.conv5_2_D(x)))
        x = F.relu(self.conv5_1_D_bn(self.conv5_1_D(x)))
        x = self.decdrop5(x)

        x = self.upconv4(x)
        if self.use_cc["4"]:
            x = torch.cat([x, cc4], 1)
        x = F.relu(self.conv4_3_D_bn(self.conv4_3_D(x)))
        x = F.relu(self.conv4_2_D_bn(self.conv4_2_D(x)))
        x = F.relu(self.conv4_1_D_bn(self.conv4_1_D(x)))
        x = self.decdrop4(x)

        x = self.upconv3(x)
        if self.use_cc["3"]:
            x = torch.cat([x, cc3], 1)
        x = F.relu(self.conv3_3_D_bn(self.conv3_3_D(x)))
        x = F.relu(self.conv3_2_D_bn(self.conv3_2_D(x)))
        x = F.relu(self.conv3_1_D_bn(self.conv3_1_D(x)))
        x = self.decdrop3(x)

        x = self.upconv2(x)
        if self.use_cc["2"]:
            x = torch.cat([x, cc2], 1)
        x = F.relu(self.conv2_2_D_bn(self.conv2_2_D(x)))
        x = F.relu(self.conv2_1_D_bn(self.conv2_1_D(x)))

        x = self.upconv1(x)
        if self.use_cc["1"]:
            x = torch.cat([x, cc1], 1)
        x = F.relu(self.conv1_2_D_bn(self.conv1_2_D(x)))
        x = F.relu(self.conv1_1_D_bn(self.conv1_1_D(x)))

        return x


@json_support
class DDFFNet(nn.Module):
    def __init__(self, focal_stack_size=10, output_dim=1, dropout=0.0,
                 bias=False, load_pretrained=True, scoring_mode="last", use_ccx=None):
        super(DDFFNet, self).__init__()

        if use_ccx is None:
            use_ccx = [False, False, True, False, False]

        self.encoder = DDFFEncoderNet(dropout=dropout, load_pretrained=load_pretrained)

        if scoring_mode.split(sep="/")[0] == "inter":
            self.inter_scoring = nn.Conv2d(focal_stack_size*512, 512, 1, bias=False)
            raise Exception("Old not working")
            # self.ccx_reduce = CCXReduceModule(focal_stack_size, use_ccx, scoring_mode.split(sep="/")[1])
        else:
            self.inter_scoring = None

        self.decoder = DDFFDecoderNet(output_dim, dropout=dropout, bias=bias, use_cc=use_ccx)

        if scoring_mode == "last":
            self.scoring = nn.Conv2d(focal_stack_size*output_dim, output_dim, 1, bias=False)
            DDFFDecoderNet.init_weights(self.scoring)
        else:
            self.scoring = None

    def get_output_mode(self):
        return "last"

    def forward(self, x):
        batch_size, fs_size, num_channels, img_size = x.shape[0], x.shape[1], x.shape[2], x.shape[3:]

        x = x.view(batch_size * fs_size, num_channels, *img_size)

        x, ccx = self.encoder(x)

        if self.inter_scoring is not None:
            x = x.view(batch_size, 512*fs_size, 7, 7)
            x = self.inter_scoring(x)
            ccx = self.ccx_reduce(ccx)

        x = self.decoder(x, ccx)

        if self.inter_scoring is None:
            x = x.view(batch_size, fs_size, *img_size)

        if self.scoring is not None:
            x = self.scoring(x)

        return x
"""
