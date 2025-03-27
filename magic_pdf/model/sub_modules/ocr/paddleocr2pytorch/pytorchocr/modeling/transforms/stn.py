"""
This code is refer from:
https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/stn_head.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .tps_spatial_transformer import TPSSpatialTransformer


def conv3x3_block(in_channels, out_channels, stride=1):
    n = 3 * 3 * out_channels
    w = math.sqrt(2. / n)
    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)
    block = nn.Sequential(conv_layer, nn.BatchNorm2d(out_channels), nn.ReLU())
    return block


class STN(nn.Module):
    def __init__(self, in_channels, num_ctrlpoints, activation='none'):
        super(STN, self).__init__()
        self.in_channels = in_channels
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.Sequential(
            conv3x3_block(in_channels, 32),  #32x64
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(32, 64),  #16x32
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(128, 256),  # 4*8
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(256, 256),  # 2*4,
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(256, 256))  # 1*2
        self.stn_fc1 = nn.Sequential(
            nn.Linear(
                2 * 256,
                512,
                bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        fc2_bias = self.init_stn()
        self.stn_fc2 = nn.Linear(
            512,
            num_ctrlpoints * 2,
            bias=True)

    def init_stn(self):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate(
            [ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        if self.activation == 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        ctrl_points = torch.Tensor(ctrl_points)
        # fc2_bias = ctrl_points.view(-1)
        fc2_bias = torch.reshape(
            ctrl_points, shape=[ctrl_points.shape[0] * ctrl_points.shape[1]])
        return fc2_bias

    def forward(self, x):
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.shape
        # x = x.view(batch_size, -1)
        x = torch.reshape(x, shape=(batch_size, -1))
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        # x = x.view(-1, self.num_ctrlpoints, 2)
        x = torch.reshape(x, shape=[-1, self.num_ctrlpoints, 2])
        return img_feat, x


class STN_ON(nn.Module):
    def __init__(self, in_channels, tps_inputsize, tps_outputsize,
                 num_control_points, tps_margins, stn_activation):
        super(STN_ON, self).__init__()
        self.tps = TPSSpatialTransformer(
            output_image_size=tuple(tps_outputsize),
            num_control_points=num_control_points,
            margins=tuple(tps_margins))
        self.stn_head = STN(in_channels=in_channels,
                            num_ctrlpoints=num_control_points,
                            activation=stn_activation)
        self.tps_inputsize = tps_inputsize
        self.out_channels = in_channels

    def forward(self, image):
        stn_input = torch.nn.functional.interpolate(
            image, self.tps_inputsize, mode="bilinear", align_corners=True)
        stn_img_feat, ctrl_points = self.stn_head(stn_input)
        x, _ = self.tps(image, ctrl_points)
        return x
