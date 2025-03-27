from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchocr.modeling.common import Activation


class TableFPN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(TableFPN, self).__init__()
        self.out_channels = 512

        self.in2_conv = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.in3_conv = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            stride = 1,
            bias=False)
        self.in4_conv = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.in5_conv = nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.p5_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.p4_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.p3_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.p2_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.fuse_conv = nn.Conv2d(
            in_channels=self.out_channels * 4,
            out_channels=512,
            kernel_size=3,
            padding=1,
            bias=False)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.interpolate(
            in5, size=in4.shape[2:4], mode="nearest", )#align_mode=1)  # 1/16
        out3 = in3 + F.interpolate(
            out4, size=in3.shape[2:4], mode="nearest", )#align_mode=1)  # 1/8
        out2 = in2 + F.interpolate(
            out3, size=in2.shape[2:4], mode="nearest", )#align_mode=1)  # 1/4

        p4 = F.interpolate(out4, size=in5.shape[2:4], mode="nearest", )#align_mode=1)
        p3 = F.interpolate(out3, size=in5.shape[2:4], mode="nearest", )#align_mode=1)
        p2 = F.interpolate(out2, size=in5.shape[2:4], mode="nearest", )#align_mode=1)
        fuse = torch.cat([in5, p4, p3, p2], dim=1)
        fuse_conv = self.fuse_conv(fuse) * 0.005
        return [c5 + fuse_conv]