

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchocr.modeling.common import Activation


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act
        if self.act is not None:
            self._act = Activation(act_type=self.act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self._act(x)
        return x


class PGHead(nn.Module):
    """
    """

    def __init__(self, in_channels, **kwargs):
        super(PGHead, self).__init__()
        self.conv_f_score1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_score{}".format(1))
        self.conv_f_score2 = ConvBNLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu',
            name="conv_f_score{}".format(2))
        self.conv_f_score3 = ConvBNLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_score{}".format(3))

        self.conv1 = nn.Conv2d(
            in_channels=128,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False)

        self.conv_f_boder1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_boder{}".format(1))
        self.conv_f_boder2 = ConvBNLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu',
            name="conv_f_boder{}".format(2))
        self.conv_f_boder3 = ConvBNLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_boder{}".format(3))
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False)
        self.conv_f_char1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_char{}".format(1))
        self.conv_f_char2 = ConvBNLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu',
            name="conv_f_char{}".format(2))
        self.conv_f_char3 = ConvBNLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_char{}".format(3))
        self.conv_f_char4 = ConvBNLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu',
            name="conv_f_char{}".format(4))
        self.conv_f_char5 = ConvBNLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_char{}".format(5))
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=37,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False)

        self.conv_f_direc1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_direc{}".format(1))
        self.conv_f_direc2 = ConvBNLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu',
            name="conv_f_direc{}".format(2))
        self.conv_f_direc3 = ConvBNLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_f_direc{}".format(3))
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False)

    def forward(self, x):
        f_score = self.conv_f_score1(x)
        f_score = self.conv_f_score2(f_score)
        f_score = self.conv_f_score3(f_score)
        f_score = self.conv1(f_score)
        f_score = torch.sigmoid(f_score)

        # f_border
        f_border = self.conv_f_boder1(x)
        f_border = self.conv_f_boder2(f_border)
        f_border = self.conv_f_boder3(f_border)
        f_border = self.conv2(f_border)

        f_char = self.conv_f_char1(x)
        f_char = self.conv_f_char2(f_char)
        f_char = self.conv_f_char3(f_char)
        f_char = self.conv_f_char4(f_char)
        f_char = self.conv_f_char5(f_char)
        f_char = self.conv3(f_char)

        f_direction = self.conv_f_direc1(x)
        f_direction = self.conv_f_direc2(f_direction)
        f_direction = self.conv_f_direc3(f_direction)
        f_direction = self.conv4(f_direction)

        predicts = {}
        predicts['f_score'] = f_score
        predicts['f_border'] = f_border
        predicts['f_char'] = f_char
        predicts['f_direction'] = f_direction
        return predicts
