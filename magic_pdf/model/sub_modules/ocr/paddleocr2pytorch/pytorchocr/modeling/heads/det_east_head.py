from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchocr.modeling.common import Activation
# import paddle
# from paddle import nn
# import paddle.nn.functional as F
# from paddle import ParamAttr


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

        self.bn = nn.BatchNorm2d(
            out_channels,)
        self.act = act
        if act is not None:
            self._act = Activation(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self._act(x)
        return x


class EASTHead(nn.Module):
    """
    """
    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTHead, self).__init__()
        self.model_name = model_name
        if self.model_name == "large":
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [64, 32, 1, 8]

        self.det_conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=num_outputs[0],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="det_head1")
        self.det_conv2 = ConvBNLayer(
            in_channels=num_outputs[0],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="det_head2")
        self.score_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name="f_score")
        self.geo_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[3],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name="f_geo")

    def forward(self, x):
        f_det = self.det_conv1(x)
        f_det = self.det_conv2(f_det)
        f_score = self.score_conv(f_det)
        f_score = torch.sigmoid(f_score)
        f_geo = self.geo_conv(f_det)
        f_geo = (torch.sigmoid(f_geo) - 0.5) * 2 * 800

        pred = {'f_score': f_score, 'f_geo': f_geo}
        return pred