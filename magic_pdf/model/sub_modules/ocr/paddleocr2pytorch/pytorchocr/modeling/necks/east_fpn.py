from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


class DeConvBNLayer(nn.Module):
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
        super(DeConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(
            out_channels,
            )
        self.act = act
        if act is not None:
            self._act = Activation(act)


    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self._act(x)
        return x


class EASTFPN(nn.Module):
    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTFPN, self).__init__()
        self.model_name = model_name
        if self.model_name == "large":
            self.out_channels = 128
        else:
            self.out_channels = 64
        self.in_channels = in_channels[::-1]
        self.h1_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[1],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_1")
        self.h2_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[2],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_2")
        self.h3_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[3],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_3")
        self.g0_deconv = DeConvBNLayer(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_0")
        self.g1_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_1")
        self.g2_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_2")
        self.g3_conv = ConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_3")

    def forward(self, x):
        f = x[::-1]

        h = f[0]
        g = self.g0_deconv(h)
        # h = paddle.concat([g, f[1]], axis=1)
        h = torch.cat([g, f[1]], dim=1)
        h = self.h1_conv(h)
        g = self.g1_deconv(h)
        # h = paddle.concat([g, f[2]], axis=1)
        h = torch.cat([g, f[2]], dim=1)
        h = self.h2_conv(h)
        g = self.g2_deconv(h)
        # h = paddle.concat([g, f[3]], axis=1)
        h = torch.cat([g, f[3]], dim=1)
        h = self.h3_conv(h)
        g = self.g3_conv(h)

        return g