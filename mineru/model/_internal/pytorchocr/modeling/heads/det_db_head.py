# Copyright (c) Opendatalab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import Activation
from ..backbones.det_mobilenet_v3 import ConvBNLayer

class Head(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.conv_bn1 = nn.BatchNorm2d(
            in_channels // 4)
        self.relu1 = Activation(act_type='relu')

        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=2,
            stride=2)
        self.conv_bn2 = nn.BatchNorm2d(
            in_channels // 4)
        self.relu2 = Activation(act_type='relu')

        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=2,
            stride=2)

    def forward(self, x, return_f=False):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu2(x)
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = torch.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class PPOCRV6DBConvBatchnormLayer(nn.Module):
    """PP-OCRv6 DBHead 使用的 Conv-BN-Act 基础层，命名对齐 safetensors。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        activation="relu",
        bias=False,
        convolution_transpose=False,
    ):
        """初始化普通卷积或反卷积、BN 和激活层。"""
        super().__init__()
        if convolution_transpose:
            self.convolution = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
        else:
            self.convolution = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act_fn = nn.ReLU() if activation == "relu" else nn.Identity()

    def forward(self, hidden_states):
        """执行 DBHead v6 分支的卷积、BN 和激活。"""
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, mode=None, kernel_list=None, fix_nan=False, **kwargs):
        """初始化 DBHead；v6 模式使用 safetensors 对齐的三层上采样 head。"""
        super(DBHead, self).__init__()
        self.k = k
        self.mode = mode
        self.fix_nan = fix_nan
        if mode == "ppocrv6":
            kernel_list = kernel_list or [3, 2, 2]
            self.conv_down = PPOCRV6DBConvBatchnormLayer(
                in_channels=in_channels,
                out_channels=in_channels // 4,
                kernel_size=kernel_list[0],
                padding=int(kernel_list[0] // 2),
            )
            self.conv_up = PPOCRV6DBConvBatchnormLayer(
                in_channels=in_channels // 4,
                out_channels=in_channels // 4,
                kernel_size=kernel_list[1],
                stride=2,
                convolution_transpose=True,
            )
            self.conv_final = nn.ConvTranspose2d(
                in_channels=in_channels // 4,
                out_channels=1,
                kernel_size=kernel_list[2],
                stride=2,
            )
            return
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)

    def step_function(self, x, y):
        """计算 DB 二值化近似阶跃函数。"""
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        """推理时返回统一的 `maps` 字段，兼容现有 OCR-det 后处理。"""
        if self.mode == "ppocrv6":
            shrink_maps = self.conv_down(x)
            shrink_maps = self.conv_up(shrink_maps)
            shrink_maps = self.conv_final(shrink_maps)
            shrink_maps = torch.sigmoid(shrink_maps)
            if self.fix_nan:
                shrink_maps = torch.nan_to_num(shrink_maps)
            return {'maps': shrink_maps}
        shrink_maps = self.binarize(x)
        return {'maps': shrink_maps}


class LocalModule(nn.Module):
    def __init__(self, in_c, mid_c, use_distance=True):
        super(self.__class__, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act='relu')
        self.last_1 = nn.Conv2d(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map, distance_map):
        outf = torch.cat([init_map, x], dim=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out

class PFHeadLocal(DBHead):
    def __init__(self, in_channels, k=50, mode='small', **kwargs):
        super(PFHeadLocal, self).__init__(in_channels, k, **kwargs)
        self.mode = mode

        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest")
        if self.mode == 'large':
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 4)
        elif self.mode == 'small':
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 8)

    def forward(self, x, targets=None):
        shrink_maps, f = self.binarize(x, return_f=True)
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps, None)
        cbn_maps = F.sigmoid(cbn_maps)
        return {'maps': 0.5 * (base_maps + cbn_maps), 'cbn_maps': cbn_maps}
