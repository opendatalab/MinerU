"""
This code is refer from:
https://github.com/whai362/PSENet/blob/python3/models/neck/fpn.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv_BN_ReLU(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.1)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()

        # Top layer
        self.toplayer_ = Conv_BN_ReLU(
            in_channels[3], out_channels, kernel_size=1, stride=1, padding=0)
        # Lateral layers
        self.latlayer1_ = Conv_BN_ReLU(
            in_channels[2], out_channels, kernel_size=1, stride=1, padding=0)

        self.latlayer2_ = Conv_BN_ReLU(
            in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)

        self.latlayer3_ = Conv_BN_ReLU(
            in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.smooth2_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.smooth3_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.out_channels = out_channels * 4
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _upsample(self, x, scale=1):
        return F.interpolate(x, scale_factor=scale, mode='bilinear')

    def _upsample_add(self, x, y, scale=1):
        return F.interpolate(x, scale_factor=scale, mode='bilinear') + y

    def forward(self, x):
        f2, f3, f4, f5 = x
        p5 = self.toplayer_(f5)

        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4, 2)
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3, 2)
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2, 2)
        p2 = self.smooth3_(p2)

        p3 = self._upsample(p3, 2)
        p4 = self._upsample(p4, 4)
        p5 = self._upsample(p5, 8)

        fuse = torch.cat([p2, p3, p4, p5], dim=1)
        return fuse
