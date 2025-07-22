import torch.nn as nn


class MobileNetV1Enhance(nn.Module):
    def __init__(self, in_channels=3, scale=0.5, **kwargs):
        super(MobileNetV1Enhance, self).__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            if_act=True,
        )

        self.conv2 = ConvBNLayer(
            in_channels=int(32 * scale),
            out_channels=int(64 * scale),
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
        )

        self.conv3 = DepthwiseSeparable(
            in_channels=int(64 * scale),
            out_channels1=int(64 * scale),
            out_channels2=int(128 * scale),
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
        )

        self.conv4 = DepthwiseSeparable(
            in_channels=int(128 * scale),
            out_channels1=int(128 * scale),
            out_channels2=int(128 * scale),
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
        )

        self.conv5 = DepthwiseSeparable(
            in_channels=int(128 * scale),
            out_channels1=int(128 * scale),
            out_channels2=int(256 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            if_act=True,
        )

        self.conv6 = DepthwiseSeparable(
            in_channels=int(256 * scale),
            out_channels1=int(256 * scale),
            out_channels2=int(256 * scale),
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
        )

        self.conv7 = DepthwiseSeparable(
            in_channels=int(256 * scale),
            out_channels1=int(256 * scale),
            out_channels2=int(512 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            if_act=True,
        )

        self.conv8 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    in_channels=int(512 * scale),
                    out_channels1=int(512 * scale),
                    out_channels2=int(512 * scale),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    if_act=True,
                )
                for _ in range(5)
            ]
        )

        self.conv9 = DepthwiseSeparable(
            in_channels=int(512 * scale),
            out_channels1=int(512 * scale),
            out_channels2=int(1024 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            if_act=True,
        )

        self.conv10 = DepthwiseSeparable(
            in_channels=int(1024 * scale),
            out_channels1=int(1024 * scale),
            out_channels2=int(1024 * scale),
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool(x)
        return x


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        if_act=True,
    ):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if self.if_act:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.relu(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels1,
        out_channels2,
        kernel_size,
        stride=1,
        padding=0,
        if_act=True,
    ):
        super(DepthwiseSeparable, self).__init__()
        self.if_act = if_act
        self.depthwise_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            if_act=False,
        )
        self.pointwise_conv = ConvBNLayer(
            in_channels=out_channels1,
            out_channels=out_channels2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=if_act,
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
