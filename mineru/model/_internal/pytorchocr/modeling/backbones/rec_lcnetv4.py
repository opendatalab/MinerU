# Copyright (c) Opendatalab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn


NET_CONFIG_DET = {
    "small": {
        "stem_channels": [3, 24, 48],
        "block_configs": [
            [[3, 48, 48, 1, True], [3, 48, 48, 1, False]],
            [[3, 48, 96, 2, False], [3, 96, 96, 1, True], [3, 96, 96, 1, False]],
            [
                [3, 96, 192, 2, False],
                [3, 192, 192, 1, True],
                [3, 192, 192, 1, False],
                [3, 192, 192, 1, True],
                [3, 192, 192, 1, False],
            ],
            [[3, 192, 384, 2, False], [3, 384, 384, 1, True], [3, 384, 384, 1, False]],
        ],
    },
}


NET_CONFIG_REC = {
    "small": {
        "stem_channels": [3, 48, 96],
        "block_configs": [
            [[3, 96, 96, 1, True]],
            [[3, 96, 96, 1, False], [3, 96, 96, 1, False]],
            [
                [3, 96, 192, (2, 1), False],
                [3, 192, 192, 1, True],
                [3, 192, 192, 1, False],
                [3, 192, 192, 1, True],
                [3, 192, 192, 1, False],
                [3, 192, 192, 1, True],
                [3, 192, 192, 1, False],
            ],
            [[3, 192, 384, (2, 1), False], [3, 384, 384, 1, True], [3, 384, 384, 1, False]],
        ],
    },
    "medium": {
        "stem_channels": [3, 64, 128],
        "block_configs": [
            [[3, 128, 128, 1, True]],
            [[3, 128, 256, 1, False], [3, 256, 256, 1, False], [3, 256, 256, 1, True]],
            [
                [3, 256, 512, (2, 1), False],
                [3, 512, 512, 1, True],
                [3, 512, 512, 1, False],
                [3, 512, 512, 1, True],
                [3, 512, 512, 1, False],
                [3, 512, 512, 1, True],
                [3, 512, 512, 1, False],
            ],
            [[3, 512, 768, (2, 1), False], [3, 768, 768, 1, True], [3, 768, 768, 1, False]],
        ],
    },
}


def _build_activation(name):
    """按 PP-OCRv6 配置创建无参数激活层，便于和 safetensors 权重命名保持解耦。"""
    if name is None:
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "hardsigmoid":
        return nn.Hardsigmoid()
    raise ValueError(f"Unsupported activation: {name}")


def _to_stride(stride):
    """把 Paddle/Transformers 配置里的 stride 统一成 PyTorch 可接受的格式。"""
    if isinstance(stride, list):
        return tuple(stride)
    return stride


class PPLCNetV4ConvLayer(nn.Module):
    """PP-LCNetV4 的 Conv-BN-Act 基础层，属性名对齐 HF 权重。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        activation="relu",
    ):
        """初始化卷积、归一化和激活层。"""
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=_to_stride(stride),
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = _build_activation(activation)

    def forward(self, hidden_states):
        """执行 Conv-BN-Act 前向计算。"""
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class PPLCNetV4SqueezeExcitationModule(nn.Module):
    """PP-LCNetV4 的 SE 模块，保留 `convolutions.0/2` 权重命名。"""

    def __init__(self, channel, reduction=4):
        """初始化全局池化和两层 1x1 卷积。"""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Hardsigmoid(),
            ]
        )

    def forward(self, hidden_states):
        """根据通道注意力缩放输入特征。"""
        residual = hidden_states
        hidden_states = self.avg_pool(hidden_states)
        for layer in self.convolutions:
            hidden_states = layer(hidden_states)
        return residual * hidden_states


class PPLCNetV4LargeStem(nn.Module):
    """PP-LCNetV4 branch stem，属性名对齐 `encoder.convolution.stem*`。"""

    def __init__(self, stem_channels):
        """初始化 v6 small/medium 使用的分支 stem。"""
        super().__init__()
        self.stem1 = PPLCNetV4ConvLayer(stem_channels[0], stem_channels[1], kernel_size=3, stride=2)
        self.stem2a = PPLCNetV4ConvLayer(stem_channels[1], stem_channels[1] // 2, kernel_size=2, stride=1)
        self.stem2b = PPLCNetV4ConvLayer(stem_channels[1] // 2, stem_channels[1], kernel_size=2, stride=1)
        self.stem3 = PPLCNetV4ConvLayer(stem_channels[1] * 2, stem_channels[1], kernel_size=3, stride=2)
        self.stem4 = PPLCNetV4ConvLayer(stem_channels[1], stem_channels[2], kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, pixel_values):
        """执行分支 stem 的 pad、pool 和 concat 流程。"""
        embedding = self.stem1(pixel_values)
        embedding = F.pad(embedding, (0, 1, 0, 1))
        emb_stem_2a = self.stem2a(embedding)
        emb_stem_2a = F.pad(emb_stem_2a, (0, 1, 0, 1))
        emb_stem_2a = self.stem2b(emb_stem_2a)
        pooled_emb = self.pool(embedding)
        embedding = torch.cat([pooled_emb, emb_stem_2a], dim=1)
        embedding = self.stem3(embedding)
        embedding = self.stem4(embedding)
        return embedding


class PPLCNetV4DepthwiseSeparableConvLayer(nn.Module):
    """PP-LCNetV4 block 中的 token mixer 和 channel mixer。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        use_squeeze_excitation,
        reduction=4,
    ):
        """按 v6 配置初始化深度卷积、SE 和两层 point-wise 卷积。"""
        super().__init__()
        self.has_residual = in_channels == out_channels and stride == 1
        self.use_rep_dw = stride == 1 and in_channels == out_channels
        if self.use_rep_dw:
            self.token_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=True,
            )
        else:
            self.token_conv = PPLCNetV4ConvLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                activation=None,
            )
        self.token_squeeze_excitation = (
            PPLCNetV4SqueezeExcitationModule(in_channels, reduction) if use_squeeze_excitation else nn.Identity()
        )
        self.channel_conv1 = PPLCNetV4ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=1,
            stride=1,
            activation=None,
        )
        self.channel_act_fn = nn.GELU()
        self.channel_conv2 = PPLCNetV4ConvLayer(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            activation=None,
        )

    def forward(self, hidden_states):
        """执行 token mixing、channel mixing 和可选残差连接。"""
        hidden_states = self.token_conv(hidden_states)
        hidden_states = self.token_squeeze_excitation(hidden_states)
        residual = hidden_states
        hidden_states = self.channel_conv1(hidden_states)
        hidden_states = self.channel_act_fn(hidden_states)
        hidden_states = self.channel_conv2(hidden_states)
        if self.has_residual:
            hidden_states = residual + hidden_states
        return hidden_states


class PPLCNetV4Block(nn.Module):
    """PP-LCNetV4 的一个 stage，内部包含多个 depthwise separable block。"""

    def __init__(self, block_configs):
        """根据 stage 配置创建 block 列表。"""
        super().__init__()
        self.blocks = nn.ModuleList()
        for kernel_size, in_channels, out_channels, stride, use_squeeze_excitation in block_configs:
            self.blocks.append(
                PPLCNetV4DepthwiseSeparableConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=_to_stride(stride),
                    use_squeeze_excitation=use_squeeze_excitation,
                )
            )

    def forward(self, hidden_states):
        """顺序执行当前 stage 的所有 block。"""
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class PPLCNetV4Encoder(nn.Module):
    """PP-LCNetV4 编码器，保留 `encoder.convolution` 和 `encoder.blocks` 命名。"""

    def __init__(self, stem_channels, block_configs):
        """初始化 stem 和四个 stage。"""
        super().__init__()
        self.convolution = PPLCNetV4LargeStem(stem_channels)
        self.blocks = nn.ModuleList([PPLCNetV4Block(stage_configs) for stage_configs in block_configs])

    def forward(self, pixel_values):
        """返回四个 stage 的输出特征，供 det/rec 上层按需使用。"""
        hidden_states = self.convolution(pixel_values)
        feature_maps = []
        for block in self.blocks:
            hidden_states = block(hidden_states)
            feature_maps.append(hidden_states)
        return feature_maps


class PPLCNetV4(nn.Module):
    """PP-OCRv6 使用的 PPLCNetV4 backbone，支持 det small 和 rec small/medium。"""

    def __init__(self, det=False, model_size="small", in_channels=3, **kwargs):
        """按 det/rec 模式选择 v6 的固定网络配置。"""
        super().__init__()
        self.det = det
        if in_channels != 3:
            raise ValueError(f"PPLCNetV4 only supports 3 input channels, got {in_channels}.")
        config_dict = NET_CONFIG_DET if det else NET_CONFIG_REC
        if model_size not in config_dict:
            mode = "det" if det else "rec"
            raise ValueError(f"PPLCNetV4 {mode} model_size must be one of {list(config_dict)}, got {model_size}.")
        config = config_dict[model_size]
        self.encoder = PPLCNetV4Encoder(config["stem_channels"], config["block_configs"])
        stage_out_channels = [stage[-1][2] for stage in config["block_configs"]]
        self.out_channels = stage_out_channels if det else stage_out_channels[-1]

    def forward(self, x):
        """det 返回四级特征列表，rec 返回高度池化后的识别特征。"""
        feature_maps = self.encoder(x)
        if self.det:
            return feature_maps
        x = feature_maps[-1]
        if self.training:
            return F.adaptive_avg_pool2d(x, [1, 40])
        if x.shape[2] < 3:
            raise ValueError(f"Feature height {x.shape[2]} < pool kernel 3.")
        return F.avg_pool2d(x, [3, 2])
