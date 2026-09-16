# Copyright (c) Opendatalab. All rights reserved.
import torch
from torch import nn

from ..backbones.rec_svtrnet import Block, ConvBNLayer


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == 1
        x = x.squeeze(dim=2)
        # x = x.transpose([0, 2, 1])  # paddle (NTC)(batch, width, channels)
        x = x.permute(0, 2, 1)
        return x

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     # 处理四维张量，将空间维度展平为序列
    #     if H == 1:
    #         # 原来的处理逻辑，适用于H=1的情况
    #         x = x.squeeze(dim=2)
    #         x = x.permute(0, 2, 1)  # (B, W, C)
    #     else:
    #         # 处理H不为1的情况
    #         x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
    #         x = x.reshape(B, H * W, C)  # (B, H*W, C)
    #
    #     return x

class EncoderWithRNN_(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN_, self).__init__()
        self.out_channels = hidden_size * 2
        self.rnn1 = nn.LSTM(
            in_channels,
            hidden_size,
            bidirectional=False,
            batch_first=True,
            num_layers=2,
        )
        self.rnn2 = nn.LSTM(
            in_channels,
            hidden_size,
            bidirectional=False,
            batch_first=True,
            num_layers=2,
        )

    def forward(self, x):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(torch.flip(x, [1]))
        return torch.cat([out1, torch.flip(out2, [1])], 2)


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True
        )  # batch_first:=True

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            bias=True,
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class EncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels,
        dims=64,  # XS
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        kernel_size=[3, 3],
        attn_drop_rate=0.1,
        drop_path=0.0,
        qk_scale=None,
    ):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act="swish",
        )
        self.conv2 = ConvBNLayer(
            in_channels // 8, hidden_dims, kernel_size=1, act="swish"
        )

        self.svtr_block = nn.ModuleList(
            [
                Block(
                    dim=hidden_dims,
                    num_heads=num_heads,
                    mixer="Global",
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer="swish",
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer="nn.LayerNorm",
                    epsilon=1e-05,
                    prenorm=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act="swish")
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(
            2 * in_channels, in_channels // 8, padding=1, act="swish"
        )

        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act="swish")
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # weight initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # for use guide
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)

        for blk in self.svtr_block:
            z = blk(z)

        z = self.norm(z)
        # last stage
        z = z.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))

        return z


class LightSVTRConvLayer(nn.Module):
    """PP-OCRv6 LightSVTR 使用的 Conv-BN-SiLU 基础层。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        activation="silu",
        groups=1,
    ):
        """初始化与 safetensors key 对齐的卷积、归一化和激活层。"""
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=False,
            groups=groups,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU() if activation in {"silu", "swish"} else nn.Identity()

    def forward(self, hidden_states):
        """执行卷积、BN 和激活。"""
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class LightSVTRAttention(nn.Module):
    """LightSVTR 的多头自注意力，属性名对齐 `self_attn.*` 权重。"""

    def __init__(self, hidden_size, num_heads=8, qkv_bias=True, attention_dropout=0.1):
        """初始化 qkv 投影和输出投影。"""
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        """计算全局自注意力并返回投影后的序列特征。"""
        batch_size, seq_len, embed_dim = hidden_states.shape
        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.reshape(batch_size, seq_len, 3, self.num_heads, embed_dim // self.num_heads)
        mixed_qkv = mixed_qkv.permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim).contiguous()
        return self.projection(attn_output)


class LightSVTRMLP(nn.Module):
    """LightSVTR block 内的前馈网络，属性名对齐 `mlp.fc*` 权重。"""

    def __init__(self, hidden_size, mlp_ratio=4.0, drop_rate=0.1):
        """初始化两层线性层、SiLU 激活和 dropout。"""
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, hidden_states):
        """执行 MLP 前向计算。"""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop(hidden_states)
        return hidden_states


class LightSVTRBlock(nn.Module):
    """LightSVTR 的 Transformer block，属性名对齐 v6 safetensors。"""

    def __init__(
        self,
        hidden_size,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        layer_norm_eps=1e-6,
    ):
        """初始化注意力、MLP 和两层 LayerNorm。"""
        super().__init__()
        self.self_attn = LightSVTRAttention(hidden_size, num_heads, qkv_bias, attn_drop_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = LightSVTRMLP(hidden_size, mlp_ratio, drop_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        """执行 pre-norm attention 和 pre-norm MLP 残差结构。"""
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class EncoderWithLightSVTR(nn.Module):
    """PP-OCRv6 使用的 LightSVTR neck，输出仍保持 4D 特征。"""

    def __init__(
        self,
        in_channels,
        dims=64,
        depth=1,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        qk_scale=None,
        local_kernel=7,
        use_guide=False,
        **kwargs,
    ):
        """初始化 skip/reduce/local conv、LightSVTR block 和归一化层。"""
        super().__init__()
        self.use_guide = use_guide
        self.conv_block = nn.ModuleList(
            [
                LightSVTRConvLayer(in_channels, dims, kernel_size=(1, 1), activation="silu"),
                LightSVTRConvLayer(in_channels, dims, kernel_size=(1, 1), activation="silu"),
                LightSVTRConvLayer(dims, dims, kernel_size=(1, local_kernel), activation="silu", groups=dims),
            ]
        )
        self.svtr_block = nn.ModuleList(
            [
                LightSVTRBlock(
                    hidden_size=dims,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dims, eps=1e-6)
        self.out_channels = dims

    def forward(self, x):
        """执行轻量局部卷积增强、全局注意力和 skip 残差融合。"""
        if self.use_guide:
            x = x.detach()
        residual = self.conv_block[0](x)
        hidden_states = self.conv_block[1](x)
        hidden_states = hidden_states + self.conv_block[2](hidden_states)
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).permute(0, 2, 1)
        for block in self.svtr_block:
            hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return hidden_states + residual


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support_encoder_dict = {
                "reshape": Im2Seq,
                "fc": EncoderWithFC,
                "rnn": EncoderWithRNN,
                "svtr": EncoderWithSVTR,
                "lightsvtr": EncoderWithLightSVTR,
            }
            assert encoder_type in support_encoder_dict, "{} must in {}".format(
                encoder_type, support_encoder_dict.keys()
            )

            if encoder_type in ("svtr", "lightsvtr"):
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs
                )
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size
                )
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type not in ("svtr", "lightsvtr"):
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x
