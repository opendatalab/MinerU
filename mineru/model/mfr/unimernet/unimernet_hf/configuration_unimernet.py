# Copyright (c) Opendatalab. All rights reserved.
"""显式解析 UniMERNet 子配置，避免修改 Transformers 的全局 Auto 注册表。"""

from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig, VisionEncoderDecoderConfig

from .unimer_mbart.configuration_unimer_mbart import UnimerMBartConfig
from .unimer_swin.configuration_unimer_swin import UnimerSwinConfig


class UnimernetConfig(VisionEncoderDecoderConfig):
    """保留现有组合配置的 JSON 标识，并固定编码器和解码器的实现归属。"""

    model_type = "vision-encoder-decoder"
    has_no_defaults_at_init = True
    sub_configs = {"encoder": UnimerSwinConfig, "decoder": UnimerMBartConfig}

    def __post_init__(self, **kwargs: Any) -> None:
        """复制并构造两个子配置，保留原始模型文件及调用方字典。"""
        encoder = kwargs.pop("encoder", None)
        decoder = kwargs.pop("decoder", None)
        if encoder is None or decoder is None:
            raise ValueError("UniMERNet requires both encoder and decoder configurations")
        self.encoder = encoder if isinstance(encoder, UnimerSwinConfig) else UnimerSwinConfig(**deepcopy(encoder))
        self.decoder = decoder if isinstance(decoder, UnimerMBartConfig) else UnimerMBartConfig(**deepcopy(decoder))
        PretrainedConfig.__post_init__(self, **kwargs)


__all__ = ["UnimernetConfig"]
