# Copyright (c) Opendatalab. All rights reserved.
import os
from pathlib import Path

import torch

from .modeling.architectures.base_model import BaseModel

class BaseOCRV20:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()


    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)

    @staticmethod
    def _is_safetensors_path(weights_path):
        """判断权重文件是否为 safetensors 格式。"""
        return Path(weights_path).suffix == ".safetensors"

    @staticmethod
    def _load_weight_file(weights_path):
        """根据文件后缀选择 safetensors 或 torch 原生加载方式。"""
        if BaseOCRV20._is_safetensors_path(weights_path):
            from safetensors.torch import load_file

            return load_file(str(weights_path), device="cpu")
        try:
            return torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(weights_path, map_location="cpu")

    @staticmethod
    def _normalize_ppocrv6_state_dict(weights, weights_path):
        """归一化 HF OCR safetensors 的外层 `model.` 前缀。"""
        if not BaseOCRV20._is_safetensors_path(weights_path):
            return weights
        if not any(key.startswith("model.") for key in weights.keys()):
            return weights
        return {
            key.removeprefix("model."): value
            for key, value in weights.items()
        }

    def read_pytorch_weights(self, weights_path):
        """读取 PyTorch OCR 权重，并兼容 PP-OCRv6 safetensors。"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = self._load_weight_file(weights_path)
        return self._normalize_ppocrv6_state_dict(weights, weights_path)

    def get_out_channels(self, weights):
        """从权重结构推断识别输出通道数。"""
        if "head.head.weight" in weights:
            # PP-OCRv6 safetensors 的识别分类层固定命名为 head.head。
            return weights["head.head.weight"].shape[0]
        if list(weights.keys())[-1].endswith('.weight') and len(list(weights.values())[-1].shape) == 2:
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        # print('weights is loaded.')

    def load_pytorch_weights(self, weights_path):
        """加载 PyTorch OCR 权重，按后缀兼容 safetensors。"""
        self.net.load_state_dict(self.read_pytorch_weights(weights_path))
        # print('model is loaded: {}'.format(weights_path))

    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer
