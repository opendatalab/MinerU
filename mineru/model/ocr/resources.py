# Copyright (c) Opendatalab. All rights reserved.
"""Torch 与 ONNX 共用的 OCR 字符资源；导入不加载推理框架。"""

from pathlib import Path

PPOCRV6_DICT_PATH = Path(__file__).with_name("data") / "ppocrv6_dict.txt"

__all__ = ["PPOCRV6_DICT_PATH"]
