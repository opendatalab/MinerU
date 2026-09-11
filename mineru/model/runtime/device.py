# Copyright (c) Opendatalab. All rights reserved.
"""模型运行设备与独立的小模型后端选择。"""

from __future__ import annotations

import importlib.util
import os
from typing import Literal


def get_device() -> str:
    """返回显式配置或当前环境中可用的首选模型设备。"""
    configured_device = os.getenv("MINERU_DEVICE_MODE")
    if configured_device is not None:
        return configured_device

    try:
        import torch
    except ImportError:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    try:
        import torch_npu

        if torch_npu.npu.is_available():
            return "npu"
    except Exception:
        pass
    for device_name in ("gcu", "musa", "mlu", "sdaa"):
        try:
            device_api = getattr(torch, device_name)
            if device_api.is_available():
                return device_name
        except Exception:
            pass
    return "cpu"


TORCH_REQUIRED_MODULES: tuple[str, ...] = ("torch", "torchvision", "transformers", "accelerate", "safetensors")


def module_available(name: str) -> bool:
    """只检查模块是否安装，避免自动选择时初始化无关的重依赖。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_small_model_backend(backend: str | None = None) -> Literal["onnx", "torch"]:
    """独立解析小模型后端；显式选择可用于离线下载，依赖由运行入口预检。"""
    from ...config import config

    selected = config.model.small_backend if backend is None else backend
    if selected == "onnx":
        return "onnx"
    if selected == "torch":
        return "torch"
    if selected != "auto":
        raise ValueError(f"Unsupported small backend '{selected}'. Expected one of: auto, onnx, torch.")
    if all(module_available(name) for name in TORCH_REQUIRED_MODULES) and get_device().split(":")[0] != "cpu":
        return "torch"
    return "onnx"


__all__ = ["TORCH_REQUIRED_MODULES", "get_device", "module_available", "resolve_small_model_backend"]
