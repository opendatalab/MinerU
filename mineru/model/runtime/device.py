# Copyright (c) Opendatalab. All rights reserved.
"""模型运行设备与独立的小模型后端选择。"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
import os
from typing import Literal


# 通过 getattr(torch, name) 探测可用性的附加加速后端，按优先级排列；xpu 为 Intel GPU 原生后端。
PROBE_TORCH_DEVICE_APIS: tuple[str, ...] = ("xpu", "gcu", "musa", "mlu", "sdaa")


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
    for device_name in PROBE_TORCH_DEVICE_APIS:
        try:
            device_api = getattr(torch, device_name)
            if device_api.is_available():
                return device_name
        except Exception:
            pass
    return "cpu"


TORCH_REQUIRED_MODULES: tuple[str, ...] = ("torch", "torchvision", "transformers", "accelerate", "safetensors")


@dataclass(frozen=True)
class TorchDeviceSelection:
    """记录经过可用性检查的 Torch 设备，不包含模型性能结论。"""

    device: str = "cpu"
    reason: str = "torch_cpu"
    warning: str | None = None

    @property
    def accelerated(self) -> bool:
        """只有验证成功的非 CPU 设备才允许按加速设备推荐。"""
        return self.device.split(":", 1)[0] != "cpu"


def resolve_torch_device() -> TorchDeviceSelection:
    """核验显式或自动 Torch 设备，不分配模型或执行校准，也不改写配置。"""
    try:
        selected = get_device()
        kind, _, index_text = selected.partition(":")
        if kind == "cpu":
            return TorchDeviceSelection(reason="configured_cpu" if os.getenv("MINERU_DEVICE_MODE") else "torch_cpu")
        import torch

        if kind == "mps":
            available = torch.backends.mps.is_available() and (not index_text or int(index_text) == 0)
        else:
            if kind == "npu":
                import torch_npu

                api = torch_npu.npu
            elif kind in {"cuda", *PROBE_TORCH_DEVICE_APIS}:
                api = getattr(torch, kind, None)
            else:
                api = None
            available = api is not None and api.is_available()
            if available:
                index = int(index_text) if index_text else 0
                available = index >= 0 and index < api.device_count()
        if available:
            return TorchDeviceSelection(selected, "torch_accelerator")
        return TorchDeviceSelection(reason="torch_device_unavailable", warning=f"Torch device {selected!r} is unavailable")
    except Exception as error:
        return TorchDeviceSelection(reason="torch_device_error", warning=f"Torch device check failed: {error}")


def module_available(name: str) -> bool:
    """只检查模块是否安装，避免自动选择时初始化无关的重依赖。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_small_model_backend(backend: str | None = None) -> Literal["onnx", "torch"]:
    """独立解析小模型后端；显式选择可用于离线下载，依赖由运行入口预检。"""
    from ...config import get_config

    selected = get_config().model.small_backend if backend is None else backend
    if selected == "onnx":
        return "onnx"
    if selected == "torch":
        return "torch"
    if selected != "auto":
        raise ValueError(f"Unsupported small backend '{selected}'. Expected one of: auto, onnx, torch.")
    if all(module_available(name) for name in TORCH_REQUIRED_MODULES) and get_device().split(":")[0] != "cpu":
        return "torch"
    return "onnx"


__all__ = [
    "PROBE_TORCH_DEVICE_APIS",
    "TORCH_REQUIRED_MODULES",
    "TorchDeviceSelection",
    "get_device",
    "module_available",
    "resolve_small_model_backend",
    "resolve_torch_device",
]
