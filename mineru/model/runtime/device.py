# Copyright (c) Opendatalab. All rights reserved.
"""模型运行设备与轻量/完整模型栈选择。"""

from __future__ import annotations

import os
from typing import Literal, cast


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


def get_model_stack() -> Literal["light", "full"]:
    """按配置和实际设备解析轻量或完整模型推理栈。"""
    from ...config import config

    configured_stack = config.model.stack
    if configured_stack in ("light", "full"):
        return cast(Literal["light", "full"], configured_stack)
    return "light" if get_device() == "cpu" else "full"


__all__ = ["get_device", "get_model_stack"]
