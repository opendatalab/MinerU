# Copyright (c) Opendatalab. All rights reserved.
"""模型推理显存探测与缓存释放。"""

from __future__ import annotations

import gc
import os
from typing import Any

from loguru import logger


def _optional_torch() -> tuple[Any | None, Any | None]:
    """惰性加载 torch 与可选 NPU 扩展，避免公共门面触发重依赖。"""
    try:
        import torch
    except ImportError:
        return None, None
    try:
        import torch_npu
    except ImportError:
        torch_npu = None
    return torch, torch_npu


def clean_memory(device: str = "cuda") -> None:
    """释放指定模型设备的框架缓存并执行 Python 垃圾回收。"""
    torch, torch_npu = _optional_torch()
    if torch is None:
        gc.collect()
        return

    device_name = str(device)
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_name.startswith("npu") and torch_npu is not None and torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()
    elif device_name.startswith("mps"):
        torch.mps.empty_cache()
    else:
        for accelerator_name in ("gcu", "musa", "mlu", "sdaa"):
            if not device_name.startswith(accelerator_name):
                continue
            accelerator = getattr(torch, accelerator_name, None)
            if accelerator is not None and accelerator.is_available():
                accelerator.empty_cache()
            break
    gc.collect()


def get_vram(device: str) -> int:
    """返回指定设备的整数 GB 显存，无法探测时保守返回一。"""
    configured_vram = os.getenv("MINERU_VIRTUAL_VRAM_SIZE")
    if configured_vram is not None:
        try:
            total_memory = int(configured_vram)
            if total_memory > 0:
                return total_memory
            logger.warning(
                f"MINERU_VIRTUAL_VRAM_SIZE value '{configured_vram}' is not positive, falling back to auto-detection"
            )
        except ValueError:
            logger.warning(
                f"MINERU_VIRTUAL_VRAM_SIZE value '{configured_vram}' is not a valid integer, falling back to auto-detection"
            )

    torch, torch_npu = _optional_torch()
    if torch is None:
        return 1

    device_name = str(device)
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.get_device_properties(device).total_memory / (1024**3))
    if device_name.startswith("npu") and torch_npu is not None and torch_npu.npu.is_available():
        return round(torch_npu.npu.get_device_properties(device).total_memory / (1024**3))
    for accelerator_name in ("gcu", "musa", "mlu", "sdaa"):
        if not device_name.startswith(accelerator_name):
            continue
        accelerator = getattr(torch, accelerator_name, None)
        if accelerator is not None and accelerator.is_available():
            return round(accelerator.get_device_properties(device).total_memory / (1024**3))
    return 1


__all__ = ["clean_memory", "get_vram"]
