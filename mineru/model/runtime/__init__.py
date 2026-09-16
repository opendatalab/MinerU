# Copyright (c) Opendatalab. All rights reserved.
"""模型设备、资源与本地推理上下文的私有运行时。"""

from .contracts import AtomicModelName
from .device import get_device, resolve_small_model_backend
from .memory import clean_memory, get_vram

__all__ = [
    "AtomicModelName",
    "clean_memory",
    "get_device",
    "resolve_small_model_backend",
    "get_vram",
]
