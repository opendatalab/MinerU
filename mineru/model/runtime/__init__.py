# Copyright (c) Opendatalab. All rights reserved.
"""模型设备、资源与本地推理上下文的私有运行时。"""

from .contracts import AtomicModelName
from .device import get_device, get_model_stack
from .hybrid import HybridLocalModelContext, HybridLocalModelContextSingleton
from .memory import clean_memory, get_vram

__all__ = [
    "AtomicModelName",
    "HybridLocalModelContext",
    "HybridLocalModelContextSingleton",
    "clean_memory",
    "get_device",
    "get_model_stack",
    "get_vram",
]
