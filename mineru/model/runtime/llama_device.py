# Copyright (c) Opendatalab. All rights reserved.
"""复用小模型后端的 CPU/GPU 结论，具体显卡交由 llama.cpp 原生运行时选择。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
import threading
from typing import Literal

from loguru import logger

from .intel_gpu import INTEL_VENDOR_ID, intel_architecture
from .vulkan import vulkan_environment_lock

SERIALIZE_ENV = "GGML_VK_SERIALIZE_SUBMISSIONS"
DEFAULT_GPU_THREADS = 4
_lock = threading.RLock()
_runtime_environment: tuple[int, tuple[tuple[str, str | None], ...]] | None = None
_BACKEND_ENV = (
    "GGML_VK_VISIBLE_DEVICES",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_LOADER_VENDOR_ID_FILTER",
    "VK_LOADER_DEVICE_ID_FILTER",
    "MESA_VK_DEVICE_SELECT",
    SERIALIZE_ENV,
)


@dataclass(frozen=True)
class LlamaSelection:
    """只决定运行模式和兼容环境，不携带显卡库存或原生设备名称。"""

    device: Literal["native", "cpu", "gpu"] = "native"
    reason: str = "native"
    environment: tuple[tuple[str, str], ...] = ()


def default_llama_threads(selection: LlamaSelection) -> int | None:
    """GPU 减少 CPU 线程开销；CPU 和未决原生路由保留 llama.cpp 的自动计算。"""
    if selection.device == "gpu" or selection.reason == "explicit_gpu":
        return DEFAULT_GPU_THREADS
    return None


def resolve_llama_device(
    *,
    n_gpu_layers: int | None = None,
    device_names: tuple[str, ...] | None = None,
    allow_download: bool = True,
) -> LlamaSelection:
    """显式参数优先；自动模式复用 ONNX 选型或 Torch 可用性，不探测 ggml。"""
    with _lock:
        selection = _resolve(n_gpu_layers=n_gpu_layers, device_names=device_names, allow_download=allow_download)
        logger.info("llama.cpp device policy: {} ({})", selection.device, selection.reason)
        return selection


def _resolve(*, n_gpu_layers: int | None, device_names: tuple[str, ...] | None, allow_download: bool) -> LlamaSelection:
    requested = os.environ.get("MINERU_LLAMA_CPP_BACKEND", "auto")
    if sys.platform not in {"linux", "win32"}:
        return LlamaSelection(reason="platform_native")
    if requested not in {"auto", "cpu", "vulkan", "metal"}:
        raise ValueError("MINERU_LLAMA_CPP_BACKEND must be auto, cpu, vulkan or metal")
    if n_gpu_layers == 0 or requested == "cpu" or device_names == ("none",):
        return LlamaSelection("cpu", reason="explicit_cpu")
    if requested != "auto" or n_gpu_layers is not None or device_names is not None:
        return LlamaSelection(reason="explicit_gpu")

    from .device import resolve_small_model_backend, resolve_torch_device

    if resolve_small_model_backend() == "torch":
        choice = resolve_torch_device()
        if choice.warning:
            logger.warning("llama.cpp follows Torch device decision: {}", choice.warning)
        return LlamaSelection("gpu" if choice.accelerated else "cpu", reason=choice.reason)

    from .onnx_device import resolve_onnx_device

    choice = resolve_onnx_device() if allow_download else resolve_onnx_device(allow_download=False)
    adapter = choice.adapter
    environment = ()
    if (
        choice.device == "gpu"
        and sys.platform == "linux"
        and choice.backend == "webgpu"
        and adapter is not None
        and adapter.vendor_id == INTEL_VENDOR_ID
        and intel_architecture(adapter.device_id) in {"legacy", "xe_lp", "xe_lpg"}
        and SERIALIZE_ENV not in os.environ
    ):
        environment = ((SERIALIZE_ENV, "1"),)
    return LlamaSelection(choice.device, reason=f"onnx_{choice.reason}", environment=environment)


@contextmanager
def llama_engine_scope(selection: LlamaSelection) -> Iterator[None]:
    """串行初始化并恢复临时设置；手动 GPU 也不能读到并发 CPU 构造的环境。"""
    global _runtime_environment
    if selection.device == "native" and sys.platform not in {"linux", "win32"}:
        yield
        return
    with _lock, vulkan_environment_lock():
        changes = dict(selection.environment)
        if selection.device == "cpu":
            changes["MINERU_LLAMA_CPP_BACKEND"] = "cpu"
        previous = {name: os.environ.get(name) for name in changes}
        try:
            os.environ.update(changes)
            signature = (os.getpid(), tuple((name, os.environ.get(name)) for name in _BACKEND_ENV))
            if (
                _runtime_environment is not None
                and _runtime_environment[0] == os.getpid()
                and _runtime_environment != signature
            ):
                raise RuntimeError("llama.cpp GPU environment changed after initialization; use a fresh process")
            _runtime_environment = signature
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


__all__ = ["DEFAULT_GPU_THREADS", "LlamaSelection", "default_llama_threads", "llama_engine_scope", "resolve_llama_device"]
