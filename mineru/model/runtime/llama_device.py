# Copyright (c) Opendatalab. All rights reserved.
"""llama.cpp 实际设备选择、集显 ONNX 决策复用与旧 Intel Vulkan 提交策略。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
import subprocess
import sys
import threading
from typing import Literal

from loguru import logger

from .intel_gpu import INTEL_VENDOR_ID, intel_architecture
from .llama_probe import LlamaDevice, PROBE_MARKER
from .vulkan import vulkan_environment_lock

SERIALIZE_ENV = "GGML_VK_SERIALIZE_SUBMISSIONS"
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
    """可审查的选择计划；environment 只在创建引擎或启动子进程时应用。"""

    device: Literal["native", "cpu", "gpu"] = "native"
    devices: tuple[LlamaDevice, ...] = ()
    reason: str = "native"
    environment: tuple[tuple[str, str], ...] = ()
    device_aliases: tuple[tuple[str, str], ...] = ()
    model_devices: tuple[str, ...] | None = None
    cpu_model: bool = False


def discover_llama_devices(environment: Mapping[str, str] | None = None) -> tuple[LlamaDevice, ...]:
    """隔离原生注册与驱动错误，返回 ggml 过滤后真正可选的设备。"""
    with vulkan_environment_lock():
        snapshot = dict(os.environ if environment is None else environment)
    result = subprocess.run(
        [sys.executable, "-m", "mineru.model.runtime.llama_probe"],
        env=snapshot,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    payload = next(
        (line.removeprefix(PROBE_MARKER) for line in result.stdout.splitlines() if line.startswith(PROBE_MARKER)), None
    )
    if payload is None:
        raise RuntimeError("llama.cpp device probe returned no inventory")
    return tuple(LlamaDevice(**item) for item in json.loads(payload))


def needs_serialized_submissions(device: LlamaDevice) -> bool:
    """仅 Linux Vulkan 的明确 Xe-LP/更早 PCI 架构触发规避；未知型号不猜测。"""
    return (
        sys.platform == "linux"
        and device.backend == "vulkan"
        and device.vendor_id == INTEL_VENDOR_ID
        and device.device_id is not None
        and intel_architecture(device.device_id) in {"legacy", "xe_lp"}
    )


def _preferred(devices: tuple[LlamaDevice, ...], backend: str) -> tuple[LlamaDevice, ...]:
    """与包内 Engine 的独显优先、单后端多设备规则保持一致。"""
    for integrated in (False, True):
        for name in ("metal", "vulkan"):
            if backend not in ("auto", name):
                continue
            group = tuple(d for d in devices if d.integrated == integrated and d.backend == name)
            if group:
                return group
    return ()


def _identity(device: LlamaDevice) -> tuple[str, bool, int | None, int | None]:
    """不使用可能因可见设备过滤而变化的 VulkanN 名称作为物理身份。"""
    return device.description, device.integrated, device.vendor_id, device.device_id


def _bind(selection: LlamaSelection) -> LlamaSelection:
    """隔离选中的物理设备并用原生 ggml 再核验，防止环境开关污染其他 GPU。"""
    selected = selection.devices
    if not selected or any(d.backend != "vulkan" or d.vulkan_index is None for d in selected):
        if any(needs_serialized_submissions(d) for d in selected):
            raise RuntimeError("Cannot safely bind the legacy Intel Vulkan device; use a verified device filter")
        return replace(selection, device_aliases=tuple((d.name, d.name) for d in selected))
    changes = dict(selection.environment)
    changes["GGML_VK_VISIBLE_DEVICES"] = ",".join(str(d.vulkan_index) for d in selected)
    with vulkan_environment_lock():
        environment = {**os.environ, **changes}
    bound = discover_llama_devices(environment)
    actual = tuple(d for d in bound if d.backend == "vulkan")
    if sorted(map(_identity, actual)) != sorted(map(_identity, selected)):
        raise RuntimeError("llama.cpp Vulkan device binding changed the selected hardware")
    aliases = tuple((old.name, next(d.name for d in actual if _identity(d) == _identity(old))) for old in selected)
    return replace(selection, devices=actual, environment=tuple(changes.items()), device_aliases=aliases)


def resolve_llama_device(
    *,
    n_gpu_layers: int | None = None,
    device_names: tuple[str, ...] | None = None,
    backend: str | None = None,
    allow_download: bool = True,
) -> LlamaSelection:
    """自动模式只在没有原生可用独显时复用 ONNX；显式 CPU/GPU 设置优先。"""
    with _lock:
        selection = _resolve(
            n_gpu_layers=n_gpu_layers, device_names=device_names, backend=backend, allow_download=allow_download
        )
        target = [d.description for d in selection.devices] if selection.devices else selection.device
        logger.info("llama.cpp device policy: {} ({})", target, selection.reason)
        return selection


def _resolve(
    *, n_gpu_layers: int | None, device_names: tuple[str, ...] | None, backend: str | None, allow_download: bool
) -> LlamaSelection:
    """在原生环境互斥保护下构造设备计划，校准仍走既有 ONNX 生命周期。"""
    requested = backend if backend is not None else os.environ.get("MINERU_LLAMA_CPP_BACKEND", "auto")
    if sys.platform not in {"linux", "win32"}:
        return LlamaSelection(reason="platform_native")
    if requested not in {"auto", "cpu", "vulkan", "metal"}:
        raise ValueError("MINERU_LLAMA_CPP_BACKEND must be auto, cpu, vulkan or metal")
    if n_gpu_layers == 0 or requested == "cpu" or device_names == ("none",):
        return LlamaSelection("cpu", reason="explicit_cpu")
    try:
        inventory = discover_llama_devices()
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, TypeError) as error:
        logger.warning("llama.cpp device discovery unavailable; preserving native selection: {}", error)
        return LlamaSelection(reason="discovery_unavailable")
    if device_names is not None:
        selected = tuple(next((d for d in inventory if d.name == name), None) for name in device_names)
        if not selected or any(d is None for d in selected):
            raise ValueError(f"Unknown llama.cpp device selection: {device_names}")
        selected = tuple(d for d in selected if d is not None)
    else:
        selected = _preferred(inventory, requested)
    if not selected:
        if requested != "auto" or device_names is not None:
            raise RuntimeError(f"Requested llama.cpp {requested} GPU is unavailable")
        return LlamaSelection("cpu", reason="no_gpu")
    manual = requested != "auto" or n_gpu_layers is not None or device_names is not None
    reason = "explicit_gpu" if manual else "discrete_gpu"
    if not manual and not any(not d.integrated for d in inventory):
        from .onnx_device import resolve_onnx_device

        choice = resolve_onnx_device() if allow_download else resolve_onnx_device(allow_download=False)
        if choice.device != "gpu" or choice.adapter is None:
            return LlamaSelection("cpu", reason=f"onnx_{choice.reason}")
        matches = tuple(
            d for d in selected if (d.vendor_id, d.device_id) == (choice.adapter.vendor_id, choice.adapter.device_id)
        )
        if len(matches) != 1:
            return LlamaSelection("cpu", reason="onnx_gpu_identity_unconfirmed")
        selected, reason = matches, f"onnx_{choice.backend}_{choice.reason}"
    old = tuple(d for d in selected if needs_serialized_submissions(d))
    if old and len(old) != len(selected):
        if manual:
            raise ValueError("Legacy Intel and other GPUs need different Vulkan submission policies; select one device group")
        # The global ggml switch cannot be restricted per device. Prefer the other
        # discrete GPUs over mixing Xe-LP DG1 with AMD/NVIDIA/modern Intel.
        selected = tuple(d for d in selected if not needs_serialized_submissions(d))
        old = ()
    changes = {SERIALIZE_ENV: "1"} if old and SERIALIZE_ENV not in os.environ else {}
    return _bind(LlamaSelection("gpu", selected, reason, tuple(changes.items())))


@contextmanager
def llama_engine_scope(selection: LlamaSelection) -> Iterator[None]:
    """临时应用进程环境并还原；已缓存的原生后端不得被换卡或修改提交策略。"""
    global _runtime_environment
    if selection.device == "native":
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


__all__ = [
    "LlamaSelection",
    "discover_llama_devices",
    "llama_engine_scope",
    "needs_serialized_submissions",
    "resolve_llama_device",
]
