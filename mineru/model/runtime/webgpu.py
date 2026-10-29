# Copyright (c) Opendatalab. All rights reserved.
"""Lazy native WebGPU registration and process-wide session serialization."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
import sys
import threading
from typing import TYPE_CHECKING

from .intel_gpu import supports_webgpu_architecture
from .gpu import (
    GpuClassification,
    GpuIdentity,
    gpu_identities_match,
    gpu_kind_rank,
    is_discrete_gpu,
    match_gpu_identities,
    with_gpu_classification,
)

if TYPE_CHECKING:
    import onnxruntime as ort

WEBGPU_PROVIDER = "WebGpuExecutionProvider"
WEBGPU_OPTIONS: dict[str, str] = {
    "preferredLayout": "NHWC",
    "enableGraphCapture": "0",
    "dawnBackendType": "Vulkan",
    "powerPreference": "high-performance",
}


@dataclass(frozen=True)
class WebGPUDevice(GpuClassification):
    """Serializable EP registration identity, NOT a Dawn physical-adapter selector."""

    vendor_id: int
    device_id: int
    metadata: tuple[tuple[str, str], ...] = ()
    # 物理设备约束用于多卡选型及独显免校准；只传递身份，不保存原生句柄。
    vulkan_device: tuple[int, int] | None = None

    @property
    def identity(self) -> GpuIdentity:
        """复用 EP 已提供的身份元数据，不额外查询 UUID 或 PCI 地址。"""
        return GpuIdentity(self.vendor_id, self.device_id, bus_id=dict(self.metadata).get("pci_bus_id"))


_lock = threading.RLock()
_registered_pid: int | None = None
_runtime_pid: int | None = None
_runtime_device: tuple[int, int] | None = None


def execution_lock() -> threading.RLock:
    """All WebGPU sessions share Dawn state; never run two sessions concurrently."""
    return _lock


def _devices() -> list[ort.OrtEpDevice]:
    """Register once in this process; keep native EP handles local to this call."""
    import onnxruntime as ort
    import onnxruntime_ep_webgpu as plugin

    global _registered_pid
    if _registered_pid != os.getpid():
        ort.register_execution_provider_library(WEBGPU_PROVIDER, plugin.get_library_path())
        _registered_pid = os.getpid()
    return [
        ep
        for ep in ort.get_ep_devices()
        if ep.ep_name == WEBGPU_PROVIDER and ep.device.type == ort.OrtHardwareDeviceType.GPU and ep.device.vendor_id != 0
    ]


def _describe(ep: ort.OrtEpDevice) -> WebGPUDevice:
    device = ep.device
    return WebGPUDevice(device.vendor_id, device.device_id, tuple(sorted(device.metadata.items())))


def discover_webgpu_devices() -> tuple[WebGPUDevice, ...]:
    """Return hashable hardware identities without starting a model or writing config."""
    with execution_lock():
        return tuple(_describe(ep) for ep in _devices())


def select_webgpu_device(devices: tuple[WebGPUDevice, ...]) -> WebGPUDevice | None:
    """按 Vulkan 可见类型优先选择独显，并为免校准和多卡会话绑定实际设备。"""
    if not devices:
        return None
    if sys.platform != "linux":
        return devices[0]
    allowed = tuple(d for d in devices if supports_webgpu_architecture(d.vendor_id, d.device_id))
    if not allowed:
        return None
    from .vulkan import enumerate_vulkan_devices, vulkan_environment_lock

    # ORT 库存可能包含 ICD 已隐藏的设备；分类必须来自 Vulkan 实际可见的物理设备。
    try:
        with execution_lock(), vulkan_environment_lock():
            visible = enumerate_vulkan_devices()
    except (OSError, RuntimeError):
        if len(allowed) != len(devices):
            raise  # 有受限架构时无法确认物理绑定，不允许继续默认选卡。
        return with_gpu_classification(devices[0], GpuClassification())
    matches = match_gpu_identities([d.identity for d in allowed], [d.identity for d in visible])
    candidates: list[tuple[int, WebGPUDevice]] = []
    for descriptor, matched in zip(allowed, matches):
        physical_indices = [
            i for i, physical in enumerate(visible) if gpu_identities_match(descriptor.identity, physical.identity)
        ]
        if not physical_indices:
            continue
        # 物理型号存在但无法双向唯一关联时，仍可校准，不允许继承某张卡的独显分类。
        candidate = with_gpu_classification(descriptor, visible[matched] if matched is not None else GpuClassification())
        bind = is_discrete_gpu(candidate) or len(visible) > 1 or len(allowed) != len(devices)
        identity = (candidate.vendor_id, candidate.device_id)
        candidates.append((physical_indices[0], replace(candidate, vulkan_device=identity if bind else None)))
    candidates.sort(key=lambda item: (gpu_kind_rank(item[1].kind), item[0]))
    return candidates[0][1] if candidates else None


@contextmanager
def webgpu_session_scope(device: WebGPUDevice) -> Iterator[None]:
    """Serialize initialization and enforce the same GPU restriction in every process."""
    global _runtime_pid, _runtime_device

    if sys.platform == "linux" and not supports_webgpu_architecture(device.vendor_id, device.device_id):
        raise RuntimeError("Linux WebGPU requires Intel Xe-LPG/Xe-HPG or a recognized later architecture")
    if device.vulkan_device is not None and device.vulkan_device != (device.vendor_id, device.device_id):
        raise ValueError("WebGPU registration identity and Vulkan restriction differ")
    with execution_lock():
        if _runtime_pid == os.getpid():
            if _runtime_device != device.vulkan_device:
                raise RuntimeError("WebGPU physical adapter cannot change after initialization; use a new process")
        if device.vulkan_device is None:
            _runtime_pid, _runtime_device = os.getpid(), None
            yield
            return
        from .vulkan import restrict_vulkan_device

        # Reapply on every creation: the EP can release its context after the
        # last session is destroyed, even though our process identity is cached.
        with restrict_vulkan_device(*device.vulkan_device):
            # Keep this identity even after a failed session creation: native
            # context initialization may have partially completed already.
            _runtime_pid, _runtime_device = os.getpid(), device.vulkan_device
            yield


def attach_webgpu_provider(options: ort.SessionOptions, device: WebGPUDevice) -> None:
    """Resolve a fresh native handle and attach the production provider parameters."""
    with execution_lock():
        identity = with_gpu_classification(replace(device, vulkan_device=None), GpuClassification())
        ep = next((ep for ep in _devices() if _describe(ep) == identity), None)
        if ep is None:
            raise RuntimeError("WebGPU hardware registration device is no longer available")
        # 物理约束由调用方 webgpu_session_scope 验证；EP 注册句柄本身不绑定显卡。
        options.add_provider_for_devices([ep], WEBGPU_OPTIONS)


__all__ = [
    "WEBGPU_OPTIONS",
    "WEBGPU_PROVIDER",
    "WebGPUDevice",
    "attach_webgpu_provider",
    "discover_webgpu_devices",
    "execution_lock",
    "select_webgpu_device",
    "webgpu_session_scope",
]
