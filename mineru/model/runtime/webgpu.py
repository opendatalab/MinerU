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
class WebGPUDevice:
    """Serializable EP registration identity, NOT a Dawn physical-adapter selector."""

    vendor_id: int
    device_id: int
    metadata: tuple[tuple[str, str], ...] = ()
    # Only needed when old Intel and eligible GPUs coexist. No native handles.
    vulkan_device: tuple[int, int] | None = None


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
    """Apply the Linux Intel policy without treating an EP handle as a GPU binding."""
    if not devices:
        return None
    if sys.platform != "linux":
        return devices[0]
    allowed = tuple(d for d in devices if supports_webgpu_architecture(d.vendor_id, d.device_id))
    if not allowed:
        return None
    if len(allowed) == len(devices):
        return devices[0]  # Preserve ordinary AMD/NVIDIA/modern-Intel selection.
    from .vulkan import enumerate_vulkan_devices

    # ORT enumerates OS devices, even those hidden by VK_DRIVER_FILES. Query
    # the loader before deciding which physical GPUs can actually be used.
    with execution_lock():
        visible = enumerate_vulkan_devices()
    rank = {2: 0, 1: 1, 3: 2}  # Dawn high-performance: discrete before integrated.
    for physical in sorted(visible, key=lambda d: rank[d.device_type]):
        descriptor = next((d for d in allowed if (d.vendor_id, d.device_id) == (physical.vendor_id, physical.device_id)), None)
        if descriptor is not None:
            return replace(descriptor, vulkan_device=(physical.vendor_id, physical.device_id))
    return None


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
        identity = replace(device, vulkan_device=None)
        ep = next((ep for ep in _devices() if _describe(ep) == identity), None)
        if ep is None:
            raise RuntimeError("WebGPU hardware registration device is no longer available")
        # The caller's webgpu_session_scope restricts mixed-architecture systems.
        # This registration handle itself still does not bind a physical adapter.
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
