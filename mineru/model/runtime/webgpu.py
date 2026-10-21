"""Lazy native WebGPU registration and process-wide session serialization."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from typing import TYPE_CHECKING

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


_lock = threading.RLock()
_registered_pid: int | None = None


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


def attach_webgpu_provider(options: ort.SessionOptions, device: WebGPUDevice) -> None:
    """Resolve a fresh native handle and attach the production provider parameters."""
    with execution_lock():
        ep = next((ep for ep in _devices() if _describe(ep) == device), None)
        if ep is None:
            raise RuntimeError("WebGPU hardware registration device is no longer available")
        # The plugin selects the physical adapter via powerPreference, not this EP handle.
        options.add_provider_for_devices([ep], WEBGPU_OPTIONS)


__all__ = [
    "WEBGPU_OPTIONS",
    "WEBGPU_PROVIDER",
    "WebGPUDevice",
    "attach_webgpu_provider",
    "discover_webgpu_devices",
    "execution_lock",
]
