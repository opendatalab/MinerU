# Copyright (c) Opendatalab. All rights reserved.
"""No-compute Vulkan enumeration and scoped physical-device restrictions for Dawn.

Uses Vulkan 1.0 properties and Vulkan 1.1 identity/budget queries.
No logical device, shader or GPU work is created.
Callers must serialize scopes with the WebGPU runtime lock.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import ctypes as ct
from dataclasses import dataclass
import os
import sys
import threading
from typing import Any

from .gpu import GpuClassification, GpuIdentity, classify_vulkan_gpu, with_gpu_classification

_environment_lock = threading.RLock()


def vulkan_environment_lock() -> threading.RLock:
    """跨 ONNX/llama 原生初始化串行化进程级 Vulkan 过滤环境。"""
    return _environment_lock


@dataclass(frozen=True)
class VulkanDevice(GpuClassification):
    vendor_id: int
    device_id: int
    name: str
    index: int = -1
    dedicated_memory_bytes: int | None = None
    memory_budget_bytes: int | None = None
    memory_error: str | None = None
    luid: str | None = None

    @property
    def identity(self) -> GpuIdentity:
        """携带 Windows 原生 LUID，避免同型号 DXGI 适配器记录造成身份歧义。"""
        return GpuIdentity(self.vendor_id, self.device_id, luid=self.luid, name=self.name)


class _ApplicationInfo(ct.Structure):
    _fields_ = [
        ("sType", ct.c_uint32),
        ("pNext", ct.c_void_p),
        ("pApplicationName", ct.c_char_p),
        ("applicationVersion", ct.c_uint32),
        ("pEngineName", ct.c_char_p),
        ("engineVersion", ct.c_uint32),
        ("apiVersion", ct.c_uint32),
    ]


class _InstanceCreateInfo(ct.Structure):
    _fields_ = [
        ("sType", ct.c_uint32),
        ("pNext", ct.c_void_p),
        ("flags", ct.c_uint32),
        ("pApplicationInfo", ct.POINTER(_ApplicationInfo)),
        ("enabledLayerCount", ct.c_uint32),
        ("ppEnabledLayerNames", ct.c_void_p),
        ("enabledExtensionCount", ct.c_uint32),
        ("ppEnabledExtensionNames", ct.c_void_p),
    ]


class _PropertiesPrefix(ct.Structure):
    _fields_ = [
        ("apiVersion", ct.c_uint32),
        ("driverVersion", ct.c_uint32),
        ("vendorID", ct.c_uint32),
        ("deviceID", ct.c_uint32),
        ("deviceType", ct.c_uint32),
        ("deviceName", ct.c_char * 256),
    ]


class _PropertiesStorage(ct.Union):
    # VkPhysicalDeviceProperties is an unextended Vulkan 1.0 structure (824 bytes
    # on Linux x86_64). Reserve aligned storage for its unused limits/sparse fields.
    # Unlike VkPhysicalDeviceProperties2, it cannot acquire fields via extensions.
    _fields_ = [("prefix", _PropertiesPrefix), ("storage", ct.c_uint64 * 128)]


class _Properties2(ct.Structure):
    _fields_ = [("sType", ct.c_uint32), ("pNext", ct.c_void_p), ("properties", _PropertiesStorage)]


class _DeviceIDProperties(ct.Structure):
    _fields_ = [
        ("sType", ct.c_uint32),
        ("pNext", ct.c_void_p),
        ("deviceUUID", ct.c_uint8 * 16),
        ("driverUUID", ct.c_uint8 * 16),
        ("deviceLUID", ct.c_uint8 * 8),
        ("deviceNodeMask", ct.c_uint32),
        ("deviceLUIDValid", ct.c_uint32),
    ]


def _device_luid(library: Any, handle: ct.c_void_p) -> str | None:
    """只接受 Vulkan 声明有效的 Windows LUID，并按 DXGI 的 high:low 格式编码。"""
    identity = _DeviceIDProperties(sType=1000071004)
    info = _Properties2(sType=1000059001, pNext=ct.cast(ct.pointer(identity), ct.c_void_p))
    properties = library.vkGetPhysicalDeviceProperties2
    properties.argtypes, properties.restype = [ct.c_void_p, ct.POINTER(_Properties2)], None
    properties(handle, ct.byref(info))
    if not identity.deviceLUIDValid:
        return None
    value = bytes(identity.deviceLUID)
    return f"{int.from_bytes(value[4:], 'little'):08x}:{int.from_bytes(value[:4], 'little'):08x}"


class _MemoryType(ct.Structure):
    _fields_ = [("flags", ct.c_uint32), ("heap_index", ct.c_uint32)]


class _MemoryHeap(ct.Structure):
    _fields_ = [("size", ct.c_uint64), ("flags", ct.c_uint32)]


class _MemoryProperties(ct.Structure):
    _fields_ = [
        ("type_count", ct.c_uint32),
        ("types", _MemoryType * 32),
        ("heap_count", ct.c_uint32),
        ("heaps", _MemoryHeap * 16),
    ]


class _MemoryProperties2(ct.Structure):
    _fields_ = [("sType", ct.c_uint32), ("pNext", ct.c_void_p), ("memory", _MemoryProperties)]


class _MemoryBudget(ct.Structure):
    _fields_ = [
        ("sType", ct.c_uint32),
        ("pNext", ct.c_void_p),
        ("budget", ct.c_uint64 * 16),
        ("usage", ct.c_uint64 * 16),
    ]


class _ExtensionProperties(ct.Structure):
    _fields_ = [("name", ct.c_char * 256), ("version", ct.c_uint32)]


def _device_memory(library: Any, handle: ct.c_void_p, *, discrete: bool) -> tuple[int | None, int | None]:
    """按唯一设备的本地堆读取容量和预算；共享内存无预算扩展时保持未知。"""
    properties = library.vkGetPhysicalDeviceMemoryProperties
    properties.argtypes, properties.restype = [ct.c_void_p, ct.POINTER(_MemoryProperties)], None
    memory = _MemoryProperties()
    properties(handle, ct.byref(memory))
    local = [index for index in range(min(memory.heap_count, 16)) if memory.heaps[index].flags & 1]
    total = sum(memory.heaps[index].size for index in local) or None
    if discrete:
        return total, None
    extensions = library.vkEnumerateDeviceExtensionProperties
    extensions.argtypes = [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_uint32), ct.POINTER(_ExtensionProperties)]
    extensions.restype = ct.c_int32
    count = ct.c_uint32()
    _check(extensions(handle, None, ct.byref(count), None), "count memory extensions")
    rows = (_ExtensionProperties * count.value)()
    _check(extensions(handle, None, ct.byref(count), rows), "read memory extensions")
    if not any(row.name == b"VK_EXT_memory_budget" for row in rows):
        return None, None
    budget = _MemoryBudget(sType=1000237000)
    info = _MemoryProperties2(sType=1000059006, pNext=ct.cast(ct.pointer(budget), ct.c_void_p))
    properties2 = library.vkGetPhysicalDeviceMemoryProperties2
    properties2.argtypes, properties2.restype = [ct.c_void_p, ct.POINTER(_MemoryProperties2)], None
    properties2(handle, ct.byref(info))
    # 预算为进程可用上限，不减去当前用量；每个物理堆只计一次。
    local = [index for index in range(min(info.memory.heap_count, 16)) if info.memory.heaps[index].flags & 1]
    if not local or any(budget.budget[index] == 0 for index in local):
        return None, None
    return None, sum(min(budget.budget[index], info.memory.heaps[index].size) for index in local)


def _check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"Vulkan {operation} failed ({result})")


def enumerate_vulkan_devices(*, include_memory: bool = False) -> tuple[VulkanDevice, ...]:
    """遵守 ICD 和设备过滤枚举 GPU；仅推荐容量时额外读取本地堆或共享预算。"""
    library = ct.WinDLL("vulkan-1.dll") if sys.platform == "win32" else ct.CDLL("libvulkan.so.1")
    create = library.vkCreateInstance
    create.argtypes = [ct.POINTER(_InstanceCreateInfo), ct.c_void_p, ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_int32
    destroy = library.vkDestroyInstance
    destroy.argtypes = [ct.c_void_p, ct.c_void_p]
    destroy.restype = None
    enumerate_devices = library.vkEnumeratePhysicalDevices
    enumerate_devices.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_void_p)]
    enumerate_devices.restype = ct.c_int32
    properties = library.vkGetPhysicalDeviceProperties
    properties.argtypes = [ct.c_void_p, ct.POINTER(_PropertiesStorage)]
    properties.restype = None
    app = _ApplicationInfo(sType=0, pApplicationName=b"MinerU WebGPU policy", apiVersion=(1 << 22) | (1 << 12))
    info = _InstanceCreateInfo(sType=1, pApplicationInfo=ct.pointer(app))
    instance = ct.c_void_p()
    _check(create(ct.byref(info), None, ct.byref(instance)), "create instance")
    try:
        for _ in range(3):
            count = ct.c_uint32()
            _check(enumerate_devices(instance, ct.byref(count), None), "count devices")
            handles = (ct.c_void_p * count.value)()
            status = enumerate_devices(instance, ct.byref(count), handles)
            if status == 5:  # VK_INCOMPLETE: the inventory changed between calls.
                continue
            _check(status, "enumerate devices")
            result = []
            for index, handle in enumerate(handles[: count.value]):
                storage = _PropertiesStorage()
                properties(handle, ct.byref(storage))
                value = storage.prefix
                if value.deviceType in (0, 1, 2, 3):  # 保留未知原生类型，由运行后端过滤可用性；排除软件 CPU。
                    total, budget, memory_error = None, None, None
                    if include_memory:
                        try:
                            total, budget = _device_memory(library, handle, discrete=value.deviceType == 2)
                        except Exception as error:
                            memory_error = str(error)
                    result.append(
                        with_gpu_classification(
                            VulkanDevice(
                                value.vendorID,
                                value.deviceID,
                                value.deviceName.decode("utf-8", errors="replace"),
                                index,
                                total,
                                budget,
                                memory_error,
                                luid=_device_luid(library, handle) if sys.platform == "win32" else None,
                            ),
                            classify_vulkan_gpu(value.deviceType),
                        )
                    )
            return tuple(result)
        raise RuntimeError("Vulkan GPU inventory kept changing")
    finally:
        destroy(instance, None)


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    with vulkan_environment_lock():
        previous = {name: os.environ.get(name) for name in values}
        try:
            os.environ.update(values)
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


@contextmanager
def restrict_vulkan_device(vendor_id: int, device_id: int) -> Iterator[None]:
    """Verify a restriction before Dawn initialization; never infer success from env vars.

    Loader filters require Vulkan-Loader >= 1.4.326. Mesa device-select is an
    independently verified fallback for older loaders. Existing user filters
    must already allow the requested device and are restored on every exit.
    """
    visible = enumerate_vulkan_devices()
    if not any((d.vendor_id, d.device_id) == (vendor_id, device_id) for d in visible):
        raise RuntimeError("Selected WebGPU physical device is no longer visible to Vulkan")
    restrictions = (
        {"VK_LOADER_VENDOR_ID_FILTER": hex(vendor_id), "VK_LOADER_DEVICE_ID_FILTER": hex(device_id)},
        {"MESA_VK_DEVICE_SELECT": f"{vendor_id:04x}:{device_id:04x}!"},
    )
    for values in restrictions:
        with _environment(values):
            filtered = enumerate_vulkan_devices()
            if filtered and all((d.vendor_id, d.device_id) == (vendor_id, device_id) for d in filtered):
                yield
                return
    raise RuntimeError("Cannot restrict Dawn to the supported GPU; update Vulkan loader or enable Mesa device-select")


__all__ = ["VulkanDevice", "enumerate_vulkan_devices", "restrict_vulkan_device", "vulkan_environment_lock"]
