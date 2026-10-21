"""No-compute Vulkan enumeration and scoped physical-device restrictions for Dawn.

Uses the stable Vulkan 1.0 C ABI. No logical device, shader or GPU work is created.
Callers must serialize scopes with the WebGPU runtime lock.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import ctypes as ct
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class VulkanDevice:
    vendor_id: int
    device_id: int
    device_type: int
    name: str


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


def _check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"Vulkan {operation} failed ({result})")


def enumerate_vulkan_devices() -> tuple[VulkanDevice, ...]:
    """Enumerate GPUs visible to the loader, honoring the user's ICD/layer filters."""
    library = ct.CDLL("libvulkan.so.1")
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
            for handle in handles[: count.value]:
                storage = _PropertiesStorage()
                properties(handle, ct.byref(storage))
                value = storage.prefix
                if value.deviceType in (1, 2, 3):  # integrated, discrete, virtual GPU; exclude software CPU
                    result.append(
                        VulkanDevice(
                            value.vendorID, value.deviceID, value.deviceType, value.deviceName.decode("utf-8", errors="replace")
                        )
                    )
            return tuple(result)
        raise RuntimeError("Vulkan GPU inventory kept changing")
    finally:
        destroy(instance, None)


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
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


__all__ = ["VulkanDevice", "enumerate_vulkan_devices", "restrict_vulkan_device"]
