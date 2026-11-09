"""原生独显容量与保留的硬件诊断 ABI 回归。"""

from __future__ import annotations

import ctypes as ct
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from mineru.model.runtime import vulkan, windows_gpu
from mineru.model.runtime.gpu import PciAddress


@pytest.mark.parametrize("stage", ["success", "open", "query", "invalid", "close"])
def test_windows_pci_query_abi_and_handle_lifetime(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    """验证系统接口结构布局，失败/无效地址均不伪造坐标，已打开句柄始终关闭。"""

    def open_adapter(output: Any) -> int:
        assert output._obj.luid.low == 0x1709C and output._obj.luid.high == 1
        output._obj.handle = 123
        return -1 if stage == "open" else 0

    def query_adapter(output: Any) -> int:
        query = output._obj
        assert query.handle == 123 and query.type == 6 and query.size == 12
        address = ct.cast(query.data, ct.POINTER(windows_gpu._AdapterAddress)).contents
        address.bus, address.device, address.function = (0xFFFFFFFF, 0xFFFF, 0xFFFF) if stage == "invalid" else (195, 0, 1)
        return -1 if stage == "query" else 0

    def close_adapter(output: Any) -> int:
        assert output._obj.handle == 123
        return -1 if stage == "close" else 0

    library = SimpleNamespace(
        D3DKMTOpenAdapterFromLuid=Mock(side_effect=open_adapter),
        D3DKMTQueryAdapterInfo=Mock(side_effect=query_adapter),
        D3DKMTCloseAdapter=Mock(side_effect=close_adapter),
    )
    monkeypatch.setattr(windows_gpu.C, "WinDLL", lambda *args, **kwargs: library, raising=False)
    if stage == "success":
        assert windows_gpu._pci_address(windows_gpu._Luid(0x1709C, 1)) == PciAddress(195, 0, function=1)
    else:
        with pytest.raises(ValueError if stage == "invalid" else OSError):
            windows_gpu._pci_address(windows_gpu._Luid(0x1709C, 1))
    assert library.D3DKMTCloseAdapter.call_count == int(stage != "open")
    assert ct.sizeof(windows_gpu._OpenAdapterFromLuid) == 12
    assert windows_gpu._QueryAdapterInfo.data.offset == 8


@pytest.mark.parametrize("vendor", [0x10DE, 0x1002, 0x8086])
def test_windows_discovery_carries_native_64bit_memory(monkeypatch: pytest.MonkeyPatch, vendor: int) -> None:
    """模拟系统 DXGI 调用，验证三家独显均保留超过 32 位的总容量。"""
    monkeypatch.setattr(windows_gpu, "sys", SimpleNamespace(platform="win32"))

    def create(iid: Any, output: Any) -> int:
        """使用旧版工厂路径，避免测试依赖宿主 Windows 接口。"""
        output._obj.value = 1
        return -1 if factory.call_count == 1 else 0

    def enumerate_adapter(pointer: Any, index: int, output: Any) -> int:
        """枚举一张独显后正常结束。"""
        output._obj.value = 2
        return 0 if index == 0 else 0x887A0002

    factory = Mock(side_effect=create)
    library = SimpleNamespace(CreateDXGIFactory1=factory, D3D12CreateDevice=Mock())
    monkeypatch.setattr(windows_gpu.C, "WinDLL", lambda *args, **kwargs: library, raising=False)
    monkeypatch.setattr(windows_gpu, "_method", lambda *args: enumerate_adapter)
    monkeypatch.setattr(windows_gpu, "_architecture", Mock(return_value=(True, False)))
    description = windows_gpu._Description()
    description.name, description.vendor, description.device = "GPU", vendor, 1
    description.video_memory = 8 * 1024**3
    monkeypatch.setattr(windows_gpu, "_description", Mock(return_value=description))
    release = Mock()
    monkeypatch.setattr(windows_gpu, "_release", release)
    selected = windows_gpu.preferred_gpu(windows_gpu.discover_gpus())
    assert selected.kind == "discrete" and selected.vendor_id == vendor
    assert selected.dedicated_memory_bytes == 8 * 1024**3
    assert release.call_count == 2


@pytest.mark.parametrize("include_memory", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_vulkan_discovery_only_reads_discrete_local_heaps(
    monkeypatch: pytest.MonkeyPatch, include_memory: bool, failed: bool
) -> None:
    """完整枚举夹具验证仅查询独显、本地堆不重复计数、失败保留原因且释放实例。"""
    monkeypatch.setattr(vulkan, "sys", SimpleNamespace(platform="linux"))

    def create(info: Any, allocator: Any, output: Any) -> int:
        """创建测试实例，不创建逻辑设备或分配显存。"""
        output._obj.value = 1
        return 0

    def enumerate_devices(instance: Any, count: Any, handles: Any) -> int:
        """返回一张独显和一张集显。"""
        count._obj.value = 2
        if handles is not None:
            handles[0], handles[1] = 2, 3
        return 0

    def properties(handle: int, output: Any) -> None:
        """独显和集显使用不同的原生类型。"""
        output._obj.prefix.vendorID = 0x1002
        output._obj.prefix.deviceID = handle
        output._obj.prefix.deviceType = 2 if handle == 2 else 1
        output._obj.prefix.deviceName = b"GPU"

    def memory(handle: int, output: Any) -> None:
        """两个内存类型指向同一个堆，另一个大堆属于主机内存。"""
        assert handle == 2
        if failed:
            raise OSError("memory properties unavailable")
        value = output._obj
        value.type_count, value.heap_count = 2, 3
        value.types[0].heap_index = value.types[1].heap_index = 0
        value.heaps[0].flags, value.heaps[0].size = 1, 8 * 1024**3
        value.heaps[1].flags, value.heaps[1].size = 1, 2 * 1024**3
        value.heaps[2].flags, value.heaps[2].size = 0, 128 * 1024**3

    library = SimpleNamespace(
        vkCreateInstance=Mock(side_effect=create),
        vkDestroyInstance=Mock(),
        vkEnumeratePhysicalDevices=Mock(side_effect=enumerate_devices),
        vkGetPhysicalDeviceProperties=Mock(side_effect=properties),
        vkGetPhysicalDeviceMemoryProperties=Mock(side_effect=memory),
    )
    monkeypatch.setattr(vulkan.ct, "CDLL", lambda *args: library)
    discrete, integrated = vulkan.enumerate_vulkan_devices(include_memory=include_memory)
    assert discrete.kind == "discrete" and integrated.kind == "integrated"
    assert discrete.dedicated_memory_bytes == (10 * 1024**3 if include_memory and not failed else None)
    assert bool(discrete.memory_error) is (include_memory and failed)
    assert integrated.dedicated_memory_bytes is None and integrated.memory_error is None
    assert library.vkGetPhysicalDeviceMemoryProperties.call_count == int(include_memory)
    library.vkDestroyInstance.assert_called_once()
