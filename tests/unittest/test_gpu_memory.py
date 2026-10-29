"""选中设备容量、共享预算与原生 ABI 的回归，避免依赖宿主显卡。"""

from __future__ import annotations

import ctypes as ct
from dataclasses import replace
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from mineru.model.runtime import gpu_memory, vulkan, windows_gpu
from mineru.model.runtime.gpu import GpuIdentity, PciAddress


@pytest.mark.parametrize("device", ["cuda:1", "cuda"])
def test_torch_reads_exact_selected_device_bytes(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    """显式序号和默认当前设备都不借用零号大卡，虚拟显存和四舍五入不参与。"""
    values = (24 * 1024**3, 5_905_580_032 - 1)
    properties = Mock(side_effect=lambda index: SimpleNamespace(name=f"GPU {index}", total_memory=values[index]))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 1, get_device_properties=properties))
    )
    monkeypatch.setenv("MINERU_VIRTUAL_VRAM_SIZE", "128")
    result = gpu_memory.torch_gpu_memory(device)
    assert result.device == "cuda:1" and result.memory_bytes == values[1] and not result.sufficient
    assert result.kind == "dedicated" and result.source == "torch.cuda.total_memory"
    properties.assert_called_once_with(1)


@pytest.mark.parametrize("amount", [None, 0, 5_905_580_032 - 1, 5_905_580_032, 6 * 1024**3 - 1, 8 * 1024**3])
def test_metal_uses_recommended_budget(monkeypatch: pytest.MonkeyPatch, amount: int | None) -> None:
    """Metal 只读推荐工作集，不拿系统总内存或当前空闲内存替代。"""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(recommended_max_memory=lambda: amount)))
    result = gpu_memory.torch_gpu_memory("mps")
    assert result.memory_bytes == (amount or None)
    assert result.sufficient is (amount is not None and amount >= 5_905_580_032)
    assert result.kind == "budget" and result.source == "metal.recommended_working_set"
    assert bool(result.warning) is (not amount)


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("amount", [None, 5_905_580_032 - 1, 5_905_580_032, 6 * 1024**3 - 1])
def test_native_capacity_belongs_to_selected_gpu(
    monkeypatch: pytest.MonkeyPatch, platform: str, integrated: bool, amount: int | None
) -> None:
    """库存中的其他大卡不授予能力；集显的大共享上限不能替代预算。"""
    monkeypatch.setattr(gpu_memory, "sys", SimpleNamespace(platform=platform))
    identity = GpuIdentity(0x8086, 1, luid="selected" if platform == "win32" else None)
    kind = "integrated" if integrated else "discrete"
    if platform == "win32":
        rows = [
            {
                "vendor_id": 0x8086,
                "device_id": 2,
                "name": "other",
                "luid": "other",
                "kind": "discrete",
                "dedicated_memory_bytes": 24 * 1024**3,
            },
            {
                "vendor_id": 0x8086,
                "device_id": 1,
                "name": "selected",
                "luid": "selected",
                "kind": kind,
                "dedicated_memory_bytes": 512 * 1024**2 if integrated else amount,
                "memory_budget_bytes": amount if integrated else None,
                "shared_memory_bytes": 128 * 1024**3,
            },
        ]
        monkeypatch.setattr(windows_gpu, "inspect_gpus", lambda **kwargs: rows)
    else:
        devices = (
            vulkan.VulkanDevice(0x8086, 2, "other", kind="discrete", dedicated_memory_bytes=24 * 1024**3),
            vulkan.VulkanDevice(
                0x8086,
                1,
                "selected",
                kind=kind,
                dedicated_memory_bytes=None if integrated else amount,
                memory_budget_bytes=amount if integrated else None,
            ),
        )
        monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", lambda **kwargs: devices)
    result = gpu_memory.native_gpu_memory(identity, device="selected")
    assert result.memory_bytes == amount
    assert result.kind == ("budget" if integrated else "dedicated")
    assert result.sufficient is (amount is not None and amount >= 5_905_580_032)


def test_ambiguous_native_gpu_does_not_borrow_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """同型号多卡缺少强身份时，即使某卡容量达标也不能猜测归属。"""
    monkeypatch.setattr(gpu_memory, "sys", SimpleNamespace(platform="linux"))
    first = vulkan.VulkanDevice(0x8086, 1, "GPU", kind="discrete", dedicated_memory_bytes=4 * 1024**3)
    monkeypatch.setattr(
        vulkan, "enumerate_vulkan_devices", lambda **kwargs: (first, replace(first, dedicated_memory_bytes=24 * 1024**3))
    )
    result = gpu_memory.native_gpu_memory(first.identity, device="selected")
    assert result.memory_bytes is None and not result.sufficient and "uniquely matched" in result.warning


def test_xpu_integrated_uses_native_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch 集显的全局内存不能替代原生预算，型号用于唯一关联原生设备。"""
    props = SimpleNamespace(name="Intel GPU", device_id=1, is_integrated_gpu=True, total_memory=64 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(xpu=SimpleNamespace(get_device_properties=lambda index: props)))
    query = Mock(return_value=gpu_memory.GpuMemory("xpu:1", GpuIdentity(0x8086, 1), 4 * 1024**3, "budget", "native"))
    monkeypatch.setattr(gpu_memory, "native_gpu_memory", query)
    result = gpu_memory.torch_gpu_memory("xpu:1")
    assert result.memory_bytes == 4 * 1024**3 and not result.sufficient
    query.assert_called_once_with(GpuIdentity(0x8086, 1, name="Intel GPU"), device="xpu:1")


@pytest.mark.parametrize("device", ["cuda:1", "cuda"])
@pytest.mark.parametrize("amount", [None, 5_905_580_032 - 1, 5_905_580_032])
def test_rocm_integrated_reads_matching_dxgi_budget(monkeypatch: pytest.MonkeyPatch, device: str, amount: int | None) -> None:
    """整数 PCI 字段定位真实 DXGI 适配器，重复条目或 Torch 大容量都不替代其预算。"""
    monkeypatch.setattr(gpu_memory, "sys", SimpleNamespace(platform="win32"))
    properties = Mock(
        return_value=SimpleNamespace(
            name="8060S", is_integrated=1, pci_bus_id=195, pci_device_id=0, pci_domain_id=0, total_memory=96 * 1024**3
        )
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 1, get_device_properties=properties))
    )
    physical = {
        "vendor_id": 0x1002,
        "device_id": 5510,
        "name": "8060S",
        "luid": "physical",
        "pci_address": {"bus": 195, "device": 0, "domain": None, "function": 0},
        "kind": "integrated",
        "dedicated_memory_bytes": 32 * 1024**3,
        "shared_memory_bytes": 128 * 1024**3,
        "memory_budget_bytes": amount,
    }
    duplicate = {**physical, "luid": "duplicate", "pci_address": None, "memory_budget_bytes": 128 * 1024**3}
    monkeypatch.setattr(windows_gpu, "inspect_gpus", lambda **kwargs: [duplicate, physical])
    result = gpu_memory.torch_gpu_memory(device)
    assert result.memory_bytes == amount and result.sufficient is (amount == 5_905_580_032)
    assert result.device == "cuda:1" and result.identity.luid == "physical"
    assert result.source == "dxgi.local_budget" and result.kind == "budget"
    properties.assert_called_once_with(1)


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


@pytest.mark.parametrize("native", [False, True])
def test_memory_probe_errors_are_reported(monkeypatch: pytest.MonkeyPatch, native: bool) -> None:
    """驱动异常输出未知容量和警告，不伪造零容量或吞掉原因。"""
    failure = Mock(side_effect=RuntimeError("driver unavailable"))
    monkeypatch.setattr(gpu_memory, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", failure)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(recommended_max_memory=failure)))
    result = gpu_memory.native_gpu_memory(GpuIdentity(1, 2), device="gpu") if native else gpu_memory.torch_gpu_memory("mps")
    assert result.memory_bytes is None and not result.sufficient and "driver unavailable" in result.warning


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("extension", [False, True])
def test_vulkan_memory_abi_uses_local_heaps_once(discrete: bool, extension: bool) -> None:
    """模拟原生填充结构体，验证共享预算扩展、字节宽度及本地堆去重。"""

    def fill_memory(memory: Any) -> None:
        """两个类型指向同一个 GPU 堆，另一个大堆是主机内存。"""
        memory.type_count, memory.heap_count = 2, 2
        memory.types[0].heap_index = memory.types[1].heap_index = 0
        memory.heaps[0].flags, memory.heaps[0].size = 1, 8 * 1024**3
        memory.heaps[1].flags, memory.heaps[1].size = 0, 128 * 1024**3

    def properties(handle: Any, output: Any) -> None:
        """按 Vulkan 1.0 ABI 回填显存堆。"""
        fill_memory(output._obj)

    def extensions(handle: Any, layer: Any, count: Any, rows: Any) -> int:
        """分别模拟支持和不支持预算扩展的运行时。"""
        count._obj.value = int(extension)
        if rows is not None and extension:
            rows[0].name = b"VK_EXT_memory_budget"
        return 0

    def properties2(handle: Any, output: Any) -> None:
        """预算显著低于总容量且有用量，查询应保留预算上限。"""
        assert output._obj.sType == 1000059006
        fill_memory(output._obj.memory)
        budget = ct.cast(output._obj.pNext, ct.POINTER(vulkan._MemoryBudget)).contents
        assert budget.sType == 1000237000
        budget.budget[0], budget.usage[0] = 6 * 1024**3, 2 * 1024**3
        budget.budget[1] = 100 * 1024**3

    lib = SimpleNamespace(
        vkGetPhysicalDeviceMemoryProperties=Mock(side_effect=properties),
        vkEnumerateDeviceExtensionProperties=Mock(side_effect=extensions),
        vkGetPhysicalDeviceMemoryProperties2=Mock(side_effect=properties2),
    )
    assert vulkan._device_memory(lib, ct.c_void_p(1), discrete=discrete) == (
        (8 * 1024**3, None) if discrete else (None, 6 * 1024**3 if extension else None)
    )
    if discrete or not extension:
        lib.vkGetPhysicalDeviceMemoryProperties2.assert_not_called()
    assert ct.sizeof(vulkan._MemoryProperties) == 520
    assert vulkan._MemoryProperties.heaps.offset == 264


@pytest.mark.parametrize("failed", [False, True])
def test_dxgi_budget_queries_same_adapter_and_releases(monkeypatch: pytest.MonkeyPatch, failed: bool) -> None:
    """模拟 COM 查询与释放，预算失败仍释放接口且不使用共享容量替代。"""

    def query(pointer: Any, iid: Any, output: Any) -> int:
        """返回同一适配器的 IDXGIAdapter3 接口。"""
        assert pointer.value == 123
        output._obj.value = 456
        return 0

    def read(pointer: Any, node: int, segment: int, output: Any) -> int:
        """查询本地段预算，不减去用量或加入非本地内存。"""
        assert (pointer.value, node, segment) == (456, 0, 0)
        output._obj.budget, output._obj.usage = 6 * 1024**3, 2 * 1024**3
        return -1 if failed else 0

    def method(pointer: Any, slot: int, *args: Any) -> Any:
        """只允许 QueryInterface 和 QueryVideoMemoryInfo 两个方法。"""
        return {0: query, 14: read}[slot]

    monkeypatch.setattr(windows_gpu, "_method", method)
    release = Mock()
    monkeypatch.setattr(windows_gpu, "_release", release)
    if failed:
        with pytest.raises(OSError, match="HRESULT"):
            windows_gpu._memory_budget(ct.c_void_p(123))
    else:
        assert windows_gpu._memory_budget(ct.c_void_p(123)) == 6 * 1024**3
    assert release.call_count == 1 and release.call_args.args[0].value == 456
