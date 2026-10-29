# Copyright (c) Opendatalab. All rights reserved.
"""为自动档位提供选定设备的精确内存证据，不使用虚拟显存或系统内存替代预算。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys
from typing import Literal

from .gpu import GpuIdentity, PciAddress, match_gpu_identities

MIN_GPU_MEMORY_BYTES = 11 * 1024**3 // 2  # 5.5 GiB，始终使用整数字节比较。


@dataclass(frozen=True)
class GpuMemory:
    """一次设备查询的容量、口径和来源；未知容量保留为空并携带原因。"""

    device: str
    identity: GpuIdentity
    memory_bytes: int | None = None
    kind: Literal["dedicated", "budget", "unknown"] = "unknown"
    source: str = "unknown"
    warning: str | None = None

    @property
    def sufficient(self) -> bool:
        """按原始字节判断 5.5 GiB 边界，不四舍五入、不累加不同显卡。"""
        return self.memory_bytes is not None and self.memory_bytes >= MIN_GPU_MEMORY_BYTES


def _validated(memory: GpuMemory) -> GpuMemory:
    """未知或非法容量统一降为未知，保留探测来源和具体错误。"""
    if type(memory.memory_bytes) is int and memory.memory_bytes > 0:
        return memory
    return replace(memory, memory_bytes=None, warning=memory.warning or "GPU memory capacity or budget is unavailable")


def native_gpu_memory(identity: GpuIdentity, *, device: str) -> GpuMemory:
    """通过双向唯一身份关联查询选中 GPU，不能关联时禁止借用其他设备的容量。"""
    result = GpuMemory(device, identity)
    try:
        if sys.platform == "win32":
            from .windows_gpu import inspect_gpus

            rows = inspect_gpus(include_memory_budget=True)
            identities = [
                GpuIdentity(
                    row["vendor_id"],
                    row["device_id"],
                    luid=row["luid"],
                    name=row["name"],
                    pci_address=PciAddress(**row["pci_address"]) if row.get("pci_address") is not None else None,
                )
                for row in rows
            ]
            matched = match_gpu_identities((identity,), identities, allow_name=True)[0]
            if matched is not None:
                row = rows[matched]
                dedicated = row["kind"] == "discrete"
                result = GpuMemory(
                    device,
                    identities[matched],
                    row.get("dedicated_memory_bytes") if dedicated else row.get("memory_budget_bytes"),
                    "dedicated" if dedicated else "budget",
                    "dxgi.dedicated_video_memory" if dedicated else "dxgi.local_budget",
                    row.get("memory_error"),
                )
        elif sys.platform == "linux":
            from .vulkan import enumerate_vulkan_devices, vulkan_environment_lock

            with vulkan_environment_lock():
                devices = enumerate_vulkan_devices(include_memory=True)
            matched = match_gpu_identities((identity,), [item.identity for item in devices], allow_name=True)[0]
            if matched is not None:
                item = devices[matched]
                dedicated = item.kind == "discrete"
                result = GpuMemory(
                    device,
                    item.identity,
                    item.dedicated_memory_bytes if dedicated else item.memory_budget_bytes,
                    "dedicated" if dedicated else "budget",
                    "vulkan.device_local_heap" if dedicated else "vulkan.heap_budget",
                    item.memory_error,
                )
        if result.source == "unknown":
            result = replace(result, warning="GPU memory device identity could not be uniquely matched")
    except Exception as error:
        result = replace(result, warning=f"GPU memory discovery failed: {error}")
    return _validated(result)


def torch_gpu_memory(device: str) -> GpuMemory:
    """惰性读取 Torch 选定设备的容量；MPS 与集显使用运行时预算且不创建模型。"""
    result = GpuMemory(device, GpuIdentity(name=device))
    try:
        import torch

        kind, _, index_text = device.partition(":")
        if kind == "mps":
            return _validated(
                GpuMemory(
                    device, result.identity, torch.mps.recommended_max_memory(), "budget", "metal.recommended_working_set"
                )
            )
        if kind == "npu":
            import torch_npu

            api = torch_npu.npu
        else:
            api = getattr(torch, kind)
        index = int(index_text) if index_text else api.current_device()
        properties = api.get_device_properties(index)
        selected = f"{kind}:{index}"
        bus = getattr(properties, "pci_bus_id", None)
        slot = getattr(properties, "pci_device_id", None)
        pci_address = None
        if type(bus) is int and type(slot) is int:
            # ROCm 给出整数 bus/device/domain，没有 function；不能把 bus 编号当完整 BDF 字符串。
            try:
                pci_address = PciAddress(bus, slot, domain=getattr(properties, "pci_domain_id", None))
            except ValueError:
                pass  # 可选身份字段不可用，不影响独显直接读取容量；原生匹配仍需唯一。
        identity = GpuIdentity(
            vendor_id=0x8086 if kind == "xpu" else None,
            device_id=getattr(properties, "device_id", None),
            bus_id=bus if isinstance(bus, str) else None,
            name=getattr(properties, "name", None),
            pci_address=pci_address,
        )
        integrated = getattr(properties, "is_integrated_gpu", getattr(properties, "is_integrated", None))
        if integrated or (kind == "xpu" and integrated is None):
            return native_gpu_memory(identity, device=selected)
        result = GpuMemory(
            selected, identity, getattr(properties, "total_memory", None), "dedicated", f"torch.{kind}.total_memory"
        )
    except Exception as error:
        result = replace(result, warning=f"GPU memory discovery failed: {error}")
    return _validated(result)


__all__ = ["GpuMemory", "MIN_GPU_MEMORY_BYTES", "native_gpu_memory", "torch_gpu_memory"]
