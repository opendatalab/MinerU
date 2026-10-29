# Copyright (c) Opendatalab. All rights reserved.
"""Lazy Windows GPU discovery; DXGI indices are never persisted."""

from __future__ import annotations

import ctypes as C
from dataclasses import asdict, dataclass
import sys
from typing import Any
import uuid

from .gpu import GpuClassification, GpuIdentity, classify_d3d12_gpu, gpu_kind_rank, with_gpu_classification


@dataclass(frozen=True)
class GpuAdapter(GpuClassification):
    """One hardware adapter, with its ordinary DXGI index and preference rank."""

    index: int
    luid: str
    name: str
    vendor_id: int
    device_id: int
    preference: int

    @property
    def identity(self) -> GpuIdentity:
        """提供已有 LUID 与 PCI 型号，设备序号不参与物理身份匹配。"""
        return GpuIdentity(self.vendor_id, self.device_id, luid=self.luid, name=self.name)

    @property
    def vendor(self) -> str:
        return {0x8086: "intel", 0x1002: "amd", 0x10DE: "nvidia"}.get(self.vendor_id, "unknown")


class _Guid(C.Structure):
    _fields_ = [("a", C.c_uint32), ("b", C.c_uint16), ("c", C.c_uint16), ("d", C.c_uint8 * 8)]


class _Luid(C.Structure):
    _fields_ = [("low", C.c_uint32), ("high", C.c_int32)]


class _Description(C.Structure):
    _fields_ = [
        ("name", C.c_wchar * 128),
        ("vendor", C.c_uint32),
        ("device", C.c_uint32),
        ("subsystem", C.c_uint32),
        ("revision", C.c_uint32),
        ("video_memory", C.c_size_t),
        ("system_memory", C.c_size_t),
        ("shared_memory", C.c_size_t),
        ("luid", _Luid),
        ("flags", C.c_uint32),
    ]


class _Architecture(C.Structure):
    _fields_ = [
        ("node", C.c_uint32),
        ("tile_based", C.c_int32),
        ("uma", C.c_int32),
        ("coherent", C.c_int32),
        ("isolated", C.c_int32),
    ]


class _VideoMemoryInfo(C.Structure):
    _fields_ = [
        ("budget", C.c_uint64),
        ("usage", C.c_uint64),
        ("available_for_reservation", C.c_uint64),
        ("reservation", C.c_uint64),
    ]


def _memory_budget(adapter: C.c_void_p) -> int:
    """读取同一 UMA 适配器的本地内存预算，查询接口始终成对释放。"""
    adapter3 = C.c_void_p()
    query = _method(adapter, 0, C.c_int32, C.POINTER(_Guid), C.POINTER(C.c_void_p))
    _check(query(adapter, C.byref(_guid("645967a4-1392-4310-a798-8053ce3e93fd")), C.byref(adapter3)))
    try:
        memory = _VideoMemoryInfo()
        read = _method(adapter3, 14, C.c_int32, C.c_uint32, C.c_int32, C.POINTER(_VideoMemoryInfo))
        _check(read(adapter3, 0, 0, C.byref(memory)))
        return memory.budget
    finally:
        _release(adapter3)


def _guid(value: str) -> _Guid:
    return _Guid.from_buffer_copy(uuid.UUID(value).bytes_le)


def _method(pointer: C.c_void_p, slot: int, result: Any, *args: Any) -> Any:
    vtable = C.cast(pointer, C.POINTER(C.POINTER(C.c_void_p))).contents
    return C.WINFUNCTYPE(result, C.c_void_p, *args)(vtable[slot])


def _release(pointer: C.c_void_p) -> None:
    if pointer:
        _method(pointer, 2, C.c_uint32)(pointer)


def _check(hr: int) -> None:
    if hr < 0:
        raise OSError(f"Windows GPU discovery failed: HRESULT=0x{hr & 0xFFFFFFFF:08x}")


def _description(adapter: C.c_void_p) -> _Description:
    desc = _Description()
    _check(_method(adapter, 10, C.c_int32, C.POINTER(_Description))(adapter, C.byref(desc)))
    return desc


def _luid(desc: _Description) -> str:
    return f"{desc.luid.high & 0xFFFFFFFF:08x}:{desc.luid.low:08x}"


def _architecture(adapter: C.c_void_p, create: Any, *, classify: bool = True) -> tuple[bool, bool | None]:
    device = C.c_void_p()
    iid = _guid("189819f1-1db6-4b57-be54-1821339b85f7")
    if not classify:
        # A null ppDevice checks D3D12 capability without creating a device (success is S_FALSE).
        return create(adapter, 0xB000, C.byref(iid), None) >= 0, None
    hr = create(adapter, 0xB000, C.byref(iid), C.byref(device))
    if hr < 0:
        return False, None
    try:
        info = _Architecture()
        check = _method(device, 13, C.c_int32, C.c_int32, C.c_void_p, C.c_uint32)
        # D3D12_FEATURE_ARCHITECTURE1, falling back to ARCHITECTURE on older drivers.
        hr = check(device, 16, C.byref(info), C.sizeof(info))
        if hr < 0:
            hr = check(device, 1, C.byref(info), C.sizeof(info) - C.sizeof(C.c_int32))
        return True, bool(info.uma) if hr >= 0 else None
    finally:
        _release(device)


def discover_gpus(*, classify: bool = True) -> tuple[GpuAdapter, ...]:
    """Discover D3D12 hardware; configured GPU mode only needs capability and DXGI performance order."""
    if sys.platform != "win32":
        return ()
    dxgi = C.WinDLL("dxgi", use_last_error=True)
    d3d = C.WinDLL("d3d12", use_last_error=True)
    create = dxgi.CreateDXGIFactory1
    create.argtypes = [C.POINTER(_Guid), C.POINTER(C.c_void_p)]
    create.restype = C.c_int32
    create_device = d3d.D3D12CreateDevice
    create_device.argtypes = [C.c_void_p, C.c_int32, C.POINTER(_Guid), C.POINTER(C.c_void_p)]
    create_device.restype = C.c_int32
    factory = C.c_void_p()
    modern = True
    iid = _guid("c1b6694f-ff09-44a9-b03c-77900a0a1d17")
    if create(C.byref(iid), C.byref(factory)) < 0:
        modern = False
        iid = _guid("770aae78-f26f-4dba-a829-253c83d1b387")
        _check(create(C.byref(iid), C.byref(factory)))
    try:
        ranks: dict[str, int] = {}
        if modern:
            enum_preference = _method(factory, 29, C.c_int32, C.c_uint32, C.c_int32, C.POINTER(_Guid), C.POINTER(C.c_void_p))
            adapter_iid = _guid("29038f61-3839-4626-91fd-086879011a05")
            for rank in range(64):
                adapter = C.c_void_p()
                hr = enum_preference(factory, rank, 2, C.byref(adapter_iid), C.byref(adapter))
                if hr & 0xFFFFFFFF == 0x887A0002:
                    break
                _check(hr)
                try:
                    ranks[_luid(_description(adapter))] = rank
                finally:
                    _release(adapter)
        rows: list[GpuAdapter] = []
        enum = _method(factory, 12, C.c_int32, C.c_uint32, C.POINTER(C.c_void_p))
        for index in range(64):
            adapter = C.c_void_p()
            hr = enum(factory, index, C.byref(adapter))
            if hr & 0xFFFFFFFF == 0x887A0002:
                break
            _check(hr)
            try:
                desc = _description(adapter)
                if desc.flags & 2:
                    continue
                # Old factories lack performance ordering, so classify to retain discrete-GPU priority.
                supported, uma = _architecture(adapter, create_device, classify=classify or not modern)
                if supported:
                    luid = _luid(desc)
                    rows.append(
                        with_gpu_classification(
                            GpuAdapter(index, luid, desc.name, desc.vendor, desc.device, ranks.get(luid, index)),
                            classify_d3d12_gpu(uma),
                        )
                    )
            finally:
                _release(adapter)
        return tuple(sorted(rows, key=lambda item: (gpu_kind_rank(item.kind), item.preference, item.index)))
    finally:
        _release(factory)


def preferred_gpu(adapters: tuple[GpuAdapter, ...]) -> GpuAdapter | None:
    """Prefer a confirmed discrete GPU, then Windows performance preference."""
    return min(adapters, key=lambda item: (gpu_kind_rank(item.kind), item.preference, item.index), default=None)


def inspect_gpus(*, include_memory_budget: bool = False) -> list[dict[str, Any]]:
    """诊断库存保留不支持 D3D12 的硬件，并读取 DXGI 的 64 位显存容量。"""
    if sys.platform != "win32":
        return []
    dxgi = C.WinDLL("dxgi", use_last_error=True)
    d3d = C.WinDLL("d3d12", use_last_error=True)
    create = dxgi.CreateDXGIFactory1
    create.argtypes = [C.POINTER(_Guid), C.POINTER(C.c_void_p)]
    create.restype = C.c_int32
    create_device = d3d.D3D12CreateDevice
    create_device.argtypes = [C.c_void_p, C.c_int32, C.POINTER(_Guid), C.POINTER(C.c_void_p)]
    create_device.restype = C.c_int32
    factory = C.c_void_p()
    _check(create(C.byref(_guid("770aae78-f26f-4dba-a829-253c83d1b387")), C.byref(factory)))
    result: list[dict[str, Any]] = []
    try:
        enum = _method(factory, 12, C.c_int32, C.c_uint32, C.POINTER(C.c_void_p))
        for index in range(64):
            adapter = C.c_void_p()
            hr = enum(factory, index, C.byref(adapter))
            if hr & 0xFFFFFFFF == 0x887A0002:
                break
            _check(hr)
            try:
                desc = _description(adapter)
                if desc.flags & 2:
                    continue
                supported, uma = _architecture(adapter, create_device)
                budget, memory_error = None, None
                if include_memory_budget and uma is True:
                    try:
                        budget = _memory_budget(adapter)
                    except Exception as error:
                        memory_error = str(error)
                result.append(
                    {
                        "name": desc.name,
                        "vendor_id": desc.vendor,
                        "device_id": desc.device,
                        "luid": _luid(desc),
                        "d3d12_available": supported,
                        **asdict(classify_d3d12_gpu(uma)),
                        "dedicated_memory_bytes": desc.video_memory,
                        "shared_memory_bytes": desc.shared_memory,
                        "memory_budget_bytes": budget,
                        "memory_error": memory_error,
                    }
                )
            finally:
                _release(adapter)
    finally:
        _release(factory)
    return result


__all__ = ["GpuAdapter", "discover_gpus", "inspect_gpus", "preferred_gpu"]
