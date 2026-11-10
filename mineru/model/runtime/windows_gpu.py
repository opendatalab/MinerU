# Copyright (c) Opendatalab. All rights reserved.
"""Lazy Windows GPU discovery; DXGI indices are never persisted."""

from __future__ import annotations

import ctypes as C
from dataclasses import asdict, dataclass, field, replace
import sys
from typing import Any
import uuid

from .gpu import (
    GpuClassification,
    GpuIdentity,
    PciAddress,
    classify_d3d12_gpu,
    gpu_kind_rank,
    is_discrete_gpu,
    minimum_gpu_memory_bytes,
    with_gpu_classification,
)


@dataclass(frozen=True)
class GpuAdapter(GpuClassification):
    """One hardware adapter, with its ordinary DXGI index and preference rank."""

    index: int
    luid: str
    name: str
    vendor_id: int
    device_id: int
    preference: int
    # 容量只参与自动选型，不改变设备身份、排序或校准证据。
    dedicated_memory_bytes: int | None = field(default=None, compare=False)

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


class _OpenAdapterFromLuid(C.Structure):
    _fields_ = [("luid", _Luid), ("handle", C.c_uint32)]


class _QueryAdapterInfo(C.Structure):
    _fields_ = [("handle", C.c_uint32), ("type", C.c_int32), ("data", C.c_void_p), ("size", C.c_uint32)]


class _AdapterAddress(C.Structure):
    _fields_ = [("bus", C.c_uint32), ("device", C.c_uint32), ("function", C.c_uint32)]


class _CloseAdapter(C.Structure):
    _fields_ = [("handle", C.c_uint32)]


def _pci_address(luid: _Luid) -> PciAddress:
    """以 DXGI LUID 查询同一适配器的物理 PCI 地址，不加载任何推理后端。"""
    library = C.WinDLL("gdi32", use_last_error=True)
    open_adapter = library.D3DKMTOpenAdapterFromLuid
    open_adapter.argtypes, open_adapter.restype = [C.POINTER(_OpenAdapterFromLuid)], C.c_int32
    query_adapter = library.D3DKMTQueryAdapterInfo
    query_adapter.argtypes, query_adapter.restype = [C.POINTER(_QueryAdapterInfo)], C.c_int32
    close_adapter = library.D3DKMTCloseAdapter
    close_adapter.argtypes, close_adapter.restype = [C.POINTER(_CloseAdapter)], C.c_int32

    def check_status(status: int) -> None:
        if status < 0:
            raise OSError(f"Windows GPU PCI query failed: NTSTATUS=0x{status & 0xFFFFFFFF:08x}")

    opened = _OpenAdapterFromLuid(luid, 0)
    check_status(open_adapter(C.byref(opened)))
    try:
        address = _AdapterAddress()
        # KMTQAITYPE_ADAPTERADDRESS = 6; Windows 不提供 PCI domain，不能自行补零。
        query = _QueryAdapterInfo(opened.handle, 6, C.addressof(address), C.sizeof(address))
        check_status(query_adapter(C.byref(query)))
        return PciAddress(address.bus, address.device, function=address.function)
    finally:
        check_status(close_adapter(C.byref(_CloseAdapter(opened.handle))))


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
                            GpuAdapter(
                                index,
                                luid,
                                desc.name,
                                desc.vendor,
                                desc.device,
                                ranks.get(luid, index),
                                dedicated_memory_bytes=desc.video_memory,
                            ),
                            classify_d3d12_gpu(uma),
                        )
                    )
            finally:
                _release(adapter)
        return tuple(sorted(rows, key=lambda item: (gpu_kind_rank(item.kind), item.preference, item.index)))
    finally:
        _release(factory)


def preferred_gpu(adapters: tuple[GpuAdapter, ...]) -> GpuAdapter | None:
    """优先独显及系统性能排序；同型号独显取最小总容量，无需逐卡身份关联。"""
    selected = min(adapters, key=lambda item: (gpu_kind_rank(item.kind), item.preference, item.index), default=None)
    if selected is None or not is_discrete_gpu(selected):
        return selected
    capacities = [
        item.dedicated_memory_bytes
        for item in adapters
        if is_discrete_gpu(item) and (item.vendor_id, item.device_id) == (selected.vendor_id, selected.device_id)
    ]
    return replace(selected, dedicated_memory_bytes=minimum_gpu_memory_bytes(capacities))


def inspect_gpus() -> list[dict[str, Any]]:
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
                pci_address, pci_error = None, None
                try:
                    pci_address = _pci_address(desc.luid)
                except Exception as error:
                    pci_error = str(error)
                result.append(
                    {
                        "name": desc.name,
                        "vendor_id": desc.vendor,
                        "device_id": desc.device,
                        "luid": _luid(desc),
                        "pci_address": asdict(pci_address) if pci_address is not None else None,
                        "pci_error": pci_error,
                        "d3d12_available": supported,
                        **asdict(classify_d3d12_gpu(uma)),
                        "dedicated_memory_bytes": desc.video_memory,
                        "shared_memory_bytes": desc.shared_memory,
                    }
                )
            finally:
                _release(adapter)
    finally:
        _release(factory)
    return result


__all__ = ["GpuAdapter", "discover_gpus", "inspect_gpus", "preferred_gpu"]
