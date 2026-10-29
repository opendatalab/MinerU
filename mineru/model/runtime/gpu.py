# Copyright (c) Opendatalab. All rights reserved.
"""共享 GPU 分类、已有身份匹配和候选优先级；不导入或初始化推理运行时。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeVar

GpuKind = Literal["discrete", "integrated", "virtual", "unknown"]
GPU_KIND_ORDER: tuple[GpuKind, ...] = ("discrete", "integrated", "virtual", "unknown")
_KIND_PRIORITY: dict[GpuKind, int] = {"discrete": 0, "integrated": 1, "virtual": 2, "unknown": 3}


@dataclass(frozen=True, kw_only=True)
class GpuClassification:
    """各后端共用的分类事实，统一内存属性与 GPU 类型分别保存。"""

    kind: GpuKind = "unknown"
    kind_source: str = "unknown"
    unified_memory: bool | None = None


@dataclass(frozen=True)
class PciAddress:
    """PCI 物理位置；未提供的 domain/function 保留未知，不猜测为零。"""

    bus: int
    device: int
    domain: int | None = None
    function: int | None = None

    def __post_init__(self) -> None:
        if self.bus is None or self.device is None:
            raise ValueError("PCI bus and device are required")
        for value, maximum in ((self.bus, 255), (self.device, 31), (self.domain, 65535), (self.function, 7)):
            if value is not None and (type(value) is not int or not 0 <= value <= maximum):
                raise ValueError("Invalid PCI address")


@dataclass(frozen=True)
class GpuIdentity:
    """只使用已有设备身份；名称仅可用于原生库存之间的唯一关联。"""

    vendor_id: int | None = None
    device_id: int | None = None
    luid: str | None = None
    bus_id: str | None = None
    name: str | None = None
    pci_address: PciAddress | None = None


_Gpu = TypeVar("_Gpu", bound=GpuClassification)


def with_gpu_classification(device: _Gpu, classification: GpuClassification) -> _Gpu:
    """替换统一分类字段，保留后端的设备索引、注册描述和绑定约束。"""
    return replace(
        device,
        kind=classification.kind,
        kind_source=classification.kind_source,
        unified_memory=classification.unified_memory,
    )


def gpu_kind_rank(kind: GpuKind) -> int:
    """返回统一优先级；同类型的性能排序或分组仍由各后端维护。"""
    return _KIND_PRIORITY[kind]


def is_discrete_gpu(device: GpuClassification) -> bool:
    """只有已确认的独显分类才允许进入独显策略，未知或虚拟类型均不算。"""
    return device.kind == "discrete"


def classify_d3d12_gpu(uma: bool | None) -> GpuClassification:
    """在 Windows 适配边界将 UMA 查询结果转换为统一分类，失败保留未知。"""
    kind: GpuKind = "unknown" if uma is None else "integrated" if uma else "discrete"
    return GpuClassification(kind=kind, kind_source="d3d12", unified_memory=uma)


def classify_vulkan_gpu(device_type: int | None) -> GpuClassification:
    """仅在原生适配边界解释 Vulkan 类型，不据此推断未查询的统一内存属性。"""
    kinds: dict[int, GpuKind] = {1: "integrated", 2: "discrete", 3: "virtual"}
    if device_type is None:
        return GpuClassification()
    return GpuClassification(kind=kinds.get(device_type, "unknown"), kind_source="vulkan")


def classify_macos_gpu(vendor: str | None, architecture: str, *, integrated: bool) -> GpuClassification:
    """转换现有 macOS 系统事实，Apple Silicon 的统一内存不等同于性能较弱。"""
    unified = vendor == "Apple" and architecture in {"arm64", "aarch64"}
    return GpuClassification(
        kind="integrated" if unified or integrated else "unknown",
        kind_source="system_profiler",
        unified_memory=True if unified else None,
    )


def _normalized_bus_id(value: str) -> str:
    """统一已采集 PCI 地址的十六进制宽度，不额外读取系统或扩展身份信息。"""
    try:
        domain, bus, address = value.split(":")
        slot, function = address.split(".")
        return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{int(slot, 16):02x}.{int(function, 16):x}"
    except ValueError:
        return value.lower()


def _pci_address(identity: GpuIdentity) -> PciAddress | None:
    """兼容既有完整 BDF 字符串与运行时只提供部分坐标的 PCI 地址。"""
    if identity.pci_address is not None:
        return identity.pci_address
    if isinstance(identity.bus_id, str):
        try:
            domain, bus, address = identity.bus_id.split(":")
            device, function = address.split(".")
            return PciAddress(int(bus, 16), int(device, 16), int(domain, 16), int(function, 16))
        except ValueError:
            pass
    return None


def _identity_match_strength(left: GpuIdentity, right: GpuIdentity, *, allow_name: bool) -> int:
    """LUID、PCI 地址、型号、名称依次降级；共有身份冲突时禁止弱匹配。"""
    for first, second in ((left.vendor_id, right.vendor_id), (left.device_id, right.device_id)):
        if first is not None and second is not None and first != second:
            return -1
    if left.luid and right.luid:
        return 5 if left.luid.lower() == right.luid.lower() else -1
    left_pci, right_pci = _pci_address(left), _pci_address(right)
    if left_pci is not None and right_pci is not None:
        coordinates = (
            (left_pci.bus, right_pci.bus),
            (left_pci.device, right_pci.device),
            (left_pci.domain, right_pci.domain),
            (left_pci.function, right_pci.function),
        )
        shared = [(first, second) for first, second in coordinates if first is not None and second is not None]
        return len(shared) if all(first == second for first, second in shared) else -1
    if left.bus_id and right.bus_id:
        return 4 if _normalized_bus_id(left.bus_id) == _normalized_bus_id(right.bus_id) else -1
    if all(value is not None for value in (left.vendor_id, left.device_id, right.vendor_id, right.device_id)):
        return 1
    return 0 if allow_name and left.name and left.name == right.name else -1


def gpu_identities_match(left: GpuIdentity, right: GpuIdentity, *, allow_name: bool = False) -> bool:
    """优先比较双方共有的强身份，存在冲突时不得退回型号或名称匹配。"""
    return _identity_match_strength(left, right, allow_name=allow_name) >= 0


def match_gpu_identities(
    sources: Sequence[GpuIdentity], targets: Sequence[GpuIdentity], *, allow_name: bool = False
) -> tuple[int | None, ...]:
    """只有两侧均唯一对应时返回目标索引，同型号多卡和重复库存均不猜测归属。"""
    candidates: list[list[int]] = []
    for source in sources:
        strengths = [_identity_match_strength(source, target, allow_name=allow_name) for target in targets]
        strongest = max(strengths, default=-1)
        candidates.append([index for index, strength in enumerate(strengths) if strength >= 0 and strength == strongest])
    return tuple(
        matches[0] if len(matches) == 1 and sum(matches[0] in other for other in candidates) == 1 else None
        for matches in candidates
    )


__all__ = [
    "GPU_KIND_ORDER",
    "GpuClassification",
    "GpuIdentity",
    "GpuKind",
    "PciAddress",
    "classify_d3d12_gpu",
    "classify_macos_gpu",
    "classify_vulkan_gpu",
    "gpu_identities_match",
    "gpu_kind_rank",
    "is_discrete_gpu",
    "match_gpu_identities",
    "with_gpu_classification",
]
