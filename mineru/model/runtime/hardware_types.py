# Copyright (c) Opendatalab. All rights reserved.
"""硬件诊断的可序列化报告类型，不导入推理运行时。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .onnx_calibration_cache import CalibrationInfo
from .gpu import GpuIdentity, GpuKind

HardwareInfoSection = Literal["environment", "cpu_memory", "packages", "gpus", "runtimes", "configuration"]


@dataclass
class CpuInfo:
    """区分机器核数、进程可用核数和不同来源的频率。"""

    model: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    available_cores: int | None = None
    base_mhz: float | None = None
    current_mhz: float | None = None
    max_mhz: float | None = None


@dataclass
class MemoryInfo:
    """内存统一以字节表示，未知值不当作零。"""

    total_bytes: int | None = None
    available_bytes: int | None = None


@dataclass
class GpuInfo:
    """系统发现的物理或虚拟显卡，不代表当前运行时一定可用。"""

    name: str | None = None
    vendor: str | None = None
    device_id: str | None = None
    kind: GpuKind = "unknown"
    kind_source: str = "unknown"
    bus_id: str | None = None
    driver: str | None = None
    dedicated_memory_bytes: int | None = None
    shared_memory_bytes: int | None = None
    unified_memory: bool | None = None
    source: str = "unknown"

    @property
    def identity(self) -> GpuIdentity:
        """将报告已有的 PCI 型号和地址转换为共享身份，不从名称猜测厂商或类型。"""
        vendor, chip = None, None
        if self.device_id:
            try:
                vendor_text, chip_text = self.device_id.split(":")
                vendor, chip = int(vendor_text, 16), int(chip_text, 16)
            except ValueError:
                pass
        return GpuIdentity(vendor, chip, bus_id=self.bus_id, name=self.name)


@dataclass
class SystemHardware:
    """平台原生只读采集结果和局部失败说明。"""

    cpu: CpuInfo = field(default_factory=CpuInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    gpus: list[GpuInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PackageInfo:
    """已安装发行包的版本和安装位置，不以导入成功代替安装状态。"""

    name: str
    version: str
    location: str


@dataclass(frozen=True)
class RuntimeProbe:
    """隔离子进程的探测结果，保留未安装、超时及导入失败的区别。"""

    name: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class HardwareInfo:
    """供文本和 JSON 共用的完整问题反馈报告。"""

    system: dict[str, str]
    python: dict[str, str]
    mineru: dict[str, str]
    cpu: CpuInfo
    memory: MemoryInfo
    gpus: list[GpuInfo]
    packages: list[PackageInfo]
    runtimes: list[RuntimeProbe]
    effective_config: dict[str, Any]
    calibration: CalibrationInfo
    warnings: list[str]
    schema_version: int = 1


__all__ = [
    "CpuInfo",
    "GpuInfo",
    "HardwareInfo",
    "HardwareInfoSection",
    "MemoryInfo",
    "PackageInfo",
    "RuntimeProbe",
    "SystemHardware",
]
