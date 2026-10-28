# Copyright (c) Opendatalab. All rights reserved.
"""标准库与平台原生工具实现的只读 CPU、内存和系统显卡库存采集。"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
from typing import Any

from .cpu_threads import effective_cpu_count
from .hardware_types import GpuInfo, SystemHardware
from .gpu import GpuIdentity, classify_macos_gpu, gpu_identities_match, match_gpu_identities

_VENDORS = {"0x8086": "Intel", "0x10de": "NVIDIA", "0x1002": "AMD"}


def _read(path: Path) -> str | None:
    """读取可选系统字段，缺失或无权限时保留未知而不填充伪造值。"""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _number(value: Any) -> float | None:
    """解析平台报告的数值，空值、无效值和负数均保持未知。"""
    try:
        number = float(value)
        return number if number >= 0 and number < float("inf") else None
    except (ValueError, TypeError):
        return None


def _integer(value: Any) -> int | None:
    """统一平台核数和字节值，避免对未知数值执行整型转换。"""
    value = _number(value)
    return int(value) if value is not None else None


def _command(command: list[str], report: SystemHardware, *, partial: bool = False) -> str | None:
    """有界运行只读平台工具，局部错误写入报告而不阻断后续采集。"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
        if result.returncode:
            report.warnings.append(f"{command[0]} exited {result.returncode}: {result.stderr.strip()[:300]}")
            return result.stdout if partial else None
        return result.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        report.warnings.append(f"{command[0]} unavailable: {error}")
        return None


def _memory_size(value: Any) -> int | None:
    """解析平台给出的内存容量单位，不把未知或统一内存自动解释为独立显存。"""
    match = re.search(r"([\d.]+)\s*(bytes|[KMGT]i?B)\b", str(value), re.IGNORECASE)
    if not match:
        return None
    units = {"bytes": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
    return int(float(match[1]) * units[match[2].lower().replace("ib", "b")])


def _linux_cpu(report: SystemHardware, proc_root: Path, sys_root: Path) -> None:
    """从 Linux 处理器拓扑、配额和频率节点读取事实，保留异构或受限环境的未知值。"""
    content = _read(proc_root / "cpuinfo")
    if content is None:
        report.warnings.append("Cannot read /proc/cpuinfo; CPU identification is incomplete")
    records: list[dict[str, str]] = []
    for block in (content or "").split("\n\n"):
        record = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        records.append({key.strip(): value.strip() for key, value in record.items()})
    report.cpu.model = next(
        (r.get("model name") or r.get("Hardware") for r in records if r.get("model name") or r.get("Hardware")), None
    )
    topology = {(r["physical id"], r["core id"]) for r in records if "physical id" in r and "core id" in r}
    if topology:
        report.cpu.physical_cores = len(topology)
    frequencies = [_number(r.get("cpu MHz")) for r in records]
    current = [value for value in frequencies if value is not None]
    if current:
        report.cpu.current_mhz = sum(current) / len(current)
    for attribute, node in (("base_mhz", "base_frequency"), ("max_mhz", "cpuinfo_max_freq")):
        values = [_number(_read(path)) for path in (sys_root / "devices/system/cpu").glob(f"cpu[0-9]*/cpufreq/{node}")]
        known = [value for value in values if value is not None]
        if known:
            setattr(report.cpu, attribute, max(known) / 1000)
    memory = {}
    for line in (_read(proc_root / "meminfo") or "").splitlines():
        key, _, raw = line.partition(":")
        memory[key] = _memory_size(raw)
    report.memory.total_bytes = memory.get("MemTotal")
    report.memory.available_bytes = memory.get("MemAvailable")


def _linux_gpus(report: SystemHardware, sys_root: Path) -> None:
    """列出系统 PCI 显卡并补充驱动信息，不只保留运行时优先选择的显卡。"""
    root = sys_root / "bus/pci/devices"
    if not root.exists():
        report.warnings.append("PCI inventory is unavailable; GPU inventory may be incomplete")
        return
    for path in sorted(root.iterdir()):
        if not (_read(path / "class") or "").startswith("0x03"):
            continue
        vendor, chip = _read(path / "vendor"), _read(path / "device")
        driver = path / "driver"
        vram = _integer(_read(path / "mem_info_vram_total"))
        report.gpus.append(
            GpuInfo(
                vendor=_VENDORS.get(vendor or "", vendor),
                device_id=f"{vendor}:{chip}" if vendor and chip else None,
                bus_id=path.name,
                driver=driver.resolve().name if driver.exists() else None,
                dedicated_memory_bytes=vram,
                source="sysfs",
            )
        )
    executable = shutil.which("lspci")
    if executable:
        output = _command([executable, "-D", "-mm"], report)
        for line in (output or "").splitlines():
            columns = shlex.split(line)
            if len(columns) < 4:
                continue
            for gpu in report.gpus:
                if gpu.bus_id == columns[0]:
                    gpu.name = f"{columns[2]} {columns[3]}"


def parse_macos_displays(payload: dict[str, Any], report: SystemHardware, architecture: str) -> None:
    """解析系统显示设备 JSON，只保留显卡属性，不包含显示器序列号等无关数据。"""
    for row in payload.get("SPDisplaysDataType", []):
        vendor = row.get("spdisplays_vendor", "").removeprefix("sppci_vendor_") or None
        classification = classify_macos_gpu(
            vendor, architecture, integrated=row.get("spdisplays_integrated") == "spdisplays_yes"
        )
        unified = classification.unified_memory
        memory = _memory_size(row.get("spdisplays_vram") or row.get("spdisplays_vram_shared"))
        report.gpus.append(
            GpuInfo(
                name=row.get("sppci_model") or row.get("_name"),
                vendor=vendor,
                device_id=row.get("spdisplays_device-id"),
                kind=classification.kind,
                kind_source=classification.kind_source,
                bus_id=row.get("sppci_bus"),
                driver=row.get("spdisplays_driver_version"),
                dedicated_memory_bytes=None if unified or row.get("spdisplays_vram_shared") else memory,
                shared_memory_bytes=report.memory.total_bytes
                if unified
                else memory
                if row.get("spdisplays_vram_shared")
                else None,
                unified_memory=unified,
                source="system_profiler",
            )
        )


def _macos_hardware(report: SystemHardware) -> None:
    """通过 sysctl、vm_stat 与 system_profiler 获取 macOS 原生报告。"""
    keys = [
        "machdep.cpu.brand_string",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "hw.cpufrequency",
        "hw.cpufrequency_max",
        "hw.memsize",
        "hw.pagesize",
    ]
    output = _command(["sysctl", *keys], report, partial=True)
    values = dict(line.split(": ", 1) for line in (output or "").splitlines() if ": " in line)
    report.cpu.model = values.get("machdep.cpu.brand_string")
    report.cpu.physical_cores = _integer(values.get("hw.physicalcpu"))
    report.cpu.logical_cores = _integer(values.get("hw.logicalcpu")) or report.cpu.logical_cores
    for attribute, key in (("base_mhz", "hw.cpufrequency"), ("max_mhz", "hw.cpufrequency_max")):
        value = _number(values.get(key))
        if value is not None:
            setattr(report.cpu, attribute, value / 1_000_000)
    report.memory.total_bytes = _integer(values.get("hw.memsize"))
    vm = _command(["vm_stat"], report)
    page_size = _integer(values.get("hw.pagesize"))
    if vm and page_size:
        pages = dict(re.findall(r"^(Pages (?:free|inactive|speculative)):\s+(\d+)", vm, re.MULTILINE))
        if len(pages) == 3:
            report.memory.available_bytes = sum(int(value) for value in pages.values()) * page_size
    displays = _command(["system_profiler", "SPDisplaysDataType", "-json"], report)
    if displays:
        parse_macos_displays(json.loads(displays), report, platform.machine())


def parse_windows_hardware(payload: dict[str, Any], report: SystemHardware) -> None:
    """解析 CIM 结果，AdapterRAM 的 32 位限制不用于宣称独立显存容量。"""
    cpus = payload.get("cpus") or []
    if isinstance(cpus, dict):
        cpus = [cpus]
    if cpus:
        report.cpu.model = "; ".join(dict.fromkeys(row.get("Name", "").strip() for row in cpus)) or None
        for attribute, key in (("physical_cores", "NumberOfCores"), ("logical_cores", "NumberOfLogicalProcessors")):
            counts = [_integer(row.get(key)) for row in cpus]
            if all(value is not None for value in counts):
                setattr(report.cpu, attribute, sum(counts))
        for attribute, key in (("current_mhz", "CurrentClockSpeed"), ("max_mhz", "MaxClockSpeed")):
            frequencies = [_number(row.get(key)) for row in cpus]
            known = [value for value in frequencies if value is not None]
            if known:
                setattr(report.cpu, attribute, max(known))
    memory = payload.get("memory") or {}
    for attribute, key in (("total_bytes", "TotalVisibleMemorySize"), ("available_bytes", "FreePhysicalMemory")):
        amount = _integer(memory.get(key))
        if amount is not None:
            setattr(report.memory, attribute, amount * 1024)
    rows = payload.get("gpus") or []
    if isinstance(rows, dict):
        rows = [rows]
    for row in rows:
        identity = re.search(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})", row.get("PNPDeviceID") or "", re.IGNORECASE)
        report.gpus.append(
            GpuInfo(
                name=row.get("Name"),
                vendor=row.get("AdapterCompatibility"),
                device_id=f"0x{identity[1].lower()}:0x{identity[2].lower()}" if identity else None,
                driver=row.get("DriverVersion"),
                source="cim",
            )
        )


def _windows_hardware(report: SystemHardware) -> None:
    """仅使用系统自带 PowerShell/CIM 采集，不依赖额外 WMI Python 包。"""
    queries = {
        "cpus": "Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,CurrentClockSpeed,MaxClockSpeed",
        "memory": "Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory",
        "gpus": "Win32_VideoController | Select-Object Name,AdapterCompatibility,PNPDeviceID,DriverVersion",
    }
    payload: dict[str, Any] = {}
    for key, query in queries.items():
        script = (
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); $ErrorActionPreference = 'Stop'; "
            f"Get-CimInstance {query} | ConvertTo-Json -Depth 4 -Compress"
        )
        output = _command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], report)
        if output:
            try:
                payload[key] = json.loads(output.lstrip("\ufeff"))
            except ValueError as error:
                report.warnings.append(f"CIM {key} returned invalid data: {error}")
    parse_windows_hardware(payload, report)


def _nvidia_details(report: SystemHardware) -> None:
    """利用可选厂商工具补充名称、驱动和显存，按 PCI 地址匹配以免串卡。"""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return
    output = _command(
        [executable, "--query-gpu=pci.bus_id,name,driver_version,memory.total", "--format=csv,noheader,nounits"], report
    )
    rows = [tuple(value.strip() for value in row) for row in csv.reader(io.StringIO(output or "")) if len(row) == 4]
    identities = [GpuIdentity(bus_id=row[0], name=row[1]) for row in rows]
    targets = [gpu.identity for gpu in report.gpus]
    matches = match_gpu_identities(identities, targets, allow_name=True)
    for index, ((bus, name, driver, memory), matched) in enumerate(zip(rows, matches)):
        if matched is None:
            if any(gpu_identities_match(identities[index], target, allow_name=True) for target in targets):
                continue
            gpu = GpuInfo(name=name, vendor="NVIDIA", bus_id=bus, source="nvidia-smi")
            report.gpus.append(gpu)
        else:
            gpu = report.gpus[matched]
        gpu.name, gpu.driver, gpu.bus_id = name, driver, bus
        amount = _number(memory)
        if amount is not None:
            gpu.dedicated_memory_bytes = int(amount * 1024**2)


def collect_system_hardware() -> SystemHardware:
    """分项采集本机只读硬件事实，未支持平台或局部失败不会伪造空库存。"""
    report = SystemHardware()
    report.cpu.logical_cores = os.cpu_count()
    try:
        report.cpu.available_cores = effective_cpu_count()
    except OSError as error:
        report.warnings.append(f"CPU availability check failed: {error}")
    system = platform.system()
    try:
        if system == "Linux":
            try:
                _linux_cpu(report, Path("/proc"), Path("/sys"))
            except (OSError, ValueError, TypeError) as error:
                report.warnings.append(f"Linux CPU/memory information is incomplete: {error}")
            _linux_gpus(report, Path("/sys"))
        elif system == "Darwin":
            _macos_hardware(report)
        elif system == "Windows":
            _windows_hardware(report)
        else:
            report.warnings.append(f"Native hardware inventory is unsupported on {system}")
    except (OSError, ValueError, TypeError) as error:
        report.warnings.append(f"Native hardware inventory is incomplete: {error}")
    _nvidia_details(report)
    return report


__all__ = ["collect_system_hardware", "parse_macos_displays", "parse_windows_hardware"]
