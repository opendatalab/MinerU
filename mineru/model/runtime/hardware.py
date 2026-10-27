"""只读聚合问题反馈所需的环境信息；不调用模型加载或自动校准。"""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any

from .hardware_probe import run_probe
from .hardware_system import collect_system_hardware
from .hardware_types import GpuInfo, HardwareInfo, PackageInfo, RuntimeProbe
from .onnx_calibration_cache import inspect_calibration_record

if TYPE_CHECKING:
    from .webgpu import WebGPUDevice
    from .windows_gpu import GpuAdapter

PACKAGE_NAMES = (
    "mineru",
    "docvortex",
    "mineru-llama-cpp",
    "mineru-vl-utils",
    "torch",
    "transformers",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-ep-webgpu",
    "vllm",
    "lmdeploy",
    "mlx-vlm",
    "pypdfium2",
)
DEVICE_ENVIRONMENT = (
    "MINERU_DEVICE_MODE",
    "MINERU_LLAMA_CPP_BACKEND",
    "MINERU_CPU_NUM_THREADS",
    "MINERU_INTRA_OP_NUM_THREADS",
    "MINERU_INTER_OP_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
    "ASCEND_RT_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_LOADER_VENDOR_ID_FILTER",
    "VK_LOADER_DEVICE_ID_FILTER",
    "MESA_VK_DEVICE_SELECT",
    "GGML_VK_SERIALIZE_SUBMISSIONS",
)


def installed_packages() -> list[PackageInfo]:
    """仅列出明确清单内已安装的发行包，不为了获取版本而导入重依赖。"""
    result = []
    for name in PACKAGE_NAMES:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        result.append(PackageInfo(name, distribution.version, str(distribution.locate_file(""))))
    return result


def config_summary() -> dict[str, Any]:
    """按白名单输出设备相关配置及来源，远程地址只暴露是否启用。"""
    from ...config import get_config, get_config_file_path, get_config_source

    config = get_config()
    fields = {
        "model.small_backend": config.model.small_backend,
        "model.onnx_device": config.model.onnx_device,
        "model.vlm.engine": config.model.vlm.engine,
    }
    return {
        "config_file": get_config_file_path(),
        "settings": {key: {"value": value, "source": get_config_source(key)} for key, value in fields.items()},
        "remote_vlm": {"enabled": bool(config.model.vlm.server_url), "source": get_config_source("model.vlm.server_url")},
        "environment": {key: os.environ[key] for key in DEVICE_ENVIRONMENT if key in os.environ},
    }


def merge_gpu_inventory(gpus: list[GpuInfo], probe: RuntimeProbe) -> None:
    """按明确身份补充原生类型和显存；无法唯一匹配的多卡库存不猜测归属。"""
    if probe.status != "ok":
        return
    vendors = {0x8086: "Intel", 0x10DE: "NVIDIA", 0x1002: "AMD"}
    for row in probe.data.get("devices", []):
        vendor, chip = row.get("vendor_id"), row.get("device_id")
        if vendor is None or chip is None:
            continue
        identity = f"0x{vendor:04x}:0x{chip:04x}"
        matches = [gpu for gpu in gpus if gpu.device_id == identity]
        name = row.get("name")
        if len(matches) > 1:
            matches = [gpu for gpu in matches if gpu.name == name]
            if len(matches) != 1:
                continue
        kind = row.get("kind") or {1: "integrated", 2: "discrete", 3: "virtual"}.get(row.get("device_type"), "unknown")
        if matches:
            target = matches[0]
            target.name = target.name or name
            target.kind = kind
        else:
            target = GpuInfo(
                name=name, vendor=vendors.get(vendor, hex(vendor)), device_id=identity, kind=kind, source=probe.name
            )
            gpus.append(target)
        for attribute in ("dedicated_memory_bytes", "shared_memory_bytes"):
            if row.get(attribute) is not None:
                setattr(target, attribute, row[attribute])


def _calibration_adapters(probes: list[RuntimeProbe]) -> tuple[GpuAdapter | WebGPUDevice, ...] | None:
    """从隔离探测结果重建纯数据描述，不在报告进程初始化 ORT 或驱动。"""
    from .webgpu import WebGPUDevice
    from .windows_gpu import GpuAdapter

    onnx = next((probe for probe in probes if probe.name == "onnx"), None)
    if onnx is None or onnx.status != "ok" or "adapters" not in onnx.data:
        return None
    rows = onnx.data["adapters"]
    if "selected_adapter" in onnx.data:
        rows = [onnx.data["selected_adapter"]] if onnx.data["selected_adapter"] is not None else []
    if onnx.data.get("backend") == "dml":
        return tuple(GpuAdapter(**row) for row in rows)
    return tuple(WebGPUDevice(**row) for row in rows)


def collect_hardware_info() -> HardwareInfo:
    """汇总静态信息及有界运行时探测，单项失败保留其余诊断内容。"""
    from ...version import __version__

    native = collect_system_hardware()
    packages = installed_packages()
    names = {package.name for package in packages}
    runtime_names = ["selection", "torch", "onnx"]
    if "mineru-llama-cpp" in names:
        runtime_names.append("llama-cpp")
    runtime_names.extend(name for name in ("vllm", "lmdeploy", "mlx-vlm") if name in names)
    if sys.platform in {"win32", "linux"}:
        runtime_names.append("dxgi" if sys.platform == "win32" else "vulkan")
    probes = [run_probe(name) for name in runtime_names]
    for probe in probes:
        if probe.name in {"vulkan", "dxgi"}:
            merge_gpu_inventory(native.gpus, probe)
        if probe.status in {"error", "timeout"}:
            native.warnings.append(f"{probe.name}: {probe.error}")
        if probe.data.get("adapter_error"):
            native.warnings.append(f"ONNX adapter inventory is incomplete: {probe.data['adapter_error']}")
    config = config_summary()
    selection = next(probe for probe in probes if probe.name == "selection")
    config["resolved"] = selection.data if selection.status == "ok" else None
    calibration = inspect_calibration_record(_calibration_adapters(probes))
    distribution = ""
    if sys.platform == "linux":
        try:
            distribution = platform.freedesktop_os_release().get("PRETTY_NAME", "")
        except OSError:
            pass
    return HardwareInfo(
        system={
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "os_version": platform.mac_ver()[0]
            if sys.platform == "darwin"
            else platform.win32_ver()[1]
            if sys.platform == "win32"
            else platform.release(),
            "distribution": distribution,
            "architecture": platform.machine(),
        },
        python={"version": platform.python_version(), "executable": sys.executable},
        mineru={"version": __version__, "module_path": str(Path(__file__).resolve().parents[2] / "__init__.py")},
        cpu=native.cpu,
        memory=native.memory,
        gpus=native.gpus,
        packages=packages,
        runtimes=probes,
        effective_config=config,
        calibration=calibration,
        warnings=native.warnings,
    )


__all__ = [
    "DEVICE_ENVIRONMENT",
    "PACKAGE_NAMES",
    "collect_hardware_info",
    "config_summary",
    "installed_packages",
    "merge_gpu_inventory",
]
