"""硬件推荐和只读诊断命令；帮助展示不导入或探测模型运行时。"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
import sys
from typing import TYPE_CHECKING, Any

import typer

from ...cli.contracts import CliContext
from ...utils.i18n import t
from ._base import RuntimeCommand

if TYPE_CHECKING:
    from ...model.runtime.hardware_types import HardwareInfoSection

app = typer.Typer(
    help=t("Detect recommended tiers or inspect hardware and runtime information."),
    no_args_is_help=True,
    add_completion=False,
)


def _value(value: Any) -> str:
    """统一可读报告中的缺失值，避免将未知信息显示成零或空字符串。"""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return "unknown" if value is None or value == "" else str(value)


def render_detection(data: dict[str, Any]) -> str:
    """从结构化推荐直接渲染可读结果，不从日志推导推荐或校准状态。"""
    calibration = data["calibration"]
    lines = [
        f"推荐启动上限：{data['ceiling']}",
        f"PDF/图片档位：{', '.join(data['pdf_image_tiers'])}",
        f"请求默认档位：{data['default_request_tier']}",
        f"推荐依据：{data['reason']}",
        f"可用 CPU：{data['cpu_count']}",
        f"小模型后端：{data['small_backend']}",
        f"VLM 引擎：{data['vlm_engine']}",
        f"实际设备：{data['device']}",
        f"校准状态：{calibration['status']} ({calibration['reason']})",
    ]
    lines.extend(f"提示：{warning}" for warning in data.get("warnings", []))
    return "\n".join(lines)


def _memory(value: int | None) -> str:
    """将已知字节容量转换为三位小数的 GiB，未知值不附加伪造单位。"""
    return "unknown" if value is None else f"{value / 1024**3:.3f} GiB"


def _torch_device_summary(data: dict[str, Any]) -> str:
    """仅展示选中的 Torch 设备及其型号、容量，严格匹配设备索引而不猜测其他卡。"""
    device = (data.get("selected_device") or {}).get("device")
    if not device:
        return "unknown"
    kind, _, index_text = device.partition(":")
    try:
        index = int(index_text) if index_text else 0
    except ValueError:
        return device
    rows = data.get("backends", {}).get(kind, {}).get("devices", [])
    selected = next((row for row in rows if row.get("index") == index), None)
    if selected is None:
        return device
    details = [selected["name"]] if selected.get("name") else []
    if selected.get("total_memory_bytes") is not None:
        details.append(_memory(selected["total_memory_bytes"]))
    label = f"{kind}:{index}"
    return f"{label} ({', '.join(details)})" if details else label


def _render_runtime_probe(probe: dict[str, Any]) -> list[str]:
    """将各运行时转成重点摘要，完整适配器身份和原生库存只保留在 JSON 中。"""
    name, status, data = probe["name"], probe["status"], probe["data"]
    if name in {"dxgi", "vulkan"}:
        # 原生库存已合并进系统显卡；采集失败由报告末尾的 warnings 保留。
        return []
    if name == "selection" and status == "ok":
        return [f"  Small model backend: {_value(data.get('small_backend'))}; VLM engine: {_value(data.get('vlm_engine'))}"]
    labels = {
        "selection": "Backend selection",
        "torch": "Torch",
        "onnx": "ONNX Runtime",
        "llama-cpp": "llama.cpp",
        "vllm": "vLLM",
        "lmdeploy": "LMDeploy",
        "mlx-vlm": "MLX-VLM",
    }
    states = {"ok": "ok", "not_installed": "not installed", "error": "error", "timeout": "timeout"}
    version = f"{data['version']}, " if data.get("version") else ""
    lines = [f"  {labels.get(name, name)}: {version}{states.get(status, status)}"]
    if probe["error"]:
        lines.append(f"    {probe['error']}")
    if status != "ok":
        return lines
    if name == "torch":
        lines.append(f"    Current device: {_torch_device_summary(data)}")
        selected = data.get("selected_device") or {}
        if (selected.get("device") or "").split(":", 1)[0] == "cuda":
            if data.get("rocm_build"):
                lines.append(f"    ROCm build: {data['rocm_build']}")
            elif data.get("cuda_build"):
                lines.append(f"    CUDA build: {data['cuda_build']}")
        if selected.get("warning"):
            lines.append(f"    Warning: {selected['warning']}")
        lines.extend(
            f"    Warning: {backend} probe failed: {details['error']}"
            for backend, details in data.get("backends", {}).items()
            if details.get("error")
        )
    elif name == "onnx":
        providers = [
            {"DmlExecutionProvider": "DirectML", "WebGpuExecutionProvider": "WebGPU"}.get(
                provider, provider.removesuffix("ExecutionProvider")
            )
            for provider in data.get("providers", [])
        ]
        if providers:
            lines.append(f"    Available providers: {', '.join(providers)}")
        selected = data.get("selected_device") or {}
        device = selected.get("device")
        adapter = selected.get("adapter") or data.get("selected_adapter") or {}
        if selected.get("reason") in {"calibration_required", "calibration_pending"}:
            lines.append("    Device selection: pending calibration")
            if adapter.get("name"):
                lines.append(f"    Candidate device: {adapter['name']}")
        elif device == "gpu":
            label = f"{selected.get('backend') or data.get('backend')}:gpu"
            detail = f" ({adapter['name']})" if adapter.get("name") else ""
            lines.append(f"    Device selection: {label}{detail}")
        else:
            lines.append(f"    Device selection: {_value(device)}")
        if data.get("adapter_error"):
            lines.append(f"    Warning: {data['adapter_error']}")
    elif name == "llama-cpp" and "devices" in data:
        lines.append("    Visible devices:" if data["devices"] else "    No visible GPU found")
        for device in data["devices"]:
            label = device.get("name") or device.get("backend") or "GPU"
            detail = f": {device['description']}" if device.get("description") else ""
            lines.append(f"      {label}{detail}")
    return lines


def _render_device_selection(data: dict[str, Any]) -> list[str]:
    """展示当前后端的只读设备选择；仅在存在校准证据时展开详细记录。"""
    backend = (data["effective_config"].get("resolved") or {}).get("small_backend")
    probe = next((probe for probe in data["runtimes"] if probe["name"] == backend), None)
    selection = probe["data"].get("selected_device") if probe and probe["status"] == "ok" else None
    lines = ["Device selection and calibration:", f"  Small model backend: {_value(backend)}"]
    reason = selection.get("reason") if selection else None
    reasons = {
        "torch_accelerator": "A Torch accelerator is available",
        "torch_cpu": "No usable Torch accelerator found; using CPU",
        "torch_device_unavailable": "The configured Torch device is unavailable; falling back to CPU",
        "torch_device_error": "Torch device checks failed; falling back to CPU",
        "configured_cpu": "CPU selected by configuration",
        "configured_gpu": "GPU selected by configuration; automatic benchmarking skipped",
        "discrete_gpu": "The confirmed discrete GPU candidate meets the rules for skipping calibration",
        "unsupported_platform": "ONNX GPU execution is unsupported on this platform; using CPU",
        "directml_unavailable": "DirectML provider is unavailable; using CPU",
        "no_hardware_gpu": "No usable ONNX GPU found; using CPU",
        "webgpu_architecture_policy": "No GPU passed the WebGPU architecture and visibility checks; using CPU",
        "calibration_required": "The candidate GPU requires OCR calibration before final device selection",
        "calibration_pending": "No usable calibration threshold is configured for the current backend",
        "selection_unavailable": "Device probing failed; selection could not be determined",
    }
    if selection:
        device = selection.get("device")
        if reason in {"calibration_required", "calibration_pending"}:
            device = "pending calibration"
        elif backend == "onnx" and device == "gpu":
            device = f"{selection['backend']}:gpu"
        lines.append(f"  Selected device: {_value(device)}")
        lines.append(f"  Selection reason: {reasons.get(reason, _value(reason))} ({_value(reason)})")
        if selection.get("warning"):
            lines.append(f"  Device warning: {selection['warning']}")
        if backend == "onnx" and selection.get("device") == "gpu":
            lines.append("  Validation scope: read-only selection; GPU inference session has not been validated")
    else:
        lines.append("  Selected device: unknown")
        detail = f"{probe['status']}, {_value(probe['error'])}" if probe else "No device selection result from this runtime"
        lines.append(f"  Selection reason: {detail}")

    if backend == "torch":
        lines.append("  Calibration: not applicable (Torch does not require ONNX OCR calibration)")
    elif reason in {"discrete_gpu", "configured_gpu"}:
        lines.append("  Calibration: skipped")
    elif reason in {"calibration_required", "calibration_pending"}:
        lines.append("  Calibration: pending (info performs read-only checks without benchmarking)")
    elif selection and selection.get("device") == "cpu":
        lines.append("  Calibration: not performed (CPU is currently selected)")

    calibration = data["calibration"]
    if calibration["status"] == "present":
        label = (
            "ONNX calibration record (not used for current Torch device selection)"
            if backend == "torch"
            else "ONNX calibration record"
        )
        lines.append(f"  {label}:")
        lines.extend(f"    {key}: {_value(value)}" for key, value in calibration.items() if value is not None)
    elif calibration["status"] == "invalid":
        lines.extend(
            [
                "  Calibration record: unavailable",
                f"    path: {calibration['path']}",
                f"    error: {_value(calibration['error'])}",
            ]
        )
    elif backend != "torch" and reason not in {"discrete_gpu", "configured_gpu"}:
        lines.append("  Calibration record: no detailed record")
    return lines


def _render_info_section(section: HardwareInfoSection, data: dict[str, Any]) -> str:
    """用统一分节渲染实现完整报告与逐段输出，保持文本字段和顺序一致。"""
    if section == "environment":
        return "\n".join(
            [
                f"System: {data['system']['name']} {data['system']['release']} ({data['system']['architecture']})",
                f"OS version: {_value(data['system'].get('os_version'))}",
                f"Distribution: {_value(data['system']['distribution'])}",
                f"Python: {data['python']['version']}",
                f"MinerU: {data['mineru']['version']}",
            ]
        )
    if section == "cpu_memory":
        cpu, memory = data["cpu"], data["memory"]
        return "\n".join(
            [
                f"CPU: {_value(cpu['model'])}",
                f"CPU cores: physical {_value(cpu['physical_cores'])}, logical {_value(cpu['logical_cores'])}, "
                f"available to process {_value(cpu['available_cores'])}",
                f"CPU frequency (MHz): base {_value(cpu['base_mhz'])}, current {_value(cpu['current_mhz'])}, "
                f"max {_value(cpu['max_mhz'])}",
                f"Memory: total {_memory(memory['total_bytes'])}, available {_memory(memory['available_bytes'])}",
            ]
        )
    if section == "packages":
        return "\n".join(
            ["Installed packages:", *(f"  {package['name']} {package['version']}" for package in data["packages"])]
        )
    if section == "gpus":
        lines = ["System GPUs:"]
        if not data["gpus"]:
            lines.append("  None found; see warnings for any probe errors")
        for gpu in data["gpus"]:
            lines.extend(
                [
                    f"  {_value(gpu['name'])} [{gpu['kind']}], vendor {_value(gpu['vendor'])}, ID {_value(gpu['device_id'])}",
                    f"    Source {gpu['source']}, driver {_value(gpu['driver'])}, bus {_value(gpu['bus_id'])}",
                    f"    Dedicated memory {_memory(gpu['dedicated_memory_bytes'])}, "
                    f"shared memory {_memory(gpu['shared_memory_bytes'])}, unified memory {_value(gpu['unified_memory'])}",
                ]
            )
        return "\n".join(lines)
    if section == "runtimes":
        lines = ["Runtimes:"] if data.get("runtime_header", True) else []
        for probe in data["runtimes"]:
            lines.extend(_render_runtime_probe(probe))
        return "\n".join(lines)
    lines = ["Effective configuration:"]
    lines.extend(f"  {key}: {_value(value)}" for key, value in data["effective_config"].items())
    lines.extend(_render_device_selection(data))
    lines.extend(f"Warning: {warning}" for warning in data["warnings"])
    return "\n".join(lines)


def render_info(data: dict[str, Any]) -> str:
    """使用与 JSON 相同的原始数据生成精简路径和容量的完整文本报告。"""
    sections: tuple[HardwareInfoSection, ...] = ("environment", "cpu_memory", "packages", "gpus", "runtimes", "configuration")
    return "\n".join(_render_info_section(section, data) for section in sections)


def _emit_info_section(section: HardwareInfoSection, data: dict[str, Any]) -> None:
    """每节完成后立即刷新 stdout，管道重定向时也能读取到早期结果。"""
    rendered = _render_info_section(section, data)
    if rendered:
        print(rendered, file=sys.stdout, flush=True)


def _info_data(*, stream: bool = False) -> dict[str, Any] | None:
    """文本逐节展示，JSON 仍返回完整报告；帮助入口不导入采集层。"""
    from ...model.runtime.hardware import collect_hardware_info

    report = collect_hardware_info(on_section=_emit_info_section if stream else None)
    return None if stream else asdict(report)


def _detect_data() -> dict[str, Any]:
    """仅在执行 detect 时调用共享推荐工作进程。"""
    from ...model.runtime.hardware_probe import run_detection

    return run_detection()


@app.command(
    "detect", cls=RuntimeCommand, help=t("Recommend the API tier ceiling using hardware and applicable OCR calibration.")
)
def detect_cmd(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """执行与 API 自动启动一致的后端感知推荐。"""
    from ...cli.runtime import run_cli

    run_cli(CliContext(json_mode=json_mode), _detect_data, render=render_detection)


@app.command(
    "info", cls=RuntimeCommand, help=t("Read hardware, installed versions and runtime diagnostics without calibration.")
)
def info_cmd(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """只读输出可用于问题反馈的环境报告。"""
    from ...cli.runtime import run_cli

    run_cli(CliContext(json_mode=json_mode), partial(_info_data, stream=not json_mode))


__all__ = ["app", "detect_cmd", "info_cmd", "render_detection", "render_info"]
