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
    return "未知" if value is None or value == "" else str(value)


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
    return "未知" if value is None else f"{value / 1024**3:.3f} GiB"


def _torch_device_summary(data: dict[str, Any]) -> str:
    """仅展示选中的 Torch 设备及其型号、容量，严格匹配设备索引而不猜测其他卡。"""
    device = (data.get("selected_device") or {}).get("device")
    if not device:
        return "未知"
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
    return f"{label}（{'，'.join(details)}）" if details else label


def _render_runtime_probe(probe: dict[str, Any]) -> list[str]:
    """将各运行时转成重点摘要，完整适配器身份和原生库存只保留在 JSON 中。"""
    name, status, data = probe["name"], probe["status"], probe["data"]
    if name in {"dxgi", "vulkan"}:
        # 原生库存已合并进系统显卡；采集失败由报告末尾的 warnings 保留。
        return []
    if name == "selection" and status == "ok":
        return [f"  小模型后端：{_value(data.get('small_backend'))}；VLM 引擎：{_value(data.get('vlm_engine'))}"]
    labels = {
        "selection": "后端选择",
        "torch": "Torch",
        "onnx": "ONNX Runtime",
        "llama-cpp": "llama.cpp",
        "vllm": "vLLM",
        "lmdeploy": "LMDeploy",
        "mlx-vlm": "MLX-VLM",
    }
    states = {"ok": "正常", "not_installed": "未安装", "error": "失败", "timeout": "超时"}
    version = f"{data['version']}，" if data.get("version") else ""
    lines = [f"  {labels.get(name, name)}：{version}{states.get(status, status)}"]
    if probe["error"]:
        lines.append(f"    {probe['error']}")
    if status != "ok":
        return lines
    if name == "torch":
        lines.append(f"    当前设备：{_torch_device_summary(data)}")
        selected = data.get("selected_device") or {}
        if (selected.get("device") or "").split(":", 1)[0] == "cuda":
            if data.get("rocm_build"):
                lines.append(f"    ROCm 构建版本：{data['rocm_build']}")
            elif data.get("cuda_build"):
                lines.append(f"    CUDA 构建版本：{data['cuda_build']}")
        if selected.get("warning"):
            lines.append(f"    提示：{selected['warning']}")
        lines.extend(
            f"    提示：{backend} 探测失败：{details['error']}"
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
            lines.append(f"    可用 Provider：{', '.join(providers)}")
        selected = data.get("selected_device") or {}
        device = selected.get("device")
        adapter = selected.get("adapter") or data.get("selected_adapter") or {}
        if selected.get("reason") in {"calibration_required", "calibration_pending"}:
            lines.append("    设备选择：待校准")
            if adapter.get("name"):
                lines.append(f"    候选设备：{adapter['name']}")
        elif device == "gpu":
            label = f"{selected.get('backend') or data.get('backend')}:gpu"
            detail = f"（{adapter['name']}）" if adapter.get("name") else ""
            lines.append(f"    设备选择：{label}{detail}")
        else:
            lines.append(f"    设备选择：{_value(device)}")
        if data.get("adapter_error"):
            lines.append(f"    提示：{data['adapter_error']}")
    elif name == "llama-cpp" and "devices" in data:
        lines.append("    可见设备：" if data["devices"] else "    未发现可见 GPU")
        for device in data["devices"]:
            label = device.get("name") or device.get("backend") or "GPU"
            detail = f"：{device['description']}" if device.get("description") else ""
            lines.append(f"      {label}{detail}")
    return lines


def _render_device_selection(data: dict[str, Any]) -> list[str]:
    """展示当前后端的只读设备选择；仅在存在校准证据时展开详细记录。"""
    backend = (data["effective_config"].get("resolved") or {}).get("small_backend")
    probe = next((probe for probe in data["runtimes"] if probe["name"] == backend), None)
    selection = probe["data"].get("selected_device") if probe and probe["status"] == "ok" else None
    lines = ["设备选择与校准：", f"  小模型后端：{_value(backend)}"]
    reason = selection.get("reason") if selection else None
    reasons = {
        "torch_accelerator": "Torch 加速设备可用",
        "torch_cpu": "未发现可用的 Torch 加速设备，使用 CPU",
        "torch_device_unavailable": "指定的 Torch 设备不可用，回退 CPU",
        "torch_device_error": "Torch 设备检查失败，回退 CPU",
        "configured_cpu": "配置指定使用 CPU",
        "configured_gpu": "配置指定使用 GPU，跳过自动测速",
        "discrete_gpu": "已确认的候选独显满足免校准规则",
        "unsupported_platform": "当前平台不支持 ONNX GPU 路径，使用 CPU",
        "directml_unavailable": "DirectML provider 不可用，使用 CPU",
        "no_hardware_gpu": "未发现可用的 ONNX GPU，使用 CPU",
        "webgpu_architecture_policy": "没有通过 WebGPU 架构与可见性检查的 GPU，使用 CPU",
        "calibration_required": "候选 GPU 需要 OCR 校准后才能确定最终设备",
        "calibration_pending": "当前后端尚未配置可用的校准阈值",
        "selection_unavailable": "设备探测失败，无法确定选择结果",
    }
    if selection:
        device = selection.get("device")
        if reason in {"calibration_required", "calibration_pending"}:
            device = "待校准"
        elif backend == "onnx" and device == "gpu":
            device = f"{selection['backend']}:gpu"
        lines.append(f"  选择设备：{_value(device)}")
        lines.append(f"  选择依据：{reasons.get(reason, _value(reason))} ({_value(reason)})")
        if selection.get("warning"):
            lines.append(f"  设备提示：{selection['warning']}")
        if backend == "onnx" and selection.get("device") == "gpu":
            lines.append("  验证范围：只读选型，尚未验证 GPU 推理会话")
    else:
        lines.append("  选择设备：未知")
        detail = f"{probe['status']}，{_value(probe['error'])}" if probe else "未取得对应运行时的设备选择结果"
        lines.append(f"  选择依据：{detail}")

    if backend == "torch":
        lines.append("  校准：不适用（Torch 后端无需 ONNX OCR 校准）")
    elif reason in {"discrete_gpu", "configured_gpu"}:
        lines.append("  校准：免校准")
    elif reason in {"calibration_required", "calibration_pending"}:
        lines.append("  校准：待完成（info 只读检查不会执行测速）")
    elif selection and selection.get("device") == "cpu":
        lines.append("  校准：未执行（当前选择 CPU）")

    calibration = data["calibration"]
    if calibration["status"] == "present":
        label = "ONNX 校准记录（不参与当前 Torch 设备选择）" if backend == "torch" else "ONNX 校准记录"
        lines.append(f"  {label}：")
        lines.extend(f"    {key}: {_value(value)}" for key, value in calibration.items() if value is not None)
    elif calibration["status"] == "invalid":
        lines.extend(
            [
                "  校准记录：不可用",
                f"    path: {calibration['path']}",
                f"    error: {_value(calibration['error'])}",
            ]
        )
    elif backend != "torch" and reason not in {"discrete_gpu", "configured_gpu"}:
        lines.append("  校准记录：无详细记录")
    return lines


def _render_info_section(section: HardwareInfoSection, data: dict[str, Any]) -> str:
    """用统一分节渲染实现完整报告与逐段输出，保持文本字段和顺序一致。"""
    if section == "environment":
        return "\n".join(
            [
                f"系统：{data['system']['name']} {data['system']['release']} ({data['system']['architecture']})",
                f"系统版本：{_value(data['system'].get('os_version'))}",
                f"发行版：{_value(data['system']['distribution'])}",
                f"Python：{data['python']['version']}",
                f"MinerU：{data['mineru']['version']}",
            ]
        )
    if section == "cpu_memory":
        cpu, memory = data["cpu"], data["memory"]
        return "\n".join(
            [
                f"CPU：{_value(cpu['model'])}",
                f"CPU 核数：物理 {_value(cpu['physical_cores'])}，逻辑 {_value(cpu['logical_cores'])}，"
                f"进程可用 {_value(cpu['available_cores'])}",
                f"CPU 频率 (MHz)：基础 {_value(cpu['base_mhz'])}，当前 {_value(cpu['current_mhz'])}，"
                f"最大 {_value(cpu['max_mhz'])}",
                f"内存：总量 {_memory(memory['total_bytes'])}，可用 {_memory(memory['available_bytes'])}",
            ]
        )
    if section == "packages":
        return "\n".join(["已安装软件：", *(f"  {package['name']} {package['version']}" for package in data["packages"])])
    if section == "gpus":
        lines = ["系统显卡："]
        if not data["gpus"]:
            lines.append("  未发现；如有探测错误，请参见提示")
        for gpu in data["gpus"]:
            lines.extend(
                [
                    f"  {_value(gpu['name'])} [{gpu['kind']}]，厂商 {_value(gpu['vendor'])}，标识 {_value(gpu['device_id'])}",
                    f"    来源 {gpu['source']}，驱动 {_value(gpu['driver'])}，总线 {_value(gpu['bus_id'])}",
                    f"    独立显存 {_memory(gpu['dedicated_memory_bytes'])}，"
                    f"共享内存 {_memory(gpu['shared_memory_bytes'])}，统一内存 {_value(gpu['unified_memory'])}",
                ]
            )
        return "\n".join(lines)
    if section == "runtimes":
        lines = ["运行时："] if data.get("runtime_header", True) else []
        for probe in data["runtimes"]:
            lines.extend(_render_runtime_probe(probe))
        return "\n".join(lines)
    lines = ["有效配置："]
    lines.extend(f"  {key}: {_value(value)}" for key, value in data["effective_config"].items())
    lines.extend(_render_device_selection(data))
    lines.extend(f"提示：{warning}" for warning in data["warnings"])
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
