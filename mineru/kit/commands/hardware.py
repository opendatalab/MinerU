"""硬件推荐和只读诊断命令；帮助展示不导入或探测模型运行时。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import typer

from ...cli.contracts import CliContext
from ...cli.runtime import run_cli
from ...utils.i18n import t

app = typer.Typer(help=t("Detect recommended tiers or inspect hardware and runtime information."), no_args_is_help=True)


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
    memory = data.get("gpu_memory")
    if memory is not None:
        lines.extend(
            [
                f"GPU 内存设备：{memory['device']}，身份：{_value(memory['identity'])}",
                f"GPU 内存：{_value(memory['memory_bytes'])} bytes，口径：{memory['kind']}，来源：{memory['source']}",
                "Standard/Advanced 内存门槛：5.5 GiB (5905580032 bytes)",
            ]
        )
    lines.extend(f"提示：{warning}" for warning in data.get("warnings", []))
    return "\n".join(lines)


def render_info(data: dict[str, Any]) -> str:
    """使用与 JSON 相同的字段生成可复制的问题反馈文本。"""
    cpu, memory = data["cpu"], data["memory"]
    lines = [
        f"系统：{data['system']['name']} {data['system']['release']} ({data['system']['architecture']})",
        f"系统版本：{_value(data['system'].get('os_version'))}",
        f"发行版：{_value(data['system']['distribution'])}",
        f"Python：{data['python']['version']} ({data['python']['executable']})",
        f"MinerU：{data['mineru']['version']} ({data['mineru']['module_path']})",
        f"CPU：{_value(cpu['model'])}",
        f"CPU 核数：物理 {_value(cpu['physical_cores'])}，逻辑 {_value(cpu['logical_cores'])}，"
        f"进程可用 {_value(cpu['available_cores'])}",
        f"CPU 频率 (MHz)：基础 {_value(cpu['base_mhz'])}，当前 {_value(cpu['current_mhz'])}，最大 {_value(cpu['max_mhz'])}",
        f"内存 (bytes)：总量 {_value(memory['total_bytes'])}，可用 {_value(memory['available_bytes'])}",
        "系统显卡：",
    ]
    if not data["gpus"]:
        lines.append("  未发现；如有探测错误，请参见提示")
    for gpu in data["gpus"]:
        lines.extend(
            [
                f"  {_value(gpu['name'])} [{gpu['kind']}]，厂商 {_value(gpu['vendor'])}，标识 {_value(gpu['device_id'])}",
                f"    来源 {gpu['source']}，驱动 {_value(gpu['driver'])}，总线 {_value(gpu['bus_id'])}",
                f"    独立显存 {_value(gpu['dedicated_memory_bytes'])} bytes，"
                f"共享内存 {_value(gpu['shared_memory_bytes'])} bytes，统一内存 {_value(gpu['unified_memory'])}",
            ]
        )
    lines.append("已安装软件：")
    lines.extend(f"  {package['name']} {package['version']} ({package['location']})" for package in data["packages"])
    lines.append("当前进程运行时探测：")
    for probe in data["runtimes"]:
        lines.append(f"  {probe['name']}: {probe['status']}")
        if probe["error"]:
            lines.append(f"    {probe['error']}")
        for key, value in probe["data"].items():
            lines.append(f"    {key}: {_value(value)}")
    lines.append("有效配置：")
    lines.extend(f"  {key}: {_value(value)}" for key, value in data["effective_config"].items())
    lines.append("OCR 校准记录：")
    lines.extend(f"  {key}: {_value(value)}" for key, value in data["calibration"].items())
    lines.extend(f"提示：{warning}" for warning in data["warnings"])
    return "\n".join(lines)


def _info_data() -> dict[str, Any]:
    """仅在执行 info 时惰性导入采集层，帮助入口保持无探测副作用。"""
    from ...model.runtime.hardware import collect_hardware_info

    return asdict(collect_hardware_info())


def _detect_data() -> dict[str, Any]:
    """仅在执行 detect 时调用共享推荐工作进程。"""
    from ...model.runtime.hardware_probe import run_detection

    return run_detection()


@app.command("detect", help=t("Recommend the API tier ceiling using hardware and applicable OCR calibration."))
def detect_cmd(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """执行与 API 自动启动一致的后端感知推荐。"""
    run_cli(CliContext(json_mode=json_mode), _detect_data, render=render_detection)


@app.command("info", help=t("Read hardware, installed versions and runtime diagnostics without calibration."))
def info_cmd(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """只读输出可用于问题反馈的环境报告。"""
    run_cli(CliContext(json_mode=json_mode), _info_data, render=render_info)


__all__ = ["app", "detect_cmd", "info_cmd", "render_detection", "render_info"]
