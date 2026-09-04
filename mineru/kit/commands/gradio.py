"""`mineru-kit gradio` 的轻量命令门面。"""

from __future__ import annotations

import importlib.util
import os
from typing import Literal

import typer

from ...model.ocr.language import validate_public_ocr_lang
from ...types import SERVER_TIERS, ServerTier
from ...utils.stdio import configure_standard_streams
from ..errors import exit_with_message


def _require_gradio_dependencies() -> None:
    """检查 Gradio 可选依赖，并在缺失时给出安装提示。"""
    missing = [name for name in ("gradio", "gradio_pdf") if importlib.util.find_spec(name) is None]
    if missing:
        exit_with_message(
            "dependency_missing",
            "Gradio support requires the optional dependencies; install with `pip install 'mineru[gradio]'`.",
        )


def _validate_server_tier(value: str) -> ServerTier:
    """校验自动启动的本地 API server 能力档位。"""
    if value not in SERVER_TIERS:
        exit_with_message(
            "invalid_request",
            f"Unsupported API server tier '{value}'. Supported tiers: {', '.join(SERVER_TIERS)}",
            "api_server_tier",
        )
    return value  # type: ignore[return-value]


def _validate_positive_option(value: int, *, name: str) -> int:
    """校验本地托管服务使用的正整数参数。"""
    if value <= 0:
        exit_with_message("invalid_request", f"{name} must be greater than zero", name)
    return value


def _validate_server_port(value: int) -> int:
    """校验 Gradio 监听端口处于 TCP 有效范围。"""
    if value < 1 or value > 65535:
        exit_with_message("invalid_request", "server_port must be between 1 and 65535", "server_port")
    return value


def gradio_cmd(
    api_url: str | None = typer.Option(None, "--api-url", help="External MinerU V1 API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="Bearer API key; falls back to MINERU_API_KEY"),
    server_name: str = typer.Option("127.0.0.1", "--server-name", help="Gradio bind host"),
    server_port: int | None = typer.Option(
        None, "--server-port", help="Gradio bind port; omitted: auto-select from 7860 or GRADIO_SERVER_PORT"
    ),
    output_dir: str = typer.Option("./output", "--output-dir", help="Directory for Gradio artifacts"),
    enable_example: bool = typer.Option(True, "--enable-example/--no-enable-example", help="Show local examples"),
    enable_api: bool = typer.Option(True, "--enable-api/--no-enable-api", help="Expose the Gradio event API"),
    latex_delimiters_type: Literal["a", "b", "all"] = typer.Option(
        "all", "--latex-delimiters-type", help="LaTeX delimiters used by the Markdown preview"
    ),
    api_server_tier: str = typer.Option("standard", "--api-server-tier", "--tier", help="Managed API server capability tier"),
    api_server_no_flash: bool = typer.Option(
        False, "--api-server-no-flash/--no-flash", help="Disable Flash on the managed server"
    ),
    api_server_concurrency: int = typer.Option(
        1, "--api-server-concurrency", "--concurrency", help="Managed server job concurrency"
    ),
    api_server_language: str = typer.Option("ch", "--api-server-language", "--language", help="Managed server OCR language"),
    api_server_ocr_mode: Literal["auto", "txt", "ocr"] = typer.Option(
        "auto", "--api-server-ocr-mode", "--ocr-mode", help="Managed server OCR mode"
    ),
    api_server_disable_image_analysis: bool = typer.Option(
        False,
        "--api-server-disable-image-analysis/--disable-image-analysis",
        help="Disable managed server image analysis",
    ),
    api_server_preload_models: bool = typer.Option(
        False, "--api-server-preload-models/--preload-models", help="Preload managed server models"
    ),
) -> None:
    """启动基于 MinerU V1 API 的 Gradio 文档解析界面。"""
    _require_gradio_dependencies()
    if server_port is not None:
        _validate_server_port(server_port)
    _validate_positive_option(api_server_concurrency, name="api_server_concurrency")
    normalized_tier = _validate_server_tier(api_server_tier)
    try:
        normalized_language = validate_public_ocr_lang(api_server_language)
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "api_server_language")

    from ..gradio.app import launch_gradio

    try:
        resolved_api_key = api_key if api_key is not None else os.environ.get("MINERU_API_KEY")
        launch_gradio(
            api_url=api_url,
            api_key=resolved_api_key,
            server_name=server_name,
            server_port=server_port,
            output_dir=output_dir,
            enable_example=enable_example,
            enable_api=enable_api,
            latex_delimiters_type=latex_delimiters_type,
            api_server_tier=normalized_tier,
            api_server_no_flash=api_server_no_flash,
            api_server_concurrency=api_server_concurrency,
            api_server_language=normalized_language,
            api_server_ocr_mode=api_server_ocr_mode,
            api_server_disable_image_analysis=api_server_disable_image_analysis,
            api_server_preload_models=api_server_preload_models,
        )
    except Exception as exc:
        exit_with_message("gradio_start_failed", str(exc))


def main() -> None:
    """以独立 console script 运行新版 Gradio 命令。"""
    configure_standard_streams()
    typer.run(gradio_cmd)


__all__ = ["gradio_cmd", "main"]
