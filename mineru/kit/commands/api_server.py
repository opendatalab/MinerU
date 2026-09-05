from __future__ import annotations

from typing import cast

import click
import typer

from ...parser import api_server as parser_api_server
from ...types import SERVER_TIERS, ServerTier
from ...model.ocr.language import PUBLIC_OCR_LANGUAGES, validate_public_ocr_lang
from ...utils.stdio import configure_standard_streams
from ..errors import exit_with_message

API_SERVER_LANGUAGES = PUBLIC_OCR_LANGUAGES


def _normalize_tier_option(tier: str | None) -> ServerTier | None:
    """规范化 mineru-kit 传入的单个 server tier。"""
    if tier is None:
        return None
    if tier not in SERVER_TIERS:
        exit_with_message(
            "invalid_request",
            f"Unsupported server tier '{tier}'. Supported server tiers: {', '.join(SERVER_TIERS)}",
            "tier",
        )
    return cast(ServerTier, tier)


def api_server_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    upload_dir: str = typer.Option("", "--upload-dir", help="Upload directory"),
    tier: str | None = typer.Option(
        None,
        "--tier",
        help="Server capability tier: flash, basic, or standard",
    ),
    no_flash: bool = typer.Option(False, "--no-flash", help="Disable Flash tier advertisement and execution"),
    no_advanced: bool = typer.Option(False, "--no-advanced", help="Disable Advanced tier advertisement and execution"),
    concurrency: int = typer.Option(1, "--concurrency", help="Maximum concurrent parse jobs"),
    url_timeout: int = typer.Option(60, "--url-timeout", help="Timeout for URL source downloads"),
    allow_local_source: bool = typer.Option(False, "--allow-local-source", help="Allow local source paths"),
    max_inline_bytes: int = typer.Option(1024 * 1024, "--max-inline-bytes", help="Maximum decoded bytes for inline sources"),
    allow_http_source: bool = typer.Option(False, "--allow-http-source", help="Allow URL sources to use plain HTTP"),
    language: str = typer.Option(
        "ch",
        "--language",
        help="Hybrid medium OCR language hint; accepted by other efforts for compatibility",
    ),
    disable_image_analysis: bool = typer.Option(False, "--disable-image-analysis", help="Disable image analysis"),
    preload_models: bool = typer.Option(
        False, "--preload-models", help="Initialize VLM client and local Hybrid models at startup"
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="Optional fixed API key"),
    vlm_server_url: str | None = typer.Option(None, "--vlm-server-url", help="Remote VLM URL; empty value selects local VLM"),
    vlm_api_key: str | None = typer.Option(None, "--vlm-api-key", help="Bearer key for the remote VLM server"),
    vlm_model: str | None = typer.Option(None, "--vlm-model", help="Remote VLM model name; empty value enables discovery"),
    vlm_http_timeout: int | None = typer.Option(
        None, "--vlm-http-timeout", min=1, help="VLM HTTP timeout in seconds (default: 600)"
    ),
    vlm_max_concurrency: int | None = typer.Option(
        None, "--vlm-max-concurrency", min=1, help="VLM inference concurrency (default: 100)"
    ),
) -> None:
    """转发显式启动参数，启动 self-hosted MinerU 解析 API 服务。"""
    try:
        normalized_language = validate_public_ocr_lang(language)
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "language")
    tier_value = _normalize_tier_option(tier)
    tier_args = ["--tier", tier_value] if tier_value is not None else []
    vlm_args: list[str] = []
    for option, value in (
        ("--vlm-server-url", vlm_server_url),
        ("--vlm-api-key", vlm_api_key),
        ("--vlm-model", vlm_model),
        ("--vlm-http-timeout", vlm_http_timeout),
        ("--vlm-max-concurrency", vlm_max_concurrency),
    ):
        if value is not None:
            vlm_args.extend([option, str(value)])
    try:
        parser_api_server.main.main(
            args=[
                "--host",
                host,
                "--port",
                str(port),
                "--concurrency",
                str(concurrency),
                "--url-timeout",
                str(url_timeout),
                "--max-inline-bytes",
                str(max_inline_bytes),
                "--language",
                normalized_language,
                *tier_args,
                *vlm_args,
                *(["--no-flash"] if no_flash else []),
                *(["--no-advanced"] if no_advanced else []),
                *(["--allow-local-source"] if allow_local_source else []),
                *(["--allow-http-source"] if allow_http_source else []),
                *(["--upload-dir", upload_dir] if upload_dir else []),
                *(["--disable-image-analysis"] if disable_image_analysis else []),
                *(["--preload-models"] if preload_models else []),
                *(["--api-key", api_key] if api_key else []),
            ],
            prog_name="mineru-kit api-server",
            standalone_mode=False,
        )
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        raise typer.Exit(exc.code) from None
    except click.ClickException as exc:
        exit_with_message("invalid_request", exc.format_message())


def main() -> None:
    """配置标准流后，以独立命令运行新版 API 服务入口。"""
    configure_standard_streams()
    typer.run(api_server_cmd)


__all__ = ["API_SERVER_LANGUAGES", "api_server_cmd", "main"]
