from __future__ import annotations

from typing import cast

import click
import typer

from ...parser import api_server as parser_api_server
from ...types import SERVER_TIERS, ServerTier
from ...utils.ocr_language import PUBLIC_OCR_LANGUAGES, validate_public_ocr_lang
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
    ocr_mode: str = typer.Option("auto", "--ocr-mode", help="OCR mode: auto, txt, ocr"),
    disable_image_analysis: bool = typer.Option(False, "--disable-image-analysis", help="Disable image analysis"),
    preload_models: bool = typer.Option(False, "--preload-models", help="Load local models during server startup"),
    api_key: str | None = typer.Option(None, "--api-key", help="Optional fixed API key"),
) -> None:
    """Start the self-hosted MinerU parse API server."""
    try:
        normalized_language = validate_public_ocr_lang(language)
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "language")
    tier_value = _normalize_tier_option(tier)
    tier_args = ["--tier", tier_value] if tier_value is not None else []
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
                "--ocr-mode",
                ocr_mode,
                *tier_args,
                *(["--no-flash"] if no_flash else []),
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
