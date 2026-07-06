from __future__ import annotations

from typing import Literal, Sequence

import typer

from ...parser import api_server as parser_api_server
from ...parser.tier import PARSER_BACKENDS
from ...utils.backend_options import DEFAULT_HYBRID_EFFORT, resolve_backend_and_effort
from ...utils.ocr_language import PUBLIC_OCR_LANGUAGES, validate_public_ocr_lang
from ..errors import exit_with_message

API_SERVER_BACKENDS = tuple(backend for backend in PARSER_BACKENDS if backend != "flash")
API_SERVER_LANGUAGES = PUBLIC_OCR_LANGUAGES
API_SERVER_TIERS = ("standard", "pro")


def _normalize_tier_options(tier: Sequence[str] | str | None) -> list[str]:
    """规范化 mineru-kit 传入的 tier 参数，支持 Typer 重复选项和旧单值调用。"""
    if tier is None:
        return []
    raw_tiers = [tier] if isinstance(tier, str) else list(tier)
    tiers: list[str] = []
    for item in raw_tiers:
        if item not in API_SERVER_TIERS:
            exit_with_message(
                "invalid_request",
                f"Unsupported tier '{item}'. Supported tiers: {', '.join(API_SERVER_TIERS)}",
                "tier",
            )
        if item not in tiers:
            tiers.append(item)
    return tiers


def api_server_cmd(
    host: str = "127.0.0.1",
    port: int = 8000,
    upload_dir: str = "",
    backend: str | None = None,
    tier: Sequence[Literal["standard", "pro"]] | Literal["standard", "pro"] | None = None,
    concurrency: int = 1,
    url_timeout: int = 60,
    max_wait: int = 600,
    language: str = "ch",
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
    effort: Literal["medium", "high", "extra_high"] = DEFAULT_HYBRID_EFFORT,
    disable_table: bool = False,
    disable_formula: bool = False,
    disable_image_analysis: bool = False,
    api_key: str | None = None,
) -> None:
    try:
        normalized_backend, normalized_effort = (
            resolve_backend_and_effort(backend, effort) if backend is not None else (None, effort)
        )
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "backend")
    try:
        normalized_language = validate_public_ocr_lang(language)
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "language")
    tier_values = _normalize_tier_options(tier)
    effective_tiers = ["standard"] if not tier_values and normalized_backend is None else tier_values
    tier_args = [arg for tier_value in effective_tiers for arg in ("--tier", tier_value)]
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
                "--max-wait",
                str(max_wait),
                "--language",
                normalized_language,
                "--ocr-mode",
                ocr_mode,
                "--effort",
                normalized_effort,
                *tier_args,
                *(["--upload-dir", upload_dir] if upload_dir else []),
                *(["--backend", normalized_backend] if normalized_backend else []),
                *(["--disable-table"] if disable_table else []),
                *(["--disable-formula"] if disable_formula else []),
                *(["--disable-image-analysis"] if disable_image_analysis else []),
                *(["--api-key", api_key] if api_key else []),
            ],
            prog_name="mineru-kit api-server",
            standalone_mode=False,
        )
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        raise typer.Exit(exc.code) from None
