from __future__ import annotations

from typing import Literal

import typer

from ...parser import api_server as parser_api_server
from ...parser.tier import PARSER_BACKENDS
from ...utils.backend_options import normalize_public_backend
from ..errors import exit_with_message

API_SERVER_BACKENDS = tuple(backend for backend in PARSER_BACKENDS if backend != "flash")
API_SERVER_LANGUAGES = (
    "ch",
    "ch_server",
    "ch_lite",
    "en",
    "korean",
    "japan",
    "chinese_cht",
    "ta",
    "te",
    "ka",
    "th",
    "el",
    "latin",
    "arabic",
    "east_slavic",
    "cyrillic",
    "devanagari",
)


def api_server_cmd(
    host: str = "127.0.0.1",
    port: int = 8000,
    upload_dir: str = "",
    backend: str | None = None,
    tier: Literal["standard", "pro"] | None = None,
    concurrency: int = 1,
    url_timeout: int = 60,
    max_wait: int = 600,
    language: str = "ch",
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
    effort: Literal["medium", "high"] = "medium",
    disable_table: bool = False,
    disable_formula: bool = False,
    disable_image_analysis: bool = False,
    api_key: str | None = None,
) -> None:
    try:
        normalized_backend = normalize_public_backend(backend) if backend is not None else None
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "backend")
    if language not in API_SERVER_LANGUAGES:
        exit_with_message("invalid_request", f"Unsupported language '{language}'.", "language")
    effective_tier = "standard" if tier is None and normalized_backend is None else tier
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
                language,
                "--ocr-mode",
                ocr_mode,
                "--effort",
                effort,
                *(["--tier", effective_tier] if effective_tier else []),
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
