# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from pathlib import Path

from ..filetypes import ensure_tier_supported_for_parse_extension
from ..types import Tier
from .api_client import MinerUApiParser
from .base import MIDDLE_JSON_SCHEMA_VERSION, DocumentParser, ParseResult
from .html import HtmlParser
from .office import DocxParser, PptxParser, XlsxParser
from .pdf import PdfFlashParser, PdfHybridParser, PdfPipelineParser, PdfVlmParser
from .tier import PARSER_BACKENDS, backend_for_tier, resolve_runtime_options, resolve_tier_and_backend
from ..utils.backend_options import DEFAULT_HYBRID_EFFORT

__all__ = [
    "backend_for_tier",
    "PARSER_BACKENDS",
    "DocumentParser",
    "DocxParser",
    "HtmlParser",
    "MinerUApiParser",
    "MIDDLE_JSON_SCHEMA_VERSION",
    "ParseResult",
    "PdfFlashParser",
    "PdfHybridParser",
    "PdfPipelineParser",
    "PdfVlmParser",
    "PptxParser",
    "XlsxParser",
    "parse",
    "parse_async",
    "resolve_tier_and_backend",
]

_OFFICE_SUFFIXES = frozenset({"docx", "pptx", "xlsx"})
_HTML_SUFFIXES = frozenset({"html", "htm"})
_PDF_INPUT_SUFFIXES = frozenset({"pdf", "png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"})
_PATH_PRIORITY_SUFFIXES = _OFFICE_SUFFIXES | _HTML_SUFFIXES | _PDF_INPUT_SUFFIXES


def _resolve_input_suffix(path: Path) -> str:
    from ..utils.guess_suffix_or_lang import guess_suffix_by_path

    guessed_suffix = guess_suffix_by_path(path)
    path_suffix = path.suffix.lower().lstrip(".")
    if path_suffix in _PATH_PRIORITY_SUFFIXES:
        return path_suffix
    return guessed_suffix


def _build_parser(
    path: str | Path,
    *,
    tier: Tier | None = None,
    backend: str | None = None,
    language: str = "ch",
    ocr_mode: str = "auto",
    effort: str = DEFAULT_HYBRID_EFFORT,
    disable_image_analysis: bool = False,
    server_url: str | None = None,
    method: str | None = None,
    lang: str | None = None,
    image_analysis: bool | None = None,
) -> DocumentParser:
    path = Path(path)
    suffix = _resolve_input_suffix(path)
    resolved_ocr_mode = method or ocr_mode
    resolved_language = lang or language
    resolved_image_analysis = (not disable_image_analysis) if image_analysis is None else image_analysis

    if suffix in _OFFICE_SUFFIXES:
        if tier is not None or backend is not None:
            runtime = resolve_runtime_options(tier=tier, backend=backend, effort=effort)
            ensure_tier_supported_for_parse_extension(runtime.tier, suffix)
        parser_cls: type[DocumentParser] = {
            "docx": DocxParser,
            "pptx": PptxParser,
            "xlsx": XlsxParser,
        }[suffix]
        return parser_cls()
    elif suffix in _HTML_SUFFIXES:
        if tier is not None or backend is not None:
            runtime = resolve_runtime_options(tier=tier, backend=backend, effort=effort)
            ensure_tier_supported_for_parse_extension(runtime.tier, suffix)
        return HtmlParser()

    if suffix not in _PDF_INPUT_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or path.suffix or 'unknown'}")

    runtime = resolve_runtime_options(tier=tier, backend=backend, effort=effort)
    if runtime.backend.startswith("hybrid-"):
        return PdfHybridParser(
            backend=runtime.backend,
            method=resolved_ocr_mode,
            lang=resolved_language,
            server_url=server_url,
            image_analysis=resolved_image_analysis,
            effort=runtime.effort,
        )
    elif runtime.backend == "flash":
        return PdfFlashParser()
    else:
        raise ValueError(f"Unknown backend: {runtime.backend}")


def parse(
    path: str | Path,
    *,
    tier: Tier | None = None,
    backend: str | None = None,
    language: str = "ch",
    ocr_mode: str = "auto",
    effort: str = DEFAULT_HYBRID_EFFORT,
    disable_image_analysis: bool = False,
    server_url: str | None = None,
    page_range: str = "",
    method: str | None = None,
    lang: str | None = None,
    image_analysis: bool | None = None,
) -> ParseResult:
    parser = _build_parser(
        path,
        tier=tier,
        backend=backend,
        language=language,
        ocr_mode=ocr_mode,
        effort=effort,
        disable_image_analysis=disable_image_analysis,
        server_url=server_url,
        method=method,
        lang=lang,
        image_analysis=image_analysis,
    )
    return parser.parse(path, page_range=page_range)


async def parse_async(
    path: str | Path,
    *,
    tier: Tier | None = None,
    backend: str | None = None,
    language: str = "ch",
    ocr_mode: str = "auto",
    effort: str = DEFAULT_HYBRID_EFFORT,
    disable_image_analysis: bool = False,
    server_url: str | None = None,
    page_range: str = "",
    method: str | None = None,
    lang: str | None = None,
    image_analysis: bool | None = None,
) -> ParseResult:
    parser = _build_parser(
        path,
        tier=tier,
        backend=backend,
        language=language,
        ocr_mode=ocr_mode,
        effort=effort,
        disable_image_analysis=disable_image_analysis,
        server_url=server_url,
        method=method,
        lang=lang,
        image_analysis=image_analysis,
    )
    return await parser.parse_async(path, page_range=page_range)
