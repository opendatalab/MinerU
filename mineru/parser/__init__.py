# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from pathlib import Path

from .api_client import MinerUApiParser
from .base import DocumentParser, ParseResult
from .html import HtmlParser
from .office import DocxParser, PptxParser, XlsxParser
from .pdf import PdfFlashParser, PdfHybridParser, PdfPipelineParser, PdfVlmParser

__all__ = [
    "DocumentParser",
    "DocxParser",
    "HtmlParser",
    "MinerUApiParser",
    "ParseResult",
    "PdfFlashParser",
    "PdfHybridParser",
    "PdfPipelineParser",
    "PdfVlmParser",
    "PptxParser",
    "XlsxParser",
    "parse",
]


def parse(
    path: str | Path,
    *,
    backend: str = "hybrid-auto-engine",
    method: str = "auto",
    lang: str = "ch",
    formula_enable: bool = True,
    table_enable: bool = True,
    image_analysis: bool = True,
    server_url: str | None = None,
    page_range: str = "",
) -> ParseResult:
    from ..utils.guess_suffix_or_lang import guess_suffix_by_path

    path = Path(path)
    suffix = guess_suffix_by_path(path)

    if suffix in ("docx", "pptx", "xlsx"):
        parser_cls: type[DocumentParser] = {
            "docx": DocxParser,
            "pptx": PptxParser,
            "xlsx": XlsxParser,
        }[suffix]
        parser = parser_cls()
    elif suffix in ("html", "htm"):
        parser = HtmlParser()
    elif backend == "pipeline":
        parser = PdfPipelineParser(
            backend=backend,
            method=method,
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )
    elif backend.startswith("vlm-"):
        parser = PdfVlmParser(
            backend=backend,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            image_analysis=image_analysis,
        )
    elif backend.startswith("hybrid-"):
        parser = PdfHybridParser(
            backend=backend,
            method=method,
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            image_analysis=image_analysis,
        )
    elif backend == "flash":
        parser = PdfFlashParser()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return parser.parse(path, page_range=page_range)
