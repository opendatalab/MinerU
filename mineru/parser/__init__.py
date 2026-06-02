# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from pathlib import Path

from .base import DocumentParser
from .html_parser import HtmlParser
from .office_parser import DocxParser, OfficeBaseParser, PptxParser, XlsxParser
from .parse_result import ParseResult
from .pdf_parser import PdfBaseParser, PdfHybridParser, PdfPipelineParser, PdfVlmParser
from .types import Block, Line, PageInfo, Span
from .api_parser import MinerUApiParser

__all__ = [
    "Block",
    "DocumentParser",
    "DocxParser",
    "HtmlParser",
    "Line",
    "MinerUApiParser",
    "OfficeBaseParser",
    "PageInfo",
    "ParseResult",
    "PdfBaseParser",
    "PdfHybridParser",
    "PdfPipelineParser",
    "PdfVlmParser",
    "PptxParser",
    "Span",
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
    start_page_id: int = 0,
    end_page_id: int | None = None,
    output_dir: str = "./output",
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
        parser = parser_cls(output_dir=output_dir)
    elif suffix in ("html", "htm"):
        parser = HtmlParser(output_dir=output_dir)
    elif backend == "pipeline":
        parser = PdfPipelineParser(
            backend=backend,
            method=method,
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            output_dir=output_dir,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    elif backend.startswith("vlm-"):
        parser = PdfVlmParser(
            backend=backend,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            image_analysis=image_analysis,
            output_dir=output_dir,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
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
            output_dir=output_dir,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return parser.parse(path)