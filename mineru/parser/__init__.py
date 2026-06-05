# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from pathlib import Path

from mineru.types import Block, Line, PageInfo, Span

from .api_parser import MinerUApiParser
from .base import DocumentParser
from .html_parser import HtmlParser
from .office_parser import DocxParser, OfficeBaseParser, PptxParser, XlsxParser
from .parse_result import ParseResult
from .pdf_parser import PdfBaseParser, PdfHybridParser, PdfPipelineParser, PdfVlmParser

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
    elif backend == "flash":
        return _parse_flash(path, start_page_id, end_page_id, output_dir)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return parser.parse(path)


def _parse_flash(path: str | Path, start_page_id: int, end_page_id: int | None, output_dir: str = "./output") -> ParseResult:
    """Flash parse — extract text via pypdfium2 and build proper ParseResult."""
    from pathlib import Path as P

    from mineru.types import Block, Line, PageInfo, Span

    from .. import version
    from ..backend.flash.pdf_extractor import extract_text

    path = P(path)
    text = extract_text(str(path), start_page=start_page_id, end_page=end_page_id)
    pages_text = text.split("\n\n")  # each page separated by double newline

    page_count = len(pages_text)
    current_page = start_page_id

    pages: list[PageInfo] = []
    for pt in pages_text:
        if not pt.strip():
            continue
        block = Block(
            type="text",
            lines=[Line(spans=[Span(type="text", content=pt.strip())])],
        )
        page = PageInfo(
            page_idx=current_page,
            para_blocks=[block],
        )
        pages.append(page)
        current_page += 1

    return ParseResult(
        pages=pages,
        _backend="flash",
        _version_name=version.__version__,
        _file_name=path.stem,
    )
