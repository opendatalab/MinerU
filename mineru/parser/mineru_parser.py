# Copyright (c) Opendatalab. All rights reserved.
"""统一文档解析器，委托 mineru.backend.analyze 处理 PDF/图片/DOCX/PPTX/XLSX。"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..backend.analyze import aio_doc_analyze, doc_analyze
from ..errors import InvalidRequestError
from ..filetypes import IMAGE_EXTENSIONS
from ..types import MiddleJson, PageInfo, Tier
from ..utils.backend_options import effort_for_tier
from .base import DocumentParser, ParseResult

logger = logging.getLogger(__name__)

_Effort = Literal["flash", "low", "medium", "high", "xhigh"]
_ParseMode = Literal["auto", "txt", "ocr"]
_FileSuffix = Literal["pdf", "docx", "pptx", "xlsx"]
_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({"pdf", "docx", "pptx", "xlsx"})


@dataclass
class _PreparedInput:
    """记录文档输入准备结果，避免跨文档复用 parser 实例状态。"""

    file_name: str
    file_bytes: bytes
    file_suffix: _FileSuffix
    retained_page_indices: list[int] | None = None
    broken_page_indices: list[int] | None = None


class MinerUParser(DocumentParser):
    """统一文档解析器，支持 PDF/图片/DOCX/PPTX/XLSX。

    通过 file_suffix 路由到 backend.analyze 的统一 doc_analyze 入口，
    保留 PDF 输入的图片转 PDF、页范围重写、坏页补齐等预处理逻辑。
    """

    def __init__(
        self,
        *,
        tier: Tier = "standard",
        parse_mode: _ParseMode = "auto",
        image_analysis: bool = True,
    ) -> None:
        self.tier: Tier = tier
        self.effort: _Effort = effort_for_tier(tier)  # type: ignore[assignment]
        self.parse_mode: _ParseMode = parse_mode
        self.image_analysis: bool = image_analysis

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        prepared = self._prepare_input(path, page_range)
        middle_json, model_output = self._run_analysis(prepared)
        if prepared.file_suffix == "pdf":
            self._insert_broken_pages(
                middle_json.pages,
                prepared.retained_page_indices,
                prepared.broken_page_indices,
            )
        return self._build_result(
            middle_json,
            model_output,
            retained_page_indices=prepared.retained_page_indices,
            broken_page_indices=prepared.broken_page_indices,
        )

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        prepared = await asyncio.to_thread(self._prepare_input, path, page_range)
        middle_json, model_output = await self._arun_analysis(prepared)
        if prepared.file_suffix == "pdf":
            self._insert_broken_pages(
                middle_json.pages,
                prepared.retained_page_indices,
                prepared.broken_page_indices,
            )
        return self._build_result(
            middle_json,
            model_output,
            retained_page_indices=prepared.retained_page_indices,
            broken_page_indices=prepared.broken_page_indices,
        )

    def _run_analysis(self, prepared: _PreparedInput) -> tuple[MiddleJson, Any]:
        return doc_analyze(
            prepared.file_bytes,
            effort=self.effort,
            parse_mode=self.parse_mode,
            image_analysis=self.image_analysis,
            page_index_map=prepared.retained_page_indices,
            file_suffix=prepared.file_suffix,
        )

    async def _arun_analysis(self, prepared: _PreparedInput) -> tuple[MiddleJson, Any]:
        return await aio_doc_analyze(
            prepared.file_bytes,
            effort=self.effort,
            parse_mode=self.parse_mode,
            image_analysis=self.image_analysis,
            page_index_map=prepared.retained_page_indices,
            file_suffix=prepared.file_suffix,
        )

    def _prepare_input(self, path: Path, page_range: str = "") -> _PreparedInput:
        from ..utils.guess_suffix_or_lang import guess_suffix_by_path
        from ..utils.pdf_document import PDFDocument

        file_name = path.stem
        file_bytes = path.read_bytes()
        suffix = guess_suffix_by_path(path)

        if suffix in IMAGE_EXTENSIONS:
            conversion_started_at = time.perf_counter()
            input_size = len(file_bytes)
            logger.debug(
                "Image input conversion started filename=%s suffix=%s input_bytes=%d",
                path.name,
                suffix,
                input_size,
            )
            file_bytes = PDFDocument.from_image(file_bytes).bytes
            logger.debug(
                "Image input conversion completed filename=%s suffix=%s input_bytes=%d pdf_bytes=%d elapsed_ms=%d",
                path.name,
                suffix,
                input_size,
                len(file_bytes),
                round((time.perf_counter() - conversion_started_at) * 1000),
            )
            suffix = "pdf"

        file_bytes, retained_page_indices, broken_page_indices = self._maybe_adjust_pdf_bytes(
            file_bytes,
            suffix,
            page_range,
        )
        if suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix or path.suffix or 'unknown'}")
        return _PreparedInput(
            file_name=file_name,
            file_bytes=file_bytes,
            file_suffix=suffix,  # type: ignore[arg-type]
            retained_page_indices=retained_page_indices,
            broken_page_indices=broken_page_indices,
        )

    def _maybe_adjust_pdf_bytes(
        self,
        file_bytes: bytes,
        suffix: str,
        page_range: str = "",
    ) -> tuple[bytes, list[int] | None, list[int] | None]:
        """仅 PDF 走页范围重写；Office 文件直接返回原字节。"""
        if suffix != "pdf":
            return file_bytes, None, None

        from ..utils.pdf_document import PDFDocument
        from ..utils.pdf_page_id import parse_page_range

        doc = PDFDocument(file_bytes)
        page_indices = parse_page_range(page_range, doc.page_count)
        if page_range.strip() and not page_indices:
            raise InvalidRequestError(
                "page_range_invalid",
                f"Page range does not select any pages: {page_range}",
                "page_range",
            )

        if page_indices == list(range(doc.page_count)):
            return file_bytes, None, None

        from ..utils.pdfium_guard import safe_rewrite_pdf_bytes_with_pdfium_result

        rewrite_result = safe_rewrite_pdf_bytes_with_pdfium_result(file_bytes, page_indices=page_indices)
        if rewrite_result.used_original:
            return rewrite_result.pdf_bytes or file_bytes, None, rewrite_result.broken_page_indices
        return (
            rewrite_result.pdf_bytes or file_bytes,
            rewrite_result.retained_page_indices,
            rewrite_result.broken_page_indices,
        )

    def _insert_broken_pages(
        self,
        pages: list[PageInfo],
        retained_page_indices: list[int] | None = None,
        broken_page_indices: list[int] | None = None,
    ) -> None:
        """按 PDF 重写结果补齐坏页空占位，不再修改 backend 已生成的页号。"""
        if retained_page_indices is None or not broken_page_indices:
            return

        pages_by_index = {page.page_idx: page for page in pages}
        ordered_page_indices = sorted(set(pages_by_index) | set(broken_page_indices))
        pages[:] = [
            pages_by_index.get(page_idx, PageInfo(page_idx=page_idx))
            for page_idx in ordered_page_indices
        ]

    def _build_result(
        self,
        middle_json: MiddleJson,
        model_output: Any = None,
        *,
        retained_page_indices: list[int] | None = None,
        broken_page_indices: list[int] | None = None,
    ) -> ParseResult:
        return ParseResult(
            middle_json=middle_json,
            _model_output=model_output,
            _retained_page_indices=retained_page_indices,
            _broken_page_indices=broken_page_indices,
        )


__all__ = ["MinerUParser"]
