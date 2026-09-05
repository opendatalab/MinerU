# Copyright (c) Opendatalab. All rights reserved.
"""统一文档解析器，委托 backend.analyze 处理 PDF、EPUB、HTML、OFD、图片、CSV 与 Office/RTF/ODF。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ..backend.analyze import aio_doc_analyze, doc_analyze
from ..config import VlmConfig, config
from ..errors import InvalidRequestError
from ..filetypes import IMAGE_EXTENSIONS, PAGE_RANGE_PARSE_EXTENSIONS
from ..model.flash.html import HtmlSourceContext
from ..types import FILE_SUFFIXES, FileSuffix, MiddleJson, ModelJson, PageInfo, Tier
from .tier import effort_for_tier
from .base import DocumentParser, ParseResult

logger = logging.getLogger(__name__)

_Effort = Literal["flash", "medium", "high", "xhigh"]
_ParseMode = Literal["auto", "txt", "ocr"]


@dataclass
class _PreparedInput:
    """记录文档输入准备结果，避免跨文档复用 parser 实例状态。"""

    file_name: str
    file_bytes: bytes
    file_suffix: FileSuffix
    source_context: HtmlSourceContext | None = None
    retained_page_indices: list[int] | None = None
    broken_page_indices: list[int] | None = None


class MinerUParser(DocumentParser):
    """统一文档解析器，支持 PDF、EPUB、HTML、OFD、图片、CSV 与 Office/RTF/ODF 文档。

    通过 file_suffix 路由到 backend.analyze 的统一 doc_analyze 入口，
    保留 PDF 输入的图片转 PDF、页范围重写和坏页补齐；其他格式仅支持整本解析。
    """

    def __init__(
        self,
        *,
        tier: Tier = "standard",
        parse_mode: _ParseMode = "auto",
        image_analysis: bool = True,
        vlm_config: VlmConfig | None = None,
    ) -> None:
        """保存当前解析器的 VLM 配置副本，避免其他应用或调用修改连接设置。"""
        self.tier: Tier = tier
        self.effort: _Effort = effort_for_tier(tier)  # type: ignore[assignment]
        self.parse_mode: _ParseMode = parse_mode
        self.image_analysis: bool = image_analysis
        self.vlm_config = (vlm_config if vlm_config is not None else config.model.vlm).model_copy(deep=True)

    def parse(
        self,
        path: str | Path,
        *,
        page_range: str = "",
        source_context: HtmlSourceContext | None = None,
    ) -> ParseResult:
        """解析本地路径，并允许内部调用方覆盖 HTML 原始来源上下文。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        prepared = self._prepare_input(path, page_range, source_context)
        middle_json, model_output = self._run_analysis(prepared)
        if prepared.file_suffix == "pdf":
            self._insert_broken_pages(
                middle_json.pages,
                prepared.retained_page_indices,
                prepared.broken_page_indices,
            )
        return self._build_result(middle_json, model_output)

    async def parse_async(
        self,
        path: str | Path,
        *,
        page_range: str = "",
        source_context: HtmlSourceContext | None = None,
    ) -> ParseResult:
        """异步解析本地路径，并保留 HTML 下载来源或本地资源根。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        prepared = await asyncio.to_thread(self._prepare_input, path, page_range, source_context)
        middle_json, model_output = await self._arun_analysis(prepared)
        if prepared.file_suffix == "pdf":
            self._insert_broken_pages(
                middle_json.pages,
                prepared.retained_page_indices,
                prepared.broken_page_indices,
            )
        return self._build_result(middle_json, model_output)

    def _run_analysis(self, prepared: _PreparedInput) -> tuple[MiddleJson, ModelJson]:
        """将准备后的输入和实例连接配置传入同步分析入口。"""
        return doc_analyze(
            prepared.file_bytes,
            effort=self.effort,
            parse_mode=self.parse_mode,
            image_analysis=self.image_analysis,
            page_index_map=prepared.retained_page_indices,
            file_suffix=prepared.file_suffix,
            source_context=prepared.source_context,
            vlm_config=self.vlm_config,
        )

    async def _arun_analysis(self, prepared: _PreparedInput) -> tuple[MiddleJson, ModelJson]:
        """将准备后的输入和实例连接配置传入异步分析入口。"""
        return await aio_doc_analyze(
            prepared.file_bytes,
            effort=self.effort,
            parse_mode=self.parse_mode,
            image_analysis=self.image_analysis,
            page_index_map=prepared.retained_page_indices,
            file_suffix=prepared.file_suffix,
            source_context=prepared.source_context,
            vlm_config=self.vlm_config,
        )

    def _prepare_input(
        self,
        path: Path,
        page_range: str = "",
        source_context: HtmlSourceContext | None = None,
    ) -> _PreparedInput:
        """读取路径、检测类型并补齐 HTML 来源或 PDF 页范围状态。"""
        from .file_type import guess_suffix_by_path
        from .page_range import normalize_page_range_input

        page_range = normalize_page_range_input(page_range)

        file_name = path.stem
        file_bytes = path.read_bytes()
        suffix = guess_suffix_by_path(path)
        source_suffix = suffix

        if suffix in IMAGE_EXTENSIONS:
            from ..model.flash.pdf.document import PDFDocument

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

        if suffix not in FILE_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix or path.suffix or 'unknown'}")
        if source_suffix not in PAGE_RANGE_PARSE_EXTENSIONS and page_range.strip():
            raise InvalidRequestError(
                "page_range_invalid",
                f"Page range is only supported for PDF files; '{source_suffix}' uses full-document parsing.",
                "page_range",
            )
        file_bytes, retained_page_indices, broken_page_indices = self._maybe_adjust_pdf_bytes(
            file_bytes,
            suffix,
            page_range,
        )
        resolved_source_context = source_context
        if suffix == "html" and resolved_source_context is None:
            from ..model.flash.html import HtmlSourceContext

            resolved_path = path.resolve()
            resolved_source_context = HtmlSourceContext(
                source_uri=resolved_path.as_uri(),
                local_resource_root=resolved_path.parent,
            )
        return _PreparedInput(
            file_name=file_name,
            file_bytes=file_bytes,
            file_suffix=cast(FileSuffix, suffix),
            source_context=resolved_source_context,
            retained_page_indices=retained_page_indices,
            broken_page_indices=broken_page_indices,
        )

    def _maybe_adjust_pdf_bytes(
        self,
        file_bytes: bytes,
        suffix: str,
        page_range: str = "",
    ) -> tuple[bytes, list[int] | None, list[int] | None]:
        """仅 PDF 走页范围重写；其他格式直接返回原字节。"""
        if suffix != "pdf":
            return file_bytes, None, None

        from ..model.flash.pdf.document import PDFDocument
        from .page_range import parse_page_range

        with PDFDocument(file_bytes) as doc:
            page_count = doc.page_count
        page_indices = parse_page_range(page_range, page_count)

        if page_indices == list(range(page_count)):
            return file_bytes, None, None

        from ..model.flash.pdf.pdfium import safe_rewrite_pdf_bytes_with_pdfium_result

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
        pages[:] = [pages_by_index.get(page_idx, PageInfo(page_idx=page_idx)) for page_idx in ordered_page_indices]

    def _build_result(
        self,
        middle_json: MiddleJson,
        model_output: Any = None,
    ) -> ParseResult:
        return ParseResult(
            middle_json=middle_json,
            _model_output=model_output,
        )


__all__ = ["MinerUParser"]
