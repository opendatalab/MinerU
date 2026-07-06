# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import InvalidRequestError
from ..types import PageInfo
from ..utils.backend_options import (
    CANONICAL_HYBRID_ENGINE,
    DEFAULT_HYBRID_EFFORT,
    LOCAL_HYBRID_EFFORT,
    MAX_HYBRID_EFFORT,
    resolve_backend_and_effort,
    validate_effort,
)
from ..utils.image_payload import ImagePayloadCache
from .base import DocumentParser, ParseResult

_IMAGE_SUFFIXES = frozenset({"png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"})


@dataclass
class _PreparedPdfInput:
    """记录 PDF 输入准备结果，避免跨文档复用 parser 实例状态。"""

    file_name: str
    pdf_bytes: bytes
    retained_page_indices: list[int] | None = None
    broken_page_indices: list[int] | None = None


def _resolve_hybrid_backend(backend: str, *, is_async: bool = False) -> str:
    """根据同步/异步调用形态解析公开 Hybrid backend 到具体执行 engine。"""
    resolved = backend[7:] if backend.startswith("hybrid-") else backend
    if resolved in {"engine", "auto-engine"}:
        from ..utils.engine_utils import get_vlm_engine

        resolved = get_vlm_engine(inference_engine="auto", is_async=is_async)
    return resolved


class PdfBaseParser(DocumentParser):
    _parse_method: str = ""
    _backend: str = ""

    def __init__(
        self,
        *,
        backend: str = "hybrid-engine",
        method: str = "auto",
        lang: str = "ch",
        effort: str = DEFAULT_HYBRID_EFFORT,
        image_analysis: bool = True,
        server_url: str | None = None,
    ):
        self.backend = backend
        self.method = method
        self.lang = lang
        self.effort = validate_effort(effort)
        self.image_analysis = image_analysis
        self.server_url = server_url

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        prepared = self._prepare_input(path, page_range)
        image_cache = ImagePayloadCache()
        middle_json = self._run_analysis(
            prepared.pdf_bytes,
            page_index_map=prepared.retained_page_indices,
            image_cache=image_cache,
        )
        self._insert_broken_pages(
            middle_json,
            prepared.retained_page_indices,
            prepared.broken_page_indices,
        )
        return self._build_result(
            middle_json,
            prepared.pdf_bytes,
            prepared.file_name,
            retained_page_indices=prepared.retained_page_indices,
            broken_page_indices=prepared.broken_page_indices,
            image_cache=image_cache,
        )

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        prepared = await asyncio.to_thread(self._prepare_input, path, page_range)
        image_cache = ImagePayloadCache()
        middle_json = await self._arun_analysis(
            prepared.pdf_bytes,
            page_index_map=prepared.retained_page_indices,
            image_cache=image_cache,
        )
        self._insert_broken_pages(
            middle_json,
            prepared.retained_page_indices,
            prepared.broken_page_indices,
        )
        return self._build_result(
            middle_json,
            prepared.pdf_bytes,
            prepared.file_name,
            retained_page_indices=prepared.retained_page_indices,
            broken_page_indices=prepared.broken_page_indices,
            image_cache=image_cache,
        )

    @abstractmethod
    def _run_analysis(
        self,
        pdf_bytes: bytes,
        page_index_map: list[int] | None = None,
        image_cache: ImagePayloadCache | None = None,
    ) -> list[PageInfo]:
        """Execute backend-specific analysis. Returns (middle_json, model_output)."""

    async def _arun_analysis(
        self,
        pdf_bytes: bytes,
        page_index_map: list[int] | None = None,
        image_cache: ImagePayloadCache | None = None,
    ) -> list[PageInfo]:
        return await asyncio.to_thread(
            self._run_analysis,
            pdf_bytes,
            page_index_map=page_index_map,
            image_cache=image_cache,
        )

    def _prepare_input(self, path: Path, page_range: str = "") -> _PreparedPdfInput:
        from ..utils.guess_suffix_or_lang import guess_suffix_by_path
        from ..utils.pdf_document import PDFDocument

        file_name = path.stem
        pdf_bytes = path.read_bytes()

        suffix = guess_suffix_by_path(path)
        if suffix in _IMAGE_SUFFIXES:
            pdf_bytes = PDFDocument.from_image(pdf_bytes).bytes

        pdf_bytes, retained_page_indices, broken_page_indices = self._maybe_adjust_pdf_bytes(
            pdf_bytes,
            suffix,
            page_range,
        )
        return _PreparedPdfInput(
            file_name=file_name,
            pdf_bytes=pdf_bytes,
            retained_page_indices=retained_page_indices,
            broken_page_indices=broken_page_indices,
        )

    def _maybe_adjust_pdf_bytes(
        self,
        pdf_bytes: bytes,
        suffix: str,
        page_range: str = "",
    ) -> tuple[bytes, list[int] | None, list[int] | None]:
        if suffix != "pdf":
            return pdf_bytes, None, None

        from ..utils.pdf_document import PDFDocument
        from ..utils.pdf_page_id import parse_page_range

        doc = PDFDocument(pdf_bytes)
        page_indices = parse_page_range(page_range, doc.page_count)
        if page_range.strip() and not page_indices:
            raise InvalidRequestError("page_range_invalid", f"Page range does not select any pages: {page_range}", "page_range")

        if page_indices == list(range(doc.page_count)):
            return pdf_bytes, None, None

        from ..utils.pdfium_guard import safe_rewrite_pdf_bytes_with_pdfium_result

        rewrite_result = safe_rewrite_pdf_bytes_with_pdfium_result(pdf_bytes, page_indices=page_indices)
        if rewrite_result.used_original:
            return rewrite_result.pdf_bytes or pdf_bytes, None, rewrite_result.broken_page_indices
        return (
            rewrite_result.pdf_bytes or pdf_bytes,
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

        backend = next((page._backend for page in pages if page._backend), None)
        pages_by_index = {page.page_idx: page for page in pages}
        ordered_page_indices = sorted(set(pages_by_index) | set(broken_page_indices))
        pages[:] = [
            pages_by_index.get(page_idx, PageInfo(page_idx=page_idx, _backend=backend)) for page_idx in ordered_page_indices
        ]

    def _build_result(
        self,
        middle_json: list[PageInfo],
        pdf_bytes: bytes,
        file_name: str,
        *,
        retained_page_indices: list[int] | None = None,
        broken_page_indices: list[int] | None = None,
        image_cache: ImagePayloadCache | None = None,
    ) -> ParseResult:
        return ParseResult(
            pages=middle_json,
            _retained_page_indices=retained_page_indices,
            _broken_page_indices=broken_page_indices,
            _image_cache=image_cache,
        )


class PdfHybridParser(PdfBaseParser):
    """PDF / image parser using the Hybrid local/VLM backend."""

    _parse_method: str = ""
    _backend = "hybrid"

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        self._parse_method = f"hybrid_{self.method}"
        return super().parse(path, page_range=page_range)

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        self._parse_method = f"hybrid_{self.method}"
        return await super().parse_async(path, page_range=page_range)

    def _run_analysis(
        self,
        pdf_bytes: bytes,
        page_index_map: list[int] | None = None,
        image_cache: ImagePayloadCache | None = None,
    ) -> list[PageInfo]:
        from ..backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze

        backend = (
            self.backend
            if self.effort == LOCAL_HYBRID_EFFORT
            else _resolve_hybrid_backend(self.backend, is_async=False)
        )
        server_url = self.server_url if backend.endswith("client") else None
        middle_json, model_list, _vlm_ocr_enable = hybrid_doc_analyze(
            pdf_bytes,
            backend=backend,
            parse_method=self.method,
            language=self.lang,
            effort=self.effort,
            server_url=server_url,
            image_analysis=self.image_analysis,
            page_index_map=page_index_map,
            image_cache=image_cache,
        )

        return middle_json

    async def _arun_analysis(
        self,
        pdf_bytes: bytes,
        page_index_map: list[int] | None = None,
        image_cache: ImagePayloadCache | None = None,
    ) -> list[PageInfo]:
        from ..backend.hybrid.hybrid_analyze import aio_doc_analyze as hybrid_aio_doc_analyze

        backend = (
            self.backend if self.effort == LOCAL_HYBRID_EFFORT else _resolve_hybrid_backend(self.backend, is_async=True)
        )
        server_url = self.server_url if backend.endswith("client") else None
        middle_json, model_list, _vlm_ocr_enable = await hybrid_aio_doc_analyze(
            pdf_bytes,
            backend=backend,
            parse_method=self.method,
            language=self.lang,
            effort=self.effort,
            server_url=server_url,
            image_analysis=self.image_analysis,
            page_index_map=page_index_map,
            image_cache=image_cache,
        )

        return middle_json


class PdfPipelineParser(PdfHybridParser):
    """保留旧 SDK 类名，内部统一委托 Hybrid medium 解析。"""

    def __init__(self, **kwargs: Any) -> None:
        """将旧 PdfPipelineParser 构造参数归一到 Hybrid medium，避免继续加载旧 backend。"""
        kwargs.pop("backend", None)
        kwargs.pop("effort", None)
        super().__init__(backend=CANONICAL_HYBRID_ENGINE, effort=LOCAL_HYBRID_EFFORT, **kwargs)


class PdfVlmParser(PdfHybridParser):
    """保留旧 SDK 类名，内部统一委托 Hybrid extra_high 解析。"""

    def __init__(self, **kwargs: Any) -> None:
        """将旧 PdfVlmParser 构造参数归一到 Hybrid extra_high，避免继续暴露独立 VLM backend。"""
        backend = kwargs.pop("backend", "vlm-engine")
        kwargs.pop("effort", None)
        resolved_backend, resolved_effort = resolve_backend_and_effort(backend, MAX_HYBRID_EFFORT)
        super().__init__(backend=resolved_backend, effort=resolved_effort, **kwargs)


class PdfFlashParser(PdfBaseParser):
    """PDF / image parser using the flash backend."""

    _backend = "flash"

    def _run_analysis(
        self,
        pdf_bytes: bytes,
        page_index_map: list[int] | None = None,
        image_cache: ImagePayloadCache | None = None,
    ) -> list[PageInfo]:
        from ..backend.flash.pdf_extractor import extract_pages_text
        from ..types import Block, Line, PageInfo, Span
        from ..utils.page_index import resolve_output_page_idx

        filepath = self._pdf_bytes_to_tempfile(pdf_bytes)
        try:
            pages_text = extract_pages_text(filepath)
        finally:
            try:
                os.unlink(filepath)
            except OSError:
                pass

        pages: list[PageInfo] = []
        block_idx = 0
        for index, pt in enumerate(pages_text):
            page_idx = resolve_output_page_idx(index, page_index_map)
            para_blocks: list[Block] = []
            if pt.strip():
                span = Span(type="text", bbox=(0.0, 0.0, 0.0, 0.0), content=pt.strip())
                line = Line(bbox=(0.0, 0.0, 0.0, 0.0), spans=[span])
                block = Block(index=block_idx, type="text", bbox=(0.0, 0.0, 0.0, 0.0), lines=[line])
                para_blocks.append(block)
                block_idx += 1
            page = PageInfo(
                page_idx=page_idx,
                para_blocks=para_blocks,
            )
            pages.append(page)

        return pages

    @staticmethod
    def _pdf_bytes_to_tempfile(pdf_bytes: bytes) -> str:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(pdf_bytes)
        tmp.close()
        return tmp.name
