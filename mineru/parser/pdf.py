# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import PageInfo
from .base import DocumentParser, ParseResult

_IMAGE_SUFFIXES = frozenset({"png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"})


@dataclass
class _PreparedPdfInput:
    """记录 PDF 输入准备结果，避免跨文档复用 parser 实例状态。"""

    file_name: str
    pdf_bytes: bytes
    retained_page_indices: list[int] | None = None
    broken_page_indices: list[int] | None = None


def _resolve_vlm_backend(backend: str) -> str:
    """Strip the ``vlm-`` prefix and resolve ``auto-engine`` to the concrete engine."""
    if not backend.startswith("vlm-"):
        return backend
    resolved = backend[4:]
    if resolved == "auto-engine":
        from ..utils.engine_utils import get_vlm_engine

        resolved = get_vlm_engine(inference_engine="auto", is_async=False)
    return resolved


def _resolve_hybrid_backend(backend: str) -> str:
    """Strip the ``hybrid-`` prefix and resolve ``auto-engine`` to the concrete engine."""
    resolved = backend[7:] if backend.startswith("hybrid-") else backend
    if resolved == "auto-engine":
        from ..utils.engine_utils import get_vlm_engine

        resolved = get_vlm_engine(inference_engine="auto", is_async=False)
    return resolved


class PdfBaseParser(DocumentParser):
    _parse_method: str = ""
    _backend: str = ""

    def __init__(
        self,
        *,
        backend: str = "hybrid-auto-engine",
        method: str = "auto",
        lang: str = "ch",
        formula_enable: bool = True,
        table_enable: bool = True,
        image_analysis: bool = True,
        server_url: str | None = None,
    ):
        self.backend = backend
        self.method = method
        self.lang = lang
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.image_analysis = image_analysis
        self.server_url = server_url

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        prepared = self._prepare_input(path, page_range)
        middle_json = self._run_analysis(prepared.pdf_bytes, image_writer=None)
        self._fix_page_indices(
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
        )

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        prepared = await asyncio.to_thread(self._prepare_input, path, page_range)
        middle_json = await self._arun_analysis(prepared.pdf_bytes, image_writer=None)
        self._fix_page_indices(
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
        )

    @abstractmethod
    def _run_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        """Execute backend-specific analysis. Returns (middle_json, model_output)."""

    async def _arun_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        return await asyncio.to_thread(self._run_analysis, pdf_bytes, image_writer)

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

    def _fix_page_indices(
        self,
        pages: list[PageInfo],
        retained_page_indices: list[int] | None = None,
        broken_page_indices: list[int] | None = None,
    ) -> None:
        """按 PDF 重写结果修正原始页号，并为跳过的损坏页补空页面。"""
        if retained_page_indices is None:
            return

        fixed_pages: list[PageInfo] = []
        for i, p in enumerate(pages):
            if i < len(retained_page_indices):
                p.page_idx = retained_page_indices[i]
            fixed_pages.append(p)

        if not broken_page_indices:
            pages[:] = fixed_pages
            return

        pages_by_index = {page.page_idx: page for page in fixed_pages}
        ordered_page_indices = sorted(set(pages_by_index) | set(broken_page_indices))
        pages[:] = [
            pages_by_index.get(page_idx, PageInfo(page_idx=page_idx))
            for page_idx in ordered_page_indices
        ]

    def _build_result(
        self,
        middle_json: list[PageInfo],
        pdf_bytes: bytes,
        file_name: str,
        *,
        retained_page_indices: list[int] | None = None,
        broken_page_indices: list[int] | None = None,
    ) -> ParseResult:
        return ParseResult(
            pages=middle_json,
            _retained_page_indices=retained_page_indices,
            _broken_page_indices=broken_page_indices,
        )


class PdfVlmParser(PdfBaseParser):
    """PDF / image parser using the VLM (Vision Language Model) backend."""

    _parse_method = "vlm"
    _backend = "vlm"

    def _run_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        from ..backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze

        backend = _resolve_vlm_backend(self.backend)
        os.environ["MINERU_VLM_FORMULA_ENABLE"] = str(self.formula_enable).lower()
        os.environ["MINERU_VLM_TABLE_ENABLE"] = str(self.table_enable).lower()

        return vlm_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            server_url=self.server_url,
            image_analysis=self.image_analysis,
        )[0]

    async def _arun_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        from ..backend.vlm.vlm_analyze import aio_doc_analyze as vlm_aio_doc_analyze

        backend = _resolve_vlm_backend(self.backend)
        os.environ["MINERU_VLM_FORMULA_ENABLE"] = str(self.formula_enable).lower()
        os.environ["MINERU_VLM_TABLE_ENABLE"] = str(self.table_enable).lower()

        middle_json, _ = await vlm_aio_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            server_url=self.server_url,
            image_analysis=self.image_analysis,
        )
        return middle_json


class PdfPipelineParser(PdfBaseParser):
    """PDF / image parser using the pipeline (CV+OCR) backend."""

    _parse_method: str = ""
    _backend = "pipeline"

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        self._parse_method = self.method
        return super().parse(path, page_range=page_range)

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        self._parse_method = self.method
        return await super().parse_async(path, page_range=page_range)

    def parse_batch(self, paths: list[str | Path], *, page_range: str = "") -> list[ParseResult]:
        self._parse_method = self.method

        from ..backend.pipeline.pipeline_analyze import doc_analyze_streaming

        prepared_inputs: list[_PreparedPdfInput] = []
        pdf_bytes_list: list[bytes] = []
        for p in paths:
            prepared = self._prepare_input(Path(p), page_range)
            prepared_inputs.append(prepared)
            pdf_bytes_list.append(prepared.pdf_bytes)

        results_by_index: dict[int, list[PageInfo]] = {}

        def on_doc_ready(doc_index: int, model_list: Any, middle_json: list[PageInfo], _ocr_enable: bool) -> None:
            results_by_index[doc_index] = middle_json

        doc_analyze_streaming(
            pdf_bytes_list,
            [None] * len(paths),
            [self.lang] * len(paths),
            on_doc_ready,
            parse_method=self.method,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
        )

        parse_results: list[ParseResult] = []
        for idx in range(len(paths)):
            middle_json = results_by_index[idx]
            prepared = prepared_inputs[idx]
            self._fix_page_indices(
                middle_json,
                prepared.retained_page_indices,
                prepared.broken_page_indices,
            )
            result = self._build_result(
                middle_json,
                prepared.pdf_bytes,
                prepared.file_name,
                retained_page_indices=prepared.retained_page_indices,
                broken_page_indices=prepared.broken_page_indices,
            )
            parse_results.append(result)

        return parse_results

    async def parse_batch_async(self, paths: list[str | Path], *, page_range: str = "") -> list[ParseResult]:
        import asyncio

        return await asyncio.to_thread(self.parse_batch, paths, page_range=page_range)

    def _run_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        from ..backend.pipeline.pipeline_analyze import doc_analyze_streaming

        result_holder: dict = {}

        def on_doc_ready(_doc_index, model_list, middle_json, _ocr_enable):
            result_holder["middle_json"] = middle_json

        doc_analyze_streaming(
            [pdf_bytes],
            [image_writer],
            [self.lang],
            on_doc_ready,
            parse_method=self.method,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
        )

        return result_holder["middle_json"]


class PdfHybridParser(PdfBaseParser):
    """PDF / image parser using the hybrid (pipeline + VLM) backend."""

    _parse_method: str = ""
    _backend = "hybrid"

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        self._parse_method = f"hybrid_{self.method}"
        return super().parse(path, page_range=page_range)

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        self._parse_method = f"hybrid_{self.method}"
        return await super().parse_async(path, page_range=page_range)

    def _run_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        from ..backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze

        backend = _resolve_hybrid_backend(self.backend)
        server_url = self.server_url if backend.endswith("client") else None
        os.environ["MINERU_VLM_FORMULA_ENABLE"] = "true"
        os.environ["MINERU_VLM_TABLE_ENABLE"] = str(self.table_enable).lower()

        middle_json, model_list, _vlm_ocr_enable = hybrid_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            parse_method=self.method,
            language=self.lang,
            inline_formula_enable=self.formula_enable,
            server_url=server_url,
            image_analysis=self.image_analysis,
        )

        return middle_json

    async def _arun_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        from ..backend.hybrid.hybrid_analyze import aio_doc_analyze as hybrid_aio_doc_analyze

        backend = _resolve_hybrid_backend(self.backend)
        server_url = self.server_url if backend.endswith("client") else None
        os.environ["MINERU_VLM_FORMULA_ENABLE"] = "true"
        os.environ["MINERU_VLM_TABLE_ENABLE"] = str(self.table_enable).lower()

        middle_json, model_list, _vlm_ocr_enable = await hybrid_aio_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            parse_method=self.method,
            language=self.lang,
            inline_formula_enable=self.formula_enable,
            server_url=server_url,
            image_analysis=self.image_analysis,
        )

        return middle_json


class PdfFlashParser(PdfBaseParser):
    """PDF / image parser using the flash (CPU-only pypdfium2) backend."""

    _backend = "flash"

    def _run_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        from ..backend.flash.pdf_extractor import extract_pages_text
        from ..types import Block, Line, PageInfo, Span

        filepath = self._pdf_bytes_to_tempfile(pdf_bytes)
        pages_text = extract_pages_text(filepath)

        pages: list[PageInfo] = []
        block_idx = 0
        for index, pt in enumerate(pages_text):
            para_blocks: list[Block] = []
            if pt.strip():
                span = Span(type="text", bbox=(0.0, 0.0, 0.0, 0.0), content=pt.strip())
                line = Line(bbox=(0.0, 0.0, 0.0, 0.0), spans=[span])
                block = Block(index=block_idx, type="text", bbox=(0.0, 0.0, 0.0, 0.0), lines=[line])
                para_blocks.append(block)
                block_idx += 1
            page = PageInfo(
                page_idx=index,
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
