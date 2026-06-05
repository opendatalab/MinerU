# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any

from mineru.types import PageInfo

from .base import DocumentParser
from .parse_result import ParseResult

_IMAGE_SUFFIXES = frozenset({"png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"})


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

    def parse(self, path: str | Path) -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        file_name, pdf_bytes = self._prepare_input(path)
        middle_json = self._run_analysis(pdf_bytes, image_writer=None)
        return self._build_result(middle_json, pdf_bytes, file_name)

    async def parse_async(self, path: str | Path) -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        file_name, pdf_bytes = await asyncio.to_thread(self._prepare_input, path)
        middle_json = await self._arun_analysis(pdf_bytes, image_writer=None)
        return self._build_result(middle_json, pdf_bytes, file_name)

    @abstractmethod
    def _run_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        """Execute backend-specific analysis. Returns (middle_json, model_output)."""

    async def _arun_analysis(self, pdf_bytes: bytes, image_writer: Any) -> list[PageInfo]:
        return await asyncio.to_thread(self._run_analysis, pdf_bytes, image_writer)

    def _prepare_input(self, path: Path) -> tuple[str, bytes]:
        from ..utils.guess_suffix_or_lang import guess_suffix_by_path
        from ..utils.pdf_document import PDFDocument

        file_name = path.stem
        pdf_bytes = path.read_bytes()

        suffix = guess_suffix_by_path(path)
        if suffix in _IMAGE_SUFFIXES:
            pdf_bytes = PDFDocument.from_image(pdf_bytes).bytes

        pdf_bytes = self._maybe_adjust_pdf_bytes(pdf_bytes, suffix)
        return file_name, pdf_bytes

    def _maybe_adjust_pdf_bytes(self, pdf_bytes: bytes, suffix: str) -> bytes:
        if suffix != "pdf":
            return pdf_bytes

        from ..utils.pdf_document import PDFDocument

        doc = PDFDocument(pdf_bytes)
        end = self.end_page_id if self.end_page_id is not None else doc.page_count - 1
        if self.start_page_id > 0 or self.end_page_id is not None:
            extracted = doc.extract_page_range(self.start_page_id, end)
            if extracted.bytes:
                return extracted.bytes
        return pdf_bytes

    def _build_result(
        self,
        middle_json: list[PageInfo],
        pdf_bytes: bytes,
        file_name: str,
    ) -> ParseResult:
        from ..utils.pdf_document import PDFDocument
        from ..version import __version__

        return ParseResult(
            pages=middle_json,
            _backend=self._backend,
            _version_name=__version__,
            _pdf_doc=PDFDocument(pdf_bytes),
            _file_name=file_name,
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

    def parse(self, path: str | Path) -> ParseResult:
        self._parse_method = self.method
        return super().parse(path)

    async def parse_async(self, path: str | Path) -> ParseResult:
        self._parse_method = self.method
        return await super().parse_async(path)

    def parse_batch(self, paths: list[str | Path]) -> list[ParseResult]:
        self._parse_method = self.method

        from ..backend.pipeline.pipeline_analyze import doc_analyze_streaming

        file_names: list[str] = []
        pdf_bytes_list: list[bytes] = []
        for p in paths:
            fn, pb = self._prepare_input(Path(p))
            file_names.append(fn)
            pdf_bytes_list.append(pb)

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
            result = self._build_result(middle_json, pdf_bytes_list[idx], file_names[idx])
            parse_results.append(result)

        return parse_results

    async def parse_batch_async(self, paths: list[str | Path]) -> list[ParseResult]:
        import asyncio

        return await asyncio.to_thread(self.parse_batch, paths)

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

    def parse(self, path: str | Path) -> ParseResult:
        self._parse_method = f"hybrid_{self.method}"
        return super().parse(path)

    async def parse_async(self, path: str | Path) -> ParseResult:
        self._parse_method = f"hybrid_{self.method}"
        return await super().parse_async(path)

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
