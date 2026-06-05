# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import os
from typing import Any

import pypdfium2 as pdfium
from PIL import Image

from ..types import BBox, PageInfo
from .draw_bbox import draw_layout_bbox, draw_span_bbox
from .pdf_classify import classify, get_text_quality_signal_pdfium
from .pdf_classify import extract_pages as _extract_pages
from .pdf_image_tools import get_crop_img, image_to_bytes, images_bytes_to_pdf_bytes
from .pdf_text_tool import get_lines_from_chars, get_page_chars
from .pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
    pdfium_guard,
    rewrite_pdf_bytes_with_pdfium,
)


class PDFDocument:
    """A PDF file loaded in memory, with lazy pypdfium2 access.

    All pypdfium2 operations are serialised under a module-level lock for
    thread safety.  Call ``close()`` when done, or use as a context manager.
    """

    def __init__(self, pdf_bytes: bytes) -> None:
        self._pdf_bytes = pdf_bytes
        self._pdf_doc = None  # type: object | None

    # ------------------------------------------------------------------ #
    #  Factory
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_image(image_bytes: bytes) -> "PDFDocument":
        return PDFDocument(images_bytes_to_pdf_bytes(image_bytes))

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if self._pdf_doc is not None:
            close_pdfium_document(self._pdf_doc)
            self._pdf_doc = None

    def __enter__(self) -> "PDFDocument":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def page_count(self) -> int:
        return get_pdfium_document_page_count(self._ensure_open())

    @property
    def bytes(self) -> bytes:
        return self._pdf_bytes

    # ------------------------------------------------------------------ #
    #  Metadata
    # ------------------------------------------------------------------ #

    def page_size(self, page_idx: int) -> tuple[float, float]:
        page = self._get_page(page_idx)
        with pdfium_guard():
            rect: tuple[float, float, float, float] = page.get_bbox()
        return (abs(rect[2] - rect[0]), abs(rect[1] - rect[3]))

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def render_page(self, page_idx: int, *, scale: int = 2) -> Image.Image:
        page = self._get_page(page_idx)
        with pdfium_guard():
            bitmap = page.render(scale=scale)
            try:
                return bitmap.to_pil()
            finally:
                bitmap.close()

    def render_pages(self, start: int = 0, end: int | None = None, *, scale: int = 2) -> list[Image.Image]:
        end = end if end is not None else self.page_count - 1
        return [self.render_page(i, scale=scale) for i in range(start, end + 1)]

    async def render_page_async(self, page_idx: int, *, scale: int = 2) -> Image.Image:
        return await asyncio.to_thread(self.render_page, page_idx, scale=scale)

    async def render_pages_async(self, start: int = 0, end: int | None = None, *, scale: int = 2) -> list[Image.Image]:
        end = end if end is not None else self.page_count - 1
        tasks = [self.render_page_async(i, scale=scale) for i in range(start, end + 1)]
        return await asyncio.gather(*tasks)

    def crop_image(self, bbox: BBox, page_idx: int, *, scale: int = 2) -> bytes:
        pil_img = self.render_page(page_idx, scale=scale)
        crop = get_crop_img(bbox, pil_img, scale=scale)
        return image_to_bytes(crop, image_format="JPEG")

    # ------------------------------------------------------------------ #
    #  Text
    # ------------------------------------------------------------------ #

    def get_page_chars(self, page_idx: int) -> dict[str, Any]:
        page = self._get_page(page_idx)
        return get_page_chars(page)

    def get_page_lines(self, page_idx: int) -> list[dict[str, Any]]:
        chars_dict = self.get_page_chars(page_idx)
        return get_lines_from_chars(chars_dict["chars"])

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    def classify(self) -> str:
        return classify(self._pdf_bytes)

    def get_text_quality(self) -> dict[str, Any]:
        doc = self._ensure_open()
        page_indices = list(range(self.page_count))
        return get_text_quality_signal_pdfium(doc, page_indices)

    # ------------------------------------------------------------------ #
    #  Page extraction
    # ------------------------------------------------------------------ #

    def extract_page_range(self, start: int, end: int) -> "PDFDocument":
        new_bytes = rewrite_pdf_bytes_with_pdfium(
            self._pdf_bytes,
            start_page_id=start,
            end_page_id=end,
        )
        return PDFDocument(new_bytes)

    def sample_pages(self, max_pages: int = 3) -> "PDFDocument":
        new_bytes = _extract_pages(self._pdf_bytes)
        if max_pages > 0 and new_bytes:
            new_doc = PDFDocument(new_bytes)
            count = new_doc.page_count
            if count > max_pages:
                return new_doc.extract_page_range(0, max_pages - 1)
            return new_doc
        return PDFDocument(b"")

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    def draw_layout_bbox(self, pages: list[PageInfo], output_path: str) -> None:
        out_dir = os.path.dirname(output_path) or "."
        filename = os.path.basename(output_path)
        draw_layout_bbox(pages, self._pdf_bytes, out_dir, filename)

    def draw_span_bbox(self, pages: list[PageInfo], output_path: str) -> None:
        out_dir = os.path.dirname(output_path) or "."
        filename = os.path.basename(output_path)
        draw_span_bbox(pages, self._pdf_bytes, out_dir, filename)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _ensure_open(self) -> pdfium.PdfDocument:
        if self._pdf_doc is None:
            self._pdf_doc = open_pdfium_document(pdfium.PdfDocument, self._pdf_bytes)
        return self._pdf_doc

    def _get_page(self, page_idx: int) -> pdfium.PdfPage:
        return self._ensure_open()[page_idx]
