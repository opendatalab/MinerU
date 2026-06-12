# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import base64
import dataclasses
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..render import render_content_list, render_content_list_v2, render_markdown
from ..types import PageInfo

_INLINE_IMAGE_DATA_URI_RE = re.compile(r"data:image/([^;]+);base64,([^\"]+)", re.DOTALL)


@dataclass
class ParseResult:
    """The parsed result of a document.

    Holds the typed middle representation and exposes markdown / content-list
    / images as lazily-computed methods.  Call ``save(writer)`` to persist.
    """

    pages: list[PageInfo]
    _pdf_doc: object | None = None
    _model_output: Any = None
    _images_cache: dict[str, bytes] | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ParseResult:
        # TODO
        ...

    def to_dict(self) -> dict[str, Any]:
        return {"pages": [dataclasses.asdict(p) for p in self.pages]}

    @staticmethod
    def from_json(s: str) -> ParseResult:
        # TODO
        ...

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def markdown(self, *, add_markers: bool = False) -> str:
        return render_markdown(self.pages, add_markers=add_markers)

    def content_list(self) -> list[dict[str, Any]]:
        return render_content_list(self.pages)

    def content_list_v2(self) -> list[dict[str, Any]]:
        return render_content_list_v2(self.pages)

    def save(self, writer: Any) -> None:
        writer.write_string("markdown.md", self.markdown())
        writer.write_string("middle_json.json", self.to_json())

        writer.write_string(
            "content_list.json",
            json.dumps(self.content_list(), ensure_ascii=False, indent=4),
        )
        writer.write_string(
            "structured_content.json",
            json.dumps(self.content_list_v2(), ensure_ascii=False, indent=4),
        )

        if self._model_output is not None:
            writer.write_string(
                "model_output.json",
                json.dumps(self._model_output, ensure_ascii=False, indent=4),
            )

        for img_path, img_bytes in self.images().items():
            writer.write(img_path, img_bytes)

    def images(self) -> dict[str, bytes]:
        if self._images_cache is not None:
            return self._images_cache
        if self._pdf_doc is not None:
            return self._extract_pdf_images()
        return self._extract_office_images()

    def _extract_pdf_images(self) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for page_info in self.pages:
            pil_img = self._pdf_doc.render_page(page_info.page_idx)  # type: ignore[union-attr]
            for block in page_info.preproc_blocks:
                result.update(self._crop_block_spans(block, pil_img, 2))
            for block in page_info.para_blocks:
                result.update(self._crop_block_spans(block, pil_img, 2))
        return result

    @staticmethod
    def _crop_block_spans(block: Any, pil_img: Any, scale: int) -> dict[str, bytes]:
        from ..utils.pdf_image_tools import get_crop_img, image_to_bytes

        result: dict[str, bytes] = {}
        for line in block.lines:
            for span in line.spans:
                if not span.image_path or span.bbox is None:
                    continue
                crop = get_crop_img(span.bbox, pil_img, scale=scale)
                result[span.image_path] = image_to_bytes(crop, image_format="JPEG")
        for child in block.blocks:
            result.update(ParseResult._crop_block_spans(child, pil_img, scale))
        return result

    def _extract_office_images(self) -> dict[str, bytes]:
        from ..utils.hash_utils import str_sha256

        result: dict[str, bytes] = {}
        for page_info in self.pages:
            for block in page_info.para_blocks:
                for span in self._iter_block_spans(block):
                    result.update(self._decode_span_base64_images(span, str_sha256))
        return result

    @staticmethod
    def _iter_block_spans(block: Any):
        for line in block.lines:
            yield from line.spans
        for child in block.blocks:
            yield from ParseResult._iter_block_spans(child)

    @staticmethod
    def _decode_span_base64_images(span: Any, hash_fn) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        if span.image_base64:
            m = _INLINE_IMAGE_DATA_URI_RE.match(span.image_base64)
            if m:
                try:
                    result[span.image_path or f"{hash_fn(span.image_base64)}.{m.group(1)}"] = base64.b64decode(m.group(2))
                except Exception:
                    pass
        if span.html:
            for m in _INLINE_IMAGE_DATA_URI_RE.finditer(span.html):
                try:
                    result[f"{hash_fn(m.group(0))}.{m.group(1)}"] = base64.b64decode(m.group(2))
                except Exception:
                    pass
        return result


class DocumentParser(ABC):
    """Abstract base class for all document parsers.

    Subclasses implement ``parse()`` for a specific document category (PDF, DOCX, PPTX, XLSX).
    """

    _closed: bool = False

    @abstractmethod
    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        """Parse a document and return structured results.

        Parameters
        ----------
        path:
            Path to the document file.
        page_range:
            1-based page range string (``"1~5,-3~-1"``).  Empty means all pages.
        """

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        """Asynchronously parse a document.

        The default implementation delegates to ``parse()`` via ``asyncio.to_thread``.
        Subclasses may override for native async support.
        """
        import asyncio

        return await asyncio.to_thread(self.parse, path, page_range=page_range)

    def parse_batch(self, paths: list[str | Path], *, page_range: str = "") -> list[ParseResult]:
        """Parse multiple documents synchronously.

        The default implementation calls ``parse()`` for each path in order.
        Subclasses may override for batch-optimized execution.
        """
        return [self.parse(p, page_range=page_range) for p in paths]

    async def parse_batch_async(self, paths: list[str | Path], *, page_range: str = "") -> list[ParseResult]:
        """Parse multiple documents asynchronously.

        The default implementation calls ``parse_async()`` concurrently for all paths.
        Subclasses may override for batch-optimized execution.
        """
        import asyncio

        return await asyncio.gather(*(self.parse_async(p, page_range=page_range) for p in paths))

    def close(self) -> None:
        """Release resources held by this parser instance.

        After ``close()``, the instance must not be reused.
        The default implementation is a no-op; subclasses may override.
        """
        self._closed = True

    def __enter__(self) -> "DocumentParser":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
