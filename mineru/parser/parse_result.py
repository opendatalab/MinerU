# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import base64
import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any

from ..utils.enum_class import MakeMode
from ..utils.pdf_document import PDFDocument
from .types import PageInfo

_INLINE_IMAGE_DATA_URI_RE = re.compile(r"data:image/([^;]+);base64,([^\"]+)", re.DOTALL)


def _load_union_make(backend: str):
    if backend == "pipeline":
        from ..backend.pipeline.pipeline_middle_json_mkcontent import union_make

        return union_make
    if backend in ("vlm", "hybrid"):
        from ..backend.vlm.vlm_middle_json_mkcontent import union_make

        return union_make
    from ..backend.office.office_middle_json_mkcontent import union_make

    return union_make


@dataclass
class ParseResult:
    """The parsed result of a document.

    Holds the typed middle representation and exposes markdown / content-list
    / images as lazily-computed methods.  Call ``save(writer)`` to persist.
    """

    pages: list[PageInfo]
    _backend: str
    _version_name: str
    _pdf_doc: PDFDocument | None = None
    _file_name: str = ""
    _model_output: Any = None
    _images_cache: dict[str, bytes] | None = None

    def as_dict(self) -> dict:
        return {"pdf_info": [dataclasses.asdict(p) for p in self.pages]}

    def json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)

    def markdown(self) -> str:
        union_make = _load_union_make(self._backend)
        markdown = union_make(self.pages, MakeMode.MM_MD)
        assert isinstance(markdown, str)
        return markdown

    def content_list(self) -> list[dict[str, Any]]:
        union_make = _load_union_make(self._backend)
        return union_make(self.pages, MakeMode.CONTENT_LIST)  # type: ignore[return-value]

    def content_list_v2(self) -> list[dict[str, Any]]:
        union_make = _load_union_make(self._backend)
        return union_make(self.pages, MakeMode.CONTENT_LIST_V2)  # type: ignore[return-value]

    def save(self, writer: Any) -> None:
        prefix = self._file_name

        # TODO: need rename these files
        writer.write_string(f"{prefix}.md", self.markdown())
        writer.write_string(f"{prefix}_middle.json", self.json())

        writer.write_string(
            f"{prefix}_content_list.json",
            json.dumps(self.content_list(), ensure_ascii=False, indent=4),
        )
        writer.write_string(
            f"{prefix}_content_list_v2.json",
            json.dumps(self.content_list_v2(), ensure_ascii=False, indent=4),
        )

        if self._model_output is not None:
            writer.write_string(
                f"{prefix}_model.json",
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
            pil_img = self._pdf_doc.render_page(page_info.page_idx)
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
        # span-level image_base64 (Office)
        if span.image_base64:
            m = _INLINE_IMAGE_DATA_URI_RE.match(span.image_base64)
            if m:
                try:
                    result[span.image_path or f"{hash_fn(span.image_base64)}.{m.group(1)}"] = base64.b64decode(m.group(2))
                except Exception:
                    pass
        # inline images inside table html
        if span.html:
            for m in _INLINE_IMAGE_DATA_URI_RE.finditer(span.html):
                try:
                    result[f"{hash_fn(m.group(0))}.{m.group(1)}"] = base64.b64decode(m.group(2))
                except Exception:
                    pass
        return result
