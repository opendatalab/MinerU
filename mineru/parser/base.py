# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..render import render_markdown, render_structured_content
from ..render.contracts import ImageRenderer, RenderMode
from .writer import DataWriter
from ..types import FileSuffix, MiddleJson, PageInfo
from ..utils.image_payload import ImagePayloadCache
from ..model.flash.pdf.document import PDFDocument

MIDDLE_JSON_SCHEMA_VERSION: str = "2.0"
_LEGACY_SCHEMA_VERSION: str = "1.0"
_LEGACY_DEFAULT_FILE_SUFFIX: FileSuffix = "pdf"
_LEGACY_DEFAULT_EFFORT: Literal["flash", "medium", "high", "xhigh"] = "medium"
_LEGACY_DEFAULT_PARSE_MODE: Literal["txt", "ocr"] = "txt"
_PDF_RETAINED_PAGE_INDICES_KEY = "_pdf_retained_page_indices"
_PDF_BROKEN_PAGE_INDICES_KEY = "_pdf_broken_page_indices"
_TO_DICT_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {"schema_version", _PDF_RETAINED_PAGE_INDICES_KEY, _PDF_BROKEN_PAGE_INDICES_KEY}
)


def _parse_optional_int_list(value: Any) -> list[int] | None:
    """解析内部页映射列表；旧 payload 或非法类型直接按缺省处理。"""
    if not isinstance(value, list):
        return None
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int):
            return None
        parsed.append(item)
    return parsed


@dataclass
class ParseResult:
    """The parsed result of a document.

    Holds the typed middle representation and exposes markdown / structured
    content / images as lazily-computed methods.  Call ``save(writer)`` to persist.
    """

    middle_json: MiddleJson
    _pdf_doc: PDFDocument | None = None
    _model_output: Any = None
    _image_cache: ImagePayloadCache | dict[str, bytes] | None = None
    _retained_page_indices: list[int] | None = None
    _broken_page_indices: list[int] | None = None

    @property
    def pages(self) -> list[PageInfo]:
        """顶层页面列表，委托给 MiddleJson。"""
        return self.middle_json.pages

    def __post_init__(self) -> None:
        """规范化顶层图片缓存，确保 public middle_json 不再从 span 携带图片字节。"""
        if self._image_cache is None:
            self._image_cache = ImagePayloadCache()
        elif isinstance(self._image_cache, dict):
            self._image_cache = ImagePayloadCache(self._image_cache)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ParseResult:
        if not isinstance(d, dict):
            raise ValueError("ParseResult.from_dict expects a dict.")

        schema_version = d.get("schema_version", _LEGACY_SCHEMA_VERSION)
        retained_page_indices = _parse_optional_int_list(d.get(_PDF_RETAINED_PAGE_INDICES_KEY))
        broken_page_indices = _parse_optional_int_list(d.get(_PDF_BROKEN_PAGE_INDICES_KEY))

        if schema_version == MIDDLE_JSON_SCHEMA_VERSION:
            middle_json = ParseResult._build_middle_json_from_v2(d)
        else:
            middle_json = ParseResult._build_middle_json_from_legacy(d)

        return ParseResult(
            middle_json=middle_json,
            _retained_page_indices=retained_page_indices,
            _broken_page_indices=broken_page_indices,
        )

    @staticmethod
    def _build_middle_json_from_v2(d: dict[str, Any]) -> MiddleJson:
        """从 2.0 schema 直接构造 MiddleJson，pages 与元数据字段齐全。"""
        payload = {k: v for k, v in d.items() if k not in _TO_DICT_EXCLUDED_KEYS}
        return MiddleJson.model_validate(payload)

    @staticmethod
    def _build_middle_json_from_legacy(d: dict[str, Any]) -> MiddleJson:
        """从 1.0 schema 兼容构造 MiddleJson，preproc_blocks 回推为 raw model_list 重走后处理。"""
        raw_pages = d.get("pages")
        if not isinstance(raw_pages, list):
            raise ValueError("ParseResult JSON must contain a list field named pages.")

        from ..backend.postprocess.legacy_schema_adapter import legacy_page_to_model_list
        from ..backend.postprocess.pages import model_json_to_pages
        from ..types import ModelJson
        from ..version import __version__ as mineru_version

        model_list = [legacy_page_to_model_list(page) for page in raw_pages]
        if model_list:
            model_json = ModelJson(
                pages=model_list,
                page_index_map=[],
                file_suffix=_LEGACY_DEFAULT_FILE_SUFFIX,
                effort=_LEGACY_DEFAULT_EFFORT,
                parse_mode=_LEGACY_DEFAULT_PARSE_MODE,
                mineru_version=mineru_version,
            )
            pages = model_json_to_pages(model_json)
        else:
            pages = []

        return MiddleJson(
            pages=pages,
            is_full_document=True,
            file_suffix=_LEGACY_DEFAULT_FILE_SUFFIX,
            effort=_LEGACY_DEFAULT_EFFORT,
            parse_mode=_LEGACY_DEFAULT_PARSE_MODE,
            mineru_version=mineru_version,
        )

    def to_dict(self, *, skip_defaults: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": MIDDLE_JSON_SCHEMA_VERSION}
        payload.update(self.middle_json.to_dict(skip_defaults=skip_defaults))
        if self._retained_page_indices is not None:
            payload[_PDF_RETAINED_PAGE_INDICES_KEY] = list(self._retained_page_indices)
        if self._broken_page_indices:
            payload[_PDF_BROKEN_PAGE_INDICES_KEY] = list(self._broken_page_indices)
        return payload

    @staticmethod
    def from_json(s: str) -> ParseResult:
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError("ParseResult JSON must decode to a dict.")
        return ParseResult.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)

    def markdown(
        self,
        *,
        add_markers: bool = False,
        mode: RenderMode | None = None,
        asset_base_url: str = "",
        image_renderer: ImageRenderer | None = None,
    ) -> str:
        if mode is None:
            mode = RenderMode.FULL if add_markers else RenderMode.DEFAULT
        return render_markdown(
            self.middle_json,
            mode=mode,
            asset_base_url=asset_base_url,
            image_renderer=image_renderer,
        )

    def structured_content(self, *, asset_base_url: str = "") -> dict[str, Any]:
        return render_structured_content(self.middle_json, asset_base_url=asset_base_url)

    def save(self, writer: DataWriter) -> None:
        writer.write_string("markdown.md", self.markdown())
        writer.write_string("middle_json.json", self.to_json())

        writer.write_string(
            "structured_content.json",
            json.dumps(self.structured_content(), ensure_ascii=False, indent=4),
        )

        if self._model_output is not None:
            writer.write_string(
                "model_output.json",
                json.dumps(self._model_output, ensure_ascii=False, indent=4),
            )

        for img_path, img_bytes in self.images().items():
            writer.write(img_path, img_bytes)

    def images(self) -> dict[str, bytes]:
        assert isinstance(self._image_cache, ImagePayloadCache)
        return self._image_cache.images()

    def attach_export_images(self, images: dict[str, bytes]) -> None:
        """绑定 API sidecar 下载到的图片字节，供后续 images/save 统一写出。"""
        assert isinstance(self._image_cache, ImagePayloadCache)
        self._image_cache.update(images)

    def refresh_export_cache(self, *, preserve_images: bool = False) -> None:
        """保留历史方法名；当前仅按需清空顶层图片缓存。"""
        if not preserve_images:
            self._image_cache = ImagePayloadCache()

    def export_pages(self) -> list[PageInfo]:
        """返回页面树副本，避免调用方修改污染 ParseResult.pages。"""
        return deepcopy(self.pages)


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
