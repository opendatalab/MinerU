# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ..render import render_markdown, render_structured_content
from ..render.contracts import ImageRenderer, RenderMode
from ..types import FILE_SUFFIXES, FileSuffix, MiddleJson, ModelJson, PageInfo
from ..utils.image_payload import ImagePayloadCache
from .writer import DataWriter

if TYPE_CHECKING:
    from ..model.flash.pdf.document import PDFDocument

MIDDLE_JSON_SCHEMA_VERSION: str = "2.0"
_LEGACY_SCHEMA_VERSION: str = "1.0"
_LEGACY_DEFAULT_FILE_SUFFIX: FileSuffix = "pdf"
_LEGACY_DEFAULT_EFFORT: Literal["flash", "medium", "high", "xhigh"] = "medium"
_LEGACY_DEFAULT_PARSE_MODE: Literal["txt", "ocr"] = "txt"
_TO_DICT_EXCLUDED_KEYS: frozenset[str] = frozenset({"schema_version"})


def _legacy_raw_pages(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """按 envelope 识别 3.4.5 原始 pdf_info 或后续 1.0 pages 包装。

    3.4.5 release tag 内部仍报告 3.4.4，因此不能把 ``_version_name`` 作为
    唯一判据。
    """
    schema_version = payload.get("schema_version")
    raw_pages = payload.get("pdf_info") if schema_version is None else None
    if raw_pages is None and schema_version == _LEGACY_SCHEMA_VERSION:
        raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or any(not isinstance(page, dict) for page in raw_pages):
        return None
    return raw_pages


def _legacy_page_index_map(raw_pages: list[dict[str, Any]]) -> list[int]:
    """从旧 page_idx 恢复抽页映射；连续零起始页面仍表示整本文档。"""
    page_indices: list[int] = []
    for fallback_index, page in enumerate(raw_pages):
        page_idx = page.get("page_idx", fallback_index)
        if isinstance(page_idx, bool) or not isinstance(page_idx, int) or page_idx < 0:
            raise ValueError("legacy Middle JSON page_idx values must be non-negative integers")
        page_indices.append(page_idx)
    if len(page_indices) != len(set(page_indices)) or any(
        current <= previous for previous, current in zip(page_indices, page_indices[1:])
    ):
        raise ValueError("legacy Middle JSON page_idx values must be unique and strictly increasing")
    return [] if page_indices == list(range(len(raw_pages))) else page_indices


def _legacy_file_suffix(payload: dict[str, Any]) -> FileSuffix:
    """读取旧 payload 可选后缀；缺失时沿用历史 PDF 默认值。"""
    value = payload.get("file_suffix")
    if isinstance(value, str) and value in FILE_SUFFIXES:
        return cast(FileSuffix, value)
    return _LEGACY_DEFAULT_FILE_SUFFIX


def _legacy_effort(payload: dict[str, Any]) -> Literal["flash", "medium", "high", "xhigh"]:
    """把 3.4.5 的分析强度映射到当前严格枚举。"""
    value = payload.get("_effort", payload.get("effort"))
    if value in {"flash", "medium", "high", "xhigh"}:
        return cast(Literal["flash", "medium", "high", "xhigh"], value)
    return _LEGACY_DEFAULT_EFFORT


def _legacy_parse_mode(payload: dict[str, Any]) -> Literal["txt", "ocr"]:
    """优先读取显式 parse_mode，否则按 3.4.5 的 _ocr_enable 推断。"""
    value = payload.get("parse_mode")
    if value in {"txt", "ocr"}:
        return cast(Literal["txt", "ocr"], value)
    if payload.get("_ocr_enable") is True:
        return "ocr"
    return _LEGACY_DEFAULT_PARSE_MODE


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

        schema_version = d.get("schema_version")

        if schema_version == MIDDLE_JSON_SCHEMA_VERSION:
            middle_json = ParseResult._build_middle_json_from_current(d)
        elif (raw_pages := _legacy_raw_pages(d)) is not None:
            middle_json = ParseResult._build_middle_json_from_legacy(d, raw_pages)
        else:
            raise ValueError(
                f"Unsupported Middle JSON schema_version={schema_version!r}; "
                f"expected {MIDDLE_JSON_SCHEMA_VERSION!r}. Reparse the source document."
            )

        return ParseResult(middle_json=middle_json)

    @staticmethod
    def _build_middle_json_from_current(d: dict[str, Any]) -> MiddleJson:
        """从当前版本 schema 直接构造 Span 化 MiddleJson。"""
        payload = {k: v for k, v in d.items() if k not in _TO_DICT_EXCLUDED_KEYS}
        return MiddleJson.model_validate(payload)

    @staticmethod
    def _build_middle_json_from_legacy(d: dict[str, Any], raw_pages: list[dict[str, Any]]) -> MiddleJson:
        """把 3.4.5 页面回推为 raw ModelJson，再走当前统一后处理生成 2.0。"""
        from ..backend.postprocess.legacy_schema_adapter import legacy_page_to_model_list
        from ..backend.postprocess.pages import model_json_to_pages
        from ..version import __version__ as current_mineru_version

        source_version = d.get("_version_name", d.get("mineru_version"))
        mineru_version = (
            source_version.strip() if isinstance(source_version, str) and source_version.strip() else current_mineru_version
        )
        model_json = ModelJson(
            pages=[legacy_page_to_model_list(page) for page in raw_pages],
            page_index_map=_legacy_page_index_map(raw_pages),
            file_suffix=_legacy_file_suffix(d),
            effort=_legacy_effort(d),
            parse_mode=_legacy_parse_mode(d),
            mineru_version=mineru_version,
        )
        return MiddleJson(
            pages=model_json_to_pages(model_json),
            is_full_document=model_json.is_full_document,
            file_suffix=model_json.file_suffix,
            effort=model_json.effort,
            parse_mode=model_json.parse_mode,
            mineru_version=model_json.mineru_version,
        )

    def to_dict(self, *, skip_defaults: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": MIDDLE_JSON_SCHEMA_VERSION}
        options: dict[str, Any] = {"skip_defaults": skip_defaults}
        # image block in PDF can be rendered and cropped again.
        if self.middle_json.file_suffix == "pdf":
            options["exclude_block_fields"] = {"image_base64"}
        payload.update(self.middle_json.to_dict(**options))
        return payload

    @staticmethod
    def from_json(s: str) -> ParseResult:
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError("ParseResult JSON must decode to a dict.")
        return ParseResult.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

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
            json.dumps(self.structured_content(), ensure_ascii=False, indent=2),
        )

        if self._model_output is not None:
            model_output = (
                self._model_output.to_dict(skip_defaults=False)
                if isinstance(self._model_output, ModelJson)
                else self._model_output
            )
            writer.write_string(
                "model_output.json",
                json.dumps(model_output, ensure_ascii=False, indent=2),
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

    Subclasses implement ``parse()`` for a specific document category (PDF, EPUB, HTML, CSV, or Office/RTF/ODF).
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
            PDF-only 1-based page range string (``"1~5,-3~-1"``).  Empty means all pages.
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
