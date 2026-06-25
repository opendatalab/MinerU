# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from copy import deepcopy
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..render import render_content_list, render_markdown, render_structured_content
from ..schema.middle_json import MIDDLE_JSON_SCHEMA_VERSION
from ..types import PageInfo
from ..utils.image_payload import (
    collect_image_data_uri_bytes,
    image_path_from_data_uri,
    parse_image_data_uri,
    replace_inline_data_uri_sources,
)

_PDF_RETAINED_PAGE_INDICES_KEY = "_pdf_retained_page_indices"
_PDF_BROKEN_PAGE_INDICES_KEY = "_pdf_broken_page_indices"


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

    Holds the typed middle representation and exposes markdown / content-list
    / images as lazily-computed methods.  Call ``save(writer)`` to persist.
    """

    pages: list[PageInfo]
    _pdf_doc: object | None = None
    _model_output: Any = None
    _images_cache: dict[str, bytes] | None = None
    _retained_page_indices: list[int] | None = None
    _broken_page_indices: list[int] | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ParseResult:
        if not isinstance(d, dict):
            raise ValueError("ParseResult.from_dict expects a dict.")

        if "pages" not in d:
            raise ValueError("ParseResult JSON must contain a list field named pages.")

        raw_pages = d["pages"]
        if not isinstance(raw_pages, list):
            raise ValueError("ParseResult pages must be a list.")

        pages: list[PageInfo] = []
        for raw_page in raw_pages:
            if not isinstance(raw_page, dict):
                raise ValueError("ParseResult page entries must be dicts.")
            page = PageInfo.from_dict(raw_page)
            backend = raw_page.get("_backend")
            if isinstance(backend, str):
                page._backend = backend
            pages.append(page)
        retained_page_indices = _parse_optional_int_list(d.get(_PDF_RETAINED_PAGE_INDICES_KEY))
        broken_page_indices = _parse_optional_int_list(d.get(_PDF_BROKEN_PAGE_INDICES_KEY))
        return ParseResult(
            pages=pages,
            _retained_page_indices=retained_page_indices,
            _broken_page_indices=broken_page_indices,
        )

    def to_dict(self, *, skip_defaults: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": MIDDLE_JSON_SCHEMA_VERSION,
            "pages": [page.to_dict(skip_defaults=skip_defaults) for page in self.pages],
        }
        self._append_private_pdf_page_mapping(payload)
        return payload

    def to_export_dict(self, *, skip_defaults: bool = True) -> dict[str, Any]:
        """生成 public middle_json 视图：图片已落盘引用，base64 临时载荷不再输出。"""
        payload = {
            "schema_version": MIDDLE_JSON_SCHEMA_VERSION,
            "pages": [page.to_dict(skip_defaults=skip_defaults) for page in self.export_pages()],
        }
        self._append_private_pdf_page_mapping(payload)
        return payload

    def _append_private_pdf_page_mapping(self, payload: dict[str, Any]) -> None:
        """附加 PDF 重写页映射的内部元数据，供本地可视化按实际 PDF 页序绘制。"""
        if self._retained_page_indices is not None:
            payload[_PDF_RETAINED_PAGE_INDICES_KEY] = list(self._retained_page_indices)
        if self._broken_page_indices:
            payload[_PDF_BROKEN_PAGE_INDICES_KEY] = list(self._broken_page_indices)

    @staticmethod
    def from_json(s: str) -> ParseResult:
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError("ParseResult JSON must decode to a dict.")
        return ParseResult.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)

    def to_export_json(self) -> str:
        """序列化导出 middle_json，供本地文件和 API 返回使用。"""
        return json.dumps(self.to_export_dict(), ensure_ascii=False, indent=4)

    def markdown(self, *, add_markers: bool = False) -> str:
        return render_markdown(self.pages, add_markers=add_markers)

    def content_list(self) -> list[dict[str, Any]]:
        return render_content_list(self.pages)

    def structured_content(self) -> list[list[dict[str, Any]]]:
        return render_structured_content(self.pages)

    def save(self, writer: Any) -> None:
        writer.write_string("markdown.md", self.markdown())
        writer.write_string("middle_json.json", self.to_export_json())

        writer.write_string(
            "content_list.json",
            json.dumps(self.content_list(), ensure_ascii=False, indent=4),
        )
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
        if self._images_cache is not None:
            return self._images_cache
        images = self._collect_images()
        self._images_cache = images
        return images

    def export_pages(self) -> list[PageInfo]:
        """返回可导出的页面副本：清理 base64，并把内联图片替换为本地路径。"""
        export_pages, _ = self._export_pages_and_images()
        return export_pages

    def _export_pages_and_images(self) -> tuple[list[PageInfo], dict[str, bytes]]:
        """构造导出页面和图片字节映射，避免 ParseResult.save 再渲染 PDF 页面。"""
        export_pages = deepcopy(self.pages)
        images: dict[str, bytes] = {}
        for page_info in export_pages:
            for block_list in (page_info.preproc_blocks, page_info.para_blocks, page_info.discarded_blocks):
                for block in block_list:
                    self._prepare_block_images_for_export(block, images)
        return export_pages, images

    def _collect_images(self) -> dict[str, bytes]:
        """遍历原始 middle_json 收集图片字节，不修改页面和 span 内容。"""
        images: dict[str, bytes] = {}
        for page_info in self.pages:
            for block_list in (page_info.preproc_blocks, page_info.para_blocks, page_info.discarded_blocks):
                for block in block_list:
                    self._collect_block_images(block, images)
        return images

    @staticmethod
    def _collect_block_images(block: Any, images: dict[str, bytes]) -> None:
        """递归收集 block 内所有 span 的 base64 图片载荷。"""
        for line in block.lines:
            for span in line.spans:
                ParseResult._collect_span_images(span, images)
        for child in block.blocks:
            ParseResult._collect_block_images(child, images)

    @staticmethod
    def _collect_span_images(span: Any, images: dict[str, bytes]) -> None:
        """收集单个 span 的图片字节，保持 span 原始内容不变。"""
        if span.image_base64:
            parsed = parse_image_data_uri(span.image_base64)
            img_path = span.image_path or image_path_from_data_uri(span.image_base64)
            if parsed is not None and img_path:
                images[img_path] = parsed[0]
        if span.content:
            collect_image_data_uri_bytes(span.content, images)

    @staticmethod
    def _prepare_block_images_for_export(block: Any, images: dict[str, bytes]) -> None:
        """递归处理 block 内所有 span，将 base64 图片转为 image_path 引用。"""
        for line in block.lines:
            for span in line.spans:
                ParseResult._prepare_span_images_for_export(span, images)
        for child in block.blocks:
            ParseResult._prepare_block_images_for_export(child, images)

    @staticmethod
    def _prepare_span_images_for_export(span: Any, images: dict[str, bytes]) -> None:
        """处理单个 span 的图片载荷，保留路径字段并清理 base64 临时字段。"""
        if span.image_base64:
            parsed = parse_image_data_uri(span.image_base64)
            img_path = span.image_path or image_path_from_data_uri(span.image_base64)
            if parsed is not None and img_path:
                images[img_path] = parsed[0]
                span.image_path = img_path
            span.image_base64 = ""
        if span.content:
            span.content = replace_inline_data_uri_sources(span.content, images)


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
