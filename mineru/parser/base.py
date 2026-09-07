# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations
from ..integrations.docvortex import validate_mineru_metadata

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..render import render_markdown, render_structured_content
from ..render.contracts import ImageRenderer, RenderMode
from ..types import MiddleJson, ModelJson, PageInfo
from .writer import DataWriter

MIDDLE_JSON_SCHEMA_VERSION: str = MiddleJson.model_fields["schema_version"].default


@dataclass
class ParseResult:
    """The parsed result of a document.

    Holds the typed middle representation and exposes markdown / structured
    content / images as lazily-computed methods.  Call ``save(writer)`` to persist.
    """

    middle_json: MiddleJson
    _model_output: ModelJson | None = None

    def __post_init__(self) -> None:
        """要求可选分析结果为共享类型，禁止把任意旧列表重新导出为 Model JSON。"""
        if self._model_output is not None:
            if not isinstance(self._model_output, ModelJson):
                raise TypeError("model output must be a ModelJson document")
            validate_mineru_metadata(self._model_output)

    @property
    def pages(self) -> list[PageInfo]:
        """顶层页面列表，委托给 MiddleJson。"""
        return self.middle_json.pages

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ParseResult:
        """只读取统一新版协议，保留真实生产者并校验可选产品扩展。"""
        middle_json = MiddleJson.from_dict(d)
        validate_mineru_metadata(middle_json)
        return ParseResult(middle_json=middle_json)

    def to_dict(self, *, skip_defaults: bool = True) -> dict[str, Any]:
        """输出共享协议，并保留 PDF 图片省略约定且不修改源对象。"""
        validate_mineru_metadata(self.middle_json)
        return self.middle_json.to_dict(
            skip_defaults=skip_defaults,
            exclude_block_fields={"image_base64"} if self.middle_json.metadata.file_suffix == "pdf" else None,
        )

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
            validate_mineru_metadata(self._model_output)
            model_output = self._model_output.to_dict(skip_defaults=False)
            writer.write_string(
                "model_output.json",
                json.dumps(model_output, ensure_ascii=False, indent=2),
            )

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
            PDF-only 1-based inclusive range (``"1-5,r3-r1"``). ``r1`` is the last page; empty or ``"all"`` means all pages.
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
