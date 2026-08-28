# Copyright (c) Opendatalab. All rights reserved.
"""MiddleJson 多格式 renderer 共用的公共类型与调用选项。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re
from typing import Any, Callable, TypeAlias

from ..types import BlockBase


class RenderFormat(str, Enum):
    """统一渲染入口支持的目标格式。"""

    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"
    EPUB = "epub"
    STRUCTURED_CONTENT = "structured_content"
    CONTENT_LIST = "content_list"
    CONTENT_LIST_V2 = "content_list_v2"
    PDF = "pdf"


class RenderMode(str, Enum):
    """MiddleJson renderer 共用的默认合并视图与完整分页视图。"""

    DEFAULT = "default"
    FULL = "full"


AssetResolver: TypeAlias = Callable[[str], bytes]
ImageRenderer: TypeAlias = Callable[[BlockBase], str]


def _validate_mode(mode: object) -> None:
    """校验展示型 renderer 的公共模式参数。"""
    if not isinstance(mode, RenderMode):
        raise TypeError("mode must be a RenderMode value")


def _validate_asset_base_url(asset_base_url: object) -> None:
    """校验用于拼接相对图片路径的资源根地址。"""
    if not isinstance(asset_base_url, str):
        raise TypeError("asset_base_url must be a string")


@dataclass(frozen=True, slots=True)
class MarkdownRenderOptions:
    """Markdown renderer 的统一入口选项。"""

    mode: RenderMode = RenderMode.DEFAULT
    asset_base_url: str = ""
    image_renderer: ImageRenderer | None = None

    def __post_init__(self) -> None:
        """在构造时拒绝不符合严格公共契约的选项值。"""
        _validate_mode(self.mode)
        _validate_asset_base_url(self.asset_base_url)
        if self.image_renderer is not None and not callable(self.image_renderer):
            raise TypeError("image_renderer must be callable or None")


@dataclass(frozen=True, slots=True)
class HtmlRenderOptions:
    """HTML renderer 的统一入口选项。"""

    mode: RenderMode = RenderMode.DEFAULT
    asset_base_url: str = ""
    standalone: bool = True
    document_title: str | None = None

    def __post_init__(self) -> None:
        """在构造时校验 HTML 文档形态与标题选项。"""
        _validate_mode(self.mode)
        _validate_asset_base_url(self.asset_base_url)
        if not isinstance(self.standalone, bool):
            raise TypeError("standalone must be a bool")
        if self.document_title is not None and not isinstance(self.document_title, str):
            raise TypeError("document_title must be a string or None")


@dataclass(frozen=True, slots=True)
class DocxRenderOptions:
    """DOCX renderer 的统一入口选项。"""

    mode: RenderMode = RenderMode.DEFAULT
    asset_resolver: AssetResolver | None = None

    def __post_init__(self) -> None:
        """在构造时校验分页模式与可选素材解析器。"""
        _validate_mode(self.mode)
        if self.asset_resolver is not None and not callable(self.asset_resolver):
            raise TypeError("asset_resolver must be callable or None")


_EPUB_LANGUAGE_RE = re.compile(r"(?:[A-Za-z]{2,8}|und)(?:-[A-Za-z0-9]{1,8})*\Z", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class EpubRenderOptions:
    """EPUB 3.3 renderer 的统一入口选项。"""

    mode: RenderMode = RenderMode.DEFAULT
    title: str | None = None
    authors: tuple[str, ...] = ()
    language: str = "und"
    identifier: str | None = None
    modified_at: datetime | None = None
    asset_resolver: AssetResolver | None = None

    def __post_init__(self) -> None:
        """在构造时校验 EPUB 元数据、时间与素材解析器。"""
        _validate_mode(self.mode)
        if self.title is not None and (not isinstance(self.title, str) or not self.title.strip()):
            raise TypeError("title must be a non-empty string or None")
        if not isinstance(self.authors, tuple) or any(
            not isinstance(author, str) or not author.strip() for author in self.authors
        ):
            raise TypeError("authors must be a tuple of non-empty strings")
        if not isinstance(self.language, str) or _EPUB_LANGUAGE_RE.fullmatch(self.language.strip()) is None:
            raise ValueError("language must be a supported BCP 47 language tag")
        if self.identifier is not None and (not isinstance(self.identifier, str) or not self.identifier.strip()):
            raise TypeError("identifier must be a non-empty string or None")
        if self.modified_at is not None:
            if not isinstance(self.modified_at, datetime):
                raise TypeError("modified_at must be a datetime or None")
            if self.modified_at.utcoffset() is None:
                raise ValueError("modified_at must be timezone-aware")
            if self.modified_at.year < 1000:
                raise ValueError("modified_at year must use four digits")
        if self.asset_resolver is not None and not callable(self.asset_resolver):
            raise TypeError("asset_resolver must be callable or None")


@dataclass(frozen=True, slots=True)
class PdfRenderOptions:
    """PDF renderer 的统一入口选项。"""

    mode: RenderMode = RenderMode.FULL
    asset_resolver: AssetResolver | None = None
    document_title: str | None = None

    def __post_init__(self) -> None:
        """在构造时校验分页模式、素材解析器与文档标题。"""
        _validate_mode(self.mode)
        if self.asset_resolver is not None and not callable(self.asset_resolver):
            raise TypeError("asset_resolver must be callable or None")
        if self.document_title is not None and not isinstance(self.document_title, str):
            raise TypeError("document_title must be a string or None")


@dataclass(frozen=True, slots=True)
class StructuredContentRenderOptions:
    """树形 Markdown Structured Content renderer 的统一入口选项。"""

    asset_base_url: str = ""

    def __post_init__(self) -> None:
        """在构造时校验图片资源根地址。"""
        _validate_asset_base_url(self.asset_base_url)


@dataclass(frozen=True, slots=True)
class ContentListRenderOptions:
    """扁平 Content List V1 renderer 的统一入口选项。"""

    asset_base_url: str = ""

    def __post_init__(self) -> None:
        """在构造时校验图片资源根地址。"""
        _validate_asset_base_url(self.asset_base_url)


@dataclass(frozen=True, slots=True)
class ContentListV2RenderOptions:
    """按页 Content List V2 renderer 的统一入口选项。"""

    asset_base_url: str = ""

    def __post_init__(self) -> None:
        """在构造时校验图片资源根地址。"""
        _validate_asset_base_url(self.asset_base_url)


RenderOptions: TypeAlias = (
    MarkdownRenderOptions
    | HtmlRenderOptions
    | DocxRenderOptions
    | EpubRenderOptions
    | PdfRenderOptions
    | StructuredContentRenderOptions
    | ContentListRenderOptions
    | ContentListV2RenderOptions
)
RenderOutput: TypeAlias = str | bytes | dict[str, Any] | list[dict[str, Any]] | list[list[dict[str, Any]]]


__all__ = [
    "AssetResolver",
    "ContentListRenderOptions",
    "ContentListV2RenderOptions",
    "DocxRenderOptions",
    "EpubRenderOptions",
    "HtmlRenderOptions",
    "ImageRenderer",
    "MarkdownRenderOptions",
    "PdfRenderOptions",
    "RenderFormat",
    "RenderMode",
    "RenderOptions",
    "RenderOutput",
    "StructuredContentRenderOptions",
]
