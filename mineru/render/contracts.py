# Copyright (c) Opendatalab. All rights reserved.
"""MiddleJson 多格式 renderer 共用的公共类型与调用选项。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeAlias

from ..types import BlockBase


class RenderFormat(str, Enum):
    """统一渲染入口支持的目标格式。"""

    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"
    STRUCTURED_CONTENT = "structured_content"


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


@dataclass(frozen=True, slots=True)
class StructuredContentRenderOptions:
    """树形 Markdown Structured Content renderer 的统一入口选项。"""

    asset_base_url: str = ""

    def __post_init__(self) -> None:
        """在构造时校验图片资源根地址。"""
        _validate_asset_base_url(self.asset_base_url)


RenderOptions: TypeAlias = MarkdownRenderOptions | HtmlRenderOptions | DocxRenderOptions | StructuredContentRenderOptions
RenderOutput: TypeAlias = str | bytes | dict[str, Any]


__all__ = [
    "AssetResolver",
    "DocxRenderOptions",
    "HtmlRenderOptions",
    "ImageRenderer",
    "MarkdownRenderOptions",
    "RenderFormat",
    "RenderMode",
    "RenderOptions",
    "RenderOutput",
    "StructuredContentRenderOptions",
]
