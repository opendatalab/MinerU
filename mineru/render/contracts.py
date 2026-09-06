# Copyright (c) Opendatalab. All rights reserved.
"""MinerU 九种渲染目标及共享 DocVortex 选项的公共契约。"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

from docvortex.options import LatexDelimitersConfig
from docvortex.render.contracts import (
    AssetResolver,
    DocxRenderOptions,
    EpubRenderOptions,
    HtmlRenderOptions,
    ImageRenderer,
    LatexRenderOptions,
    MarkdownRenderOptions,
    PdfRenderOptions,
    RenderMode,
    StructuredContentRenderOptions,
)


class RenderFormat(str, Enum):
    """统一渲染入口支持的目标格式。"""

    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"
    DOCX = "docx"
    EPUB = "epub"
    STRUCTURED_CONTENT = "structured_content"
    CONTENT_LIST = "content_list"
    CONTENT_LIST_V2 = "content_list_v2"
    PDF = "pdf"


def _validate_asset_base_url(asset_base_url: object) -> None:
    """校验产品专用 renderer 的素材根地址，保持既有错误类型。"""
    if not isinstance(asset_base_url, str):
        raise TypeError("asset_base_url must be a string")


@dataclass(frozen=True, slots=True)
class ContentListRenderOptions:
    """扁平 Content List V1 renderer 的统一入口选项。"""

    latex_delimiters: LatexDelimitersConfig | None = None

    asset_base_url: str = ""

    def __post_init__(self) -> None:
        """在构造时校验图片资源根地址。"""
        _validate_asset_base_url(self.asset_base_url)


@dataclass(frozen=True, slots=True)
class ContentListV2RenderOptions:
    """按页 Content List V2 renderer 的统一入口选项。"""

    latex_delimiters: LatexDelimitersConfig | None = None

    asset_base_url: str = ""

    def __post_init__(self) -> None:
        """在构造时校验图片资源根地址。"""
        _validate_asset_base_url(self.asset_base_url)


RenderOptions: TypeAlias = (
    MarkdownRenderOptions
    | HtmlRenderOptions
    | LatexRenderOptions
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
    "LatexRenderOptions",
    "MarkdownRenderOptions",
    "PdfRenderOptions",
    "RenderFormat",
    "RenderMode",
    "RenderOptions",
    "RenderOutput",
    "StructuredContentRenderOptions",
]
