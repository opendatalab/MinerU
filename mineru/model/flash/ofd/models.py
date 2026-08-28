# Copyright (c) Opendatalab. All rights reserved.
"""OFD 解析器内部使用的确定性数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lxml import etree  # type: ignore[reportMissingImports]

from ....types import BBox
from .geometry import Affine, Point, Quad


@dataclass(frozen=True, slots=True)
class FontResource:
    """保存字体资源及可选内嵌字体成员路径。"""

    resource_id: int
    font_name: str
    family_name: str
    font_part: str | None
    bold: bool = False
    italic: bool = False


@dataclass(frozen=True, slots=True)
class MediaResource:
    """保存多媒体资源的成员路径与声明格式。"""

    resource_id: int
    media_type: str
    media_format: str
    media_part: str


@dataclass(frozen=True, slots=True)
class CompositeResource:
    """保存可递归展开的复合图元资源。"""

    resource_id: int
    width: float
    height: float
    element: etree._Element


@dataclass(slots=True)
class ResourceRegistry:
    """保存当前文档或页面作用域内的资源索引。"""

    fonts: dict[int, FontResource] = field(default_factory=dict)
    media: dict[int, MediaResource] = field(default_factory=dict)
    composites: dict[int, CompositeResource] = field(default_factory=dict)
    draw_params: dict[int, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GlyphItem:
    """保存一个语义字符在页面空间中的几何。"""

    text: str
    bbox: BBox
    quad: Quad
    origin: Point
    glyph_id: int | None = None


@dataclass(slots=True)
class TextLine:
    """保存一个 TextCode 恢复出的可排序文字行。"""

    text: str
    bbox: BBox
    glyphs: list[GlyphItem]
    angle: int
    font_size: float
    paint_order: int
    object_id: int | None
    layer_type: str
    template_id: int | None
    styles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AxisLine:
    """保存页面空间中的一条可见水平或垂直线。"""

    bbox: BBox
    orientation: str
    width: float
    paint_order: int
    template_id: int | None


@dataclass(frozen=True, slots=True)
class ImageItem:
    """保存页面图片载荷、几何和绘制来源。"""

    bbox: BBox
    image_base64: str | None
    paint_order: int
    object_id: int | None
    layer_type: str
    template_id: int | None
    diagnostic: str | None = None


@dataclass(slots=True)
class OfdPageScene:
    """保存一页 OFD 的原生场景和可投影对象。"""

    page_idx: int
    physical_box: BBox
    content_box: BBox | None
    text_lines: list[TextLine] = field(default_factory=list)
    axis_lines: list[AxisLine] = field(default_factory=list)
    images: list[ImageItem] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OfdDocumentRef:
    """保存 OFD.xml 中一个 DocBody 的入口和元数据。"""

    document_part: str
    signatures_part: str | None
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class PageRef:
    """保存 Document.xml 页树中的一个页面引用。"""

    page_id: int | None
    page_part: str


@dataclass(frozen=True, slots=True)
class TemplateRef:
    """保存模板 ID 到模板页面成员的映射。"""

    template_id: int
    page_part: str


@dataclass(frozen=True, slots=True)
class PageBuildContext:
    """保存递归图元构建所需的父级状态。"""

    transform: Affine
    clip_bbox: BBox
    layer_type: str
    template_id: int | None
    draw_style: dict[str, str] = field(default_factory=dict)


__all__ = [
    "AxisLine",
    "CompositeResource",
    "FontResource",
    "GlyphItem",
    "ImageItem",
    "MediaResource",
    "OfdDocumentRef",
    "OfdPageScene",
    "PageBuildContext",
    "PageRef",
    "ResourceRegistry",
    "TemplateRef",
    "TextLine",
]
