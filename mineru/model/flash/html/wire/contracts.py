# Copyright (c) Opendatalab. All rights reserved.
"""MinerU HTML v1 canonical wire 的内部类型契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, Union

from lxml import etree  # type: ignore[reportMissingImports]

from .....types import BlockType


MINERU_HTML_VERSION = "1"
WIRE_BLOCK_CLASS = "mineru-block"
WIRE_DOCUMENT_CLASS = "mineru-document"
WIRE_INDEX_CLASS = "mineru-index"
WIRE_LIST_CONTENT_CLASS = "mineru-list-content"
WIRE_LIST_MARKER_CLASS = "mineru-list-marker"
WIRE_PAGE_BREAK_CLASS = "mineru-page-break"
WIRE_PAGE_CLASS = "mineru-page"
WIRE_VISUAL_BODY_CLASS = "mineru-visual-body"
WireRenderMode: TypeAlias = Literal["default", "full"]
WireFallbackReason: TypeAlias = Literal["unsupported_version", "non_canonical_wire"]


@dataclass(frozen=True, slots=True)
class TextWireSpec:
    """保存一个文本类顶层 block 的 canonical 节点与元数据。"""

    wrapper: etree._Element
    content_root: etree._Element
    block_type: BlockType
    page_idx: int
    block_index: int | None


@dataclass(frozen=True, slots=True)
class EquationWireSpec:
    """保存一个行间公式或公式图片 carrier。"""

    wrapper: etree._Element
    content_root: etree._Element
    page_idx: int
    block_index: int | None


@dataclass(frozen=True, slots=True)
class AnnotationWireSpec:
    """保存 visual caption/footnote 的 canonical 行内容器。"""

    element: etree._Element
    block_type: BlockType


@dataclass(frozen=True, slots=True)
class RichVisualBodyWireSpec:
    """保存普通 image/chart body 的主图片与规范化富内容片段。"""

    element: etree._Element
    parent_type: BlockType
    sub_type: str
    primary_image: etree._Element | None
    content_fragment: etree._Element | None


@dataclass(frozen=True, slots=True)
class FlowchartBodyWireSpec:
    """保存 flowchart 源码与可选 raster fallback。"""

    element: etree._Element
    source_element: etree._Element
    fallback_image: etree._Element | None


@dataclass(frozen=True, slots=True)
class TableBodyWireSpec:
    """保存 table body 的唯一 canonical 载荷。"""

    element: etree._Element
    kind: Literal["empty", "html", "text", "image"]
    payload_element: etree._Element | None


@dataclass(frozen=True, slots=True)
class CodeBodyWireSpec:
    """保存 code/algorithm body 的唯一 canonical 内容载体。"""

    element: etree._Element
    kind: Literal["code", "algorithm"]
    content_element: etree._Element


VisualBodyWireSpec: TypeAlias = Union[
    RichVisualBodyWireSpec,
    FlowchartBodyWireSpec,
    TableBodyWireSpec,
    CodeBodyWireSpec,
]
VisualChildWireSpec: TypeAlias = Union[VisualBodyWireSpec, AnnotationWireSpec]


@dataclass(frozen=True, slots=True)
class VisualWireSpec:
    """保存 visual 顶层 block 与已按 DOM 顺序解析的子节点。"""

    wrapper: etree._Element
    content_root: etree._Element
    block_type: BlockType
    page_idx: int
    block_index: int | None
    sub_type: str
    guess_lang: str
    children: tuple[VisualChildWireSpec, ...]


@dataclass(frozen=True, slots=True)
class ListLeafWireSpec:
    """保存一个列表叶子的内容 carrier、marker 与公开类型。"""

    block_type: BlockType
    block_index: int | None
    content_element: etree._Element | None
    marker: str


@dataclass(frozen=True, slots=True)
class ListWireSpec:
    """保存一个 canonical 列表及其递归子项。"""

    element: etree._Element
    block_index: int | None
    ordered: bool
    start: int
    sub_type: str
    classes: frozenset[str]
    children: tuple[Union[ListLeafWireSpec, "ListWireSpec"], ...]


@dataclass(frozen=True, slots=True)
class ListBlockWireSpec:
    """保存顶层 ListBlock wrapper 与列表树。"""

    wrapper: etree._Element
    page_idx: int
    block_index: int | None
    root: ListWireSpec


@dataclass(frozen=True, slots=True)
class IndexLeafWireSpec:
    """保存一个目录叶子的 canonical 内容 carrier 与元数据。"""

    block_type: BlockType
    block_index: int | None
    content_element: etree._Element | None
    anchor: str
    level: int | None


@dataclass(frozen=True, slots=True)
class IndexWireSpec:
    """保存一个 canonical 目录列表及其递归子项。"""

    element: etree._Element
    block_index: int | None
    children: tuple[Union[IndexLeafWireSpec, "IndexWireSpec"], ...]


@dataclass(frozen=True, slots=True)
class IndexBlockWireSpec:
    """保存顶层 IndexBlock wrapper 与目录树。"""

    wrapper: etree._Element
    page_idx: int
    block_index: int | None
    root: IndexWireSpec


PageWireSpec: TypeAlias = Union[
    TextWireSpec,
    EquationWireSpec,
    VisualWireSpec,
    ListBlockWireSpec,
    IndexBlockWireSpec,
]


@dataclass(frozen=True, slots=True)
class MineruHtmlWirePlan:
    """保存一次无资源副作用的完整 canonical wire 解析结果。"""

    root: etree._Element
    render_mode: WireRenderMode
    blocks: tuple[PageWireSpec, ...]


@dataclass(frozen=True, slots=True)
class WireDecodeResult:
    """区分未命中 wire、精确解码成功与需要通用回退。"""

    blocks: list[dict[str, object]] | None
    fallback_reason: WireFallbackReason | None = None


__all__ = [
    "AnnotationWireSpec",
    "CodeBodyWireSpec",
    "EquationWireSpec",
    "FlowchartBodyWireSpec",
    "IndexBlockWireSpec",
    "IndexLeafWireSpec",
    "IndexWireSpec",
    "ListBlockWireSpec",
    "ListLeafWireSpec",
    "ListWireSpec",
    "MINERU_HTML_VERSION",
    "MineruHtmlWirePlan",
    "PageWireSpec",
    "RichVisualBodyWireSpec",
    "TableBodyWireSpec",
    "TextWireSpec",
    "VisualBodyWireSpec",
    "VisualChildWireSpec",
    "VisualWireSpec",
    "WireDecodeResult",
    "WireFallbackReason",
    "WireRenderMode",
    "WIRE_BLOCK_CLASS",
    "WIRE_DOCUMENT_CLASS",
    "WIRE_INDEX_CLASS",
    "WIRE_LIST_CONTENT_CLASS",
    "WIRE_LIST_MARKER_CLASS",
    "WIRE_PAGE_BREAK_CLASS",
    "WIRE_PAGE_CLASS",
    "WIRE_VISUAL_BODY_CLASS",
]
