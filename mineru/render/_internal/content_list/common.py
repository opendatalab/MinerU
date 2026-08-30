# Copyright (c) Opendatalab. All rights reserved.
"""Content List V1/V2 共用的严格 MiddleJson 投影能力。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias

from ....backend.postprocess.inline import inline_plain_text
from ....config import LatexDelimitersConfig
from ....types import (
    BBox,
    AlgorithmBodyBlock,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    CodeBodyBlock,
    CodeInlineSpan,
    DocTitleBlock,
    EquationInlineSpan,
    HyperlinkSpan,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    ImagePayloadBlock,
    IndexBlock,
    InlineSpan,
    ListBlock,
    PageBlock,
    ParagraphTitleBlock,
    RefTextBlock,
    TableAnnotationBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
    TextSpan,
)
from ..common.index import strip_index_page_tail
from ..common.list_items import parse_list_item_marker
from ..markdown.assets import normalize_image_source, resolve_image_source
from ..markdown.inline import render_inline_content
from ..markdown.table import format_embedded_html

VisualBlock: TypeAlias = ImageBlock | TableBlock | ChartBlock | CodeBlock
VisualBodyBlock: TypeAlias = ImageBodyBlock | TableBodyBlock | ChartBodyBlock | CodeBodyBlock | AlgorithmBodyBlock
VisualAnnotationBlock: TypeAlias = ImageAnnotationBlock | TableAnnotationBlock | ChartAnnotationBlock | CodeAnnotationBlock
ListLeafBlock: TypeAlias = TextBlock | RefTextBlock
IndexLeafBlock: TypeAlias = TextBlock | DocTitleBlock | ParagraphTitleBlock


@dataclass(frozen=True, slots=True)
class ReferenceGroup:
    """保存同页连续参考文献块，供两个兼容 renderer 统一输出列表。"""

    blocks: tuple[RefTextBlock, ...]


@dataclass(frozen=True, slots=True)
class ListLeaf:
    """保存递归列表叶子及其相对深度。"""

    block: ListLeafBlock
    depth: int


@dataclass(frozen=True, slots=True)
class IndexLeaf:
    """保存递归目录叶子及其相对深度。"""

    block: IndexLeafBlock
    depth: int


PageRenderUnit: TypeAlias = PageBlock | ReferenceGroup


def iter_page_units(blocks: list[PageBlock]) -> Iterator[PageRenderUnit]:
    """保持阅读顺序遍历页面，并把连续 ref_text 合并为一个渲染单元。"""
    reference_blocks: list[RefTextBlock] = []

    def flush_references() -> Iterator[ReferenceGroup]:
        """输出并清空当前参考文献分组。"""
        nonlocal reference_blocks
        if not reference_blocks:
            return iter(())
        group = ReferenceGroup(tuple(reference_blocks))
        reference_blocks = []
        return iter((group,))

    for block in blocks:
        if isinstance(block, RefTextBlock):
            reference_blocks.append(block)
            continue
        yield from flush_references()
        yield block
    yield from flush_references()


def unit_bbox(unit: PageRenderUnit) -> BBox | None:
    """返回普通块 bbox，或计算参考文献分组的最小包围盒。"""
    if not isinstance(unit, ReferenceGroup):
        return unit.bbox
    bboxes = [block.bbox for block in unit.blocks if block.bbox is not None]
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def normalize_bbox(bbox: BBox | None) -> list[int] | None:
    """把 MiddleJson 的 0-1 bbox 转换为 Content List 的 0-1000 整数框。"""
    if bbox is None:
        return None
    return [int(value * 1000) for value in bbox]


def resolve_legacy_image_source(block: ImagePayloadBlock, asset_base_url: str) -> str:
    """按统一图片优先级解析并规范化旧 Content List 资源字段。"""
    source = resolve_image_source(block, asset_base_url)
    return normalize_image_source(source) if source else ""


def visual_body(block: VisualBlock) -> VisualBodyBlock:
    """返回严格视觉父块中的唯一 body。"""
    for child in block.content:
        if isinstance(child, (ImageBodyBlock, TableBodyBlock, ChartBodyBlock, CodeBodyBlock, AlgorithmBodyBlock)):
            return child
    raise ValueError(f"Missing visual body: {block.type}")


def sorted_annotations(block: VisualBlock, accepted_types: set[str]) -> list[VisualAnnotationBlock]:
    """按 index 优先、原位置兜底的顺序返回指定视觉说明。"""
    annotations: list[tuple[int, VisualAnnotationBlock]] = []
    for position, child in enumerate(block.content):
        if isinstance(child, (ImageAnnotationBlock, TableAnnotationBlock, ChartAnnotationBlock, CodeAnnotationBlock)):
            if str(child.type) in accepted_types:
                annotations.append((position, child))
    annotations.sort(
        key=lambda item: (
            item[1].index is None,
            item[1].index if item[1].index is not None else 0,
            item[0],
        )
    )
    return [child for _, child in annotations]


def render_annotation_texts(
    block: VisualBlock,
    accepted_types: set[str],
    delimiters: LatexDelimitersConfig,
) -> list[str]:
    """把一组视觉说明渲染为 V1 使用的 Markdown 字符串数组。"""
    return [render_inline_content(child.content, delimiters) for child in sorted_annotations(block, accepted_types)]


def render_annotation_spans(block: VisualBlock, accepted_types: set[str]) -> list[dict[str, object]]:
    """把一组视觉说明按顺序展平为 V2 span 数组。"""
    spans: list[dict[str, object]] = []
    for child in sorted_annotations(block, accepted_types):
        spans.extend(serialize_v2_spans(child.content))
    return spans


def render_embedded_content(
    content: str,
    *,
    asset_base_url: str,
    delimiters: LatexDelimitersConfig,
) -> str:
    """保留视觉结构内容，只重写相对图片地址和安全行内公式标签。"""
    if not content:
        return ""
    return format_embedded_html(
        content,
        asset_base_url=asset_base_url,
        delimiters=delimiters,
    ).strip()


def serialize_v2_spans(content: list[InlineSpan]) -> list[dict[str, object]]:
    """把严格 InlineSpan 转换为统一的 3.4.5 风格 V2 span。"""
    return [_serialize_v2_span(span) for span in content]


def _serialize_v2_span(span: InlineSpan) -> dict[str, object]:
    """序列化单个 V2 span，并显式收敛链接 children 与样式字段。"""
    if isinstance(span, TextSpan):
        payload: dict[str, object] = {"type": "text", "content": span.content}
        if span.styles:
            payload["style"] = list(span.styles)
        return payload
    if isinstance(span, EquationInlineSpan):
        return {"type": "equation_inline", "content": span.content.strip()}
    if isinstance(span, CodeInlineSpan):
        return {"type": "code_inline", "content": span.content}
    if isinstance(span, HyperlinkSpan):
        children = [_serialize_v2_span(child) for child in span.content]
        payload = {
            "type": "hyperlink",
            "content": inline_plain_text(span.content),
            "url": span.url,
        }
        if len(children) == 1 and children[0]["type"] == "text":
            if style := children[0].get("style"):
                payload["style"] = style
        else:
            payload["children"] = children
        return payload
    raise TypeError(f"Unsupported InlineSpan type: {type(span).__name__}")


def string_as_v2_spans(content: str) -> list[dict[str, object]]:
    """把代码字符串包装为 V2 文本 span，空字符串保持为空数组。"""
    if not content:
        return []
    return [{"type": "text", "content": content}]


def flatten_list_leaves(block: ListBlock, depth: int = 0) -> list[ListLeaf]:
    """按源顺序递归展平列表叶子。"""
    leaves: list[ListLeaf] = []
    for child in block.content:
        if isinstance(child, ListBlock):
            leaves.extend(flatten_list_leaves(child, depth + 1))
        else:
            leaves.append(ListLeaf(block=child, depth=depth))
    return leaves


def infer_list_attribute(block: ListBlock) -> str:
    """根据最浅层有效条目的 marker 推断 ordered/unordered。"""
    leaves = flatten_list_leaves(block)
    if not leaves:
        return "unordered"
    minimum_depth = min(leaf.depth for leaf in leaves)
    kinds = [
        parse_list_item_marker(leaf.block.content).kind
        for leaf in leaves
        if leaf.depth == minimum_depth and inline_plain_text(leaf.block.content).strip()
    ]
    return "ordered" if kinds and all(kind == "ordered" for kind in kinds) else "unordered"


def flatten_index_leaves(block: IndexBlock, depth: int = 0) -> list[IndexLeaf]:
    """按源顺序递归展平目录叶子。"""
    leaves: list[IndexLeaf] = []
    for child in block.content:
        if isinstance(child, IndexBlock):
            leaves.extend(flatten_index_leaves(child, depth + 1))
        else:
            leaves.append(IndexLeaf(block=child, depth=depth))
    return leaves


def normalized_index_content(block: IndexLeafBlock) -> list[InlineSpan]:
    """删除目录叶子可信页码尾缀，并保留其余行内语义。"""
    return strip_index_page_tail(block.content)


def classify_table(content: str) -> tuple[str, int]:
    """按 rowspan、colspan 和嵌套 table 判断 V2 表格类型。"""
    table_count = len(re.findall(r"<table\b", content, flags=re.IGNORECASE))
    table_nest_level = 2 if table_count > 1 else 1
    complex_table = table_nest_level > 1 or re.search(r"\b(?:rowspan|colspan)\b", content, flags=re.IGNORECASE)
    return ("complex_table" if complex_table else "simple_table", table_nest_level)


__all__ = [
    "IndexLeaf",
    "ListLeaf",
    "PageRenderUnit",
    "ReferenceGroup",
    "classify_table",
    "flatten_index_leaves",
    "flatten_list_leaves",
    "infer_list_attribute",
    "iter_page_units",
    "normalize_bbox",
    "normalized_index_content",
    "render_annotation_spans",
    "render_annotation_texts",
    "render_embedded_content",
    "resolve_legacy_image_source",
    "serialize_v2_spans",
    "sorted_annotations",
    "string_as_v2_spans",
    "unit_bbox",
    "visual_body",
]
