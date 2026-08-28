# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到按页 Content List V2 的渲染实现。"""

from __future__ import annotations

from typing import Any

from ....config import LatexDelimitersConfig, config
from ....types import (
    RAW_ALGORITHM,
    AlgorithmBodyBlock,
    BlockType,
    ChartBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    ImageBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageFootnoteBlock,
    ParagraphTitleBlock,
    TableBlock,
    TextBlock,
)
from .common import (
    PageRenderUnit,
    ReferenceGroup,
    classify_table,
    flatten_index_leaves,
    flatten_list_leaves,
    infer_list_attribute,
    iter_page_units,
    normalize_bbox,
    normalized_index_content,
    render_annotation_spans,
    render_embedded_content,
    resolve_legacy_image_source,
    serialize_v2_spans,
    string_as_v2_spans,
    unit_bbox,
    visual_body,
)

_IMAGE_CAPTION_TYPES = {str(BlockType.IMAGE_CAPTION)}
_IMAGE_FOOTNOTE_TYPES = {str(BlockType.IMAGE_FOOTNOTE)}
_TABLE_CAPTION_TYPES = {str(BlockType.TABLE_CAPTION)}
_TABLE_FOOTNOTE_TYPES = {str(BlockType.TABLE_FOOTNOTE)}
_CHART_CAPTION_TYPES = {str(BlockType.CHART_CAPTION)}
_CHART_FOOTNOTE_TYPES = {str(BlockType.CHART_FOOTNOTE)}
_CODE_CAPTION_TYPES = {str(BlockType.CODE_CAPTION)}
_CODE_FOOTNOTE_TYPES = {str(BlockType.CODE_FOOTNOTE)}


def render_content_list_v2(
    middle_json: MiddleJson,
    *,
    asset_base_url: str = "",
) -> list[list[dict[str, Any]]]:
    """把严格 MiddleJson 无副作用地渲染为按页 Content List V2。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render_content_list_v2 expects a MiddleJson instance")
    if not isinstance(asset_base_url, str):
        raise TypeError("asset_base_url must be a string")

    delimiters = config.render.latex_delimiters
    output: list[list[dict[str, Any]]] = []
    for page in middle_json.pages:
        page_items: list[dict[str, Any]] = []
        for unit in iter_page_units(page.blocks):
            item = _render_unit(unit, asset_base_url=asset_base_url, delimiters=delimiters)
            if item is not None:
                page_items.append(item)
        output.append(page_items)
    return output


def _render_unit(
    unit: PageRenderUnit,
    *,
    asset_base_url: str,
    delimiters: LatexDelimitersConfig,
) -> dict[str, Any] | None:
    """把一个页面渲染单元转换为 V2 item，并补齐归一化 bbox。"""
    item = _render_unit_content(unit, asset_base_url=asset_base_url, delimiters=delimiters)
    if item is None:
        return None
    if bbox := normalize_bbox(unit_bbox(unit)):
        item["bbox"] = bbox
    return item


def _render_unit_content(
    unit: PageRenderUnit,
    *,
    asset_base_url: str,
    delimiters: LatexDelimitersConfig,
) -> dict[str, Any] | None:
    """按严格 block 类型构造 V2 的 type + content 字段。"""
    if isinstance(unit, ReferenceGroup):
        return _reference_list_item(unit)
    if isinstance(unit, TextBlock):
        return {"type": "paragraph", "content": {"paragraph_content": serialize_v2_spans(unit.content)}}
    if isinstance(unit, (DocTitleBlock, ParagraphTitleBlock)):
        item: dict[str, Any] = {
            "type": "title",
            "content": {
                "title_content": serialize_v2_spans(unit.content),
                "level": unit.level,
            },
        }
        _add_anchor(item, unit.anchor)
        return item
    if isinstance(unit, PageFootnoteBlock):
        item = {
            "type": "page_footnote",
            "content": {"page_footnote_content": serialize_v2_spans(unit.content)},
        }
        _add_anchor(item, unit.anchor)
        return item
    if isinstance(unit, PageAuxTextBlock):
        content_type = _page_content_type(str(unit.type))
        return {
            "type": content_type,
            "content": {f"{content_type}_content": serialize_v2_spans(unit.content)},
        }
    if isinstance(unit, EquationBlock):
        return {
            "type": "equation_interline",
            "content": {
                "math_content": unit.content.strip(),
                "math_type": "latex",
                "image_source": {"path": resolve_legacy_image_source(unit, asset_base_url)},
            },
        }
    if isinstance(unit, ListBlock):
        return _render_list(unit)
    if isinstance(unit, IndexBlock):
        return _render_index(unit)
    if isinstance(unit, ImageBlock):
        return _render_image(unit, asset_base_url, delimiters)
    if isinstance(unit, TableBlock):
        return _render_table(unit, asset_base_url, delimiters)
    if isinstance(unit, ChartBlock):
        return _render_chart(unit, asset_base_url, delimiters)
    if isinstance(unit, CodeBlock):
        return _render_code(unit)
    return None


def _page_content_type(block_type: str) -> str:
    """把 MiddleJson 页面辅助块类型映射为 V2 类型。"""
    mapping = {
        str(BlockType.HEADER): "page_header",
        str(BlockType.FOOTER): "page_footer",
        str(BlockType.PAGE_NUMBER): "page_number",
        str(BlockType.ASIDE_TEXT): "page_aside_text",
    }
    return mapping[block_type]


def _reference_list_item(group: ReferenceGroup) -> dict[str, Any]:
    """把连续参考文献块转换为一个 reference_list item。"""
    return {
        "type": "list",
        "content": {
            "list_type": "reference_list",
            "list_items": [
                {"item_type": "text", "item_content": serialize_v2_spans(block.content)}
                for block in group.blocks
                if block.content
            ],
        },
    }


def _render_list(block: ListBlock) -> dict[str, Any]:
    """把递归列表展平为统一 V2 list item。"""
    list_type = "reference_list" if block.sub_type == BlockType.REF_TEXT else "text_list"
    content: dict[str, Any] = {
        "list_type": list_type,
        "list_items": [
            {"item_type": "text", "item_content": serialize_v2_spans(leaf.block.content)}
            for leaf in flatten_list_leaves(block)
            if leaf.block.content
        ],
    }
    if list_type == "text_list":
        content["attribute"] = infer_list_attribute(block)
    return {"type": "list", "content": content}


def _render_index(block: IndexBlock) -> dict[str, Any]:
    """把递归目录展平为统一 V2 index item。"""
    list_items: list[dict[str, Any]] = []
    for leaf in flatten_index_leaves(block):
        content = normalized_index_content(leaf.block)
        if not content:
            continue
        item: dict[str, Any] = {
            "item_type": "text",
            "item_content": serialize_v2_spans(content),
        }
        if isinstance(leaf.block, (DocTitleBlock, ParagraphTitleBlock)):
            _add_anchor(item, leaf.block.anchor)
        list_items.append(item)
    return {
        "type": "index",
        "content": {
            "list_type": "text_list",
            "list_items": list_items,
        },
    }


def _render_image(
    block: ImageBlock,
    asset_base_url: str,
    delimiters: LatexDelimitersConfig,
) -> dict[str, Any]:
    """构造 V2 图片 item。"""
    body = visual_body(block)
    body_content = render_embedded_content(body.content, asset_base_url=asset_base_url, delimiters=delimiters)
    content: dict[str, Any] = {
        "image_source": {"path": resolve_legacy_image_source(body, asset_base_url)},
        "image_caption": render_annotation_spans(block, _IMAGE_CAPTION_TYPES),
        "image_footnote": render_annotation_spans(block, _IMAGE_FOOTNOTE_TYPES),
    }
    if body_content or block.sub_type:
        content["content"] = body_content
    item: dict[str, Any] = {"type": "image", "content": content}
    if block.sub_type:
        item["sub_type"] = block.sub_type
    return item


def _render_table(
    block: TableBlock,
    asset_base_url: str,
    delimiters: LatexDelimitersConfig,
) -> dict[str, Any]:
    """构造 V2 表格 item并计算复杂度。"""
    body = visual_body(block)
    html = render_embedded_content(body.content, asset_base_url=asset_base_url, delimiters=delimiters)
    table_type, table_nest_level = classify_table(html)
    return {
        "type": "table",
        "content": {
            "image_source": {"path": resolve_legacy_image_source(body, asset_base_url)},
            "table_caption": render_annotation_spans(block, _TABLE_CAPTION_TYPES),
            "table_footnote": render_annotation_spans(block, _TABLE_FOOTNOTE_TYPES),
            "html": html,
            "table_type": table_type,
            "table_nest_level": table_nest_level,
        },
    }


def _render_chart(
    block: ChartBlock,
    asset_base_url: str,
    delimiters: LatexDelimitersConfig,
) -> dict[str, Any]:
    """构造 V2 图表 item。"""
    body = visual_body(block)
    item: dict[str, Any] = {
        "type": "chart",
        "content": {
            "image_source": {"path": resolve_legacy_image_source(body, asset_base_url)},
            "content": render_embedded_content(body.content, asset_base_url=asset_base_url, delimiters=delimiters),
            "chart_caption": render_annotation_spans(block, _CHART_CAPTION_TYPES),
            "chart_footnote": render_annotation_spans(block, _CHART_FOOTNOTE_TYPES),
        },
    }
    if block.sub_type:
        item["sub_type"] = block.sub_type
    return item


def _render_code(block: CodeBlock) -> dict[str, Any]:
    """构造 V2 代码或算法 item。"""
    body = visual_body(block)
    captions = render_annotation_spans(block, _CODE_CAPTION_TYPES)
    footnotes = render_annotation_spans(block, _CODE_FOOTNOTE_TYPES)
    if block.sub_type == RAW_ALGORITHM:
        if not isinstance(body, AlgorithmBodyBlock):
            raise TypeError("algorithm subtype requires AlgorithmBodyBlock")
        return {
            "type": "algorithm",
            "content": {
                "algorithm_caption": captions,
                "algorithm_content": serialize_v2_spans(body.content),
                "algorithm_footnote": footnotes,
            },
        }
    if not isinstance(body, CodeBodyBlock):
        raise TypeError("code subtype requires CodeBodyBlock")
    return {
        "type": "code",
        "content": {
            "code_caption": captions,
            "code_content": string_as_v2_spans(body.content),
            "code_footnote": footnotes,
            "code_language": block.guess_lang or "txt",
        },
    }


def _add_anchor(item: dict[str, Any], anchor: str | None) -> None:
    """把非空文档锚点写入 V2 item 或目录叶子。"""
    if anchor and anchor.strip():
        item["anchor"] = anchor.strip()


__all__ = ["render_content_list_v2"]
