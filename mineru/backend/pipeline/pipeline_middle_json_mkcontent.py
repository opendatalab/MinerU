# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Any

from ...render.markdown import _get_title_level
from ...render.merge import (
    CJK_LANGS,
    _collect_text_for_lang_detection,
    _next_line_starts_with_lowercase_text,
    _normalize_text_content,
)
from ...render.merge_visual import _format_embedded_html, _normalize_visual_content
from ...types import BBox, Block, BlockType, ContentType, ContentTypeV2, IntBBox, Line
from ...utils.language import detect_lang
from ..utils.char_utils import is_hyphen_at_line_end


def _apply_visual_sub_type(para_content: dict[str, Any], para_block: Block) -> None:
    """将视觉父块的 sub_type 透传到 content_list 输出顶层。"""
    sub_type = para_block.sub_type
    if sub_type:
        para_content["sub_type"] = sub_type


def merge_adjacent_ref_text_blocks_for_content(para_blocks: list[Block]) -> list[Block]:
    """将连续 ref_text 包装为临时父 Block，供 content_list 输出为参考文献列表。"""
    merged_blocks: list[Block] = []
    ref_group: list[Block] = []

    def flush_ref_group() -> None:
        nonlocal ref_group
        if not ref_group:
            return
        if len(ref_group) == 1:
            merged_blocks.append(ref_group[0])
        else:
            merged_blocks.append(
                Block(
                    index=ref_group[0].index,
                    type=BlockType.REF_TEXT,
                    bbox=ref_group[0].bbox,
                    blocks=list(ref_group),
                )
            )
        ref_group = []

    for para_block in para_blocks or []:
        if para_block.type == BlockType.REF_TEXT:
            ref_group.append(para_block)
            continue

        flush_ref_group()
        merged_blocks.append(para_block)

    flush_ref_group()
    return merged_blocks


def _build_bbox(para_bbox: BBox, page_size: tuple[int, int]) -> IntBBox | None:
    if not para_bbox or not page_size:
        return None
    page_width, page_height = page_size
    x0, y0, x1, y1 = para_bbox
    return (
        int(x0 * 1000 / page_width),
        int(y0 * 1000 / page_height),
        int(x1 * 1000 / page_width),
        int(y1 * 1000 / page_height),
    )


def _get_ref_text_item_blocks(para_block: Block) -> list[Block]:
    return para_block.blocks or [para_block]


def _split_list_item_blocks(para_block: Block) -> list[Block]:
    item_blocks = []
    current_lines = []

    for line_idx, line in enumerate(para_block.lines):
        if line_idx > 0 and line._is_list_start and current_lines:
            item_blocks.append(
                {
                    "type": BlockType.TEXT,
                    "lines": current_lines,
                }
            )
            current_lines = []
        current_lines.append(line)

    if current_lines:
        item_blocks.append(
            {
                "type": BlockType.TEXT,
                "lines": current_lines,
            }
        )

    return item_blocks


def _get_body_data(para_block: Block) -> tuple[str, str]:
    def get_data_from_spans(lines: list[Line]) -> tuple[str, str]:
        for line in lines:
            for span in line.spans:
                span_type = span.type
                if span_type == ContentType.TABLE:
                    return span.image_path, span.content
                if span_type == ContentType.CHART:
                    return span.image_path, span.content
                if span_type == ContentType.IMAGE:
                    return span.image_path, _normalize_visual_content(span.content)
                if span_type == ContentType.INTERLINE_EQUATION:
                    return span.image_path, span.content
        return "", ""

    if para_block.blocks:
        for block in para_block.blocks:
            block_type = block.type
            if block_type in [
                BlockType.IMAGE_BODY,
                BlockType.TABLE_BODY,
                BlockType.CHART_BODY,
                BlockType.CODE_BODY,
            ]:
                result = get_data_from_spans(block.lines)
                if result != ("", "") or block_type == BlockType.CHART_BODY:
                    return result
        return "", ""

    return get_data_from_spans(para_block.lines)


def merge_para_with_text_v2(para_block: Block) -> list[dict[str, Any]]:
    block_lang = detect_lang(_collect_text_for_lang_detection(para_block))
    para_content: list[dict[str, Any]] = []
    para_type = para_block.type

    for line_idx, line in enumerate(para_block.lines):
        for span_idx, span in enumerate(line.spans):
            span_type = span.type

            if span_type == ContentType.TEXT:
                content = _normalize_text_content(span.content)
                if not content.strip():
                    continue

                output_type = ContentTypeV2.SPAN_PHONETIC if para_type == BlockType.PHONETIC else ContentTypeV2.SPAN_TEXT
                is_last_span = span_idx == len(line.spans) - 1

                if block_lang in CJK_LANGS:
                    rendered_content = content if is_last_span else f"{content} "
                else:
                    if (
                        is_last_span
                        and is_hyphen_at_line_end(content)
                        and _next_line_starts_with_lowercase_text(para_block, line_idx)
                    ):
                        rendered_content = content[:-1]
                    elif is_last_span and is_hyphen_at_line_end(content):
                        rendered_content = content
                    else:
                        rendered_content = f"{content} "

                if para_content and para_content[-1]["type"] == output_type:
                    para_content[-1]["content"] += rendered_content
                else:
                    para_content.append(
                        {
                            "type": output_type,
                            "content": rendered_content,
                        }
                    )
            elif span_type == ContentType.INLINE_EQUATION:
                content = span.content.strip()
                if content:
                    para_content.append(
                        {
                            "type": ContentTypeV2.SPAN_EQUATION_INLINE,
                            "content": content,
                        }
                    )

    if para_content and para_content[-1]["type"] in [
        ContentTypeV2.SPAN_TEXT,
        ContentTypeV2.SPAN_PHONETIC,
    ]:
        para_content[-1]["content"] = para_content[-1]["content"].rstrip()

    return para_content


def make_blocks_to_content_list_v2(para_block: Block, img_bucket_path: str, page_size: list[int]) -> dict[str, Any] | None:
    para_type = para_block.type
    para_content = None

    if para_type in [
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_NUMBER,
        BlockType.PAGE_FOOTNOTE,
    ]:
        if para_type == BlockType.HEADER:
            content_type = ContentTypeV2.PAGE_HEADER
        elif para_type == BlockType.FOOTER:
            content_type = ContentTypeV2.PAGE_FOOTER
        elif para_type == BlockType.ASIDE_TEXT:
            content_type = ContentTypeV2.PAGE_ASIDE_TEXT
        elif para_type == BlockType.PAGE_NUMBER:
            content_type = ContentTypeV2.PAGE_NUMBER
        elif para_type == BlockType.PAGE_FOOTNOTE:
            content_type = ContentTypeV2.PAGE_FOOTNOTE
        else:
            raise ValueError(f"Unknown para_type: {para_type}")
        para_content = {
            "type": content_type,
            "content": {
                f"{content_type}_content": merge_para_with_text_v2(para_block),
            },
        }
    elif para_type == BlockType.TITLE:
        title_level = _get_title_level(para_block)
        if title_level != 0:
            para_content = {
                "type": ContentTypeV2.TITLE,
                "content": {
                    "title_content": merge_para_with_text_v2(para_block),
                    "level": title_level,
                },
            }
        else:
            para_content = {
                "type": ContentTypeV2.PARAGRAPH,
                "content": {
                    "paragraph_content": merge_para_with_text_v2(para_block),
                },
            }
    elif para_type in [
        BlockType.TEXT,
        BlockType.ABSTRACT,
    ]:
        para_content = {
            "type": ContentTypeV2.PARAGRAPH,
            "content": {
                "paragraph_content": merge_para_with_text_v2(para_block),
            },
        }
    elif para_type == BlockType.INTERLINE_EQUATION:
        image_path, math_content = _get_body_data(para_block)
        para_content = {
            "type": ContentTypeV2.EQUATION_INTERLINE,
            "content": {
                "math_content": math_content,
                "math_type": "latex",
                "image_source": {"path": f"{img_bucket_path}/{image_path}"},
            },
        }
    elif para_type == BlockType.IMAGE:
        image_caption = []
        image_footnote = []
        image_path, image_content = _get_body_data(para_block)
        for block in para_block.blocks:
            if block.type == BlockType.IMAGE_CAPTION:
                image_caption.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.IMAGE_FOOTNOTE:
                image_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            "type": ContentTypeV2.IMAGE,
            "content": {
                "image_source": {"path": f"{img_bucket_path}/{image_path}"},
                "image_caption": image_caption,
                "image_footnote": image_footnote,
            },
        }
        if image_content or para_block.sub_type:
            para_content["content"]["content"] = image_content
        _apply_visual_sub_type(para_content, para_block)
    elif para_type == BlockType.TABLE:
        table_caption = []
        table_footnote = []
        image_path, html = _get_body_data(para_block)
        table_html = _format_embedded_html(html, img_bucket_path)
        table_nest_level = 2 if table_html.count("<table") > 1 else 1
        if "colspan" in table_html or "rowspan" in table_html or table_nest_level > 1:
            table_type = ContentTypeV2.TABLE_COMPLEX
        else:
            table_type = ContentTypeV2.TABLE_SIMPLE
        for block in para_block.blocks:
            if block.type == BlockType.TABLE_CAPTION:
                table_caption.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.TABLE_FOOTNOTE:
                table_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            "type": ContentTypeV2.TABLE,
            "content": {
                "image_source": {"path": f"{img_bucket_path}/{image_path}"},
                "table_caption": table_caption,
                "table_footnote": table_footnote,
                "html": table_html,
                "table_type": table_type,
                "table_nest_level": table_nest_level,
            },
        }
    elif para_type == BlockType.CHART:
        chart_caption = []
        chart_footnote = []
        image_path, _ = _get_body_data(para_block)
        for block in para_block.blocks:
            if block.type == BlockType.CHART_CAPTION:
                chart_caption.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.CHART_FOOTNOTE:
                chart_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            "type": ContentTypeV2.CHART,
            "content": {
                "image_source": {"path": f"{img_bucket_path}/{image_path}"},
                "content": "",
                "chart_caption": chart_caption,
                "chart_footnote": chart_footnote,
            },
        }
    elif para_type == BlockType.CODE:
        code_caption = []
        code_footnote = []
        code_content = []
        for block in para_block.blocks:
            if block.type == BlockType.CODE_CAPTION:
                code_caption.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.CODE_FOOTNOTE:
                code_footnote.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.CODE_BODY:
                code_content = merge_para_with_text_v2(block)

        sub_type = para_block.sub_type
        if sub_type == BlockType.CODE:
            para_content = {
                "type": ContentTypeV2.CODE,
                "content": {
                    "code_caption": code_caption,
                    "code_content": code_content,
                    "code_footnote": code_footnote,
                    "code_language": para_block.guess_lang or "txt",
                },
            }
        elif sub_type == BlockType.ALGORITHM:
            para_content = {
                "type": ContentTypeV2.ALGORITHM,
                "content": {
                    "algorithm_caption": code_caption,
                    "algorithm_content": code_content,
                    "algorithm_footnote": code_footnote,
                },
            }
        else:
            raise ValueError(f"Unknown code sub_type: {sub_type}")
    elif para_type == BlockType.REF_TEXT:
        list_items = []
        for block in _get_ref_text_item_blocks(para_block):
            item_content = merge_para_with_text_v2(block)
            if item_content:
                list_items.append(
                    {
                        "item_type": "text",
                        "item_content": item_content,
                    }
                )
        para_content = {
            "type": ContentTypeV2.LIST,
            "content": {
                "list_type": ContentTypeV2.LIST_REF,
                "list_items": list_items,
            },
        }
    elif para_type == BlockType.LIST:
        list_items = []
        for block in _split_list_item_blocks(para_block):
            item_content = merge_para_with_text_v2(block)
            if item_content:
                list_items.append(
                    {
                        "item_type": "text",
                        "item_content": item_content,
                    }
                )
        para_content = {
            "type": ContentTypeV2.LIST,
            "content": {
                "list_type": ContentTypeV2.LIST_TEXT,
                "attribute": para_block._list_attribute or "unordered",
                "list_items": list_items,
            },
        }
    elif para_type == BlockType.INDEX:
        list_items = []
        for block in _split_list_item_blocks(para_block):
            item_content = merge_para_with_text_v2(block)
            if item_content:
                list_items.append(
                    {
                        "item_type": "text",
                        "item_content": item_content,
                    }
                )
        para_content = {
            "type": ContentTypeV2.INDEX,
            "content": {
                "list_type": ContentTypeV2.LIST_TEXT,
                "list_items": list_items,
            },
        }
    if not para_content:
        return None

    bbox = _build_bbox(para_block.bbox, page_size)
    if bbox:
        para_content["bbox"] = bbox

    return para_content
