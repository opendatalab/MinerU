# Copyright (c) Opendatalab. All rights reserved.
"""单页 raw block 的内容清理、列表整理和视觉分组流水线。"""

from __future__ import annotations

import re
from typing import Any

from mineru.types import (
    RAW_ALGORITHM,
    RAW_CAPTION,
    RAW_FOOTNOTE,
    BlockType,
    VISUAL_MAIN_TYPES,
)
from mineru.utils.guess_suffix_or_lang import guess_language_by_text

from mineru.utils.text_utils import clean_isolated_formula

from .content import clean_content, code_content_clean
from .lists import (
    fix_office_index_blocks,
    fix_office_list_blocks,
    fix_pdf_index_blocks,
    fix_pdf_list_blocks,
)
from .visual import (
    fallback_inline_caption_fragments,
    fallback_leading_table_continuation_captions,
    fallback_no_bbox_caption_fragments,
    regroup_visual_blocks,
)

BlockDict = dict[str, Any]

_REPLACED_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.REF_TEXT,
    BlockType.LIST,
    BlockType.INDEX,
    RAW_CAPTION,
    RAW_FOOTNOTE,
    BlockType.IMAGE_BODY,
    BlockType.TABLE_BODY,
    BlockType.CHART_BODY,
    BlockType.CODE_BODY,
    BlockType.IMAGE_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_CAPTION,
    BlockType.CHART_FOOTNOTE,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
}


def _has_inline_formula_content(content: str | None) -> bool:
    """判断 content 是否包含成对行内公式标记。"""
    return bool(content) and content.count("<eq>") == content.count("</eq>") and content.count("<eq>") > 0


def _normalize_raw_blocks(page_model_list: list[BlockDict]) -> list[BlockDict]:
    """原地规范化 raw block 类型、内容、索引和代码子类型。"""
    blocks: list[BlockDict] = []
    for index, block in enumerate(page_model_list):
        code_block_sub_type = None
        block_type = block.get("type", "")
        block_content = block.get("content", "")
        if block_type == BlockType.IMAGE:
            block_type = BlockType.IMAGE_BODY
        elif block_type == BlockType.TABLE:
            block_type = BlockType.TABLE_BODY
        elif block_type == BlockType.CHART:
            block_type = BlockType.CHART_BODY
        elif block_type in [BlockType.CODE, RAW_ALGORITHM]:
            code_block_sub_type = block_type
            block_content = code_content_clean(block_content)
            block_type = BlockType.CODE_BODY
        elif block_type == BlockType.EQUATION:
            block_content = clean_isolated_formula(block_content)

        if block_type in [BlockType.IMAGE_BODY, BlockType.CHART_BODY] and block_content is None:
            block_content = ""
        if block_type == BlockType.DOC_TITLE:
            block["level"] = 1
        elif block_type == BlockType.PARAGRAPH_TITLE:
            raw_level = block.get("level")
            normalized_level = raw_level if type(raw_level) is int else 2
            block["level"] = min(max(normalized_level, 2), 6)

        if block_type not in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.CHART_BODY]:
            if block_content:
                block_content = clean_content(block_content) or ""
            if block_type in [BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE] and block_content:
                block_content = re.sub(r"\n\s*", " ", block_content).strip()
            if (
                block_type == BlockType.CODE_BODY
                and code_block_sub_type == BlockType.CODE
                and _has_inline_formula_content(block_content)
            ):
                code_block_sub_type = RAW_ALGORITHM

        block["type"] = block_type
        block["content"] = block_content
        block["index"] = index
        if code_block_sub_type:
            block["sub_type"] = code_block_sub_type
        blocks.append(block)
    return blocks


def _apply_caption_fallbacks(blocks: list[BlockDict], *, use_bbox: bool) -> None:
    """按 PDF 或 Office 结构应用视觉标题兜底规则。"""
    if use_bbox:
        fallback_inline_caption_fragments(blocks, VISUAL_MAIN_TYPES)
        fallback_leading_table_continuation_captions(blocks, VISUAL_MAIN_TYPES)
    else:
        fallback_no_bbox_caption_fragments(blocks, VISUAL_MAIN_TYPES)


def _partition_textual_blocks(
    blocks: list[BlockDict],
) -> tuple[list[BlockDict], list[BlockDict], list[BlockDict], list[BlockDict]]:
    """按 text、ref_text、list、index 四类分区 raw block。"""
    text_blocks: list[BlockDict] = []
    ref_text_blocks: list[BlockDict] = []
    list_blocks: list[BlockDict] = []
    index_blocks: list[BlockDict] = []
    for block in blocks:
        block_type = block["type"]
        if block_type == BlockType.TEXT:
            text_blocks.append(block)
        elif block_type == BlockType.REF_TEXT:
            ref_text_blocks.append(block)
        elif block_type == BlockType.LIST:
            list_blocks.append(block)
        elif block_type == BlockType.INDEX:
            index_blocks.append(block)
    return text_blocks, ref_text_blocks, list_blocks, index_blocks


def _prepare_lists_and_indices(
    blocks: list[BlockDict],
    *,
    use_bbox: bool,
) -> tuple[list[BlockDict], list[BlockDict], list[BlockDict], list[BlockDict]]:
    """根据文档类型整理列表、目录以及被列表吸收的文本块。"""
    text_blocks, ref_text_blocks, list_blocks, index_blocks = _partition_textual_blocks(blocks)
    if use_bbox:
        list_blocks, text_blocks, ref_text_blocks = fix_pdf_list_blocks(
            list_blocks,
            text_blocks,
            ref_text_blocks,
        )
        index_blocks = fix_pdf_index_blocks(index_blocks)
    else:
        list_blocks = fix_office_list_blocks(list_blocks)
        index_blocks = fix_office_index_blocks(index_blocks)
    return text_blocks, ref_text_blocks, list_blocks, index_blocks


def _annotate_code_languages(code_blocks: list[BlockDict]) -> None:
    """为普通代码主体推断语言，算法块保持既有子类型。"""
    for code_block in code_blocks:
        if code_block["sub_type"] != BlockType.CODE:
            continue
        for sub_block in code_block["content"]:
            if sub_block.get("type") == BlockType.CODE_BODY:
                code_block["guess_lang"] = guess_language_by_text(sub_block.get("content", ""))
                break


def process_page_blocks(
    page_model_list: list[BlockDict],
    *,
    use_bbox: bool | None = None,
) -> list[BlockDict]:
    """按固定阶段将单页 raw model-list 转换为可对象化的顶层 blocks。"""
    resolved_use_bbox = any(block.get("bbox") for block in page_model_list) if use_bbox is None else use_bbox
    blocks = _normalize_raw_blocks(page_model_list)
    _apply_caption_fallbacks(blocks, use_bbox=resolved_use_bbox)
    text_blocks, ref_text_blocks, list_blocks, index_blocks = _prepare_lists_and_indices(
        blocks,
        use_bbox=resolved_use_bbox,
    )
    visual_groups, unmatched_child_blocks = regroup_visual_blocks(
        blocks,
        use_bbox=resolved_use_bbox,
    )
    image_blocks = visual_groups[BlockType.IMAGE]
    table_blocks = visual_groups[BlockType.TABLE]
    chart_blocks = visual_groups[BlockType.CHART]
    code_blocks = visual_groups[BlockType.CODE]
    _annotate_code_languages(code_blocks)
    for block in unmatched_child_blocks:
        block["type"] = BlockType.TEXT
        text_blocks.append(block)

    result = [block for block in blocks if block["type"] not in _REPLACED_BLOCK_TYPES]
    result.extend(
        list_blocks + text_blocks + ref_text_blocks + index_blocks + image_blocks + table_blocks + chart_blocks + code_blocks
    )
    return result
