# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re
from typing import Any

from loguru import logger

from ...types import Block, BlockType
from ...utils.guess_suffix_or_lang import guess_language_by_text
from ..utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from ..utils.content_block_draft import VlmContentBlockDraft
from ..utils.visual_magic_model_utils import (
    GENERIC_CHILD_TYPES,
    IMAGE_BLOCK_BODY,
    VISUAL_MAIN_TYPES,
    clean_content,
    code_content_clean,
    fallback_inline_caption_fragments,
    fallback_leading_table_continuation_captions,
    isolated_formula_clean,
    regroup_visual_blocks,
)


def _has_inline_formula_content(content: str | None) -> bool:
    """判断 content 是否包含成对行内公式标记，用于 code/algorithm 分类。"""
    return bool(content) and content.count("\\(") == content.count("\\)") and content.count("\\(") > 0


def _copy_raw_text_block_metadata(draft: VlmContentBlockDraft, block: Block) -> None:
    if draft.raw_type != BlockType.TEXT:
        return
    block.merge_prev = draft.merge_prev


class MagicModel:
    def __init__(
        self,
        page_model_list: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height

        blocks: list[Block] = []
        code_metadata_by_index: dict[int, tuple[str, str]] = {}

        # 解析每个块
        for index, block_info in enumerate(page_model_list):
            try:
                draft = VlmContentBlockDraft.from_content_block(block_info, width, height)
                block_bbox = draft.bbox
                block_type = draft.raw_type
                raw_block_type = draft.raw_type
                block_content = draft.content or ""
                block_angle = draft.angle
                block_sub_type = draft.sub_type if raw_block_type in ["image", "chart"] else None
            except Exception as e:
                # 如果解析失败，可能是因为格式不正确，跳过这个块
                logger.warning(f"Invalid block format: {block_info}, error: {e}")
                continue

            code_block_sub_type = None
            guess_lang = None

            if block_type in ["image_caption", "table_caption", "code_caption"]:
                block_type = BlockType.CAPTION
            elif block_type in ["image_footnote", "table_footnote"]:
                block_type = BlockType.FOOTNOTE
            elif block_type == "image":
                block_type = BlockType.IMAGE_BODY
            elif block_type == "image_block":
                block_type = IMAGE_BLOCK_BODY
            elif block_type == "table":
                block_type = BlockType.TABLE_BODY
            elif block_type == "chart":
                block_type = BlockType.CHART_BODY
            elif block_type in ["code", "algorithm"]:
                block_content = code_content_clean(block_content)
                code_block_sub_type = block_type
                block_type = BlockType.CODE_BODY
                guess_lang = guess_language_by_text(block_content)
            elif block_type == "equation":
                block_type = BlockType.INTERLINE_EQUATION

            if raw_block_type == "equation":
                block_content = isolated_formula_clean(block_content)
            elif raw_block_type not in ["image", "image_block", "table", "chart"]:
                # 文本类块继续沿用现有 content 清洗规则，但不再拆分为 line/span。
                if block_content:
                    block_content = clean_content(block_content) or ""

                if block_type in [BlockType.TITLE, BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE] and block_content:
                    block_content = re.sub(r"\n\s*", " ", block_content).strip()

                if (
                    block_type == BlockType.CODE_BODY
                    and code_block_sub_type == "code"
                    and _has_inline_formula_content(block_content)
                ):
                    code_block_sub_type = "algorithm"

            block = Block(index=index, type=block_type, bbox=block_bbox, content=block_content, angle=block_angle)
            if block_sub_type:
                block.sub_type = block_sub_type
            if block_type == BlockType.CODE_BODY and code_block_sub_type:
                code_metadata_by_index[index] = (code_block_sub_type, guess_lang or "")
            if raw_block_type == "table" and draft.cell_merge:
                block._cell_merge = draft.cell_merge
            if block_type == BlockType.TEXT:
                block._ocr_det_lines = draft.ocr_det_lines
            if block_type in [BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE]:
                block._line_avg_height = draft.line_avg_height
            _copy_raw_text_block_metadata(draft, block)

            blocks.append(block)

        fallback_inline_caption_fragments(blocks, VISUAL_MAIN_TYPES)
        fallback_leading_table_continuation_captions(blocks, VISUAL_MAIN_TYPES)

        self.image_blocks: list[Block] = []
        self.table_blocks: list[Block] = []
        self.chart_blocks: list[Block] = []
        self.interline_equation_blocks: list[Block] = []
        self.text_blocks: list[Block] = []
        self.title_blocks: list[Block] = []
        self.code_blocks: list[Block] = []
        self.discarded_blocks: list[Block] = []
        self.ref_text_blocks: list[Block] = []
        self.phonetic_blocks: list[Block] = []
        self.list_blocks: list[Block] = []

        for block in blocks:
            if block.type in VISUAL_MAIN_TYPES or block.type in GENERIC_CHILD_TYPES:
                continue
            elif block.type == BlockType.INTERLINE_EQUATION:
                self.interline_equation_blocks.append(block)
            elif block.type == BlockType.TEXT:
                self.text_blocks.append(block)
            elif block.type in [
                BlockType.TITLE,
                BlockType.DOC_TITLE,
                BlockType.PARAGRAPH_TITLE,
            ]:
                self.title_blocks.append(block)
            elif block.type == BlockType.REF_TEXT:
                self.ref_text_blocks.append(block)
            elif block.type == BlockType.PHONETIC:
                self.phonetic_blocks.append(block)
            elif block.type in [
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
                BlockType.ASIDE_TEXT,
                BlockType.PAGE_FOOTNOTE,
            ]:
                self.discarded_blocks.append(block)
            elif block.type == BlockType.LIST:
                self.list_blocks.append(block)

        self.list_blocks, self.text_blocks, self.ref_text_blocks = fix_list_blocks(
            self.list_blocks,
            self.text_blocks,
            self.ref_text_blocks,
        )

        visual_groups, unmatched_child_blocks = regroup_visual_blocks(blocks)
        self.image_blocks = visual_groups[BlockType.IMAGE]
        self.table_blocks = visual_groups[BlockType.TABLE]
        self.chart_blocks = visual_groups[BlockType.CHART]
        self.code_blocks = visual_groups[BlockType.CODE]

        for code_block in self.code_blocks:
            code_block.sub_type, guess_lang = code_metadata_by_index.get(code_block.index, ("code", "txt"))
            if code_block.sub_type == "code":
                code_block.guess_lang = guess_lang

        for block in unmatched_child_blocks:
            block.type = BlockType.TEXT
            self.text_blocks.append(block)

    def get_list_blocks(self) -> list[Block]:
        return self.list_blocks

    def get_image_blocks(self) -> list[Block]:
        return self.image_blocks

    def get_table_blocks(self) -> list[Block]:
        return self.table_blocks

    def get_chart_blocks(self) -> list[Block]:
        return self.chart_blocks

    def get_code_blocks(self) -> list[Block]:
        return self.code_blocks

    def get_ref_text_blocks(self) -> list[Block]:
        return self.ref_text_blocks

    def get_phonetic_blocks(self) -> list[Block]:
        return self.phonetic_blocks

    def get_title_blocks(self) -> list[Block]:
        return self.title_blocks

    def get_text_blocks(self) -> list[Block]:
        return self.text_blocks

    def get_interline_equation_blocks(self) -> list[Block]:
        return self.interline_equation_blocks

    def get_discarded_blocks(self) -> list[Block]:
        return self.discarded_blocks


def fix_list_blocks(
    list_blocks: list[Block], text_blocks: list[Block], ref_text_blocks: list[Block]
) -> tuple[list[Block], list[Block], list[Block]]:
    for list_block in list_blocks:
        list_block.blocks = []
        if list_block.lines:
            list_block.lines = []

    temp_text_blocks = text_blocks + ref_text_blocks
    need_remove_blocks = []
    for block in temp_text_blocks:
        for list_block in list_blocks:
            if (
                calculate_overlap_area_in_bbox1_area_ratio(
                    block.bbox,
                    list_block.bbox,
                )
                >= 0.8
            ):
                list_block.blocks.append(block)
                need_remove_blocks.append(block)
                break

    for block in need_remove_blocks:
        if block in text_blocks:
            text_blocks.remove(block)
        elif block in ref_text_blocks:
            ref_text_blocks.remove(block)

    list_blocks = [lb for lb in list_blocks if lb.blocks]

    for list_block in list_blocks:
        type_count = {}
        for sub_block in list_block.blocks:
            sub_block_type = sub_block.type
            if sub_block_type not in type_count:
                type_count[sub_block_type] = 0
            type_count[sub_block_type] += 1

        if type_count:
            list_block.sub_type = max(type_count, key=type_count.get)  # type: ignore
        else:
            list_block.sub_type = "unknown"

    return list_blocks, text_blocks, ref_text_blocks
