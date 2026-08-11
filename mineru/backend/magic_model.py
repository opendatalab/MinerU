# Copyright (c) Opendatalab. All rights reserved.
import re
from typing import Any

from loguru import logger

from mineru.backend.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.backend.utils.visual_magic_model_utils import (
    code_content_clean,
    isolated_formula_clean,
    clean_content,
    VISUAL_MAIN_TYPES,
    _bbox_for_calculation,
    regroup_visual_blocks,
    fallback_inline_caption_fragments,
    fallback_leading_table_continuation_captions,
    fallback_no_bbox_caption_fragments,
)
from mineru.types import BlockType
from mineru.utils.guess_suffix_or_lang import guess_language_by_text


def _has_inline_formula_content(content: str | None) -> bool:
    """判断 content 是否包含成对 <eq> 行内公式标记，用于 code/algorithm 分类。"""
    return bool(content) and content.count("<eq>") == content.count("</eq>") and content.count("<eq>") > 0


class MagicModel:
    def __init__(
        self,
        page_model_list: list[dict[str, Any]],
    ) -> None:
        self.blocks = []
        is_block_has_bbox = any(block_info.get("bbox") for block_info in page_model_list)
        # 解析每个块
        for index, block_info in enumerate(page_model_list):
            code_block_sub_type = None
            block_type = block_info.get("type", "")
            block_content = block_info.get("content", "")

            if block_type == "image":
                block_type = BlockType.IMAGE_BODY
            elif block_type == "table":
                block_type = BlockType.TABLE_BODY
            elif block_type == "chart":
                block_type = BlockType.CHART_BODY
            elif block_type in ["code", "algorithm"]:
                code_block_sub_type = block_type
                block_content = code_content_clean(block_content)
                block_type = BlockType.CODE_BODY
            elif block_type == "equation":
                block_type = BlockType.INTERLINE_EQUATION
                block_content = isolated_formula_clean(block_content)

            if block_type not in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.CHART_BODY]:
                # 文本类块继续沿用现有 content 清洗规则，但不再拆分为 line/span。
                if block_content:
                    block_content = clean_content(block_content) or ""
                # 对于标题类块，去除换行符和多余空格
                if block_type in [BlockType.TITLE, BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE] and block_content:
                    block_content = re.sub(r"\n\s*", " ", block_content).strip()
                # 处理 code/algorithm 分类的特殊情况：如果 code 类型的块中包含成对的行内公式标记，则将其分类为 algorithm。
                if (
                    block_type == BlockType.CODE_BODY
                    and code_block_sub_type == "code"
                    and _has_inline_formula_content(block_content)
                ):
                    code_block_sub_type = "algorithm"

            block = block_info
            block["type"] = block_type
            block["content"] = block_content
            block["index"] = index
            if code_block_sub_type:
                block["sub_type"] = code_block_sub_type
            self.blocks.append(block)

        if is_block_has_bbox:
            fallback_inline_caption_fragments(self.blocks, VISUAL_MAIN_TYPES)
            fallback_leading_table_continuation_captions(self.blocks, VISUAL_MAIN_TYPES)
        else:
            fallback_no_bbox_caption_fragments(self.blocks, VISUAL_MAIN_TYPES)


        self.text_blocks = []
        self.ref_text_blocks = []
        self.list_blocks = []
        self.index_blocks = []

        for block in self.blocks:
            if block["type"] == BlockType.TEXT:
                self.text_blocks.append(block)
            elif block["type"] == BlockType.REF_TEXT:
                self.ref_text_blocks.append(block)
            elif block["type"] == BlockType.LIST:
                self.list_blocks.append(block)
            elif block["type"] == BlockType.INDEX:
                self.index_blocks.append(block)

        if is_block_has_bbox:
            self.list_blocks, self.text_blocks, self.ref_text_blocks = fix_pdf_list_blocks(
                self.list_blocks,
                self.text_blocks,
                self.ref_text_blocks,
            )
            self.index_blocks = fix_pdf_index_blocks(self.index_blocks)
        else:
            self.list_blocks = fix_office_list_blocks(self.list_blocks)
            self.index_blocks = fix_office_index_blocks(self.index_blocks)

        visual_groups, unmatched_child_blocks = regroup_visual_blocks(
            self.blocks,
            use_bbox=is_block_has_bbox,
        )
        self.image_blocks = visual_groups[BlockType.IMAGE]
        self.table_blocks = visual_groups[BlockType.TABLE]
        self.chart_blocks = visual_groups[BlockType.CHART]
        self.code_blocks = visual_groups[BlockType.CODE]

        for code_block in self.code_blocks:
            if code_block["sub_type"] == "code":
                for sub_block in code_block["blocks"]:
                    if sub_block.get("type") == BlockType.CODE_BODY:
                        guess_lang = guess_language_by_text(sub_block.get("content", ""))
                        code_block["guess_lang"] = guess_lang
                        break

        for block in unmatched_child_blocks:
            block["type"] = BlockType.TEXT
            self.text_blocks.append(block)

        # 移除已完成分类或分组的原始块，再写回处理后的顶层块。
        replaced_block_types = {
            BlockType.TEXT,
            BlockType.REF_TEXT,
            BlockType.LIST,
            BlockType.INDEX,
            BlockType.CAPTION,
            BlockType.FOOTNOTE,
            BlockType.IMAGE_BODY,
            BlockType.TABLE_BODY,
            BlockType.CHART_BODY,
            BlockType.CODE_BODY,
        }
        self.blocks = [
            block
            for block in self.blocks
            if block["type"] not in replaced_block_types
        ]
        self.blocks.extend(
            self.list_blocks
            + self.text_blocks
            + self.ref_text_blocks
            + self.index_blocks
            + self.image_blocks
            + self.table_blocks
            + self.chart_blocks
            + self.code_blocks
        )


def fix_pdf_list_blocks(
    list_blocks, text_blocks, ref_text_blocks
):
    for list_block in list_blocks:
        list_block["content"] = []

    temp_text_blocks = text_blocks + ref_text_blocks
    need_remove_blocks = []
    for block in temp_text_blocks:
        for list_block in list_blocks:
            if (
                calculate_overlap_area_in_bbox1_area_ratio(
                    _bbox_for_calculation(block["bbox"]),
                    _bbox_for_calculation(list_block["bbox"]),
                )
                >= 0.8
            ):
                list_block["content"].append(block)
                need_remove_blocks.append(block)
                break

    for block in need_remove_blocks:
        if block in text_blocks:
            text_blocks.remove(block)
        elif block in ref_text_blocks:
            ref_text_blocks.remove(block)

    list_blocks = [lb for lb in list_blocks if lb["content"]]

    for list_block in list_blocks:
        type_count = {}
        for sub_block in list_block["content"]:
            sub_block_type = sub_block["type"]
            if sub_block_type not in type_count:
                type_count[sub_block_type] = 0
            type_count[sub_block_type] += 1

        if type_count:
            list_block["sub_type"] = max(type_count, key=type_count.get)  # type: ignore
        else:
            list_block["sub_type"] = "text"

    return list_blocks, text_blocks, ref_text_blocks


def fix_pdf_index_blocks(index_blocks):
    """将 PDF 目录块的多行内容拆分为多个文本子块。"""
    for index_block in index_blocks:
        index_block["content"] = [
            {"type": "text", "content": content}
            for content in index_block["content"].split("\n")
        ]

    return index_blocks


def fix_office_index_blocks(index_blocks):
    """递归移除 Office 目录块及其子块中的 ilevel 字段。"""
    pending_blocks = list(index_blocks)
    while pending_blocks:
        block = pending_blocks.pop()
        block.pop("ilevel", None)

        content = block.get("content")
        if isinstance(content, list):
            pending_blocks.extend(
                child_block
                for child_block in content
                if isinstance(child_block, dict)
            )

    return index_blocks


def fix_office_list_blocks(
    list_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将 Office 列表层级和起始序号写入文本内容，并移除原始列表元数据。"""

    def get_list_ilevel(list_block: dict[str, Any], parent_ilevel: int | None) -> int:
        """读取列表层级，非法值按当前结构中的递归深度回退。"""
        fallback_ilevel = 0 if parent_ilevel is None else parent_ilevel + 1
        try:
            ilevel = int(list_block.get("ilevel"))
        except (TypeError, ValueError):
            return fallback_ilevel
        return ilevel if ilevel >= 0 else fallback_ilevel

    def get_ordered_list_start(list_block: dict[str, Any]) -> int:
        """读取有序列表起始编号，保留合法的 0 并将非法值回退为 1。"""
        start = list_block.get("start")
        if start is None:
            return 1
        try:
            start = int(start)
        except (TypeError, ValueError):
            return 1
        return start if start >= 0 else 1

    def format_ordered_prefix(ordered_numbers: dict[int, int]) -> str:
        """按列表层级拼接有序编号，单级编号保留末尾句点。"""
        number_parts = [str(number) for _, number in sorted(ordered_numbers.items())]
        if len(number_parts) == 1:
            return f"{number_parts[0]}. "
        return f"{'.'.join(number_parts)} "

    def fix_list_block(
        list_block: dict[str, Any],
        inherited_ordered_numbers: dict[int, int],
        parent_ilevel: int | None,
    ) -> None:
        """递归处理单个列表块，并把当前位置的有序编号传给嵌套列表。"""
        ilevel = get_list_ilevel(list_block, parent_ilevel)
        base_ordered_numbers = {
            level: number
            for level, number in inherited_ordered_numbers.items()
            if level < ilevel
        }
        active_ordered_numbers = dict(base_ordered_numbers)
        is_ordered = list_block.get("attribute") == "ordered"
        ordered_number = get_ordered_list_start(list_block)

        content = list_block.get("content")
        if isinstance(content, list):
            for child_block in content:
                if not isinstance(child_block, dict):
                    continue

                child_type = child_block.get("type")
                if child_type == BlockType.TEXT:
                    child_content = child_block.get("content")
                    if not isinstance(child_content, str):
                        continue

                    if is_ordered:
                        active_ordered_numbers = dict(base_ordered_numbers)
                        active_ordered_numbers[ilevel] = ordered_number
                        prefix = format_ordered_prefix(active_ordered_numbers)
                        ordered_number += 1
                    else:
                        prefix = "- "
                    child_block["content"] = f"{prefix}{child_content}"
                elif child_type == BlockType.LIST:
                    fix_list_block(
                        child_block,
                        dict(active_ordered_numbers),
                        ilevel,
                    )

        list_block.pop("attribute", None)
        list_block.pop("ilevel", None)
        list_block.pop("start", None)

    for list_block in list_blocks:
        if isinstance(list_block, dict):
            fix_list_block(list_block, {}, None)

    return list_blocks
