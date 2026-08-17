# Copyright (c) Opendatalab. All rights reserved.
"""PDF 与 Office 的列表、目录和标题编号后处理。"""

from __future__ import annotations

import re
from typing import Any

from mineru.types import BlockType
from mineru.utils.bbox_utils import calculate_overlap_area_in_bbox1_area_ratio

from .visual import _bbox_for_calculation


def fix_office_paragraph_titles(model_list: list[list[dict[str, Any]]]) -> None:
    """按文档级标题序列内化 Office 自动编号，并清除私有编号元数据。"""
    counters: dict[int, int] = {}
    for page_model_list in model_list:
        for block in page_model_list:
            if block.get("type") != BlockType.PARAGRAPH_TITLE:
                continue
            raw_level = block.get("level")
            level_is_valid = isinstance(raw_level, int) and not isinstance(raw_level, bool) and raw_level >= 2
            level = raw_level if level_is_valid else 2
            block["level"] = level
            numbering_depth = level - 1
            is_numbered_style = block.pop("is_numbered_style", None)
            block.pop("section_number", None)
            content = block.get("content")
            if not isinstance(content, str):
                continue
            if is_numbered_style is True:
                for ancestor_level in range(1, numbering_depth):
                    counters.setdefault(ancestor_level, 1)
                counters[numbering_depth] = counters.get(numbering_depth, 0) + 1
                _clear_deeper_title_counters(counters, numbering_depth)
                section_number = ".".join(
                    str(counters[ancestor_level]) for ancestor_level in range(1, numbering_depth + 1)
                )
                block["content"] = f"{section_number} {content}"
                continue
            if is_numbered_style is False:
                number_match = re.match(r"^\s*(\d+(?:\.\d+)*)\b", _visible_text(content))
                if number_match is None:
                    continue
                number_parts = [int(part) for part in number_match.group(1).split(".")]
                if len(number_parts) != numbering_depth:
                    continue
                counters.update((part_level, number) for part_level, number in enumerate(number_parts, start=1))
                _clear_deeper_title_counters(counters, numbering_depth)


def fix_pdf_list_blocks(
    list_blocks: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    ref_text_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """按 bbox 把 PDF text/ref_text 归入 list，并推断列表子类型。"""
    for list_block in list_blocks:
        list_block["content"] = []
    need_remove_blocks = []
    for block in text_blocks + ref_text_blocks:
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
    list_blocks = [block for block in list_blocks if block["content"]]
    for list_block in list_blocks:
        type_count: dict[str, int] = {}
        for sub_block in list_block["content"]:
            sub_block_type = sub_block["type"]
            type_count[sub_block_type] = type_count.get(sub_block_type, 0) + 1
        list_block["sub_type"] = max(type_count, key=type_count.get) if type_count else "text"  # type: ignore[arg-type]
    return list_blocks, text_blocks, ref_text_blocks


def fix_pdf_index_blocks(index_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 PDF 目录块的多行内容拆分为多个文本子块。"""
    for index_block in index_blocks:
        index_block["content"] = [{"type": "text", "content": content} for content in index_block["content"].split("\n")]
    return index_blocks


def fix_office_index_blocks(index_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """递归移除 Office 目录私有字段，并禁止普通 text 子块携带 anchor。"""
    pending_blocks = list(index_blocks)
    while pending_blocks:
        block = pending_blocks.pop()
        block.pop("ilevel", None)
        if block.get("type") == BlockType.TEXT:
            block.pop("anchor", None)
        content = block.get("content")
        if isinstance(content, list):
            pending_blocks.extend(child for child in content if isinstance(child, dict))
    return index_blocks


def _clear_deeper_title_counters(counters: dict[int, int], level: int) -> None:
    """删除当前标题层级之后的旧计数。"""
    for counter_level in [value for value in counters if value > level]:
        del counters[counter_level]


def _visible_text(content: str) -> str:
    """移除简单富文本标签，供显式标题编号识别可见文本开头。"""
    return re.sub(r"<[^>]+>", "", content)


def fix_office_list_blocks(list_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 Office 列表层级和起始序号写入文本内容，并移除原始元数据。"""

    def get_list_ilevel(list_block: dict[str, Any], parent_ilevel: int | None) -> int:
        """读取列表层级，非法值按递归深度回退。"""
        fallback_ilevel = 0 if parent_ilevel is None else parent_ilevel + 1
        try:
            ilevel = int(list_block.get("ilevel"))
        except (TypeError, ValueError):
            return fallback_ilevel
        return ilevel if ilevel >= 0 else fallback_ilevel

    def get_ordered_list_start(list_block: dict[str, Any]) -> int:
        """读取有序列表起始编号，保留合法的零值。"""
        start = list_block.get("start")
        if start is None:
            return 1
        try:
            start = int(start)
        except (TypeError, ValueError):
            return 1
        return start if start >= 0 else 1

    def format_ordered_prefix(ordered_numbers: dict[int, int]) -> str:
        """按列表层级拼接有序编号。"""
        number_parts = [str(number) for _, number in sorted(ordered_numbers.items())]
        if len(number_parts) == 1:
            return f"{number_parts[0]}. "
        return f"{'.'.join(number_parts)} "

    def fix_list_block(
        list_block: dict[str, Any],
        inherited_ordered_numbers: dict[int, int],
        parent_ilevel: int | None,
    ) -> None:
        """递归处理单个列表块，并向嵌套列表传递有序编号。"""
        ilevel = get_list_ilevel(list_block, parent_ilevel)
        base_ordered_numbers = {level: number for level, number in inherited_ordered_numbers.items() if level < ilevel}
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
                    fix_list_block(child_block, dict(active_ordered_numbers), ilevel)
        list_block.pop("attribute", None)
        list_block.pop("ilevel", None)
        list_block.pop("start", None)

    for list_block in list_blocks:
        if isinstance(list_block, dict):
            fix_list_block(list_block, {}, None)
    return list_blocks
