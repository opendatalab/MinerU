# Copyright (c) Opendatalab. All rights reserved.
"""PDF 与 Office 的列表、目录和标题编号后处理。"""

from __future__ import annotations

import re
from typing import Any

from ...model.flash._shared.spans import inline_span_plain_text, text_spans
from ...types import BlockType, parse_inline_spans
from ...utils.geometry import calculate_overlap_area_in_bbox1_area_ratio

from .inline import inline_plain_text, slice_inline_spans
from .visual import _bbox_for_calculation


def fix_office_paragraph_titles(model_list: list[list[dict[str, Any]]]) -> None:
    """按文档级标题序列内化 Office 自动编号，并清除私有编号元数据。"""
    counters: dict[int, int] = {}
    for page_model_list in model_list:
        for block in page_model_list:
            if block.get("type") != BlockType.PARAGRAPH_TITLE:
                continue
            raw_level = block.get("level")
            normalized_level = raw_level if type(raw_level) is int else 2
            level = min(max(normalized_level, 2), 6)
            block["level"] = level
            numbering_depth = level - 1
            is_numbered_style = block.pop("is_numbered_style", None)
            block.pop("section_number", None)
            content = block.get("content")
            if not isinstance(content, list):
                continue
            if is_numbered_style is True:
                for ancestor_level in range(1, numbering_depth):
                    counters.setdefault(ancestor_level, 1)
                counters[numbering_depth] = counters.get(numbering_depth, 0) + 1
                _clear_deeper_title_counters(counters, numbering_depth)
                section_number = ".".join(str(counters[ancestor_level]) for ancestor_level in range(1, numbering_depth + 1))
                block["content"] = [*text_spans(f"{section_number} "), *content]
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


def fix_office_index_title_blocks(model_list: list[list[dict[str, Any]]]) -> None:
    """按正文目标 anchor 将 Office 目录文本叶子转换为对应正文或标题类型。"""
    target_by_anchor: dict[str, tuple[str, int | None]] = {}
    for page_model_list in model_list:
        for block in page_model_list:
            block_type = block.get("type")
            if block_type not in {BlockType.TEXT, BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
                continue
            anchor = block.get("anchor")
            level = block.get("level")
            if not isinstance(anchor, str) or not anchor.strip():
                continue
            if block_type in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE} and type(level) is not int:
                continue
            target_by_anchor.setdefault(anchor.strip(), (block_type, level if type(level) is int else None))

    for page_model_list in model_list:
        for block in page_model_list:
            if block.get("type") == BlockType.INDEX:
                _rewrite_office_index_title_leaves(block, target_by_anchor)


def _rewrite_office_index_title_leaves(
    index_block: dict[str, Any],
    target_by_anchor: dict[str, tuple[str, int | None]],
) -> None:
    """递归改写目录叶子；未匹配 anchor 时降级为不带 anchor 的普通文本。"""
    content = index_block.get("content")
    if not isinstance(content, list):
        return
    for child in content:
        if not isinstance(child, dict):
            continue
        if child.get("type") == BlockType.INDEX:
            _rewrite_office_index_title_leaves(child, target_by_anchor)
            continue
        if child.get("type") != BlockType.TEXT:
            continue
        anchor = child.get("anchor")
        normalized_anchor = anchor.strip() if isinstance(anchor, str) else ""
        target = target_by_anchor.get(normalized_anchor)
        if target is None:
            child.pop("anchor", None)
            continue
        target_type, target_level = target
        child["type"] = target_type
        if target_level is None:
            child.pop("level", None)
        else:
            child["level"] = target_level
        child["anchor"] = normalized_anchor


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
        raw_content = index_block.get("content")
        if not isinstance(raw_content, list):
            index_block["content"] = []
            continue
        spans = parse_inline_spans(raw_content)
        visible = inline_plain_text(spans)
        children: list[dict[str, Any]] = []
        start = 0
        for match in re.finditer("\n", visible):
            content = slice_inline_spans(spans, start, match.start())
            if content:
                children.append({"type": BlockType.TEXT, "content": [span.model_dump(mode="json") for span in content]})
            start = match.end()
        content = slice_inline_spans(spans, start)
        if content:
            children.append({"type": BlockType.TEXT, "content": [span.model_dump(mode="json") for span in content]})
        index_block["content"] = children
    return index_blocks


def fix_office_index_blocks(index_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """递归移除 Office 目录层级私有字段，保留已规范化的标题叶子。"""
    pending_blocks = list(index_blocks)
    while pending_blocks:
        block = pending_blocks.pop()
        block.pop("ilevel", None)
        content = block.get("content")
        if isinstance(content, list):
            pending_blocks.extend(child for child in content if isinstance(child, dict))
    return index_blocks


def _clear_deeper_title_counters(counters: dict[int, int], level: int) -> None:
    """删除当前标题层级之后的旧计数。"""
    for counter_level in [value for value in counters if value > level]:
        del counters[counter_level]


def _visible_text(content: list[dict[str, Any]]) -> str:
    """提取结构化 Span 的可见文本，供显式标题编号识别。"""
    return inline_span_plain_text(content)


def fix_office_list_blocks(list_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将每层 Office 列表的局部序号写入文本内容，并移除原始元数据。"""

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

    def fix_list_block(list_block: dict[str, Any]) -> None:
        """递归处理列表树；每个有序列表只维护当前层的独立编号。"""
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
                    if not isinstance(child_content, list):
                        child_block.pop("list_label", None)
                        continue
                    exact_label = child_block.pop("list_label", None)
                    if isinstance(exact_label, str) and exact_label.strip():
                        prefix = f"{exact_label.strip()} "
                        if is_ordered:
                            ordered_number += 1
                    elif is_ordered:
                        prefix = f"{ordered_number}. "
                        ordered_number += 1
                    else:
                        prefix = "- "
                    child_block["content"] = [*text_spans(prefix), *child_content]
                elif child_type == BlockType.LIST:
                    fix_list_block(child_block)
        list_block.pop("attribute", None)
        list_block.pop("ilevel", None)
        list_block.pop("start", None)

    for list_block in list_blocks:
        if isinstance(list_block, dict):
            fix_list_block(list_block)
    return list_blocks
