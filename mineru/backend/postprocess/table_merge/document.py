# Copyright (c) Opendatalab. All rights reserved.
"""文档页边界上的跨页表格识别与延续标记编排。"""

from __future__ import annotations

from typing import Any

from ....types import MERGE_TRANSPARENT_BLOCK_TYPES, BlockType

from .blocks import _get_or_create_table_state
from .models import BlockDict, PageInfoDict, TableMergeState
from .structure import can_merge_tables

TABLE_BOUNDARY_IGNORED_TYPES = set(MERGE_TRANSPARENT_BLOCK_TYPES)


def _clear_table_continuation_marker(table_block: BlockDict) -> None:
    """递归清除 table 根块及其子块中过期的 ``continues_prev``。"""
    table_block.pop("continues_prev", None)
    content = table_block.get("content")
    if not isinstance(content, list):
        return
    for child in content:
        if isinstance(child, dict):
            child.pop("continues_prev", None)
            _clear_nested_continuation_markers(child)


def _clear_nested_continuation_markers(block: BlockDict) -> None:
    """清除表格子树中的旧延续标记，避免标记落到嵌套子块。"""
    content = block.get("content")
    if not isinstance(content, list):
        return
    for child in content:
        if isinstance(child, dict):
            child.pop("continues_prev", None)
            _clear_nested_continuation_markers(child)


def _find_boundary_table(blocks: list[Any], *, from_end: bool) -> BlockDict | None:
    """从页边界扫描 table；噪声块可跳过，其他语义块立即阻断。"""
    ordered_blocks = reversed(blocks) if from_end else iter(blocks)
    for block in ordered_blocks:
        if not isinstance(block, dict):
            return None
        block_type = block.get("type")
        if block_type in TABLE_BOUNDARY_IGNORED_TYPES:
            continue
        if block_type == BlockType.TABLE:
            return block
        return None
    return None


def _is_consecutive_page_pair(previous_page: PageInfoDict, current_page: PageInfoDict) -> bool:
    """按显式零基 page_idx 判断页面在文档中是否严格连续。"""
    previous_page_idx = previous_page.get("page_idx")
    current_page_idx = current_page.get("page_idx")
    return type(previous_page_idx) is int and type(current_page_idx) is int and current_page_idx == previous_page_idx + 1


def merge_table(page_info_list: list[PageInfoDict]) -> None:
    """倒序识别连续页边界表格，并只在后表写入延续标记。"""
    if not isinstance(page_info_list, list):
        return

    for page_info in page_info_list:
        if not isinstance(page_info, dict):
            continue
        blocks = page_info.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == BlockType.TABLE:
                _clear_table_continuation_marker(block)

    state_cache: dict[int, TableMergeState] = {}

    for page_position in range(len(page_info_list) - 1, 0, -1):
        current_page = page_info_list[page_position]
        previous_page = page_info_list[page_position - 1]
        if not isinstance(current_page, dict) or not isinstance(previous_page, dict):
            continue
        if not _is_consecutive_page_pair(previous_page, current_page):
            continue

        current_blocks = current_page.get("blocks")
        previous_blocks = previous_page.get("blocks")
        if not isinstance(current_blocks, list) or not isinstance(previous_blocks, list):
            continue

        current_table_block = _find_boundary_table(current_blocks, from_end=False)
        previous_table_block = _find_boundary_table(previous_blocks, from_end=True)
        if current_table_block is None or previous_table_block is None:
            continue

        current_state = _get_or_create_table_state(current_table_block, state_cache)
        previous_state = _get_or_create_table_state(previous_table_block, state_cache)
        if current_state is None or previous_state is None:
            continue

        if not can_merge_tables(current_state, previous_state):
            continue

        current_table_block["continues_prev"] = True
