# Copyright (c) Opendatalab. All rights reserved.
"""跨页表格的表头、宽度和边界行结构判定。"""

from __future__ import annotations

from typing import Any

from bs4 import Tag

from mineru.types import BlockType

from .blocks import (
    _bbox_for_calculation,
    _is_continuation_caption,
    _is_post_table_non_continuation_caption,
    _table_children,
)
from .html import _colspan, _rowspan, calculate_row_rendered_segments
from .models import MAX_HEADER_ROWS, TableMergeState


def detect_table_headers(
    state1: TableMergeState, state2: TableMergeState, max_header_rows: int = MAX_HEADER_ROWS
) -> tuple[int, bool, list[list[str]]]:
    """检测并比较两个表格的表头，仅扫描前几行."""
    front_rows1 = state1.front_header_info[:max_header_rows]
    front_rows2 = state2.front_header_info[:max_header_rows]

    min_rows = min(len(front_rows1), len(front_rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for row_idx in range(min_rows):
        row1 = front_rows1[row_idx]
        row2 = front_rows2[row_idx]
        structure_match = (
            row1.cell_count == row2.cell_count
            and row1.effective_cols == row2.effective_cols
            and row1.colspans == row2.colspans
            and row1.rowspans == row2.rowspans
            and row1.normalized_texts == row2.normalized_texts
        )

        if structure_match:
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        header_rows, headers_match, header_texts = _detect_table_headers_visual(state1, state2, max_header_rows=max_header_rows)

    return header_rows, headers_match, header_texts


def _detect_table_headers_visual(
    state1: TableMergeState,
    state2: TableMergeState,
    max_header_rows: int = MAX_HEADER_ROWS,
) -> tuple[int, bool, list[list[str]]]:
    """基于视觉一致性检测表头（只比较文本内容，忽略colspan/rowspan差异）."""
    front_rows1 = state1.front_header_info[:max_header_rows]
    front_rows2 = state2.front_header_info[:max_header_rows]

    min_rows = min(len(front_rows1), len(front_rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for row_idx in range(min_rows):
        row1 = front_rows1[row_idx]
        row2 = front_rows2[row_idx]
        # OCR 识别表头时可能丢失 colspan/rowspan，这里用渲染段数约束视觉一致性。
        rendered_segments1 = calculate_row_rendered_segments(state1.rows, row_idx)
        rendered_segments2 = calculate_row_rendered_segments(state2.rows, row_idx)
        if row1.normalized_texts == row2.normalized_texts and rendered_segments1 == rendered_segments2:
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        headers_match = False

    return header_rows, headers_match, header_texts


def _expand_header_count_by_rowspan(rows: list[Tag], header_count: int) -> int:
    """按表头 rowspan 覆盖范围扩展跳过行数。

    跨页续表的第一行表头可能包含 rowspan。如果只跳过已匹配的首行，
    被该 rowspan 覆盖的后续表头行会失去占位来源，合并后形成半截表头。
    因此跳过重复表头时，需要覆盖所有由已跳过表头行跨行占据的行。
    """
    if header_count <= 0 or not rows:
        return header_count

    expanded_header_count = min(header_count, len(rows))
    row_idx = 0
    while row_idx < expanded_header_count:
        row = rows[row_idx]
        for cell in row.find_all(["td", "th"]):
            rowspan = _rowspan(cell)
            if rowspan > 1:
                expanded_header_count = max(expanded_header_count, row_idx + rowspan)
                expanded_header_count = min(expanded_header_count, len(rows))
        row_idx += 1

    return expanded_header_count


def can_merge_by_structure(
    current_state: TableMergeState,
    previous_state: TableMergeState,
    current_bbox: Any = None,
    previous_bbox: Any = None,
) -> bool:
    """仅基于表格结构判断是否可合并（不检查 caption/footnote）。

    供外部工具调用，忽略 caption 和 footnote 检查。
    """
    if (
        current_bbox is not None
        and previous_bbox is not None
        and not _table_widths_are_compatible(
            current_bbox,
            previous_bbox,
        )
    ):
        return False

    if (
        previous_state.total_cols <= 0
        or current_state.total_cols <= 0
        or previous_state.last_data_row_metrics is None
        or current_state.last_data_row_metrics is None
    ):
        return False

    if previous_state.total_cols == current_state.total_cols:
        return True

    return check_rows_match(previous_state, current_state)


def _table_widths_are_compatible(current_bbox: Any, previous_bbox: Any) -> bool:
    """使用千分位 bbox 判断两张表的宽度相对差是否小于百分之十。"""
    current_calc_bbox = _bbox_for_calculation(current_bbox)
    previous_calc_bbox = _bbox_for_calculation(previous_bbox)
    if current_calc_bbox is None or previous_calc_bbox is None:
        return False

    current_width = current_calc_bbox[2] - current_calc_bbox[0]
    previous_width = previous_calc_bbox[2] - previous_calc_bbox[0]
    min_width = min(current_width, previous_width)
    return min_width > 0 and abs(current_width - previous_width) / min_width < 0.1


def can_merge_tables(current_state: TableMergeState, previous_state: TableMergeState) -> bool:
    """根据 dict 表格的辅助文本、宽度和 HTML 结构判断是否可合并。"""
    current_table_block = current_state.owner_block
    previous_table_block = previous_state.owner_block

    if not isinstance(previous_table_block, dict) or not isinstance(current_table_block, dict):
        return False

    previous_children = _table_children(previous_table_block)
    current_children = _table_children(current_table_block)
    footnote_count = sum(1 for block in previous_children if block.get("type") == BlockType.TABLE_FOOTNOTE)
    caption_blocks = [block for block in current_children if block.get("type") == BlockType.TABLE_CAPTION]
    merge_caption_blocks = [
        block for block in caption_blocks if not _is_post_table_non_continuation_caption(current_table_block, block)
    ]
    if merge_caption_blocks:
        has_continuation_marker = any(_is_continuation_caption(block) for block in merge_caption_blocks)

        if not has_continuation_marker:
            return False

        if footnote_count > 1:
            return False
    elif footnote_count > 0:
        return False

    if not _table_widths_are_compatible(current_table_block.get("bbox"), previous_table_block.get("bbox")):
        return False

    return can_merge_by_structure(current_state, previous_state)


def check_rows_match(previous_state: TableMergeState, current_state: TableMergeState) -> bool:
    """检查表格边界行是否匹配."""
    last_row_metrics = previous_state.last_data_row_metrics
    if last_row_metrics is None:
        return False

    header_count, _, _ = detect_table_headers(previous_state, current_state)
    header_count = _expand_header_count_by_rowspan(current_state.rows, header_count)
    first_data_row_metrics = current_state.front_first_data_row_metrics.get(header_count)
    if first_data_row_metrics is None:
        return False

    previous_rendered_segments = calculate_row_rendered_segments(previous_state.rows, last_row_metrics.row_idx)
    current_rendered_segments = calculate_row_rendered_segments(current_state.rows, first_data_row_metrics.row_idx)

    return (
        last_row_metrics.effective_cols == first_data_row_metrics.effective_cols
        or last_row_metrics.actual_cols == first_data_row_metrics.actual_cols
        or previous_rendered_segments == current_rendered_segments
    )


def check_row_columns_match(row1: Tag, row2: Tag) -> bool:
    """判断两行显式单元格数量与 colspan 结构是否一致。"""
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        colspan1 = _colspan(cell1)
        colspan2 = _colspan(cell2)
        if colspan1 != colspan2:
            return False
    return True
