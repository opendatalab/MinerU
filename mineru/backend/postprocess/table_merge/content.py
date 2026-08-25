# Copyright (c) Opendatalab. All rights reserved.
"""跨页表格 HTML 内容、行列结构和单元格语义合并。"""

from __future__ import annotations

from copy import deepcopy

from bs4 import Tag

from ....types import BlockType

from .blocks import (
    _build_post_body_child_index,
    _build_table_state,
    _table_children,
)
from .html import (
    _colspan,
    _refresh_table_state_metrics,
    _rowspan,
    _scan_rows,
    _serialize_table_state_html,
    build_visual_col_mapping,
    calculate_row_columns,
    calculate_visual_columns,
)
from .models import BlockDict, TableMergeState
from .structure import (
    _expand_header_count_by_rowspan,
    can_merge_tables,
    check_row_columns_match,
    detect_table_headers,
)


def adjust_table_rows_colspan(
    rows: list[Tag],
    start_idx: int,
    end_idx: int,
    row_effective_cols: list[int],
    reference_structure: list[int],
    reference_visual_cols: int,
    target_cols: int,
    match_reference_row: Tag,
) -> None:
    """调整表格行的colspan属性以匹配目标列数."""
    reference_row_copy = deepcopy(match_reference_row)

    for row_idx in range(start_idx, end_idx):
        row = rows[row_idx]
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        current_row_effective_cols = row_effective_cols[row_idx]
        current_row_cols = calculate_row_columns(row)

        if current_row_effective_cols >= target_cols or current_row_cols >= target_cols:
            continue

        if calculate_visual_columns(row) == reference_visual_cols and check_row_columns_match(row, reference_row_copy):
            if len(cells) <= len(reference_structure):
                for cell_idx, cell in enumerate(cells):
                    if cell_idx < len(reference_structure) and reference_structure[cell_idx] > 1:
                        cell["colspan"] = str(reference_structure[cell_idx])
        else:
            cols_diff = target_cols - current_row_effective_cols
            if cols_diff > 0:
                last_cell = cells[-1]
                current_last_span = _colspan(last_cell)
                last_cell["colspan"] = str(current_last_span + cols_diff)


def _cell_has_semantic_content(cell: Tag) -> bool:
    """判断单元格是否仍包含用户可见的语义内容。"""
    if cell.get_text(strip=True):
        return True

    return cell.find(["img", "svg", "math", "eq", "table", "figure", "object", "embed", "canvas"]) is not None


def _row_has_semantic_content(row: Tag) -> bool:
    """判断整行是否仍保留未并回的语义内容。"""
    return any(_cell_has_semantic_content(cell) for cell in row.find_all(["td", "th"]))


def _insert_cell_before_visual_column(rows: list[Tag], target_row_index: int, start_vcol: int, cell: Tag) -> None:
    """将单元格插入到目标行中对应视觉列之前。"""
    target_row = rows[target_row_index]
    target_cells = target_row.find_all(["td", "th"])
    target_vcol_map = build_visual_col_mapping(rows, target_row_index)

    for idx, target_start_vcol in enumerate(target_vcol_map):
        if target_start_vcol >= start_vcol:
            target_cells[idx].insert_before(cell)
            return

    target_row.append(cell)


def _carry_rowspan_structure_to_next_row(rows: list[Tag], row_idx: int) -> None:
    """下沉空白结构占位单元格，避免删除当前行后破坏后续列对齐。"""
    next_row_idx = row_idx + 1
    if next_row_idx >= len(rows):
        return

    current_row = rows[row_idx]
    current_cells = current_row.find_all(["td", "th"])
    current_vcol_map = build_visual_col_mapping(rows, row_idx)
    carried_cells = []

    for cell, start_vcol in zip(current_cells, current_vcol_map):
        rowspan = _rowspan(cell)
        if rowspan <= 1 or _cell_has_semantic_content(cell):
            continue

        carried_cell = deepcopy(cell)
        new_rowspan = rowspan - 1
        if new_rowspan > 1:
            carried_cell["rowspan"] = str(new_rowspan)
        else:
            carried_cell.attrs.pop("rowspan", None)
        carried_cells.append((start_vcol, carried_cell))

    for start_vcol, carried_cell in sorted(carried_cells, key=lambda item: item[0], reverse=True):
        _insert_cell_before_visual_column(rows, next_row_idx, start_vcol, carried_cell)


def _clip_overlapped_blank_rowspan_cells(
    rows: list[Tag],
    initial_occupied: dict[int, set[int]],
) -> bool:
    """裁剪被上页 rowspan 覆盖的当前页空白结构占位。

    跨页表格中，上一页未结束的 rowspan 会通过 initial_occupied 占住
    当前页开头的视觉列。如果当前页表格识别又生成了同位置的空白
    rowspan 单元格，这个单元格只是结构占位；直接拼接会把同一视觉列
    当成两列。这里仅裁剪无语义内容的空白占位，真实内容单元格不处理。
    """
    if not rows or not initial_occupied:
        return False

    cells_to_remove = []
    cells_to_move = []

    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        visual_col_map = build_visual_col_mapping(rows, row_idx)
        for cell, start_vcol in zip(cells, visual_col_map):
            rowspan = _rowspan(cell)
            if rowspan <= 1 or _cell_has_semantic_content(cell):
                continue

            colspan = _colspan(cell)
            occupied_cols = set(range(start_vcol, start_vcol + colspan))
            if not occupied_cols:
                continue

            overlap_rows = 0
            while overlap_rows < rowspan:
                covered_cols = initial_occupied.get(row_idx + overlap_rows, set())
                if not occupied_cols.issubset(covered_cols):
                    break
                overlap_rows += 1

            if overlap_rows == 0:
                continue

            remaining_rowspan = rowspan - overlap_rows
            target_row_idx = row_idx + overlap_rows
            if remaining_rowspan > 0 and target_row_idx >= len(rows):
                continue

            cells_to_remove.append(cell)
            if remaining_rowspan > 0:
                moved_cell = deepcopy(cell)
                if remaining_rowspan > 1:
                    moved_cell["rowspan"] = str(remaining_rowspan)
                else:
                    moved_cell.attrs.pop("rowspan", None)
                cells_to_move.append((target_row_idx, start_vcol, moved_cell))

    if not cells_to_remove:
        return False

    for cell in cells_to_remove:
        cell.extract()

    for target_row_idx, start_vcol, moved_cell in sorted(
        cells_to_move,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        _insert_cell_before_visual_column(rows, target_row_idx, start_vcol, moved_cell)

    return True


def _apply_cell_merge(
    previous_state: TableMergeState,
    current_state: TableMergeState,
    header_count: int,
) -> bool:
    """应用 cell_merge 语义合并。

    当 cell_merge 中的值为 1 时，将下表第一数据行对应单元格的内容
    追加到上表最后一行对应单元格中。全部为 1 时删除该数据行，
    混合时清空已合并单元格的内容但保留行。

    cell_merge 按视觉列索引对齐，通过构建视觉列映射来正确匹配
    两个表格中可能因 rowspan 而具有不同 <td> 元素数量的行。
    元数据从当前页 table 根块读取，HTML 与列结构仍由唯一 table body 提供。
    """
    current_table_block = current_state.owner_block
    if not isinstance(current_table_block, dict):
        return False

    cell_merge = current_table_block.get("cell_merge")
    if not isinstance(cell_merge, list) or not cell_merge:
        return False

    rows2 = current_state.rows
    if header_count >= len(rows2):
        return False
    if not previous_state.rows:
        return False

    first_data_row = rows2[header_count]
    last_row = previous_state.rows[-1]

    cells1 = last_row.find_all(["td", "th"])
    cells2 = first_data_row.find_all(["td", "th"])

    # 构建视觉列到单元格索引的映射
    last_row_idx = len(previous_state.rows) - 1
    vcol_map1 = build_visual_col_mapping(previous_state.rows, last_row_idx)
    current_merge_rows = rows2[header_count:]
    vcol_map2 = build_visual_col_mapping(
        current_merge_rows,
        0,
        initial_occupied=previous_state.tail_occupied,
    )

    # 构建视觉列 -> 单元格索引的反向映射（展开 colspan）
    vcol_to_cell1: dict[int, int] = {}
    for ci, start_vcol in enumerate(vcol_map1):
        colspan = int(cells1[ci].get("colspan", 1))
        for c in range(start_vcol, start_vcol + colspan):
            vcol_to_cell1[c] = ci
    vcol_to_cell2: dict[int, int] = {}
    for ci, start_vcol in enumerate(vcol_map2):
        colspan = int(cells2[ci].get("colspan", 1))
        for c in range(start_vcol, start_vcol + colspan):
            vcol_to_cell2[c] = ci

    # 按唯一 (src_cell_idx, dst_cell_idx) 对执行一次转移，避免 colspan 重复处理
    transferred_pairs: set[tuple[int, int]] = set()
    for vi, merge_flag in enumerate(cell_merge):
        if merge_flag == 1:
            ci1 = vcol_to_cell1.get(vi)
            ci2 = vcol_to_cell2.get(vi)
            if ci1 is not None and ci2 is not None:
                pair = (ci1, ci2)
                if pair not in transferred_pairs:
                    for child in list(cells2[ci2].children):
                        cells1[ci1].append(child.extract())
                    transferred_pairs.add(pair)

    # 只清空确实成功转移过的源单元格
    cleared_ci2: set[int] = set()
    for vi, merge_flag in enumerate(cell_merge):
        if merge_flag == 1:
            ci1 = vcol_to_cell1.get(vi)
            ci2 = vcol_to_cell2.get(vi)
            if ci1 is not None and ci2 is not None and ci2 not in cleared_ci2:
                cells2[ci2].clear()
                cleared_ci2.add(ci2)

    if not _row_has_semantic_content(first_data_row):
        _carry_rowspan_structure_to_next_row(rows2, header_count)
        first_data_row.extract()
        if first_data_row in rows2:
            rows2.remove(first_data_row)

    return bool(transferred_pairs)


def _perform_table_content_merge(
    previous_state: TableMergeState,
    current_state: TableMergeState,
    previous_table_block: BlockDict,
    current_table_block: BlockDict,
) -> bool:
    """在两个克隆表格上执行 HTML、单元格和 footnote 的内容合并。"""
    header_count, _, _ = detect_table_headers(previous_state, current_state)
    header_count = _expand_header_count_by_rowspan(current_state.rows, header_count)

    rows1 = previous_state.rows
    rows2 = current_state.rows
    if not rows1 or header_count >= len(rows2):
        return False

    previous_adjusted = False

    if header_count < len(rows2):
        current_merge_rows = rows2[header_count:]
        if _clip_overlapped_blank_rowspan_cells(current_merge_rows, previous_state.tail_occupied):
            _refresh_table_state_metrics(current_state)

    if rows1 and rows2 and header_count < len(rows2):
        last_row1 = rows1[-1]
        first_data_row2 = rows2[header_count]
        table_cols1 = previous_state.total_cols
        table_cols2 = current_state.total_cols

        if table_cols1 > table_cols2:
            reference_structure = [int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(last_row1)
            adjust_table_rows_colspan(
                rows2,
                header_count,
                len(rows2),
                current_state.row_effective_cols,
                reference_structure,
                reference_visual_cols,
                table_cols1,
                first_data_row2,
            )
        elif table_cols2 > table_cols1:
            reference_structure = [int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(first_data_row2)
            adjust_table_rows_colspan(
                rows1,
                0,
                len(rows1),
                previous_state.row_effective_cols,
                reference_structure,
                reference_visual_cols,
                table_cols2,
                last_row1,
            )
            previous_adjusted = True

    if previous_adjusted:
        _refresh_table_state_metrics(previous_state)

    cell_merge_applied = _apply_cell_merge(previous_state, current_state, header_count)

    appended_rows = rows2[header_count:]
    append_start_idx = len(previous_state.rows)
    merged_rows = []

    if previous_state.tbody is None or current_state.tbody is None:
        return False

    for row in appended_rows:
        row.extract()
        previous_state.tbody.append(row)
        merged_rows.append(row)

    if not merged_rows and not cell_merge_applied:
        return False

    previous_state.rows.extend(merged_rows)

    if merged_rows:
        appended_scan = _scan_rows(
            merged_rows,
            initial_occupied=previous_state.tail_occupied,
            start_row_idx=append_start_idx,
        )
        previous_state.row_effective_cols.extend(appended_scan.row_effective_cols)
        previous_state.total_cols = max(previous_state.total_cols, appended_scan.total_cols)
        if appended_scan.last_nonempty_row_metrics is not None:
            previous_state.last_data_row_metrics = appended_scan.last_nonempty_row_metrics
        previous_state.tail_occupied = appended_scan.tail_occupied

    previous_content = previous_table_block.get("content")
    if not isinstance(previous_content, list):
        return False

    previous_table_block["content"] = [
        block for block in previous_content if not isinstance(block, dict) or block.get("type") != BlockType.TABLE_FOOTNOTE
    ]
    current_footnotes = [
        block for block in _table_children(current_table_block) if block.get("type") == BlockType.TABLE_FOOTNOTE
    ]
    footnote_base_index = _build_post_body_child_index(previous_table_block, 0)
    for footnote_offset, table_footnote in enumerate(current_footnotes, start=1):
        temp_table_footnote = deepcopy(table_footnote)
        temp_table_footnote.pop("_cross_page", None)
        if footnote_base_index is None:
            temp_table_footnote["index"] = 0
        else:
            temp_table_footnote["index"] = footnote_base_index + footnote_offset
        previous_table_block["content"].append(temp_table_footnote)

    previous_state.dirty = True
    return _serialize_table_state_html(previous_state)


def merge_table_content(previous_table: BlockDict, current_table: BlockDict) -> BlockDict | None:
    """纯函数式合并两张跨页表格的内容，失败时返回 ``None``。

    两个输入都会先深拷贝；返回块保留前表外层信息，只改克隆表体 HTML
    并用当前表 footnote 替换前表 footnote，不会修改任何输入对象。
    """
    if (
        not isinstance(previous_table, dict)
        or not isinstance(current_table, dict)
        or previous_table.get("type") != BlockType.TABLE
        or current_table.get("type") != BlockType.TABLE
    ):
        return None

    previous_clone = deepcopy(previous_table)
    current_clone = deepcopy(current_table)
    try:
        previous_state = _build_table_state(previous_clone)
        current_state = _build_table_state(current_clone)
        if previous_state is None or current_state is None:
            return None
        if not can_merge_tables(current_state, previous_state):
            return None
        if not _perform_table_content_merge(
            previous_state,
            current_state,
            previous_clone,
            current_clone,
        ):
            return None
    except (AssertionError, TypeError, ValueError):
        return None

    return previous_clone
