# Copyright (c) Opendatalab. All rights reserved.

"""解析 BIFF chart BRAI 中的简单单元格引用。"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .records import BiffRecord

BRAI = 0x1051


@dataclass(frozen=True, slots=True)
class ChartSourceSelection:
    """一个 chart 引用到的唯一源工作表及行列集合。"""

    sheet_index: int
    rows: tuple[int, ...]
    cols: tuple[int, ...]


def _extern_sheet_index(
    extern_sheets: list[int | None],
    extern_index: int,
) -> int | None:
    """把 ixti 解析为内部工作表索引。"""

    if 0 <= extern_index < len(extern_sheets):
        return extern_sheets[extern_index]
    return None


def _reference_from_tokens(
    tokens: bytes,
    *,
    current_sheet_index: int,
    extern_sheets: list[int | None],
) -> tuple[int, list[int], list[int]] | None:
    """解析单一 PtgRef/PtgArea 及其 3D 变体和源工作表。"""

    if not tokens:
        return None
    token = tokens[0] & 0x1F
    if token == 0x04 and len(tokens) >= 5:
        row, column_flags = struct.unpack_from("<HH", tokens, 1)
        return current_sheet_index, [int(row)], [int(column_flags & 0x00FF)]
    if token == 0x05 and len(tokens) >= 9:
        row_first, row_last, col_first, col_last = struct.unpack_from("<4H", tokens, 1)
        return (
            current_sheet_index,
            list(range(min(row_first, row_last), max(row_first, row_last) + 1)),
            list(range((col_first & 0x00FF), (col_last & 0x00FF) + 1)),
        )
    if token == 0x1A and len(tokens) >= 7:
        extern_index, row, column_flags = struct.unpack_from("<3H", tokens, 1)
        sheet_index = _extern_sheet_index(extern_sheets, int(extern_index))
        if sheet_index is None:
            return None
        return sheet_index, [int(row)], [int(column_flags & 0x00FF)]
    if token == 0x1B and len(tokens) >= 11:
        extern_index, row_first, row_last, col_first, col_last = struct.unpack_from("<5H", tokens, 1)
        sheet_index = _extern_sheet_index(extern_sheets, int(extern_index))
        if sheet_index is None:
            return None
        first_col = col_first & 0x00FF
        last_col = col_last & 0x00FF
        return (
            sheet_index,
            list(range(min(row_first, row_last), max(row_first, row_last) + 1)),
            list(range(min(first_col, last_col), max(first_col, last_col) + 1)),
        )
    return None


def chart_source_selection(
    records: list[BiffRecord],
    *,
    current_sheet_index: int,
    extern_sheets: list[int | None],
) -> ChartSourceSelection | None:
    """合并 chart 名称、分类、数值与气泡引用并要求唯一源工作表。"""

    rows: set[int] = set()
    cols: set[int] = set()
    sheet_indices: set[int] = set()
    formulas_found = False
    for record in records:
        if record.record_type != BRAI or len(record.payload) < 8:
            continue
        formula_length = int(struct.unpack_from("<H", record.payload, 6)[0])
        if formula_length <= 0:
            continue
        formulas_found = True
        tokens = record.payload[8 : 8 + formula_length]
        if len(tokens) != formula_length:
            return None
        reference = _reference_from_tokens(
            tokens,
            current_sheet_index=current_sheet_index,
            extern_sheets=extern_sheets,
        )
        if reference is None:
            return None
        sheet_index, reference_rows, reference_cols = reference
        sheet_indices.add(sheet_index)
        rows.update(reference_rows)
        cols.update(reference_cols)
    if not formulas_found or not rows or not cols or len(sheet_indices) != 1:
        return None
    return ChartSourceSelection(
        sheet_index=next(iter(sheet_indices)),
        rows=tuple(sorted(rows)),
        cols=tuple(sorted(cols)),
    )


def chart_source_axes(
    records: list[BiffRecord],
    *,
    current_sheet_index: int,
    extern_sheets: list[int | None],
) -> tuple[list[int], list[int]] | None:
    """兼容 worksheet 内嵌 chart，仅接受引用当前工作表的数据。"""

    selection = chart_source_selection(
        records,
        current_sheet_index=current_sheet_index,
        extern_sheets=extern_sheets,
    )
    if selection is None or selection.sheet_index != current_sheet_index:
        return None
    return list(selection.rows), list(selection.cols)
