# Copyright (c) Opendatalab. All rights reserved.

"""解析 BIFF chart BRAI 中的简单单元格引用。"""

from __future__ import annotations

import struct

from .records import BiffRecord

BRAI = 0x1051


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
) -> tuple[list[int], list[int]] | None:
    """解析单一 PtgRef/PtgArea 及其 3D 变体。"""

    if not tokens:
        return None
    token = tokens[0] & 0x1F
    if token == 0x04 and len(tokens) >= 5:
        row, column_flags = struct.unpack_from("<HH", tokens, 1)
        return [int(row)], [int(column_flags & 0x00FF)]
    if token == 0x05 and len(tokens) >= 9:
        row_first, row_last, col_first, col_last = struct.unpack_from("<4H", tokens, 1)
        return (
            list(range(min(row_first, row_last), max(row_first, row_last) + 1)),
            list(range((col_first & 0x00FF), (col_last & 0x00FF) + 1)),
        )
    if token == 0x1A and len(tokens) >= 7:
        extern_index, row, column_flags = struct.unpack_from("<3H", tokens, 1)
        if _extern_sheet_index(extern_sheets, int(extern_index)) != current_sheet_index:
            return None
        return [int(row)], [int(column_flags & 0x00FF)]
    if token == 0x1B and len(tokens) >= 11:
        extern_index, row_first, row_last, col_first, col_last = struct.unpack_from(
            "<5H", tokens, 1
        )
        if _extern_sheet_index(extern_sheets, int(extern_index)) != current_sheet_index:
            return None
        first_col = col_first & 0x00FF
        last_col = col_last & 0x00FF
        return (
            list(range(min(row_first, row_last), max(row_first, row_last) + 1)),
            list(range(min(first_col, last_col), max(first_col, last_col) + 1)),
        )
    return None


def chart_source_axes(
    records: list[BiffRecord],
    *,
    current_sheet_index: int,
    extern_sheets: list[int | None],
) -> tuple[list[int], list[int]] | None:
    """合并 chart 名称、分类、数值与气泡引用的行列集合。"""

    rows: set[int] = set()
    cols: set[int] = set()
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
        reference_rows, reference_cols = reference
        rows.update(reference_rows)
        cols.update(reference_cols)
    if not formulas_found or not rows or not cols:
        return None
    return sorted(rows), sorted(cols)
