# Copyright (c) Opendatalab. All rights reserved.
"""多格式 renderer 共用的有界 HTML table 占位网格解析。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TypeAlias

from bs4 import BeautifulSoup, Tag

MAX_NESTED_TABLE_DEPTH = 4
MAX_TABLE_ROWS = 500
MAX_TABLE_COLUMNS = 100
MAX_TABLE_SLOTS = 10_000
_POSITIVE_INTEGER_RE = re.compile(r"[0-9]+")

HtmlTableSource: TypeAlias = str | BeautifulSoup | Tag


class HtmlTableError(ValueError):
    """表示 HTML table 无法安全解析为严格矩形网格。"""


@dataclass(frozen=True, slots=True)
class HtmlTableCell:
    """保存原始 HTML 单元格在逻辑占位网格中的位置。"""

    tag: Tag
    row: int
    column: int
    rowspan: int
    colspan: int
    is_header: bool

    @property
    def end_row(self) -> int:
        """返回单元格占用的末行下标。"""
        return self.row + self.rowspan - 1

    @property
    def end_column(self) -> int:
        """返回单元格占用的末列下标。"""
        return self.column + self.colspan - 1


@dataclass(frozen=True, slots=True)
class HtmlTableGrid:
    """保存经过重叠、边界、规模与矩形校验的 HTML 表格网格。"""

    tag: Tag
    row_count: int
    column_count: int
    cells: tuple[HtmlTableCell, ...]
    header_rows: tuple[int, ...]


def parse_html_tables(source: HtmlTableSource) -> tuple[HtmlTableGrid, ...]:
    """解析 source 中相对当前上下文的一个或多个顶层 table。"""
    root = BeautifulSoup(source, "html.parser") if isinstance(source, str) else source
    if not isinstance(root, (BeautifulSoup, Tag)):
        raise HtmlTableError("HTML table source must be a string or BeautifulSoup Tag")
    if isinstance(root, Tag) and root.name == "table":
        table_tags = (root,)
    else:
        parent_table = root.find_parent("table") if isinstance(root, Tag) else None
        table_tags = tuple(table for table in root.find_all("table") if table.find_parent("table") is parent_table)
    if not table_tags:
        raise HtmlTableError("HTML does not contain a top-level table")
    return tuple(_parse_html_table(table) for table in table_tags)


def _parse_html_table(table: Tag) -> HtmlTableGrid:
    """把单个 table 标签解析为严格矩形占位网格。"""
    if table.name != "table":
        raise HtmlTableError("Expected a <table> tag")
    rows = tuple(row for row in table.find_all("tr") if row.find_parent("table") is table)
    if not rows or len(rows) > MAX_TABLE_ROWS:
        raise HtmlTableError(f"Table row count must be between 1 and {MAX_TABLE_ROWS}")

    occupied: dict[tuple[int, int], HtmlTableCell] = {}
    cells: list[HtmlTableCell] = []
    for row_index, row in enumerate(rows):
        column_index = 0
        for source_cell in row.find_all(("td", "th"), recursive=False):
            while (row_index, column_index) in occupied:
                column_index += 1
            rowspan = _parse_span(source_cell, "rowspan")
            colspan = _parse_span(source_cell, "colspan")
            if row_index + rowspan > len(rows):
                raise HtmlTableError(f"rowspan exceeds table bounds at row={row_index}, column={column_index}")
            if column_index + colspan > MAX_TABLE_COLUMNS:
                raise HtmlTableError(f"Table column count exceeds {MAX_TABLE_COLUMNS}")
            coordinates = tuple(
                (target_row, target_column)
                for target_row in range(row_index, row_index + rowspan)
                for target_column in range(column_index, column_index + colspan)
            )
            overlap = next((coordinate for coordinate in coordinates if coordinate in occupied), None)
            if overlap is not None:
                raise HtmlTableError(f"Cell span overlaps row={overlap[0]}, column={overlap[1]}")
            placement = HtmlTableCell(
                tag=source_cell,
                row=row_index,
                column=column_index,
                rowspan=rowspan,
                colspan=colspan,
                is_header=source_cell.name == "th",
            )
            cells.append(placement)
            occupied.update(dict.fromkeys(coordinates, placement))
            if len(occupied) > MAX_TABLE_SLOTS:
                raise HtmlTableError(f"Table occupancy exceeds {MAX_TABLE_SLOTS} slots")
            column_index += colspan

    if not occupied:
        raise HtmlTableError("Table must contain at least one cell")
    column_count = max(column for _, column in occupied) + 1
    missing = next(
        (
            (row_index, column_index)
            for row_index in range(len(rows))
            for column_index in range(column_count)
            if (row_index, column_index) not in occupied
        ),
        None,
    )
    if missing is not None:
        raise HtmlTableError(f"Table occupancy is not rectangular at row={missing[0]}, column={missing[1]}")
    header_rows = tuple(
        row_index
        for row_index, row in enumerate(rows)
        if _row_belongs_to_thead(row, table)
        or all(occupied[(row_index, column_index)].is_header for column_index in range(column_count))
    )
    return HtmlTableGrid(
        tag=table,
        row_count=len(rows),
        column_count=column_count,
        cells=tuple(cells),
        header_rows=header_rows,
    )


def _parse_span(cell: Tag, attribute: str) -> int:
    """读取严格正整数 rowspan/colspan，缺失时返回一。"""
    raw_value = cell.get(attribute, "1")
    if isinstance(raw_value, list):
        raise HtmlTableError(f"Invalid {attribute}: {raw_value!r}")
    value = str(raw_value).strip()
    if _POSITIVE_INTEGER_RE.fullmatch(value) is None:
        raise HtmlTableError(f"Invalid {attribute}: {raw_value!r}")
    span = int(value)
    if span < 1 or span > MAX_TABLE_SLOTS:
        raise HtmlTableError(f"Invalid {attribute}: {raw_value!r}")
    return span


def _row_belongs_to_thead(row: Tag, table: Tag) -> bool:
    """判断 tr 是否位于当前 table 的 thead 内。"""
    parent = row.parent
    while isinstance(parent, Tag) and parent is not table:
        if parent.name == "thead":
            return True
        parent = parent.parent
    return False


__all__ = [
    "HtmlTableCell",
    "HtmlTableError",
    "HtmlTableGrid",
    "HtmlTableSource",
    "MAX_NESTED_TABLE_DEPTH",
    "MAX_TABLE_COLUMNS",
    "MAX_TABLE_ROWS",
    "MAX_TABLE_SLOTS",
    "parse_html_tables",
]
