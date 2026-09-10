# Copyright (c) Opendatalab. All rights reserved.
"""Structured row/cell extraction from recognized table HTML.

Table recognition emits HTML only. This module turns that HTML into an
explicit grid so consumers can address cells by position without re-parsing
markup or resolving colspan/rowspan themselves.
"""

from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup


@dataclass
class TableCell:
    """A single cell placed on the resolved table grid.

    row_index/col_index are grid coordinates, so a cell displaced by a
    rowspan from an earlier row reports where it actually renders rather
    than its ordinal position within its own <tr>.
    """

    text: str
    row_index: int
    col_index: int
    colspan: int = 1
    rowspan: int = 1
    is_header: bool = False
    nested_html: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "colspan": self.colspan,
            "rowspan": self.rowspan,
            "is_header": self.is_header,
        }
        if self.nested_html:
            data["nested_html"] = self.nested_html
        return data


@dataclass
class TableRow:
    row_index: int
    cells: list[TableCell] = field(default_factory=list)
    is_header_row: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "is_header_row": self.is_header_row,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass
class TableStructure:
    rows: list[TableRow] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    header_row_count: int = 0
    has_merged_cells: bool = False
    has_nested_tables: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "header_row_count": self.header_row_count,
            "has_merged_cells": self.has_merged_cells,
            "has_nested_tables": self.has_nested_tables,
            "rows": [row.to_dict() for row in self.rows],
        }

    def cell_at(self, row_index: int, col_index: int) -> TableCell | None:
        """Cell covering a grid coordinate, including cells spanned into it."""
        for row in self.rows:
            for cell in row.cells:
                if (
                    cell.row_index <= row_index < cell.row_index + cell.rowspan
                    and cell.col_index <= col_index < cell.col_index + cell.colspan
                ):
                    return cell
        return None


def _span_attr(cell, name: str) -> int:
    """Span attributes come from model output and may be absent or malformed."""
    raw = cell.get(name, 1)
    if isinstance(raw, list):
        raw = raw[0] if raw else 1
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 1
    return value if value >= 1 else 1


def _direct_rows(table_tag) -> list:
    return [tr for tr in table_tag.find_all("tr") if tr.find_parent("table") is table_tag]


def _direct_cells(row_tag) -> list:
    return [c for c in row_tag.find_all(["td", "th"]) if c.find_parent("tr") is row_tag]


def _cell_text(cell) -> str:
    return cell.get_text(separator=" ", strip=True)


def _nested_table_html(cell) -> str | None:
    nested = cell.find("table")
    return str(nested) if nested is not None else None


def _is_header_cell(cell) -> bool:
    return cell.name == "th" or cell.find_parent("thead") is not None


def parse_table_html(html: str) -> TableStructure | None:
    """Parse recognized table HTML into a resolved grid.

    Returns None when the HTML holds no table rows.
    """
    if not html or not html.strip():
        return None

    soup = BeautifulSoup(html, "html.parser")
    table_tag = soup.find("table")
    rows = _direct_rows(table_tag) if table_tag is not None else soup.find_all("tr")
    if not rows:
        return None

    structure = TableStructure()
    occupied: dict[int, set[int]] = {}
    column_count = 0

    for row_index, row_tag in enumerate(rows):
        row_occupied = occupied.setdefault(row_index, set())
        table_row = TableRow(row_index=row_index)
        col_index = 0

        for cell_tag in _direct_cells(row_tag):
            while col_index in row_occupied:
                col_index += 1

            colspan = _span_attr(cell_tag, "colspan")
            rowspan = _span_attr(cell_tag, "rowspan")
            nested_html = _nested_table_html(cell_tag)

            cell = TableCell(
                text=_cell_text(cell_tag),
                row_index=row_index,
                col_index=col_index,
                colspan=colspan,
                rowspan=rowspan,
                is_header=_is_header_cell(cell_tag),
                nested_html=nested_html,
            )
            table_row.cells.append(cell)

            if colspan > 1 or rowspan > 1:
                structure.has_merged_cells = True
            if nested_html:
                structure.has_nested_tables = True

            for row_offset in range(rowspan):
                occupied.setdefault(row_index + row_offset, set()).update(
                    range(col_index, col_index + colspan)
                )

            col_index += colspan
            column_count = max(column_count, col_index)

        table_row.is_header_row = bool(table_row.cells) and all(
            cell.is_header for cell in table_row.cells
        )
        structure.rows.append(table_row)
        column_count = max(column_count, max(row_occupied) + 1 if row_occupied else 0)

    structure.row_count = len(structure.rows)
    structure.column_count = column_count
    structure.header_row_count = _leading_header_row_count(structure.rows)
    return structure


def _leading_header_row_count(rows: list[TableRow]) -> int:
    count = 0
    for row in rows:
        if not row.is_header_row:
            break
        count += 1
    return count


def parse_table_html_to_dict(html: str) -> dict[str, Any] | None:
    structure = parse_table_html(html)
    return structure.to_dict() if structure else None
