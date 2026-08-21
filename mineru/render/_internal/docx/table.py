# Copyright (c) Opendatalab. All rights reserved.
"""DOCX renderer 的 HTML 表格占位网格解析与原生表格物化。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

from bs4 import BeautifulSoup
from bs4.element import Tag
from docx.document import Document as DocxDocument
from docx.exceptions import InvalidSpanError
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Twips
from docx.table import Table, _Cell


DEFAULT_TABLE_WIDTH_TWIPS = 8640
MAX_NESTED_TABLE_DEPTH = 4
_POSITIVE_INTEGER_RE = re.compile(r"[0-9]+")

HtmlTableSource: TypeAlias = str | BeautifulSoup | Tag


class DocxTableError(ValueError):
    """表示 HTML 表格结构或 DOCX 表格几何无法安全物化。"""


@dataclass(frozen=True, slots=True)
class HtmlTableCell:
    """保存一个 HTML 原始单元格在逻辑占位网格中的位置。"""

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
    """保存一个经过完整矩形校验的 HTML 表格占位网格。"""

    tag: Tag
    row_count: int
    column_count: int
    cells: tuple[HtmlTableCell, ...]
    occupancy: tuple[tuple[HtmlTableCell, ...], ...]
    header_rows: tuple[int, ...]


class NestedTableWriter(Protocol):
    """定义单元格回调可调用的嵌套表格写入函数。"""

    def __call__(
        self,
        source: HtmlTableSource,
        width_twips: int | None = None,
    ) -> tuple[Table, ...]:
        """把 source 中的直接嵌套表格写入当前 Word 单元格。"""
        ...


class CellFillCallback(Protocol):
    """定义由上层 renderer 填充原始单元格内容的回调。"""

    def __call__(
        self,
        cell: _Cell,
        source: Tag,
        write_nested: NestedTableWriter,
    ) -> None:
        """使用原始 td/th 标签填充 origin cell，并按需递归写入嵌套表格。"""
        ...


def parse_html_table(table: Tag) -> HtmlTableGrid:
    """把单个 table 标签解析为经过重叠、越界和矩形校验的占位网格。"""
    if table.name != "table":
        raise DocxTableError("Expected a <table> tag")

    rows = tuple(row for row in table.find_all("tr") if row.find_parent("table") is table)
    if not rows:
        raise DocxTableError("Table must contain at least one row")

    occupied: dict[tuple[int, int], HtmlTableCell] = {}
    cells: list[HtmlTableCell] = []

    for row_index, row in enumerate(rows):
        column_index = 0
        source_cells = row.find_all(("td", "th"), recursive=False)
        for source_cell in source_cells:
            while (row_index, column_index) in occupied:
                column_index += 1

            rowspan = _parse_span(source_cell, "rowspan")
            colspan = _parse_span(source_cell, "colspan")
            if row_index + rowspan > len(rows):
                raise DocxTableError(f"rowspan exceeds table bounds at row={row_index}, column={column_index}")

            coordinates = tuple(
                (target_row, target_column)
                for target_row in range(row_index, row_index + rowspan)
                for target_column in range(column_index, column_index + colspan)
            )
            overlap = next((coordinate for coordinate in coordinates if coordinate in occupied), None)
            if overlap is not None:
                raise DocxTableError(f"Cell span overlaps an occupied coordinate at row={overlap[0]}, column={overlap[1]}")

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
            column_index += colspan

    if not occupied:
        raise DocxTableError("Table must contain at least one cell")

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
        raise DocxTableError(f"Table occupancy must be rectangular; missing row={missing[0]}, column={missing[1]}")

    occupancy = tuple(
        tuple(occupied[(row_index, column_index)] for column_index in range(column_count)) for row_index in range(len(rows))
    )
    header_rows = tuple(
        row_index
        for row_index, row in enumerate(rows)
        if _row_belongs_to_thead(row, table) or all(cell.is_header for cell in occupancy[row_index])
    )
    return HtmlTableGrid(
        tag=table,
        row_count=len(rows),
        column_count=column_count,
        cells=tuple(cells),
        occupancy=occupancy,
        header_rows=header_rows,
    )


def parse_html_tables(source: HtmlTableSource) -> tuple[HtmlTableGrid, ...]:
    """解析 source 中相对当前上下文的一个或多个顶层 table。"""
    root = BeautifulSoup(source, "html.parser") if isinstance(source, str) else source
    if not isinstance(root, (BeautifulSoup, Tag)):
        raise DocxTableError("HTML table source must be a string or BeautifulSoup Tag")

    if isinstance(root, Tag) and root.name == "table":
        table_tags = (root,)
    else:
        parent_table = root.find_parent("table") if isinstance(root, Tag) else None
        table_tags = tuple(table for table in root.find_all("table") if table.find_parent("table") is parent_table)
    if not table_tags:
        raise DocxTableError("HTML does not contain a top-level table")
    return tuple(parse_html_table(table) for table in table_tags)


def materialize_docx_tables(
    container: Any,
    source: HtmlTableSource,
    *,
    width_twips: int = DEFAULT_TABLE_WIDTH_TWIPS,
    fill_cell: CellFillCallback,
) -> tuple[Table, ...]:
    """把 source 中的全部顶层表格物化到 Document、Header/Footer 或 Cell。"""
    grids = parse_html_tables(source)
    return _materialize_grids(
        container,
        grids,
        width_twips=width_twips,
        fill_cell=fill_cell,
        depth=1,
    )


def materialize_docx_table(
    container: Any,
    grid: HtmlTableGrid,
    *,
    width_twips: int = DEFAULT_TABLE_WIDTH_TWIPS,
    fill_cell: CellFillCallback,
) -> Table:
    """把一个已解析表格网格物化到指定 python-docx 容器。"""
    return _materialize_grid(
        container,
        grid,
        width_twips=width_twips,
        fill_cell=fill_cell,
        depth=1,
    )


def _materialize_grids(
    container: Any,
    grids: tuple[HtmlTableGrid, ...],
    *,
    width_twips: int,
    fill_cell: CellFillCallback,
    depth: int,
) -> tuple[Table, ...]:
    """在同一递归层级内按源码顺序物化全部表格网格。"""
    if depth > MAX_NESTED_TABLE_DEPTH:
        raise DocxTableError(f"Nested table depth exceeds {MAX_NESTED_TABLE_DEPTH}")
    _validate_width(width_twips)
    return tuple(
        _materialize_grid(
            container,
            grid,
            width_twips=width_twips,
            fill_cell=fill_cell,
            depth=depth,
        )
        for grid in grids
    )


def _materialize_grid(
    container: Any,
    grid: HtmlTableGrid,
    *,
    width_twips: int,
    fill_cell: CellFillCallback,
    depth: int,
) -> Table:
    """创建单个 Word 表格、应用合并几何并回调填充所有 origin cell。"""
    if depth > MAX_NESTED_TABLE_DEPTH:
        raise DocxTableError(f"Nested table depth exceeds {MAX_NESTED_TABLE_DEPTH}")
    column_widths = _split_width(width_twips, grid.column_count)
    table = _add_table(container, grid.row_count, grid.column_count, width_twips)
    _configure_table_geometry(table, width_twips, column_widths)
    _apply_cell_merges(table, grid)
    _normalize_cell_widths(table, column_widths)
    _clear_fixed_row_heights(table)
    _mark_header_rows(table, grid.header_rows)

    for placement in grid.cells:
        origin_cell = table.cell(placement.row, placement.column)
        origin_width = sum(column_widths[placement.column : placement.column + placement.colspan])

        def write_nested(
            nested_source: HtmlTableSource,
            nested_width_twips: int | None = None,
            *,
            _origin_cell: _Cell = origin_cell,
            _origin_width: int = origin_width,
            _depth: int = depth,
        ) -> tuple[Table, ...]:
            """在当前 origin cell 内继续物化直接嵌套表格。"""
            if _depth >= MAX_NESTED_TABLE_DEPTH:
                raise DocxTableError(f"Nested table depth exceeds {MAX_NESTED_TABLE_DEPTH}")
            nested_grids = parse_html_tables(nested_source)
            return _materialize_grids(
                _origin_cell,
                nested_grids,
                width_twips=_origin_width if nested_width_twips is None else nested_width_twips,
                fill_cell=fill_cell,
                depth=_depth + 1,
            )

        fill_cell(origin_cell, placement.tag, write_nested)
    return table


def _parse_span(cell: Tag, attribute: str) -> int:
    """读取严格正整数 rowspan/colspan，缺失属性时返回一。"""
    raw_value = cell.get(attribute, "1")
    if isinstance(raw_value, list):
        raise DocxTableError(f"Invalid {attribute}: {raw_value!r}")
    value = str(raw_value).strip()
    if _POSITIVE_INTEGER_RE.fullmatch(value) is None:
        raise DocxTableError(f"Invalid {attribute}: {raw_value!r}")
    span = int(value)
    if span < 1:
        raise DocxTableError(f"Invalid {attribute}: {raw_value!r}")
    return span


def _row_belongs_to_thead(row: Tag, table: Tag) -> bool:
    """判断当前 tr 是否位于本 table 的 thead 内。"""
    parent = row.parent
    while isinstance(parent, Tag) and parent is not table:
        if parent.name == "thead":
            return True
        parent = parent.parent
    return False


def _validate_width(width_twips: int) -> None:
    """校验调用方传入的 DXA/twips 表格宽度。"""
    if isinstance(width_twips, bool) or not isinstance(width_twips, int) or width_twips <= 0:
        raise DocxTableError("width_twips must be a positive integer")


def _split_width(width_twips: int, column_count: int) -> tuple[int, ...]:
    """把总宽度确定性地均分到列，并保证列宽之和严格等于总宽。"""
    _validate_width(width_twips)
    if column_count <= 0:
        raise DocxTableError("Table must contain at least one column")
    if width_twips < column_count:
        raise DocxTableError("width_twips is too small for the table column count")
    base_width, remainder = divmod(width_twips, column_count)
    return tuple(base_width + (1 if column_index < remainder else 0) for column_index in range(column_count))


def _add_table(container: Any, row_count: int, column_count: int, width_twips: int) -> Table:
    """按 python-docx 容器的不同 add_table 签名创建空表格。"""
    if not hasattr(container, "add_table"):
        raise DocxTableError("DOCX container must provide add_table()")
    try:
        if isinstance(container, (DocxDocument, _Cell)):
            return container.add_table(rows=row_count, cols=column_count)
        return container.add_table(
            rows=row_count,
            cols=column_count,
            width=Twips(width_twips),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise DocxTableError("Failed to create DOCX table in the target container") from exc


def _configure_table_geometry(
    table: Table,
    width_twips: int,
    column_widths: tuple[int, ...],
) -> None:
    """设置固定布局、Table Grid 样式及确定性的 tblW、tblGrid、tcW。"""
    try:
        table.style = "Table Grid"
    except KeyError as exc:
        raise DocxTableError("DOCX template does not define the 'Table Grid' style") from exc
    table.autofit = False

    table_width = table._tbl.tblPr.find(qn("w:tblW"))
    if table_width is None:
        table_width = OxmlElement("w:tblW")
        table._tbl.tblPr.insert(0, table_width)
    table_width.set(qn("w:type"), "dxa")
    table_width.set(qn("w:w"), str(width_twips))

    grid_columns = table._tbl.tblGrid.gridCol_lst
    if len(grid_columns) != len(column_widths):
        raise DocxTableError("DOCX table grid column count is inconsistent")
    for grid_column, column_width in zip(grid_columns, column_widths):
        grid_column.w = Twips(column_width)

    for row in table._tbl.tr_lst:
        if len(row.tc_lst) != len(column_widths):
            raise DocxTableError("DOCX table row is inconsistent before cell merging")
        for cell, column_width in zip(row.tc_lst, column_widths):
            cell_width = cell.get_or_add_tcPr().get_or_add_tcW()
            cell_width.type = "dxa"
            cell_width.w = column_width


def _apply_cell_merges(table: Table, grid: HtmlTableGrid) -> None:
    """按已验证 origin placement 为 Word 表格应用水平和垂直合并。"""
    for placement in grid.cells:
        if placement.rowspan == 1 and placement.colspan == 1:
            continue
        try:
            table.cell(placement.row, placement.column).merge(table.cell(placement.end_row, placement.end_column))
        except (IndexError, InvalidSpanError, ValueError) as exc:
            raise DocxTableError(f"Failed to merge cell at row={placement.row}, column={placement.column}") from exc


def _normalize_cell_widths(table: Table, column_widths: tuple[int, ...]) -> None:
    """合并后按 gridSpan 重新写入每个物理 tc 的确定性 DXA 宽度。"""
    for row_index, row in enumerate(table._tbl.tr_lst):
        column_index = 0
        for cell in row.tc_lst:
            colspan = cell.grid_span
            if colspan < 1 or column_index + colspan > len(column_widths):
                raise DocxTableError(f"DOCX cell geometry is invalid at row={row_index}")
            width = sum(column_widths[column_index : column_index + colspan])
            cell_width = cell.get_or_add_tcPr().get_or_add_tcW()
            cell_width.type = "dxa"
            cell_width.w = width
            column_index += colspan
        if column_index != len(column_widths):
            raise DocxTableError(f"DOCX row width is inconsistent at row={row_index}")


def _clear_fixed_row_heights(table: Table) -> None:
    """删除所有固定 trHeight，使 Word 根据单元格内容自然扩展行高。"""
    for row in table._tbl.tr_lst:
        row_properties = row.trPr
        if row_properties is None:
            continue
        for height in tuple(row_properties.findall(qn("w:trHeight"))):
            row_properties.remove(height)


def _mark_header_rows(table: Table, header_rows: tuple[int, ...]) -> None:
    """把 thead 或全 th 行标记为可跨页重复的 Word 表头行。"""
    for row_index in header_rows:
        row_properties = table.rows[row_index]._tr.get_or_add_trPr()
        header = row_properties.find(qn("w:tblHeader"))
        if header is None:
            header = OxmlElement("w:tblHeader")
            row_properties.append(header)
        header.set(qn("w:val"), "1")


__all__ = [
    "CellFillCallback",
    "DEFAULT_TABLE_WIDTH_TWIPS",
    "DocxTableError",
    "HtmlTableCell",
    "HtmlTableGrid",
    "MAX_NESTED_TABLE_DEPTH",
    "NestedTableWriter",
    "materialize_docx_table",
    "materialize_docx_tables",
    "parse_html_table",
    "parse_html_tables",
]
