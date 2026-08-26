# Copyright (c) Opendatalab. All rights reserved.
"""构造受限 ODF 表格网格并序列化为安全 HTML。"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Iterator

from lxml import etree  # type: ignore[reportMissingImports]

from ..limits import MAX_GRID_SLOTS
from .constants import MAX_EXPANSION_TEXT_BYTES, qname
from .errors import OdfResourceLimitError
from .models import GridCell, TableGrid


CellRenderer = Callable[[etree._Element], str]
_HTML_TEXT_RE = re.compile(r"<[^>]+>")
_CELL_ADDRESS_RE = re.compile(r"\$?(?P<col>[A-Za-z]+)\$?(?P<row>[1-9][0-9]*)")


def _positive_int(value: str | None, default: int = 1) -> int:
    """把不可信 ODF 计数属性解析为至少为一的整数。"""
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def _iter_rows(container: etree._Element, *, header: bool = False) -> Iterator[tuple[etree._Element, bool]]:
    """按 ODF 容器顺序递归产出普通行和表头行。"""
    for child in container:
        if not isinstance(child.tag, str):
            continue
        if child.tag == qname("table", "table-row"):
            yield child, header
        elif child.tag == qname("table", "table-header-rows"):
            yield from _iter_rows(child, header=True)
        elif child.tag in {qname("table", "table-rows"), qname("table", "table-row-group")}:
            yield from _iter_rows(child, header=header)


def _typed_value_text(cell: etree._Element) -> str:
    """在单元格无显示段落时把 ODF typed cached value 转为文本。"""
    value_type = cell.get(qname("office", "value-type"), "")
    if value_type == "percentage":
        try:
            return f"{float(cell.get(qname('office', 'value'), '0')) * 100:g}%"
        except ValueError:
            return ""
    if value_type == "currency":
        value = cell.get(qname("office", "value"), "")
        currency = cell.get(qname("office", "currency"), "")
        return f"{value} {currency}".strip()
    if value_type == "float":
        value = cell.get(qname("office", "value"), "")
        try:
            return f"{float(value):g}"
        except ValueError:
            return value
    if value_type == "date":
        return cell.get(qname("office", "date-value"), "")
    if value_type == "time":
        return cell.get(qname("office", "time-value"), "")
    if value_type == "boolean":
        return "TRUE" if cell.get(qname("office", "boolean-value"), "false").casefold() == "true" else "FALSE"
    if value_type == "string":
        return cell.get(qname("office", "string-value"), "")
    return ""


def _has_visible_html(value: str) -> bool:
    """判断单元格 HTML 是否包含非空文本或图片/公式结构。"""
    if any(token in value.casefold() for token in ("<img", "<eq", "<table", "<ul", "<ol")):
        return True
    return bool(html.unescape(_HTML_TEXT_RE.sub("", value)).strip())


def _validate_grid_extent(row_count: int, width: int) -> None:
    """在分配或遍历前校验预计矩形不会超过共享网格预算。"""
    projected_width = max(width, 1)
    if row_count > MAX_GRID_SLOTS // projected_width:
        raise OdfResourceLimitError(f"ODF resource limit exceeded: max_grid_slots={MAX_GRID_SLOTS}")


def _ensure_row(grid: TableGrid, row_index: int, width: int = 0) -> list[GridCell | None]:
    """确保网格存在指定行和最小列宽。"""
    _validate_grid_extent(max(len(grid.rows), row_index + 1), max(grid.width, width))
    while len(grid.rows) <= row_index:
        grid.rows.append([])
    row = grid.rows[row_index]
    if len(row) < width:
        row.extend([None] * (width - len(row)))
    return row


def _charge_grid(grid: TableGrid) -> None:
    """按当前矩形边界检查最大网格槽位。"""
    _validate_grid_extent(len(grid.rows), grid.width)


def parse_table_grid(table: etree._Element, render_cell: CellRenderer) -> TableGrid:
    """展开受限重复行列与合并单元格，构造规范二维网格。"""
    grid = TableGrid()
    duplicated_text_bytes = 0
    row_index = 0
    pending_empty_rows = 0
    pending_empty_width = 0
    for row_element, header in _iter_rows(table):
        row_repeat = _positive_int(row_element.get(qname("table", "number-rows-repeated")))
        _validate_grid_extent(row_index + row_repeat, grid.width)
        cell_templates: list[tuple[bool, int, GridCell | None]] = []
        for cell in row_element:
            if not isinstance(cell.tag, str):
                continue
            if cell.tag == qname("table", "covered-table-cell"):
                cell_templates.append((True, _positive_int(cell.get(qname("table", "number-columns-repeated"))), None))
                continue
            if cell.tag != qname("table", "table-cell"):
                continue
            column_repeat = _positive_int(cell.get(qname("table", "number-columns-repeated")))
            row_span = _positive_int(cell.get(qname("table", "number-rows-spanned")))
            col_span = _positive_int(cell.get(qname("table", "number-columns-spanned")))
            _validate_grid_extent(row_span, col_span)
            cell_html = render_cell(cell)
            if not _has_visible_html(cell_html):
                typed_value = _typed_value_text(cell)
                cell_html = html.escape(typed_value) if typed_value else ""
            duplicated_text_bytes += len(cell_html.encode("utf-8")) * max(0, row_repeat * column_repeat - 1)
            if duplicated_text_bytes > MAX_EXPANSION_TEXT_BYTES:
                raise OdfResourceLimitError(f"ODF resource limit exceeded: max_expansion_text_bytes={MAX_EXPANSION_TEXT_BYTES}")
            cell_templates.append(
                (
                    False,
                    column_repeat,
                    GridCell(html=cell_html, row_span=row_span, col_span=col_span, header=header),
                )
            )
        row_has_content = header or any(
            template is not None and (template.has_content or template.row_span > 1 or template.col_span > 1)
            for _, _, template in cell_templates
        )
        if not row_has_content:
            pending_empty_rows += row_repeat
            pending_empty_width = max(
                pending_empty_width,
                sum(repeat * (template.col_span if template is not None else 1) for _, repeat, template in cell_templates),
            )
            continue
        if pending_empty_rows:
            _validate_grid_extent(row_index + pending_empty_rows, max(grid.width, pending_empty_width))
            for _ in range(pending_empty_rows):
                _ensure_row(grid, row_index, pending_empty_width)
                row_index += 1
            pending_empty_rows = 0
            pending_empty_width = 0
        for _ in range(row_repeat):
            row = _ensure_row(grid, row_index)
            col_index = 0
            pending_empty_columns = 0
            for is_covered, repeat, template in cell_templates:
                if repeat > MAX_GRID_SLOTS:
                    raise OdfResourceLimitError(f"ODF resource limit exceeded: max_grid_slots={MAX_GRID_SLOTS}")
                if (
                    not is_covered
                    and template is not None
                    and not template.has_content
                    and template.row_span == 1
                    and template.col_span == 1
                ):
                    pending_empty_columns += repeat
                    continue
                if pending_empty_columns:
                    col_index += pending_empty_columns
                    _ensure_row(grid, row_index, col_index)
                    pending_empty_columns = 0
                _validate_grid_extent(row_index + 1, max(grid.width, col_index + repeat))
                for _ in range(repeat):
                    if is_covered:
                        _ensure_row(grid, row_index, col_index + 1)
                        grid.covered.add((row_index, col_index))
                        col_index += 1
                        continue
                    while (row_index, col_index) in grid.covered:
                        col_index += 1
                    assert template is not None
                    placed = GridCell(
                        html=template.html,
                        row_span=template.row_span,
                        col_span=template.col_span,
                        header=template.header,
                    )
                    _validate_grid_extent(
                        row_index + placed.row_span,
                        max(grid.width, col_index + placed.col_span),
                    )
                    row = _ensure_row(grid, row_index, col_index + placed.col_span)
                    row[col_index] = placed
                    for row_offset in range(placed.row_span):
                        covered_row = _ensure_row(grid, row_index + row_offset, col_index + placed.col_span)
                        for col_offset in range(placed.col_span):
                            if row_offset == 0 and col_offset == 0:
                                continue
                            grid.covered.add((row_index + row_offset, col_index + col_offset))
                            if covered_row[col_index + col_offset] is not None:
                                covered_row[col_index + col_offset] = None
                    col_index += placed.col_span
            if header:
                grid.header_rows = max(grid.header_rows, row_index + 1)
            _charge_grid(grid)
            row_index += 1
    return trim_table_grid(grid)


def trim_table_grid(grid: TableGrid) -> TableGrid:
    """移除尾部全空行列，同时保留已用范围内部的空白坐标。"""
    last_row = -1
    last_col = -1
    for row_index, row in enumerate(grid.rows):
        for col_index, cell in enumerate(row):
            if cell is not None and (cell.has_content or cell.row_span > 1 or cell.col_span > 1):
                last_row = max(last_row, row_index + cell.row_span - 1)
                last_col = max(last_col, col_index + cell.col_span - 1)
    if last_row < 0 or last_col < 0:
        return TableGrid()
    rows = []
    for row_index in range(min(last_row + 1, len(grid.rows))):
        row = list(grid.rows[row_index][: last_col + 1])
        if len(row) < last_col + 1:
            row.extend([None] * (last_col + 1 - len(row)))
        rows.append(row)
    covered = {(row, col) for row, col in grid.covered if row <= last_row and col <= last_col}
    return TableGrid(rows=rows, header_rows=min(grid.header_rows, len(rows)), covered=covered)


def crop_table_grid(grid: TableGrid, bounds: tuple[int, int, int, int]) -> TableGrid:
    """按闭区间行列边界裁剪网格并重映射合并占位。"""
    row_start, row_end, col_start, col_end = bounds
    if row_start < 0 or col_start < 0 or row_end < row_start or col_end < col_start:
        return TableGrid()
    rows: list[list[GridCell | None]] = []
    for source_row in range(row_start, min(row_end + 1, len(grid.rows))):
        row = grid.rows[source_row]
        selected = list(row[col_start : col_end + 1])
        if len(selected) < col_end - col_start + 1:
            selected.extend([None] * (col_end - col_start + 1 - len(selected)))
        rows.append(selected)
    covered = {
        (row - row_start, col - col_start)
        for row, col in grid.covered
        if row_start <= row <= row_end and col_start <= col <= col_end
    }
    header_rows = max(0, min(grid.header_rows - row_start, len(rows)))
    return trim_table_grid(TableGrid(rows=rows, header_rows=header_rows, covered=covered))


def split_table_regions(grid: TableGrid) -> list[TableGrid]:
    """按至少两条全空行列分隔电子表格中的离散数据区域。"""
    nonempty = [
        (row_index, col_index)
        for row_index, row in enumerate(grid.rows)
        for col_index, cell in enumerate(row)
        if cell is not None and cell.has_content
    ]
    if not nonempty:
        return []
    row_values = sorted({row for row, _ in nonempty})
    row_bands: list[tuple[int, int]] = []
    start = previous = row_values[0]
    for current in row_values[1:]:
        if current - previous > 2:
            row_bands.append((start, previous))
            start = current
        previous = current
    row_bands.append((start, previous))
    result: list[TableGrid] = []
    for row_start, row_end in row_bands:
        cols = sorted({col for row, col in nonempty if row_start <= row <= row_end})
        col_start = col_previous = cols[0]
        for current in cols[1:] + [cols[-1] + 3]:
            if current - col_previous > 2:
                region = crop_table_grid(grid, (row_start, row_end, col_start, col_previous))
                if region.rows:
                    result.append(region)
                col_start = current
            col_previous = current
    return result


def _render_html_row(grid: TableGrid, row_index: int, *, header: bool) -> str:
    """把网格中的一行序列化为 tr，并跳过合并占位。"""
    row = grid.rows[row_index]
    tag = "th" if header else "td"
    parts = ["<tr>"]
    for col_index in range(grid.width):
        if (row_index, col_index) in grid.covered:
            continue
        cell = row[col_index] if col_index < len(row) else None
        attrs = ""
        content = ""
        if cell is not None:
            if cell.row_span > 1:
                attrs += f' rowspan="{cell.row_span}"'
            if cell.col_span > 1:
                attrs += f' colspan="{cell.col_span}"'
            content = cell.html
        parts.append(f"<{tag}{attrs}>{content}</{tag}>")
    parts.append("</tr>")
    return "".join(parts)


def table_grid_to_html(grid: TableGrid) -> str:
    """把规范网格稳定序列化为带 thead/tbody 和跨度的 HTML 表格。"""
    if not grid.rows:
        return ""
    header_rows = min(grid.header_rows, len(grid.rows))
    parts = ["<table>"]
    if header_rows:
        parts.append("<thead>")
        parts.extend(_render_html_row(grid, row_index, header=True) for row_index in range(header_rows))
        parts.append("</thead>")
    if header_rows < len(grid.rows):
        parts.append("<tbody>")
        parts.extend(_render_html_row(grid, row_index, header=False) for row_index in range(header_rows, len(grid.rows)))
        parts.append("</tbody>")
    parts.append("</table>")
    return "".join(parts)


def _column_index(label: str) -> int:
    """把 A1 地址中的列字母转换为零基列号。"""
    result = 0
    for char in label.upper():
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def parse_cell_range_bounds(address: str) -> tuple[int, int, int, int] | None:
    """从 ODF cell-range-address 中提取零基闭区间边界。"""
    matches = list(_CELL_ADDRESS_RE.finditer(address or ""))
    if not matches:
        return None
    first = matches[0]
    last = matches[-1]
    row_start = int(first.group("row")) - 1
    col_start = _column_index(first.group("col"))
    row_end = int(last.group("row")) - 1
    col_end = _column_index(last.group("col"))
    return min(row_start, row_end), max(row_start, row_end), min(col_start, col_end), max(col_start, col_end)


def union_bounds(bounds: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    """返回多个表格范围的最小包围矩形。"""
    if not bounds:
        return None
    return (
        min(item[0] for item in bounds),
        max(item[1] for item in bounds),
        min(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


__all__ = [
    "crop_table_grid",
    "parse_cell_range_bounds",
    "parse_table_grid",
    "split_table_regions",
    "table_grid_to_html",
    "trim_table_grid",
    "union_bounds",
]
