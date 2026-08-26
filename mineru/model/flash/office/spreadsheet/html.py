# Copyright (c) Opendatalab. All rights reserved.
"""把完整工作表表格 IR 确定性渲染为 HTML。"""

from __future__ import annotations

import html
import re

from .models import ExcelTable

EQUATION_BOOKENDS = "<eq>{EQ}</eq>"


def _contains_block_level_html(content: str) -> bool:
    """判断单元格内容是否已经包含块级 HTML。"""
    return bool(
        re.search(
            r"<\s*(p|ul|ol|li|div|table|blockquote|pre|h[1-6])\b",
            content,
            re.IGNORECASE,
        )
    )


def _render_cell_inner_html(content: str, is_html: bool) -> str:
    """为普通或行内 HTML 内容补充稳定的段落容器。"""
    if not content:
        return "<p></p>"
    if is_html and _contains_block_level_html(content):
        return content
    return f"<p>{content}</p>"


def render_spreadsheet_table(
    excel_table: ExcelTable,
    *,
    equation_bookends: str = EQUATION_BOOKENDS,
) -> str:
    """渲染表头、合并格、媒体和公式均已物化的工作表表格。"""
    cell_map = {(cell.row, cell.col): cell for cell in excel_table.data}
    covered_cells: set[tuple[int, int]] = set()
    lines = ["<table>"]

    for row in range(excel_table.num_rows):
        lines.append("  <tr>")
        for col in range(excel_table.num_cols):
            if (row, col) in covered_cells:
                continue

            cell = cell_map.get((row, col))
            if cell is None:
                lines.append("    <td></td>")
                continue

            tag = "th" if cell.row == 0 else "td"
            attrs = []
            if cell.row_span > 1:
                attrs.append(f'rowspan="{cell.row_span}"')
            if cell.col_span > 1:
                attrs.append(f'colspan="{cell.col_span}"')
            for row_offset in range(cell.row_span):
                for col_offset in range(cell.col_span):
                    covered_cells.add((row + row_offset, col + col_offset))
            attr_str = " " + " ".join(attrs) if attrs else ""

            text_content = cell.text if cell.text_is_html else html.escape(cell.text)
            if cell.media:
                media_content = "<br>".join(cell.media)
                text_content = f"{text_content}<br>{media_content}" if text_content else media_content
            for formula in cell.equations:
                text_content += equation_bookends.format(EQ=formula)

            inner_html = _render_cell_inner_html(text_content, cell.text_is_html)
            lines.append(f"    <{tag}{attr_str}>{inner_html}</{tag}>")

        lines.append("  </tr>")

    lines.append("</table>")
    return "\n".join(lines)


__all__ = ["EQUATION_BOOKENDS", "render_spreadsheet_table"]
