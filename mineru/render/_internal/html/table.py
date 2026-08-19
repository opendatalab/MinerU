# Copyright (c) Opendatalab. All rights reserved.
"""HTML renderer 使用的严格 GFM pipe table 转换。"""

from __future__ import annotations

import re

from mineru.render._internal.html.inline import HtmlInlineResult, render_inline_content_html

_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def looks_like_gfm_table(content: str) -> bool:
    """判断文本是否具有 GFM pipe table 的表头与分隔行外形。"""
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if not _contains_unescaped_pipe(lines[0]) or not _contains_unescaped_pipe(lines[1]):
        return False
    separator_cells = _split_pipe_row(lines[1])
    return bool(separator_cells and all(_SEPARATOR_CELL_RE.fullmatch(cell.strip()) for cell in separator_cells))


def render_gfm_table_html(content: str) -> HtmlInlineResult | None:
    """把严格、等宽的简单 GFM pipe table 转换为语义 HTML。"""
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    if not _contains_unescaped_pipe(lines[0]) or not _contains_unescaped_pipe(lines[1]):
        return None
    header = _split_pipe_row(lines[0])
    separators = _split_pipe_row(lines[1])
    if not header or len(header) != len(separators):
        return None
    if not all(_SEPARATOR_CELL_RE.fullmatch(cell.strip()) for cell in separators):
        return None

    rows = [_split_pipe_row(line) for line in lines[2:]]
    if any(len(row) != len(header) for row in rows):
        return None

    alignments = [_separator_alignment(cell.strip()) for cell in separators]
    has_math = False

    def _render_row(cells: list[str], cell_tag: str) -> str:
        """渲染一行等宽单元格，并累积公式存在标记。"""
        nonlocal has_math
        rendered_cells: list[str] = []
        for index, cell in enumerate(cells):
            rendered = render_inline_content_html(_unescape_gfm_cell(cell.strip()))
            has_math = rendered.has_math or has_math
            alignment = alignments[index]
            class_attr = f' class="mineru-align-{alignment}"' if alignment else ""
            rendered_cells.append(f"<{cell_tag}{class_attr}>{rendered.html}</{cell_tag}>")
        return f"<tr>{''.join(rendered_cells)}</tr>"

    table = "".join(
        [
            '<table class="mineru-chart-table">',
            f"<thead>{_render_row(header, 'th')}</thead>",
            f"<tbody>{''.join(_render_row(row, 'td') for row in rows)}</tbody>",
            "</table>",
        ]
    )
    return HtmlInlineResult(table, has_math)


def _split_pipe_row(line: str) -> list[str]:
    """按未被奇数个反斜杠转义的竖线切分 GFM 表格行。"""
    normalized = line.strip()
    if normalized.startswith("|"):
        normalized = normalized[1:]
    if normalized.endswith("|") and not _is_escaped(normalized, len(normalized) - 1):
        normalized = normalized[:-1]

    cells: list[str] = []
    start = 0
    for index, char in enumerate(normalized):
        if char == "|" and not _is_escaped(normalized, index):
            cells.append(normalized[start:index])
            start = index + 1
    cells.append(normalized[start:])
    return cells


def _is_escaped(content: str, index: int) -> bool:
    """按 markdown-it table 规则判断 pipe 是否紧邻任意反斜杠。"""
    return index > 0 and content[index - 1] == "\\"


def _contains_unescaped_pipe(content: str) -> bool:
    """判断行内是否至少包含一个真正的 GFM 列分隔符。"""
    return any(char == "|" and not _is_escaped(content, index) for index, char in enumerate(content))


def _unescape_gfm_cell(content: str) -> str:
    """按既有 2n+1 编码逆向恢复 GFM 单元格中的原始反斜杠与竖线。"""

    def _replace(match: re.Match[str]) -> str:
        """把竖线前 2n+1 个 Markdown 反斜杠还原为 n 个。"""
        slash_count = len(match.group("slashes"))
        return "\\" * ((slash_count - 1) // 2) + "|"

    return re.sub(r"(?P<slashes>\\+)\|", _replace, content)


def _separator_alignment(cell: str) -> str | None:
    """从 GFM 分隔单元格解析 left、center 或 right 对齐。"""
    if cell.startswith(":") and cell.endswith(":"):
        return "center"
    if cell.endswith(":"):
        return "right"
    if cell.startswith(":"):
        return "left"
    return None


__all__ = ["looks_like_gfm_table", "render_gfm_table_html"]
