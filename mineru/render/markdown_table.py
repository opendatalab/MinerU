from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeVar

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

_ALLOWED_INLINE_TAGS = {
    "a",
    "b",
    "br",
    "code",
    "em",
    "eq",
    "i",
    "s",
    "span",
    "strong",
    "sub",
    "sup",
    "u",
}

_COMPLEX_BLOCK_TAGS = {
    "blockquote",
    "div",
    "dl",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "img",
    "li",
    "ol",
    "p",
    "pre",
    "table",
    "ul",
}

_MatrixCell = TypeVar("_MatrixCell")


@dataclass(frozen=True)
class _CellPlacement:
    text: str
    rowspan: int
    colspan: int
    is_header: bool


def to_markdown_table(
    html: str,
    *,
    max_empty_ratio: float = 0.3,
    max_complex_span_count: int = 5,
) -> str:
    """Convert a simple HTML table to a Markdown table.

    Falls back to the original HTML string when the table structure or cell
    content is too complex to render readably as Markdown.
    """
    raw_html = html.strip()
    if not raw_html:
        return raw_html

    soup = BeautifulSoup(raw_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return raw_html
    if _has_nested_table(table):
        return raw_html

    rows = table.find_all("tr")
    if not rows:
        return raw_html

    grid: list[list[str | None]] = []
    header_flags: list[list[bool]] = []
    span_count = 0
    has_rowspan = False
    has_colspan = False

    for row_idx, row in enumerate(rows):
        _ensure_row(grid, row_idx)
        _ensure_row(header_flags, row_idx)

        col_idx = 0
        cells = row.find_all(("th", "td"), recursive=False)
        for cell in cells:
            while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                col_idx += 1

            placement = _parse_cell(cell)
            if placement is None:
                return raw_html

            rowspan = placement.rowspan
            colspan = placement.colspan
            if rowspan > 1:
                has_rowspan = True
                span_count += rowspan - 1
            if colspan > 1:
                has_colspan = True
                span_count += colspan - 1

            for offset in range(colspan):
                _ensure_width(grid, row_idx, col_idx + offset + 1, fill_value=None)
                _ensure_width(header_flags, row_idx, col_idx + offset + 1, fill_value=False)
                grid[row_idx][col_idx + offset] = placement.text if offset == 0 else ""
                header_flags[row_idx][col_idx + offset] = placement.is_header

            for row_offset in range(1, rowspan):
                next_row_idx = row_idx + row_offset
                _ensure_row(grid, next_row_idx)
                _ensure_row(header_flags, next_row_idx)
                _ensure_width(grid, next_row_idx, col_idx + colspan, fill_value=None)
                _ensure_width(header_flags, next_row_idx, col_idx + colspan, fill_value=False)
                for offset in range(colspan):
                    grid[next_row_idx][col_idx + offset] = ""
                    header_flags[next_row_idx][col_idx + offset] = False

            col_idx += colspan

    width = max(len(row) for row in grid)
    normalized_rows: list[list[str]] = []
    normalized_headers: list[list[bool]] = []
    for row, flags in zip(grid, header_flags, strict=False):
        normalized_row = [(cell if cell is not None else "") for cell in row]
        normalized_row.extend([""] * (width - len(normalized_row)))
        normalized_rows.append(normalized_row)

        normalized_flag_row = list(flags)
        normalized_flag_row.extend([False] * (width - len(normalized_flag_row)))
        normalized_headers.append(normalized_flag_row)

    if not normalized_rows:
        return raw_html

    empty_ratio = _empty_ratio(normalized_rows)
    if has_rowspan and has_colspan and (span_count > max_complex_span_count or empty_ratio > max_empty_ratio):
        return raw_html

    header_index = _detect_header_row(table, normalized_headers)
    header = normalized_rows[header_index]
    body_rows = normalized_rows[:header_index] + normalized_rows[header_index + 1 :]

    markdown_lines = [
        _format_markdown_row(header),
        _format_markdown_row(["---"] * width),
        *[_format_markdown_row(row) for row in body_rows],
    ]
    return "\n".join(markdown_lines)


def _has_nested_table(table: Tag) -> bool:
    return table.find("table") is not None


def _parse_cell(cell: Tag) -> _CellPlacement | None:
    if _contains_complex_content(cell):
        return None
    text = _render_inline_children(cell).strip()
    text = _normalize_cell_text(text)
    return _CellPlacement(
        text=text,
        rowspan=_parse_positive_int(cell.get("rowspan")),
        colspan=_parse_positive_int(cell.get("colspan")),
        is_header=cell.name == "th",
    )


def _contains_complex_content(cell: Tag) -> bool:
    for descendant in cell.descendants:
        if not isinstance(descendant, Tag):
            continue
        if descendant is cell:
            continue
        if descendant.name in _COMPLEX_BLOCK_TAGS:
            return True
        if descendant.name not in _ALLOWED_INLINE_TAGS:
            return True
    return False


def _render_inline_children(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(_escape_markdown_table_text(str(child)))
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "br":
            parts.append("<br>")
        elif child.name == "code":
            parts.append(f"`{_escape_markdown_table_text(_normalize_whitespace(child.get_text()))}`")
        elif child.name == "a":
            href = str(child.get("href", "")).strip()
            label = _escape_markdown_table_text(_normalize_whitespace(child.get_text()))
            parts.append(f"[{label}]({href})" if href else label)
        elif child.name in {"b", "strong"}:
            parts.append(f"**{_render_inline_children(child).strip()}**")
        elif child.name in {"em", "i"}:
            parts.append(f"*{_render_inline_children(child).strip()}*")
        elif child.name in {"s"}:
            parts.append(f"~~{_render_inline_children(child).strip()}~~")
        elif child.name == "u":
            parts.append(_render_inline_children(child))
        else:
            parts.append(_render_inline_children(child))
    return "".join(parts)


def _normalize_cell_text(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *<br> *", "<br>", text)
    return text.strip()


def _escape_markdown_table_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", r"\|")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_positive_int(value: object) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return 1
    return parsed if parsed > 0 else 1


def _ensure_row(matrix: list[list[_MatrixCell]], row_idx: int) -> None:
    while len(matrix) <= row_idx:
        matrix.append([])


def _ensure_width(
    matrix: list[list[_MatrixCell]],
    row_idx: int,
    width: int,
    *,
    fill_value: _MatrixCell,
) -> None:
    _ensure_row(matrix, row_idx)
    while len(matrix[row_idx]) < width:
        matrix[row_idx].append(fill_value)


def _empty_ratio(rows: list[list[str]]) -> float:
    total = sum(len(row) for row in rows)
    if total == 0:
        return 0.0
    empty = sum(1 for row in rows for cell in row if cell == "")
    return empty / total


def _detect_header_row(table: Tag, header_flags: list[list[bool]]) -> int:
    thead = table.find("thead")
    if thead is not None:
        thead_rows = thead.find_all("tr", recursive=False)
        if thead_rows:
            return 0
    for idx, flags in enumerate(header_flags):
        if flags and all(flags):
            return idx
    return 0


def _format_markdown_row(row: list[str]) -> str:
    return f"| {' | '.join(row)} |"


__all__ = ["to_markdown_table"]
