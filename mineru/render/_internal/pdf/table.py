# Copyright (c) Opendatalab. All rights reserved.
"""HTML table 到可分页 ReportLab 原生表格的安全物化。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, TypeAlias

from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import ValidationError
from reportlab.platypus import Flowable, LongTable, Paragraph, Table, TableStyle

from ....types import CodeInlineSpan, EquationInlineSpan, HyperlinkSpan, InlineSpan, InlineStyle, TextSpan, parse_inline_spans
from ....utils.hyperlink import OFFICE_EXTERNAL_HYPERLINK_SCHEMES, sanitize_hyperlink_target
from .styles import BORDER_COLOR, SURFACE_COLOR, PdfStyleSet

MAX_NESTED_TABLE_DEPTH = 4
MAX_TABLE_ROWS = 500
MAX_TABLE_COLUMNS = 100
MAX_TABLE_SLOTS = 10_000
_POSITIVE_INTEGER_RE = re.compile(r"[0-9]+")
_BLOCK_TAGS = {"address", "article", "blockquote", "div", "figcaption", "footer", "header", "li", "p", "section"}
_SKIPPED_TAGS = {"script", "style", "template", "noscript"}

HtmlTableSource: TypeAlias = str | BeautifulSoup | Tag


class PdfTableError(ValueError):
    """表示 HTML 表格结构或 PDF 表格几何无法安全物化。"""


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


class ParagraphBuilder(Protocol):
    """定义表格单元格创建富文本 Paragraph 的回调。"""

    def __call__(self, spans: list[InlineSpan], style: object, max_width: float) -> Paragraph:
        """把单元格行内 span 构造成指定宽度的 Paragraph。"""
        ...


class HtmlImageBuilder(Protocol):
    """定义表格单元格创建离线图片或占位 Flowable 的回调。"""

    def __call__(self, source: str, max_width: float, alt_text: str) -> Flowable:
        """把 HTML img source 转换为图片或宽松占位。"""
        ...


def parse_html_tables(source: HtmlTableSource) -> tuple[HtmlTableGrid, ...]:
    """解析 source 中相对当前上下文的一个或多个顶层 table。"""
    root = BeautifulSoup(source, "html.parser") if isinstance(source, str) else source
    if not isinstance(root, (BeautifulSoup, Tag)):
        raise PdfTableError("HTML table source must be a string or BeautifulSoup Tag")
    if isinstance(root, Tag) and root.name == "table":
        table_tags = (root,)
    else:
        parent_table = root.find_parent("table") if isinstance(root, Tag) else None
        table_tags = tuple(table for table in root.find_all("table") if table.find_parent("table") is parent_table)
    if not table_tags:
        raise PdfTableError("HTML does not contain a top-level table")
    return tuple(_parse_html_table(table) for table in table_tags)


def build_pdf_tables(
    source: HtmlTableSource,
    *,
    available_width: float,
    styles: PdfStyleSet,
    build_paragraph: ParagraphBuilder,
    build_image: HtmlImageBuilder,
    depth: int = 1,
) -> tuple[Table, ...]:
    """把 HTML 表格递归转换为支持合并单元格与重复表头的 PDF 表格。"""
    if depth > MAX_NESTED_TABLE_DEPTH:
        raise PdfTableError(f"Nested table depth exceeds {MAX_NESTED_TABLE_DEPTH}")
    if available_width <= 0:
        raise PdfTableError("available_width must be positive")
    grids = parse_html_tables(source)
    return tuple(
        _build_pdf_table(
            grid,
            available_width=available_width,
            styles=styles,
            build_paragraph=build_paragraph,
            build_image=build_image,
            depth=depth,
        )
        for grid in grids
    )


def _parse_html_table(table: Tag) -> HtmlTableGrid:
    """把单个 table 标签解析为严格矩形占位网格。"""
    if table.name != "table":
        raise PdfTableError("Expected a <table> tag")
    rows = tuple(row for row in table.find_all("tr") if row.find_parent("table") is table)
    if not rows or len(rows) > MAX_TABLE_ROWS:
        raise PdfTableError(f"Table row count must be between 1 and {MAX_TABLE_ROWS}")
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
                raise PdfTableError(f"rowspan exceeds table bounds at row={row_index}, column={column_index}")
            if column_index + colspan > MAX_TABLE_COLUMNS:
                raise PdfTableError(f"Table column count exceeds {MAX_TABLE_COLUMNS}")
            coordinates = tuple(
                (target_row, target_column)
                for target_row in range(row_index, row_index + rowspan)
                for target_column in range(column_index, column_index + colspan)
            )
            overlap = next((coordinate for coordinate in coordinates if coordinate in occupied), None)
            if overlap is not None:
                raise PdfTableError(f"Cell span overlaps row={overlap[0]}, column={overlap[1]}")
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
                raise PdfTableError(f"Table occupancy exceeds {MAX_TABLE_SLOTS} slots")
            column_index += colspan
    if not occupied:
        raise PdfTableError("Table must contain at least one cell")
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
        raise PdfTableError(f"Table occupancy is not rectangular at row={missing[0]}, column={missing[1]}")
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


def _build_pdf_table(
    grid: HtmlTableGrid,
    *,
    available_width: float,
    styles: PdfStyleSet,
    build_paragraph: ParagraphBuilder,
    build_image: HtmlImageBuilder,
    depth: int,
) -> Table:
    """物化一个网格，写入单元格内容、合并区域与固定打印样式。"""
    column_widths = _column_widths(available_width, grid.column_count)
    data: list[list[object]] = [["" for _ in range(grid.column_count)] for _ in range(grid.row_count)]
    commands: list[tuple[object, ...]] = [
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in grid.header_rows:
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), SURFACE_COLOR))
    for placement in grid.cells:
        cell_width = sum(column_widths[placement.column : placement.column + placement.colspan]) - 10
        style = styles.table_header if placement.is_header else styles.table_cell
        data[placement.row][placement.column] = _cell_flowables(
            placement.tag,
            max_width=max(1.0, cell_width),
            style=style,
            styles=styles,
            build_paragraph=build_paragraph,
            build_image=build_image,
            depth=depth,
        )
        if placement.rowspan > 1 or placement.colspan > 1:
            commands.append(
                (
                    "SPAN",
                    (placement.column, placement.row),
                    (placement.end_column, placement.end_row),
                )
            )
    repeat_rows = 0
    while repeat_rows in grid.header_rows:
        repeat_rows += 1
    table_class = LongTable if depth == 1 else Table
    table = table_class(
        data,
        colWidths=column_widths,
        repeatRows=repeat_rows,
        splitByRow=1,
        splitInRow=1,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle(commands))
    return table


def _cell_flowables(
    cell: Tag,
    *,
    max_width: float,
    style: object,
    styles: PdfStyleSet,
    build_paragraph: ParagraphBuilder,
    build_image: HtmlImageBuilder,
    depth: int,
) -> list[Flowable]:
    """把单元格文本、图片与直接嵌套表按安全顺序转换为 Flowable。"""
    flowables: list[Flowable] = []
    spans = _html_cell_spans(cell)
    if spans:
        flowables.append(build_paragraph(spans, style, max_width))
    for image in cell.find_all("img"):
        if image.find_parent("table") is not cell.find_parent("table"):
            continue
        source = image.get("src", "")
        if isinstance(source, list):
            source = ""
        alt = image.get("alt", "image")
        if isinstance(alt, list):
            alt = " ".join(str(item) for item in alt)
        flowables.append(build_image(str(source), max_width, str(alt)))
    for nested in cell.find_all("table"):
        if nested.find_parent(("td", "th")) is not cell:
            continue
        flowables.extend(
            build_pdf_tables(
                nested,
                available_width=max_width,
                styles=styles,
                build_paragraph=build_paragraph,
                build_image=build_image,
                depth=depth + 1,
            )
        )
    if not flowables:
        flowables.append(build_paragraph([TextSpan(type="text", content=" ")], style, max_width))
    return flowables


def _html_cell_spans(cell: Tag) -> list[InlineSpan]:
    """把单元格中非图片、非嵌套表内容转换为严格 InlineSpan。"""
    spans: list[InlineSpan] = []
    for child in cell.children:
        spans.extend(_html_node_spans(child, styles=(), allow_links=True))
    if not spans:
        return []
    try:
        return parse_inline_spans(spans)
    except (TypeError, ValidationError, ValueError) as exc:
        raise PdfTableError("HTML table cell inline content is invalid") from exc


def _html_node_spans(node: object, *, styles: tuple[InlineStyle, ...], allow_links: bool) -> list[InlineSpan]:
    """递归解析安全 HTML 富文本标签，忽略活动内容与独立视觉节点。"""
    if isinstance(node, NavigableString):
        text = str(node)
        return [TextSpan(type="text", content=text, styles=list(styles))] if text else []
    if not isinstance(node, Tag):
        return []
    name = (node.name or "").lower()
    if name in _SKIPPED_TAGS or name in {"img", "table"}:
        return []
    if name == "br":
        return [TextSpan(type="text", content="\n", styles=list(styles))]
    if name == "eq":
        content = node.get_text()
        return [EquationInlineSpan(type="equation_inline", content=content)] if content.strip() else []
    if name == "code":
        content = node.get_text()
        return [CodeInlineSpan(type="code_inline", content=content)] if content else []
    style_name: InlineStyle | None = {
        "b": "bold",
        "strong": "bold",
        "i": "italic",
        "em": "italic",
        "u": "underline",
        "s": "strikethrough",
        "del": "strikethrough",
        "sup": "superscript",
        "sub": "subscript",
    }.get(name)
    child_styles = tuple(dict.fromkeys((*styles, style_name))) if style_name is not None else styles
    if name == "a" and allow_links:
        children = [
            span
            for child in node.children
            for span in _html_node_spans(child, styles=child_styles, allow_links=False)
            if not isinstance(span, HyperlinkSpan)
        ]
        target = sanitize_hyperlink_target(
            node.get("href"),
            allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
            allow_relative=True,
            allow_fragment=True,
        )
        if target is not None and children:
            try:
                return [HyperlinkSpan(type="hyperlink", url=target, content=children)]  # type: ignore[arg-type]
            except ValidationError:
                pass
        return children
    spans = [span for child in node.children for span in _html_node_spans(child, styles=child_styles, allow_links=allow_links)]
    if name in _BLOCK_TAGS and spans and not _spans_end_with_newline(spans):
        spans.append(TextSpan(type="text", content="\n"))
    return spans


def _spans_end_with_newline(spans: list[InlineSpan]) -> bool:
    """判断当前 span 序列是否已经以普通文本换行结束。"""
    return bool(spans and isinstance(spans[-1], TextSpan) and spans[-1].content.endswith("\n"))


def _parse_span(cell: Tag, attribute: str) -> int:
    """读取严格正整数 rowspan/colspan，缺失时返回一。"""
    raw_value = cell.get(attribute, "1")
    if isinstance(raw_value, list):
        raise PdfTableError(f"Invalid {attribute}: {raw_value!r}")
    value = str(raw_value).strip()
    if _POSITIVE_INTEGER_RE.fullmatch(value) is None:
        raise PdfTableError(f"Invalid {attribute}: {raw_value!r}")
    span = int(value)
    if span < 1 or span > MAX_TABLE_SLOTS:
        raise PdfTableError(f"Invalid {attribute}: {raw_value!r}")
    return span


def _row_belongs_to_thead(row: Tag, table: Tag) -> bool:
    """判断 tr 是否位于当前 table 的 thead 内。"""
    parent = row.parent
    while isinstance(parent, Tag) and parent is not table:
        if parent.name == "thead":
            return True
        parent = parent.parent
    return False


def _column_widths(total_width: float, column_count: int) -> list[float]:
    """把可用宽度确定性地均分到全部逻辑列。"""
    if column_count <= 0:
        raise PdfTableError("Table must contain at least one column")
    base = total_width / column_count
    return [base for _ in range(column_count)]


__all__ = [
    "HtmlTableCell",
    "HtmlTableGrid",
    "MAX_NESTED_TABLE_DEPTH",
    "PdfTableError",
    "build_pdf_tables",
    "parse_html_tables",
]
