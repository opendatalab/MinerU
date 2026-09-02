# Copyright (c) Opendatalab. All rights reserved.
"""HTML table 到 TeX Live longtable/tabular 源码的安全物化。"""

from __future__ import annotations

from bs4 import NavigableString, Tag
from pydantic import ValidationError

from ....types import CodeInlineSpan, EquationInlineSpan, HyperlinkSpan, InlineSpan, InlineStyle, TextSpan, parse_inline_spans
from ....utils.hyperlink import OFFICE_EXTERNAL_HYPERLINK_SCHEMES, sanitize_hyperlink_target
from ..common.html_table import (
    HtmlTableCell,
    HtmlTableError,
    HtmlTableGrid,
    HtmlTableSource,
    MAX_NESTED_TABLE_DEPTH,
    parse_html_tables,
)
from .assets import resolve_html_image_path, tex_image_path
from .inline import LatexAnchorRegistry, escape_latex_text, escape_latex_url, render_inline_spans

_BLOCK_TAGS = {"address", "article", "blockquote", "div", "figcaption", "footer", "header", "li", "p", "section"}
_SKIPPED_TAGS = {"script", "style", "template", "noscript"}


class LatexTableError(HtmlTableError):
    """表示 HTML table 无法安全物化为 LaTeX 表格。"""


class LatexTableRenderer:
    """持有图片路径和 anchor 上下文的递归 LaTeX 表格 renderer。"""

    def __init__(self, *, asset_base_path: str, anchors: LatexAnchorRegistry) -> None:
        """保存表格单元格渲染所需的文档级上下文。"""
        self.asset_base_path = asset_base_path
        self.anchors = anchors

    def render(self, source: HtmlTableSource) -> str:
        """把一个或多个顶层 HTML table 转换为可分页 LaTeX 表格。"""
        return self._render_source(source, depth=1)

    def _render_source(self, source: HtmlTableSource, *, depth: int) -> str:
        """递归解析当前层级的 table，并执行深度上限校验。"""
        if depth > MAX_NESTED_TABLE_DEPTH:
            raise LatexTableError(f"Nested table depth exceeds {MAX_NESTED_TABLE_DEPTH}")
        try:
            grids = parse_html_tables(source)
        except HtmlTableError as exc:
            raise LatexTableError(str(exc)) from exc
        return "\n\n".join(self._render_grid(grid, depth=depth) for grid in grids)

    def _render_grid(self, grid: HtmlTableGrid, *, depth: int) -> str:
        """把严格占位网格转换为 longtable 或嵌套 tabular。"""
        environment = "longtable" if depth == 1 else "tabular"
        column_spec = self._column_spec(grid.column_count)
        occupied = {
            (row, column): placement
            for placement in grid.cells
            for row in range(placement.row, placement.end_row + 1)
            for column in range(placement.column, placement.end_column + 1)
        }
        lines = [rf"\begin{{{environment}}}{{{column_spec}}}", r"\hline"]
        for row_index in range(grid.row_count):
            segments: list[str] = []
            column_index = 0
            while column_index < grid.column_count:
                placement = occupied[(row_index, column_index)]
                if placement.column != column_index:
                    column_index += 1
                    continue
                if placement.row == row_index:
                    segment = self._render_cell(
                        placement.tag,
                        is_header=placement.is_header,
                        rowspan=placement.rowspan,
                        colspan=placement.colspan,
                        column_count=grid.column_count,
                        depth=depth,
                    )
                else:
                    segment = self._continued_rowspan_placeholder(placement.colspan)
                segments.append(segment)
                column_index += placement.colspan
            lines.append(" & ".join(segments) + r" \\")
            lines.extend(_row_rules(occupied, row_index=row_index, column_count=grid.column_count))
        lines.append(rf"\end{{{environment}}}")
        return "\n".join(lines)

    def _render_cell(
        self,
        cell: Tag,
        *,
        is_header: bool,
        rowspan: int,
        colspan: int,
        column_count: int,
        depth: int,
    ) -> str:
        """渲染一个 origin cell，并组合 rowspan/colspan 声明。"""
        parts: list[str] = []
        spans = _html_cell_spans(cell)
        if spans:
            parts.append(render_inline_spans(spans, self.anchors))
        for image in cell.find_all("img"):
            if image.find_parent("table") is not cell.find_parent("table"):
                continue
            source = image.get("src", "")
            if isinstance(source, list):
                source = ""
            alt = image.get("alt", "image")
            if isinstance(alt, list):
                alt = " ".join(str(item) for item in alt)
            parts.append(self._render_image(str(source), str(alt)))
        for nested in cell.find_all("table"):
            if nested.find_parent(("td", "th")) is not cell:
                continue
            parts.append(self._render_source(nested, depth=depth + 1))
        content = r"\par ".join(part for part in parts if part) or "~"
        if is_header:
            content = rf"\cellcolor{{MinerUTableHeader}}\textbf{{{content}}}"
        width = _cell_width(colspan, column_count)
        if rowspan > 1:
            content = rf"\multirow{{{rowspan}}}{{*}}{{\parbox[t]{{{width}}}{{{content}}}}}"
        if colspan > 1:
            content = rf"\multicolumn{{{colspan}}}{{|p{{{width}}}|}}{{{content}}}"
        return content

    def _render_image(self, source: str, alt_text: str) -> str:
        """渲染单元格图片；非本地 sidecar 退化为可见链接或占位。"""
        path = resolve_html_image_path(source, self.asset_base_path)
        if path is not None:
            return rf"\includegraphics[width=.85\linewidth,height=.25\textheight,keepaspectratio]{{{tex_image_path(path)}}}"
        label = rf"\textit{{{escape_latex_text(f'image unavailable: {alt_text}')}}}"
        normalized = source.strip()
        if normalized.startswith(("http://", "https://")):
            return rf"\href{{{escape_latex_url(normalized)}}}{{{label}}}"
        return label

    @staticmethod
    def _column_spec(column_count: int) -> str:
        """按逻辑列数生成确定性的等宽段落列声明。"""
        if column_count <= 0:
            raise LatexTableError("Table must contain at least one column")
        width = _cell_width(1, column_count)
        column = rf">{{\raggedright\arraybackslash}}p{{{width}}}"
        return "|" + "|".join(column for _ in range(column_count)) + "|"

    @staticmethod
    def _continued_rowspan_placeholder(colspan: int) -> str:
        """为前序行延续下来的 rowspan 写入等列宽空占位。"""
        return rf"\multicolumn{{{colspan}}}{{|l|}}{{}}" if colspan > 1 else ""


def _cell_width(colspan: int, column_count: int) -> str:
    """按表格总列数计算当前单元格占用的 linewidth 比例。"""
    ratio = 0.94 * colspan / column_count
    return f"{ratio:.6f}\\linewidth"


def _row_rules(
    occupied: dict[tuple[int, int], HtmlTableCell],
    *,
    row_index: int,
    column_count: int,
) -> list[str]:
    """只在当前行结束的单元格列下画线，避免横线切穿 rowspan。"""
    ending_columns = [column for column in range(column_count) if occupied[(row_index, column)].end_row == row_index]
    if len(ending_columns) == column_count:
        return [r"\hline"]
    rules: list[str] = []
    range_start: int | None = None
    previous: int | None = None
    for column in ending_columns:
        if range_start is None:
            range_start = column
        elif previous is not None and column != previous + 1:
            rules.append(rf"\cline{{{range_start + 1}-{previous + 1}}}")
            range_start = column
        previous = column
    if range_start is not None and previous is not None:
        rules.append(rf"\cline{{{range_start + 1}-{previous + 1}}}")
    return rules


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
        raise LatexTableError("HTML table cell inline content is invalid") from exc


def _html_node_spans(node: object, *, styles: tuple[InlineStyle, ...], allow_links: bool) -> list[InlineSpan]:
    """递归解析表格单元格中的安全富文本标签。"""
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


__all__ = ["LatexTableError", "LatexTableRenderer"]
