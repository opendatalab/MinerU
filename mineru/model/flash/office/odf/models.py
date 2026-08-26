# Copyright (c) Opendatalab. All rights reserved.
"""OpenDocument 内部行内、样式与表格模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, Union


@dataclass(frozen=True, slots=True)
class TextStyle:
    """保存可继承的 ODF 行内样式最终值。"""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    superscript: bool = False
    subscript: bool = False

    def names(self) -> tuple[str, ...]:
        """按 MinerU 内联协议的稳定顺序返回已启用样式名。"""
        result: list[str] = []
        if self.bold:
            result.append("bold")
        if self.italic:
            result.append("italic")
        if self.underline:
            result.append("underline")
        if self.strikethrough:
            result.append("strikethrough")
        if self.superscript:
            result.append("superscript")
        if self.subscript:
            result.append("subscript")
        return tuple(result)


@dataclass(frozen=True, slots=True)
class TextStyleDelta:
    """保存 ODF 样式层级中可显式覆盖的三态字段。"""

    bold: bool | None = None
    italic: bool | None = None
    underline: bool | None = None
    strikethrough: bool | None = None
    superscript: bool | None = None
    subscript: bool | None = None

    def merge(self, child: TextStyleDelta) -> TextStyleDelta:
        """用子样式的非空字段覆盖当前样式。"""
        return TextStyleDelta(
            bold=self.bold if child.bold is None else child.bold,
            italic=self.italic if child.italic is None else child.italic,
            underline=self.underline if child.underline is None else child.underline,
            strikethrough=self.strikethrough if child.strikethrough is None else child.strikethrough,
            superscript=self.superscript if child.superscript is None else child.superscript,
            subscript=self.subscript if child.subscript is None else child.subscript,
        )

    def resolve(self) -> TextStyle:
        """把未声明字段按关闭处理并返回最终样式。"""
        return TextStyle(
            bold=bool(self.bold),
            italic=bool(self.italic),
            underline=bool(self.underline),
            strikethrough=bool(self.strikethrough),
            superscript=bool(self.superscript),
            subscript=bool(self.subscript),
        )


@dataclass(frozen=True, slots=True)
class ParagraphProperties:
    """保存影响逻辑分页的 ODF 段落属性。"""

    break_before: bool = False
    break_after: bool = False
    master_page_name: str | None = None


@dataclass(frozen=True, slots=True)
class ListLevel:
    """保存一个 ODF 列表层级的编号语义。"""

    ordered: bool = False
    start: int = 1
    prefix: str = ""
    suffix: str = "."


@dataclass(frozen=True, slots=True)
class InlineText:
    """保存带样式和可选超链接的行内文本。"""

    text: str
    style: TextStyle = TextStyle()
    hyperlink: str | None = None


@dataclass(frozen=True, slots=True)
class InlineMath:
    """保存不含外围标记的行内 LaTeX。"""

    latex: str


@dataclass(frozen=True, slots=True)
class InlineBreak:
    """表示段内显式换行。"""


@dataclass(frozen=True, slots=True)
class InlinePageBreak:
    """表示 ODT 内容中的显式软分页。"""


@dataclass(frozen=True, slots=True)
class InlineImage:
    """保存表格单元格中允许内联呈现的图片 data URI。"""

    data_uri: str
    alt: str = ""


InlineAtom: TypeAlias = Union[InlineText, InlineMath, InlineBreak, InlinePageBreak, InlineImage]


@dataclass(slots=True)
class GridCell:
    """保存 ODF 表格原点单元格的 HTML 与跨度。"""

    html: str = ""
    row_span: int = 1
    col_span: int = 1
    header: bool = False

    @property
    def has_content(self) -> bool:
        """返回单元格是否包含可见或结构化 HTML。"""
        return bool(self.html.strip())


@dataclass(slots=True)
class TableGrid:
    """保存带合并占位的 ODF 二维表格。"""

    rows: list[list[GridCell | None]] = field(default_factory=list)
    header_rows: int = 0
    covered: set[tuple[int, int]] = field(default_factory=set)

    @property
    def width(self) -> int:
        """返回网格最大视觉列数。"""
        return max((len(row) for row in self.rows), default=0)


__all__ = [
    "GridCell",
    "InlineAtom",
    "InlineBreak",
    "InlineImage",
    "InlineMath",
    "InlinePageBreak",
    "InlineText",
    "ListLevel",
    "ParagraphProperties",
    "TableGrid",
    "TextStyle",
    "TextStyleDelta",
]
