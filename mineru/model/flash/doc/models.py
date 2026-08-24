# Copyright (c) Opendatalab. All rights reserved.

"""Word 二进制解析器与 Converter 之间的内部语义模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DocCharStyle:
    """一个连续 DOC 文字 run 的可见字符样式。"""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    emphasis: bool = False
    strike: bool = False
    superscript: bool = False
    subscript: bool = False
    hidden: bool = False
    deleted: bool = False


@dataclass(frozen=True, slots=True)
class DocTextRun:
    """一段样式及超链接目标相同的可见文字。"""

    text: str
    style: DocCharStyle = DocCharStyle()
    hyperlink: str | None = None
    formula: bool = False


@dataclass(frozen=True, slots=True)
class DocListInfo:
    """段落从 PlfLst/PlfLfo 解析出的列表信息。"""

    identity: int
    level: int
    ordered: bool
    start: int = 1
    label: str | None = None


@dataclass(frozen=True, slots=True)
class DocTableCellFormat:
    """一格 Word 表格在行定义中的合并属性。"""

    right: int
    horizontal_first: bool = False
    horizontal_continue: bool = False
    vertical_first: bool = False
    vertical_continue: bool = False


@dataclass(frozen=True, slots=True)
class DocTableFormat:
    """一个 TTP 段落解析出的表格行定义。"""

    boundaries: tuple[int, ...] = ()
    cells: tuple[DocTableCellFormat, ...] = ()
    header: bool = False


@dataclass(slots=True)
class DocParagraph:
    """完成样式、字段和段落属性解析的段落。"""

    cp_start: int
    cp_end: int
    runs: list[DocTextRun] = field(default_factory=list)
    images: list[DocImagePayload] = field(default_factory=list)
    style_name: str = ""
    heading_level: int | None = None
    is_title: bool = False
    is_toc: bool = False
    toc_level: int | None = None
    is_caption: bool = False
    is_code: bool = False
    anchor: str | None = None
    list_info: DocListInfo | None = None
    in_table: bool = False
    table_depth: int = 0
    cell_mark: bool = False
    row_mark: bool = False
    table_format: DocTableFormat | None = None


@dataclass(frozen=True, slots=True)
class DocImagePayload:
    """一张从 PICF/OfficeArt 恢复的图片原始载荷。"""

    data: bytes
    extension: str
    content_type: str


@dataclass(slots=True)
class DocImage:
    """按主 story CP 定位的内联或浮动图片。"""

    cp: int
    payload: DocImagePayload


@dataclass(slots=True)
class DocTableCell:
    """Word 表格单元格及其嵌套内容。"""

    blocks: list[DocElement] = field(default_factory=list)
    row_span: int = 1
    col_span: int = 1


@dataclass(slots=True)
class DocTableRow:
    """Word 表格的一行。"""

    cells: list[DocTableCell] = field(default_factory=list)
    header: bool = False


@dataclass(slots=True)
class DocTable:
    """按 CP 顺序组装出的普通或嵌套表格。"""

    cp_start: int
    cp_end: int
    rows: list[DocTableRow] = field(default_factory=list)


DocElement: TypeAlias = DocParagraph | DocImage | DocTable


@dataclass(slots=True)
class DocSection:
    """一个 Word section 及其页面辅助文本。"""

    cp_start: int
    cp_end: int
    elements: list[DocElement] = field(default_factory=list)
    headers: list[DocParagraph] = field(default_factory=list)
    footers: list[DocParagraph] = field(default_factory=list)
    footnotes: list[DocParagraph] = field(default_factory=list)


@dataclass(slots=True)
class DocDocument:
    """一份可投影为 model-list 的 DOC 文档。"""

    sections: list[DocSection] = field(default_factory=list)
