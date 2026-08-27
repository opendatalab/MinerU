# Copyright (c) Opendatalab. All rights reserved.
"""RTF parser 与 MinerU raw-block converter 之间的显式语义模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, Union


@dataclass(frozen=True, slots=True)
class RtfTextStyle:
    """保存一个 RTF 字符 run 可投影到 Middle JSON 的样式。"""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    superscript: bool = False
    subscript: bool = False
    code: bool = False


@dataclass(frozen=True, slots=True)
class RtfTextRun:
    """保存已完成代码页解码的文本、样式和可选安全链接。"""

    text: str
    style: RtfTextStyle = RtfTextStyle()
    hyperlink: str | None = None


@dataclass(frozen=True, slots=True)
class RtfInlineEquation:
    """保存不含定界符的行内 LaTeX。"""

    latex: str


@dataclass(frozen=True, slots=True)
class RtfImage:
    """保存 RTF pict 载荷及可验证的来源信息。"""

    data: bytes
    content_type: str
    part_name: str
    alt: str = ""


@dataclass(frozen=True, slots=True)
class RtfNoteReference:
    """保存脚注或尾注引用的内部稳定 id。"""

    note_id: str


@dataclass(frozen=True, slots=True)
class RtfAnchor:
    """保存段落内书签锚点，converter 仅在标题上公开。"""

    name: str


@dataclass(frozen=True, slots=True)
class RtfLineBreak:
    """表示 RTF 行、列或显式分页控制携带的语义换行。"""


RtfInline: TypeAlias = Union[
    RtfTextRun,
    RtfInlineEquation,
    RtfImage,
    RtfNoteReference,
    RtfAnchor,
    RtfLineBreak,
]


@dataclass(frozen=True, slots=True)
class RtfListInfo:
    """保存一个列表段落的身份、层级、编号类型和精确标签。"""

    identity: int
    level: int
    ordered: bool
    marker: str = "decimal"
    start: int = 1
    label: str | None = None


@dataclass(slots=True)
class RtfParagraph:
    """保存一个 RTF 语义段落及其块级属性。"""

    inlines: list[RtfInline] = field(default_factory=list)
    style_name: str = ""
    outline_level: int | None = None
    is_title: bool = False
    block_style: Literal["normal", "code", "quote"] = "normal"
    list_info: RtfListInfo | None = None


@dataclass(slots=True)
class RtfDisplayEquation:
    """保存 RTF Office Math 行间公式。"""

    latex: str


@dataclass(slots=True)
class RtfTableCell:
    """保存表格 origin cell 的语义内容和合并标记。"""

    blocks: list[RtfBlock] = field(default_factory=list)
    horizontal_merge: Literal["none", "start", "continue"] = "none"
    vertical_merge: Literal["none", "start", "continue"] = "none"
    right_boundary: int | None = None


@dataclass(slots=True)
class RtfTableRow:
    """保存 RTF 表格的一行及表头标记。"""

    cells: list[RtfTableCell] = field(default_factory=list)
    header: bool = False


@dataclass(slots=True)
class RtfTable:
    """保存按源顺序排列的 RTF 表格行。"""

    rows: list[RtfTableRow] = field(default_factory=list)


RtfBlock: TypeAlias = Union[RtfParagraph, RtfDisplayEquation, RtfTable]


@dataclass(slots=True)
class RtfNote:
    """保存脚注、尾注或批注正文。"""

    id: str
    kind: Literal["footnote", "endnote", "annotation"]
    blocks: list[RtfBlock] = field(default_factory=list)


@dataclass(slots=True)
class RtfMetadata:
    """保存 RTF info destination 中允许公开的文档属性。"""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None


@dataclass(slots=True)
class RtfDocument:
    """保存单逻辑页 RTF 文档、辅助内容、注释、素材与元数据。"""

    blocks: list[RtfBlock] = field(default_factory=list)
    notes: list[RtfNote] = field(default_factory=list)
    headers: list[RtfBlock] = field(default_factory=list)
    footers: list[RtfBlock] = field(default_factory=list)
    metadata: RtfMetadata = field(default_factory=RtfMetadata)


__all__ = [
    "RtfAnchor",
    "RtfBlock",
    "RtfDisplayEquation",
    "RtfDocument",
    "RtfImage",
    "RtfInline",
    "RtfInlineEquation",
    "RtfLineBreak",
    "RtfListInfo",
    "RtfMetadata",
    "RtfNote",
    "RtfNoteReference",
    "RtfParagraph",
    "RtfTable",
    "RtfTableCell",
    "RtfTableRow",
    "RtfTextRun",
    "RtfTextStyle",
]
