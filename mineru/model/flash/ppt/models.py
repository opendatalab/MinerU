# Copyright (c) Opendatalab. All rights reserved.

"""旧版 PPT 解析阶段使用的内部语义模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class PptTextRun:
    """一个样式与超链接均已解析的文本片段。"""

    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    baseline: int | None = None
    hyperlink: str | None = None


@dataclass(frozen=True, slots=True)
class PptParagraph:
    """一个段落及其列表层级和编号属性。"""

    runs: tuple[PptTextRun, ...]
    depth: int = 0
    list_kind: Literal["ordered", "unordered"] | None = None
    start: int | None = None
    pp9rt: int = 0


@dataclass(frozen=True, slots=True)
class PptTextElement:
    """带幻灯片坐标的文本形状。"""

    paragraphs: tuple[PptParagraph, ...]
    text_type: int
    bbox: tuple[float, float, float, float]
    order: int
    shape_offset: int
    is_placeholder: bool = False


@dataclass(frozen=True, slots=True)
class PptImageElement:
    """已经绑定到具体幻灯片形状的图片。"""

    image_base64: str
    bbox: tuple[float, float, float, float]
    order: int
    shape_offset: int


@dataclass(frozen=True, slots=True)
class PptTableCell:
    """表格原点单元格及其跨行跨列范围。"""

    row: int
    col: int
    row_span: int
    col_span: int
    paragraphs: tuple[PptParagraph, ...]


@dataclass(frozen=True, slots=True)
class PptTableElement:
    """由 OfficeArt 表格组重建出的规则网格。"""

    rows: int
    cols: int
    cells: tuple[PptTableCell, ...]
    bbox: tuple[float, float, float, float]
    order: int
    shape_offsets: frozenset[tuple[int, ...]]


PptSlideElement: TypeAlias = PptTextElement | PptImageElement | PptTableElement


@dataclass(slots=True)
class PptSlide:
    """一张幻灯片的语义内容与备注。"""

    slide_id: int | None
    elements: list[PptSlideElement] = field(default_factory=list)
    notes: list[PptParagraph] = field(default_factory=list)
    hidden: bool = False


@dataclass(slots=True)
class PptPresentation:
    """旧版 PPT 的分页内部表示。"""

    slides: list[PptSlide]
    width: int = 5760
    height: int = 4320
