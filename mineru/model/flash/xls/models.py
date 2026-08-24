# Copyright (c) Opendatalab. All rights reserved.

"""旧版 Excel 解析阶段使用的内部语义模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class XlsFontStyle:
    """一个 BIFF FONT 记录中可表达的行内文字样式。"""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    superscript: bool = False
    subscript: bool = False


@dataclass(frozen=True, slots=True)
class XlsRichRun:
    """以 Python 字符索引表示的富文本区间。"""

    start: int
    end: int
    style: XlsFontStyle


@dataclass(frozen=True, slots=True)
class XlsRichText:
    """单元格或 drawing 文本及其富文本区间。"""

    text: str
    runs: tuple[XlsRichRun, ...] = ()


@dataclass(slots=True)
class XlsCell:
    """工作表中的一个可见语义单元格。"""

    row: int
    col: int
    value: XlsRichText
    hyperlink: str | None = None


@dataclass(frozen=True, slots=True)
class XlsImage:
    """绑定到工作表 cell anchor 的已序列化图片。"""

    row: int
    col: int
    image_base64: str


@dataclass(frozen=True, slots=True)
class XlsEquation:
    """绑定到工作表 cell anchor 的 Equation Editor 原生公式。"""

    row: int
    col: int
    latex: str


@dataclass(frozen=True, slots=True)
class XlsChart:
    """由嵌入 chart 引用恢复出的源数据坐标。"""

    row: int
    col: int
    source_rows: tuple[int, ...]
    source_cols: tuple[int, ...]
    image_base64: str | None = None


@dataclass(slots=True)
class XlsSheet:
    """一个 worksheet 的单元格、合并区域与 drawing 资源。"""

    name: str
    visible: bool
    cells: dict[tuple[int, int], XlsCell] = field(default_factory=dict)
    merges: list[tuple[int, int, int, int]] = field(default_factory=list)
    images: list[XlsImage] = field(default_factory=list)
    equations: list[XlsEquation] = field(default_factory=list)
    charts: list[XlsChart] = field(default_factory=list)
    recovered: bool = False


@dataclass(slots=True)
class XlsWorkbook:
    """Excel 97–2003 工作簿的内部分页表示。"""

    sheets: list[XlsSheet]
