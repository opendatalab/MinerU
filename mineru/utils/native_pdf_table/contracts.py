# Copyright (c) Opendatalab. All rights reserved.
"""Native PDF 表格结构恢复使用的中立内部数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pdftext.schema import Char

from mineru.types import BBox

NativeTableCandidateSource = Literal[
    "vector_grid",
    "sparse_hybrid",
    "sparse_multiline",
    "sparse_grid",
    "text_grid",
    "key_value",
]


@dataclass(frozen=True, slots=True)
class NativeTableRule:
    """保存页面坐标中的一条可见横线或竖线。"""

    bbox: BBox
    width: float
    orientation: Literal["horizontal", "vertical"]


@dataclass(frozen=True, slots=True)
class NativeTableRectangle:
    """保存页面坐标中的 PDF 矩形路径及其绘制属性。"""

    bbox: BBox
    segment_count: int
    fill_visible: bool
    stroke_visible: bool
    form_depth: int = 0


@dataclass(frozen=True, slots=True)
class NativeTableInput:
    """保存一次已知表格区域结构恢复所需的全部页面原语。"""

    table_bbox: BBox
    page_size: tuple[float, float]
    angle: int
    chars: tuple[Char, ...]
    drawing_lines: tuple[NativeTableRule, ...] = field(default_factory=tuple)
    rectangles: tuple[NativeTableRectangle, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class NativeTableGlyph:
    """保存正向表格局部坐标中的单个可见 PDF 字符。"""

    glyph_id: int
    source_index: int
    text: str
    bbox: BBox
    visual_row: int
    explicit_space_before: bool = False
    explicit_break_before: bool = False


@dataclass(frozen=True, slots=True)
class NativeTableToken:
    """保存同一视觉行内由连续字符组成的表格文本项。"""

    text: str
    bbox: BBox
    glyph_ids: tuple[int, ...]
    source_char_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativeTableTextRow:
    """保存表格局部坐标中的一条视觉文本行。"""

    row_index: int
    bbox: BBox
    tokens: tuple[NativeTableToken, ...]
    glyph_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativeTableText:
    """保存统一字符预处理后的字形、文本行与尺度统计。"""

    glyphs: tuple[NativeTableGlyph, ...]
    rows: tuple[NativeTableTextRow, ...]
    median_glyph_width: float
    median_glyph_height: float


@dataclass(frozen=True, slots=True)
class NativeTableCell:
    """保存一个逻辑单元格的网格位置、内容和字符来源。"""

    row: int
    col: int
    rowspan: int
    colspan: int
    bbox: BBox
    content: str
    source_char_indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class NativeTableCandidate:
    """保存单路算法生成且已完成文本落格的表格候选。"""

    source: NativeTableCandidateSource
    rows: int
    cols: int
    cells: tuple[NativeTableCell, ...]
    score: float
    text_capture: float
    structure_support: float
    row_stability: float
    column_stability: float
    order_consistency: float
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def topology(self) -> tuple[int, int, tuple[tuple[int, int, int, int], ...]]:
        """返回不含文本的稳定拓扑签名，供候选冲突仲裁。"""

        spans = tuple(sorted((cell.row, cell.col, cell.rowspan, cell.colspan) for cell in self.cells))
        return self.rows, self.cols, spans


@dataclass(frozen=True, slots=True)
class NativeTableResult:
    """保存最终采用的原生表格 HTML 及其可诊断内部结构。"""

    html: str
    rows: int
    cols: int
    cells: tuple[NativeTableCell, ...]
    source: NativeTableCandidateSource
    confidence: float
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "NativeTableCell",
    "NativeTableInput",
    "NativeTableRectangle",
    "NativeTableResult",
    "NativeTableRule",
]
