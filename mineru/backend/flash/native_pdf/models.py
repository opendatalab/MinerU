# Copyright (c) Opendatalab. All rights reserved.

"""Flash 原生 PDF 提取使用的内部数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pdftext.schema import Char

from mineru.types import BBox


@dataclass(slots=True)
class _LineItem:
    """保存单个可视文本行及其原始几何信息。"""

    text: str
    bbox: BBox
    angle: int
    source_index: int
    chars: list[Char] = field(default_factory=list)
    visual_row_id: int | None = None
    run_index: int = 0
    effective_height: float = 0.0
    font_signature: tuple[str, int] | None = None
    font_coverage: float = 0.0
    dominant_font_weight: float | None = None
    median_glyph_width: float | None = None
    split_from_row: bool = False
    semantic_type: str | None = None
    restored_inline_cluster: bool = False
    compact_formula_cluster: bool = False


@dataclass(slots=True)
class _Fragment:
    """保存表格规则使用的单元文本片段。"""

    text: str
    bbox: BBox
    local_bbox: BBox
    line_index: int
    visual_row_id: int | None = None


@dataclass(slots=True)
class _VisualRow:
    """保存同一局部水平带内的表格片段。"""

    fragments: list[_Fragment]
    center_y: float
    bbox: BBox
    visual_row_id: int | None = None


@dataclass(slots=True)
class _AxisLine:
    """保存 PDF 路径中的横竖线。"""

    bbox: BBox
    width: float
    orientation: Literal["horizontal", "vertical"]


@dataclass(slots=True)
class _LocalAxisLine:
    """保存转入当前文本方向后的横竖线。"""

    bbox: BBox
    original_bbox: BBox
    orientation: Literal["horizontal", "vertical"]
    width: float


@dataclass(slots=True)
class _TableCandidate:
    """保存已通过三横线与文本分布校验的表格候选。"""

    bbox: BBox
    local_bbox: BBox
    angle: int
    score: float
    core_bbox: BBox | None = None
    line_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class _GraphicCandidate:
    """保存由紧凑绘图线组件形成的图形文本容器候选。"""

    core_bbox: BBox
    lane_index: int
    line_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class _TextLane:
    """保存同一文本方向下的局部栏带与已归属文本行。"""

    left: float
    right: float
    lines: list[tuple[_LineItem, BBox]] = field(default_factory=list)
    is_span: bool = False


@dataclass(slots=True)
class _FormulaAnchor:
    """保存公式右缘锚点及其是否落在正文密集区下方。"""

    line: _LineItem
    bbox: BBox
    detached_below_body: bool = False


@dataclass(slots=True)
class _PageSource:
    """保存单页原生文本分析所需的文本、字符、绘图线和视觉容器。"""

    page_size: tuple[float, float]
    lines: list[_LineItem]
    chars: list[Char]
    drawing_lines: list[_AxisLine]
    image_bboxes: list[BBox] = field(default_factory=list)
    form_bboxes: list[BBox] = field(default_factory=list)


@dataclass(slots=True)
class _PreparedPage:
    """保存容器认领完成后、等待跨页与文本类型判定的轻量页面。"""

    page_size: tuple[float, float]
    remaining_lines: list[_LineItem]
    table_bboxes: list[BBox]
    drawing_lines: list[_AxisLine]
    fixed_blocks: list[dict[str, Any]]
    page_footnote_groups: list[set[int]] = field(default_factory=list)


@dataclass(slots=True)
class _MarginalCandidate:
    """保存页眉页脚带中的单行候选及其正向归一化几何。"""

    page_index: int
    line: _LineItem
    local_bbox: BBox
    local_page_size: tuple[float, float]
    region: Literal["header", "footer", "side"]


@dataclass(slots=True)
class _LaneBodyProfile:
    """保存标题判定使用的栏带正文排版基线，不包含文本内容特征。"""

    body_height: float
    body_font: tuple[str, int] | None
    body_weight: float | None
    regular_gap: float
    style_support: dict[tuple[str, int], float]
