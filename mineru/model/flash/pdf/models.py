# Copyright (c) Opendatalab. All rights reserved.

"""Flash 原生 PDF 提取使用的内部数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pdftext.schema import Char

from ....types import BBox
from .document import PDFPathInfo

if TYPE_CHECKING:
    from .text_styles import PDFTextScriptLine


@dataclass(slots=True)
class _LineItem:
    """保存单个可视文本行及其 source/ink/canonical 几何。"""

    text: str
    bbox: BBox
    angle: int
    source_index: int
    source_bbox: BBox | None = None
    ink_bbox: BBox | None = None
    baseline: float | None = None
    em_height: float = 0.0
    geometry_state: Literal["healthy", "repair_x", "trim_y", "repair_xy", "uncertain"] = "healthy"
    geometry_confidence: float = 1.0
    split_y_candidate: bool = False
    chars: list[Char] = field(default_factory=list)
    visual_row_id: int | None = None
    run_index: int = 0
    effective_height: float = 0.0
    font_signature: tuple[str, int] | None = None
    font_coverage: float = 0.0
    dominant_font_weight: float | None = None
    median_glyph_width: float | None = None
    leading_emphasis_width: float | None = None
    split_from_row: bool = False
    preserve_split_boundary: bool = False
    semantic_type: str | None = None
    restored_inline_cluster: bool = False
    compact_formula_cluster: bool = False
    formula_candidate_only: bool = False
    structural_title: bool = False
    style_scale_repaired: bool = False
    inline_math_regions: list[BBox] = field(default_factory=list)

    def __post_init__(self) -> None:
        """为旧调用与合成测试补齐可选来源几何字段。"""

        if self.source_bbox is None:
            self.source_bbox = self.bbox


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
class _TableAnnotation:
    """保存表格候选中已识别注释的类型、紧致边界和原始行身份。"""

    kind: Literal["caption", "footnote"]
    bbox: BBox
    line_indices: set[int] = field(default_factory=set)
    line_bboxes: dict[int, BBox] = field(default_factory=dict)


@dataclass(slots=True)
class _TableCandidate:
    """保存已通过相邻横线边界与文本分布校验的表格候选。"""

    bbox: BBox
    local_bbox: BBox
    angle: int
    score: float
    core_bbox: BBox | None = None
    line_indices: set[int] = field(default_factory=set)
    annotations: list[_TableAnnotation] = field(default_factory=list)


@dataclass(slots=True)
class _GraphicCandidate:
    """保存由紧凑绘图线组件形成的图形文本容器候选。"""

    core_bbox: BBox
    lane_index: int
    label_margin_scale: float = 2.5
    line_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class _CodeCandidate:
    """保存由填充背景或成对横线与稳定文本节奏确认的代码区域。"""

    bbox: BBox
    angle: int
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
    """保存公式右缘锚点及其相对正文密集区的上下位置。"""

    line: _LineItem
    bbox: BBox
    detached_below_body: bool = False
    detached_above_body: bool = False
    repeated_number_band: bool = False


@dataclass(slots=True)
class _PageSource:
    """保存单页原生文本分析所需的文本、字符、绘图线和视觉容器。"""

    page_size: tuple[float, float]
    lines: list[_LineItem]
    chars: list[Char]
    drawing_lines: list[_AxisLine]
    image_bboxes: list[BBox] = field(default_factory=list)
    signature_bboxes: list[BBox] = field(default_factory=list)
    form_bboxes: list[BBox] = field(default_factory=list)
    path_infos: list[PDFPathInfo] = field(default_factory=list)


@dataclass(slots=True)
class _PreparedPage:
    """保存容器认领完成后、等待跨页与文本类型判定的轻量页面。"""

    page_size: tuple[float, float]
    remaining_lines: list[_LineItem]
    table_bboxes: list[BBox]
    drawing_lines: list[_AxisLine]
    fixed_blocks: list[dict[str, Any]]
    canonical_formula_geometry: bool = False
    canonical_formula_source_lines: list[_LineItem] = field(default_factory=list)
    page_footnote_groups: list[set[int]] = field(default_factory=list)
    script_lines: list[PDFTextScriptLine] = field(default_factory=list)
    formula_candidate_lines: list[_LineItem] = field(default_factory=list)


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
    body_row_count: int = 0


@dataclass(frozen=True, slots=True)
class _DocumentBodyProfile:
    """保存跨页正文行高、常规字重与反复出现的常规字体。"""

    body_height: float
    body_weight: float | None
    regular_fonts: frozenset[tuple[str, int]]
    has_style_scale_repairs: bool = False


@dataclass(frozen=True, slots=True)
class _TitleStylePrototype:
    """保存跨页标题原型的字体、尺度、字重和栏内对齐特征。"""

    font_family: str
    font_flags: int
    height_ratio: float
    weight: float | None
    alignment: Literal["left", "center"]
    anchor_offset: float
    support_count: int
    support_pages: int


@dataclass(frozen=True, slots=True)
class _DocumentTitleProfile:
    """保存全文中由重复高置信标题形成的排版原型集合。"""

    prototypes: tuple[_TitleStylePrototype, ...]
