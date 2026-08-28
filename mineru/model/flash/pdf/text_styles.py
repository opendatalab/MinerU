# Copyright (c) Opendatalab. All rights reserved.
"""从 PDF 原生字符、绘图线和 Link 注解恢复行内样式与超链接。"""

from __future__ import annotations

import html
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence, TypeVar, cast

from loguru import logger

from ....types import BBox, BlockType, RAW_ALGORITHM, RAW_CAPTION, RAW_FOOTNOTE, RAW_PHONETIC
from .._shared.spans import (
    append_equation_span,
    append_hyperlink_span,
    append_text_span,
    extend_inline_spans,
    normalize_span_dicts,
)
from .document import PDFLinkAnnotation
from ....utils.text import is_hyphen_at_line_end


TEXT_DECORATION_MIN_LENGTH_HEIGHT_RATIO = 1.8
STRIKETHROUGH_CENTER_TOLERANCE_HEIGHT_RATIO = 0.2
UNDERLINE_BOTTOM_TOLERANCE_HEIGHT_RATIO = 0.2
TEXT_DECORATION_MAX_WIDTH_HEIGHT_RATIO = 0.2
TEXT_DECORATION_MIN_TEXT_COVERAGE_RATIO = 0.55
TEXT_DECORATION_ENDPOINT_TOLERANCE_HEIGHT_RATIO = 0.5
UNDERLINE_FRACTION_MAX_GAP_HEIGHT_RATIO = 0.15
UNDERLINE_FRACTION_MIN_LOWER_LINE_COVERAGE = 0.8
PDF_FONT_ITALIC_FLAG = 1 << 6
PDF_FONT_FORCE_BOLD_FLAG = 1 << 18
PDF_BOLD_MIN_WEIGHT = 600.0
PDF_BOLD_MIN_COMPARABLE_CHAR_COUNT = 2
PDF_LINK_CHAR_OVERLAP_THRESHOLD = 0.5

PDFTextStyle = Literal["bold", "italic", "underline", "strikethrough"]
_PDF_LINK_INTERVALS_KEY = "_inline_link_intervals"
_PDF_STYLE_INTERVALS_KEY = "_inline_style_intervals"
_PDF_INLINE_SPAN_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.REF_TEXT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.LIST,
    BlockType.INDEX,
    RAW_ALGORITHM,
    RAW_CAPTION,
    RAW_FOOTNOTE,
    RAW_PHONETIC,
}
PDFTextDecoration = Literal["underline", "strikethrough"]
PDF_TEXT_STYLE_ORDER: tuple[PDFTextStyle, ...] = (
    "bold",
    "italic",
    "underline",
    "strikethrough",
)
_PDF_TEXT_DECORATION_ORDER: tuple[PDFTextDecoration, ...] = (
    "underline",
    "strikethrough",
)

PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES = frozenset(
    {
        BlockType.TEXT,
        BlockType.REF_TEXT,
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        BlockType.LIST,
        BlockType.INDEX,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_FOOTNOTE,
        RAW_CAPTION,
        RAW_FOOTNOTE,
    }
)
_PDF_TEXT_STYLE_TARGET_BLOCK_TYPES: dict[PDFTextStyle, frozenset[str]] = {
    "bold": frozenset({BlockType.TEXT}),
    "italic": frozenset(),
    "underline": frozenset({BlockType.TEXT}),
    "strikethrough": PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES,
}

_PDF_FONT_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
_PDF_BOLD_FONT_NAME_RE = re.compile(
    r"bold|demi|black|heavy|(?:^|[-_,])(?:bd|bi)(?:mt)?$|gbi$|(?:^|[-_,])w[6-9]$",
    re.IGNORECASE,
)
_PDF_LIST_MARKER_CHARS = frozenset("•◦‣⁃▪▫●○■□∙·▶▷►▸▹\uf0b7")
_PDF_GEOMETRIC_TEXT_STYLES = frozenset({"underline", "strikethrough"})
_PDF_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_PDF_SEPARATOR_SPACE_CHARS = frozenset(
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
)
_PDF_ZERO_WIDTH_CHARS = frozenset({"\u200b", "\u2060", "\ufeff"})
_LIGATURE_REPLACEMENTS = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}


@dataclass(frozen=True, slots=True)
class PDFTextStyleRange:
    """保存可比较文本中的一个半开样式区间。"""

    start: int
    end: int
    styles: tuple[PDFTextStyle, ...]


@dataclass(frozen=True, slots=True)
class PDFTextStyleLine:
    """保存一个视觉文本 run 的几何、可比较文本和样式区间。"""

    bbox: BBox
    text: str
    style_ranges: tuple[PDFTextStyleRange, ...]
    source_index: int


@dataclass(frozen=True, slots=True)
class PDFTextLinkRange:
    """保存可比较文本中的一个半开超链接区间。"""

    start: int
    end: int
    target: str


@dataclass(frozen=True, slots=True)
class PDFTextLinkLine:
    """保存视觉文本 run 的几何、可比较文本和超链接区间。"""

    bbox: BBox
    text: str
    link_ranges: tuple[PDFTextLinkRange, ...]
    source_index: int


PDFTextEvidenceLine = TypeVar(
    "PDFTextEvidenceLine",
    PDFTextStyleLine,
    PDFTextLinkLine,
)


@dataclass(frozen=True, slots=True)
class _VisibleChar:
    """保存参与文本装饰线几何判断的可见字符。"""

    source_index: int
    bbox: BBox


@dataclass(slots=True)
class _LineCandidate:
    """保存字体与文本装饰线匹配阶段使用的视觉文本行指标。"""

    bbox: BBox
    chars: list[dict[str, Any]]
    visible_chars: list[_VisibleChar]
    median_height: float
    center_y: float
    bottom_y: float
    source_index: int
    font_styles: list[frozenset[PDFTextStyle]]
    decoration_ranges: dict[PDFTextDecoration, list[tuple[int, int]]]


@dataclass(frozen=True, slots=True)
class _DrawingMatch:
    """保存单条 drawing 对单个文本行的匹配结果和排序指标。"""

    style: PDFTextDecoration
    start_index: int
    end_index: int
    target_distance_ratio: float
    horizontal_overlap_ratio: float


@dataclass(frozen=True, slots=True)
class _ProjectedChar:
    """保存 model content 可比较字符到原字符串位置的映射。"""

    value: str
    raw_start: int
    raw_end: int
    existing_styles: frozenset[PDFTextStyle]
    formula_gap_before: bool
    inside_hyperlink: bool


@dataclass(frozen=True, slots=True)
class _RawStyleInterval:
    """保存 model content 原字符串中的半开样式区间。"""

    start: int
    end: int
    styles: tuple[PDFTextStyle, ...]


@dataclass(frozen=True, slots=True)
class _MatchedLinkRange:
    """保存已经对齐到 block 可比较文本的链接区间与物理行身份。"""

    start: int
    end: int
    target: str
    source_index: int


@dataclass(frozen=True, slots=True)
class _RawLinkInterval:
    """保存 model content 原字符串中的半开链接区间。"""

    start: int
    end: int
    target: str
    source_index: int


@dataclass(frozen=True, slots=True)
class _LineProjectionMatch:
    """保存物理行跨公式空洞对齐到 block 可比较文本的结果。"""

    start: int
    end: int
    source_to_projected: tuple[int | None, ...]


def _style_line_reading_order_key(
    line: PDFTextStyleLine,
) -> tuple[int, float, float]:
    """优先使用原生 source_index 排序，重复索引时再以 bbox 保持稳定。"""

    return line.source_index, line.bbox[1], line.bbox[0]


def _coerce_bbox(value: Any) -> BBox | None:
    """把 list、tuple 或 pdftext bbox 对象收敛为合法有限 bbox。"""

    raw_bbox = getattr(value, "bbox", value)
    try:
        if raw_bbox is None or len(raw_bbox) != 4:
            return None
        bbox = tuple(float(item) for item in raw_bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox  # type: ignore[return-value]


def _ordered_line_chars(line: Any) -> list[dict[str, Any]]:
    """按 char_idx 修复异常乱序字符，同时保留缺少索引时的来源顺序。"""

    chars = [char for char in getattr(line, "chars", []) if isinstance(char, dict)]
    indexed_chars = [char.get("char_idx") for char in chars]
    if (
        chars
        and all(isinstance(index, int) for index in indexed_chars)
        and any(first > second for first, second in zip(indexed_chars, indexed_chars[1:]))
    ):
        return sorted(chars, key=lambda char: int(char["char_idx"]))
    return chars


def _normalize_match_fragment(value: Any) -> str:
    """把单个字符片段规范为忽略排版空白的确定性匹配文本。"""

    output: list[str] = []
    for char in str(value or ""):
        if char in _PDF_ZERO_WIDTH_CHARS or char == "\u00ad":
            continue
        if char == "\x02":
            output.append("-")
            continue
        if char.isspace() or char in _PDF_SEPARATOR_SPACE_CHARS:
            continue
        if _PDF_CONTROL_CHAR_RE.fullmatch(char):
            continue
        output.append(_LIGATURE_REPLACEMENTS.get(char, char))
    return "".join(output)


def _canonical_styles(styles: Iterable[str]) -> tuple[PDFTextStyle, ...]:
    """按公开富文本协议顺序过滤、去重并规范样式集合。"""

    style_set = set(styles)
    return cast(
        tuple[PDFTextStyle, ...],
        tuple(style for style in PDF_TEXT_STYLE_ORDER if style in style_set),
    )


def _pdf_font_metadata(char: dict[str, Any]) -> tuple[str, int, float | None]:
    """读取单个字符的规范字体名、FontDescriptor flags 和有效字重。"""

    font = char.get("font")
    if not isinstance(font, dict):
        return "", 0, None
    font_name = _PDF_FONT_SUBSET_PREFIX_RE.sub(
        "",
        str(font.get("name") or ""),
    )
    try:
        font_flags = int(font.get("flags") or 0)
    except (TypeError, ValueError):
        font_flags = 0
    try:
        font_weight = float(font.get("weight"))
    except (TypeError, ValueError):
        font_weight = math.nan
    if not math.isfinite(font_weight) or font_weight <= 0:
        font_weight = None
    return font_name, font_flags, font_weight


def _char_font_styles(char: dict[str, Any]) -> frozenset[PDFTextStyle]:
    """只依据直接字体证据返回 PDF 字符粗体样式。"""

    font_name, font_flags, font_weight = _pdf_font_metadata(char)
    styles: set[PDFTextStyle] = set()
    if (
        font_flags & PDF_FONT_FORCE_BOLD_FLAG
        or (font_weight is not None and font_weight >= PDF_BOLD_MIN_WEIGHT)
        or bool(_PDF_BOLD_FONT_NAME_RE.search(font_name))
    ):
        styles.add("bold")
    return frozenset(styles)


def _has_list_marker_separator(
    chars: Sequence[dict[str, Any]],
    marker_source_index: int,
    next_source_index: int,
    median_height: float,
) -> bool:
    """判断行首项目符号与后续正文之间是否存在空白或明显视觉间隔。"""

    if any(str(chars[index].get("char") or "").isspace() for index in range(marker_source_index + 1, next_source_index)):
        return True
    marker_bbox = _coerce_bbox(chars[marker_source_index].get("bbox"))
    next_bbox = _coerce_bbox(chars[next_source_index].get("bbox"))
    return bool(marker_bbox is not None and next_bbox is not None and next_bbox[0] - marker_bbox[2] >= 0.5 * median_height)


def _filter_pdf_bold_runs(
    chars: Sequence[dict[str, Any]],
    font_styles: Sequence[frozenset[PDFTextStyle]],
    median_height: float,
) -> list[frozenset[PDFTextStyle]]:
    """过滤过短粗体 run 和与正文分离的行首项目符号粗体。"""

    output = list(font_styles)
    comparable_chars = [
        (source_index, fragment)
        for source_index, char in enumerate(chars)
        if (fragment := _normalize_match_fragment(char.get("char")))
    ]
    run_start = 0
    while run_start < len(comparable_chars):
        source_index, _fragment = comparable_chars[run_start]
        if "bold" not in output[source_index]:
            run_start += 1
            continue
        run_end = run_start + 1
        while run_end < len(comparable_chars):
            next_source_index, _next_fragment = comparable_chars[run_end]
            if "bold" not in output[next_source_index]:
                break
            run_end += 1

        run = comparable_chars[run_start:run_end]
        run_text = "".join(fragment for _index, fragment in run)
        is_short = len(run_text) < PDF_BOLD_MIN_COMPARABLE_CHAR_COUNT
        is_isolated_leading_marker = (
            run_start == 0
            and run_end < len(comparable_chars)
            and bool(run_text)
            and all(char in _PDF_LIST_MARKER_CHARS for char in run_text)
            and _has_list_marker_separator(
                chars,
                run[-1][0],
                comparable_chars[run_end][0],
                median_height,
            )
        )
        if is_short or is_isolated_leading_marker:
            for run_source_index, _run_fragment in run:
                output[run_source_index] = frozenset(style for style in output[run_source_index] if style != "bold")
        run_start = run_end
    return output


def _build_line_candidate(line: Any) -> _LineCandidate | None:
    """从视觉水平 line 构造字符几何候选，旋转文字和退化行返回空。"""

    if int(getattr(line, "angle", 0) or 0) % 360 != 0:
        return None
    line_bbox = _coerce_bbox(getattr(line, "bbox", None))
    if line_bbox is None:
        return None
    chars = _ordered_line_chars(line)
    visible_chars: list[_VisibleChar] = []
    for char_index, char in enumerate(chars):
        text = str(char.get("char") or "")
        bbox = _coerce_bbox(char.get("bbox"))
        if bbox is None or not text.isprintable() or text.isspace():
            continue
        visible_chars.append(_VisibleChar(source_index=char_index, bbox=bbox))
    if not visible_chars:
        return None

    heights = [char.bbox[3] - char.bbox[1] for char in visible_chars]
    median_height = statistics.median(heights)
    if median_height <= 0:
        return None
    body_chars = [char for char, height in zip(visible_chars, heights) if height >= 0.8 * median_height]
    if not body_chars:
        return None
    font_styles = [_char_font_styles(char) for char in chars]
    return _LineCandidate(
        bbox=line_bbox,
        chars=chars,
        visible_chars=visible_chars,
        median_height=median_height,
        center_y=statistics.median((char.bbox[1] + char.bbox[3]) / 2 for char in body_chars),
        bottom_y=statistics.median(char.bbox[3] for char in body_chars),
        source_index=int(getattr(line, "source_index", 0) or 0),
        font_styles=_filter_pdf_bold_runs(
            chars,
            font_styles,
            median_height,
        ),
        decoration_ranges={
            "underline": [],
            "strikethrough": [],
        },
    )


def _drawing_match_for_line(
    line: _LineCandidate,
    drawing: Any,
    style: PDFTextDecoration,
) -> _DrawingMatch | None:
    """按目标纵向锚点和公共几何规则校验单条文本装饰线。"""

    if getattr(drawing, "orientation", None) != "horizontal":
        return None
    drawing_bbox = _coerce_bbox(getattr(drawing, "bbox", None))
    if drawing_bbox is None:
        return None
    drawing_length = drawing_bbox[2] - drawing_bbox[0]
    if drawing_length < TEXT_DECORATION_MIN_LENGTH_HEIGHT_RATIO * line.median_height:
        return None
    drawing_center_y = (drawing_bbox[1] + drawing_bbox[3]) / 2
    target_y = line.bottom_y if style == "underline" else line.center_y
    target_tolerance = (
        UNDERLINE_BOTTOM_TOLERANCE_HEIGHT_RATIO if style == "underline" else STRIKETHROUGH_CENTER_TOLERANCE_HEIGHT_RATIO
    )
    target_distance_ratio = abs(drawing_center_y - target_y) / line.median_height
    if target_distance_ratio > target_tolerance:
        return None
    try:
        drawing_width = max(0.0, float(getattr(drawing, "width", 0.0) or 0.0))
    except (TypeError, ValueError):
        return None
    if drawing_width > TEXT_DECORATION_MAX_WIDTH_HEIGHT_RATIO * line.median_height:
        return None

    hit_chars = [char for char in line.visible_chars if drawing_bbox[0] <= (char.bbox[0] + char.bbox[2]) / 2 <= drawing_bbox[2]]
    if not hit_chars:
        return None
    hit_left = min(char.bbox[0] for char in hit_chars)
    hit_right = max(char.bbox[2] for char in hit_chars)
    if (hit_right - hit_left) / drawing_length < TEXT_DECORATION_MIN_TEXT_COVERAGE_RATIO:
        return None
    endpoint_distance = min(
        abs(drawing_bbox[0] - hit_left),
        abs(drawing_bbox[2] - hit_right),
    )
    if endpoint_distance > TEXT_DECORATION_ENDPOINT_TOLERANCE_HEIGHT_RATIO * line.median_height:
        return None

    overlap = max(
        0.0,
        min(line.bbox[2], drawing_bbox[2]) - max(line.bbox[0], drawing_bbox[0]),
    )
    horizontal_overlap_ratio = overlap / max(
        0.01,
        min(line.bbox[2] - line.bbox[0], drawing_length),
    )
    return _DrawingMatch(
        style=style,
        start_index=min(char.source_index for char in hit_chars),
        end_index=max(char.source_index for char in hit_chars) + 1,
        target_distance_ratio=target_distance_ratio,
        horizontal_overlap_ratio=horizontal_overlap_ratio,
    )


def _merge_source_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠或相邻的来源字符区间。"""

    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _line_style_payload(line: _LineCandidate) -> PDFTextStyleLine | None:
    """把来源字符的字体与装饰线证据转换为紧凑文本样式区间。"""

    decoration_styles: list[set[PDFTextStyle]] = [set() for _char in line.chars]
    for style in _PDF_TEXT_DECORATION_ORDER:
        for start, end in _merge_source_ranges(line.decoration_ranges[style]):
            for char_index in range(max(0, start), min(end, len(line.chars))):
                decoration_styles[char_index].add(style)
    compact_parts: list[str] = []
    compact_styles: list[tuple[PDFTextStyle, ...]] = []
    for char_index, char in enumerate(line.chars):
        fragment = _normalize_match_fragment(char.get("char"))
        if not fragment:
            continue
        compact_parts.append(fragment)
        styles = set(line.font_styles[char_index])
        styles.update(decoration_styles[char_index])
        canonical_styles = _canonical_styles(styles)
        compact_styles.extend([canonical_styles] * len(fragment))

    text = "".join(compact_parts)
    if not text:
        return None
    compact_ranges: list[PDFTextStyleRange] = []
    active_start = 0
    active_styles: tuple[PDFTextStyle, ...] = ()
    for offset, styles in enumerate([*compact_styles, ()]):
        if styles == active_styles:
            continue
        if active_styles:
            compact_ranges.append(PDFTextStyleRange(active_start, offset, active_styles))
        active_start = offset
        active_styles = styles
    return PDFTextStyleLine(
        bbox=line.bbox,
        text=text,
        style_ranges=tuple(compact_ranges),
        source_index=line.source_index,
    )


def _build_line_geometry_grids(
    candidates: Sequence[_LineCandidate],
) -> tuple[
    float,
    dict[int, list[tuple[int, PDFTextDecoration]]],
    dict[int, list[int]],
]:
    """按装饰线锚点和行顶坐标建立网格，限制每条 drawing 的局部比较范围。"""

    grid_size = max(
        1.0,
        statistics.median(line.median_height for line in candidates),
    )
    anchor_grid: dict[int, list[tuple[int, PDFTextDecoration]]] = {}
    top_grid: dict[int, list[int]] = {}
    for line_index, line in enumerate(candidates):
        top_grid.setdefault(math.floor(line.bbox[1] / grid_size), []).append(line_index)
        for style, target_y, tolerance_ratio in (
            (
                "underline",
                line.bottom_y,
                UNDERLINE_BOTTOM_TOLERANCE_HEIGHT_RATIO,
            ),
            (
                "strikethrough",
                line.center_y,
                STRIKETHROUGH_CENTER_TOLERANCE_HEIGHT_RATIO,
            ),
        ):
            tolerance = tolerance_ratio * line.median_height
            start_cell = math.floor((target_y - tolerance) / grid_size)
            end_cell = math.floor((target_y + tolerance) / grid_size)
            for cell in range(start_cell, end_cell + 1):
                anchor_grid.setdefault(cell, []).append((line_index, style))
    return grid_size, anchor_grid, top_grid


def _is_fraction_bar_candidate(
    candidates: Sequence[_LineCandidate],
    top_grid: dict[int, list[int]],
    grid_size: float,
    line_index: int,
    drawing_bbox: BBox,
) -> bool:
    """用紧邻且被横线覆盖的下方文本 run 排除公式分数线。"""

    line = candidates[line_index]
    drawing_center_y = (drawing_bbox[1] + drawing_bbox[3]) / 2
    max_lower_top = drawing_center_y + UNDERLINE_FRACTION_MAX_GAP_HEIGHT_RATIO * line.median_height
    lower_indices: set[int] = set()
    for cell in range(
        math.floor(drawing_center_y / grid_size),
        math.floor(max_lower_top / grid_size) + 1,
    ):
        lower_indices.update(top_grid.get(cell, ()))
    for lower_index in lower_indices:
        if lower_index == line_index:
            continue
        lower_line = candidates[lower_index]
        if not drawing_center_y <= lower_line.bbox[1] <= max_lower_top:
            continue
        lower_width = lower_line.bbox[2] - lower_line.bbox[0]
        horizontal_overlap = max(
            0.0,
            min(drawing_bbox[2], lower_line.bbox[2]) - max(drawing_bbox[0], lower_line.bbox[0]),
        )
        if horizontal_overlap / max(0.01, lower_width) >= UNDERLINE_FRACTION_MIN_LOWER_LINE_COVERAGE:
            return True
    return False


def detect_pdf_text_style_lines(
    lines: Sequence[Any],
    drawing_lines: Sequence[Any],
) -> list[PDFTextStyleLine]:
    """从视觉文本 run 与页面 drawing 中生成全部水平行样式证据。"""

    candidates = [candidate for line in lines if (candidate := _build_line_candidate(line)) is not None]
    if not candidates:
        return []
    horizontal_drawings = [drawing for drawing in drawing_lines if getattr(drawing, "orientation", None) == "horizontal"]
    if horizontal_drawings:
        grid_size, anchor_grid, top_grid = _build_line_geometry_grids(candidates)
        for drawing in horizontal_drawings:
            drawing_bbox = _coerce_bbox(getattr(drawing, "bbox", None))
            if drawing_bbox is None:
                continue
            drawing_center_y = (drawing_bbox[1] + drawing_bbox[3]) / 2
            candidate_anchors = anchor_grid.get(
                math.floor(drawing_center_y / grid_size),
                [],
            )
            matches: list[tuple[int, _DrawingMatch]] = []
            for line_index, style in candidate_anchors:
                match = _drawing_match_for_line(
                    candidates[line_index],
                    drawing,
                    style,
                )
                if match is None:
                    continue
                if style == "underline" and _is_fraction_bar_candidate(
                    candidates,
                    top_grid,
                    grid_size,
                    line_index,
                    drawing_bbox,
                ):
                    continue
                matches.append((line_index, match))
            if not matches:
                continue
            line_index, best_match = min(
                matches,
                key=lambda item: (
                    item[1].target_distance_ratio,
                    -item[1].horizontal_overlap_ratio,
                    candidates[item[0]].source_index,
                    _PDF_TEXT_DECORATION_ORDER.index(item[1].style),
                ),
            )
            candidates[line_index].decoration_ranges[best_match.style].append((best_match.start_index, best_match.end_index))

    payloads = [payload for line in candidates if (payload := _line_style_payload(line)) is not None]
    if not any(line.style_ranges for line in payloads):
        return []
    return sorted(
        payloads,
        key=_style_line_reading_order_key,
    )


def _bbox_intersection_area(first: BBox, second: BBox) -> float:
    """返回两个合法 bbox 的相交面积。"""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _link_region_hits_char(region: BBox, char_bbox: BBox) -> bool:
    """按字符中心或字符面积覆盖率判断 Link 区域是否命中字符。"""

    center_x = (char_bbox[0] + char_bbox[2]) / 2
    center_y = (char_bbox[1] + char_bbox[3]) / 2
    if region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]:
        return True
    char_area = max(
        0.01,
        (char_bbox[2] - char_bbox[0]) * (char_bbox[3] - char_bbox[1]),
    )
    return _bbox_intersection_area(region, char_bbox) / char_area >= PDF_LINK_CHAR_OVERLAP_THRESHOLD


def _link_targets_for_char(
    char_bbox: BBox,
    annotations: Sequence[PDFLinkAnnotation],
) -> set[str]:
    """返回命中字符的全部不同链接目标，供冲突检测使用。"""

    return {
        annotation.target
        for annotation in annotations
        if any(_link_region_hits_char(region, char_bbox) for region in annotation.bboxes)
    }


def _compact_link_ranges(
    compact_targets: Sequence[str | None],
) -> tuple[PDFTextLinkRange, ...]:
    """把逐字符链接目标压缩为同目标连续区间。"""

    ranges: list[PDFTextLinkRange] = []
    active_start = 0
    active_target: str | None = None
    for offset, target in enumerate([*compact_targets, None]):
        if target == active_target:
            continue
        if active_target is not None:
            ranges.append(
                PDFTextLinkRange(
                    start=active_start,
                    end=offset,
                    target=active_target,
                )
            )
        active_start = offset
        active_target = target
    return tuple(ranges)


def _build_link_line_payload(
    line: Any,
    annotations: Sequence[PDFLinkAnnotation],
    fallback_source_index: int,
) -> PDFTextLinkLine | None:
    """把一个视觉文本 run 与 Link 区域相交结果转换为紧凑链接证据。"""

    try:
        angle = int(getattr(line, "angle", 0) or 0) % 360
    except (TypeError, ValueError):
        return None
    if angle not in {0, 90, 180, 270}:
        return None
    line_bbox = _coerce_bbox(getattr(line, "bbox", None))
    if line_bbox is None:
        return None
    nearby_annotations = [
        annotation
        for annotation in annotations
        if any(_bbox_intersection_area(line_bbox, region) > 0 for region in annotation.bboxes)
    ]
    if not nearby_annotations:
        return None

    compact_parts: list[str] = []
    compact_targets: list[str | None] = []
    for char in _ordered_line_chars(line):
        fragment = _normalize_match_fragment(char.get("char"))
        if not fragment:
            continue
        char_bbox = _coerce_bbox(char.get("bbox"))
        targets = _link_targets_for_char(char_bbox, nearby_annotations) if char_bbox is not None else set()
        # 同一字符落入不同目标时不猜测 PDF 点击层级，保留为普通文本。
        target = next(iter(targets)) if len(targets) == 1 else None
        compact_parts.append(fragment)
        compact_targets.extend([target] * len(fragment))

    text = "".join(compact_parts)
    link_ranges = _compact_link_ranges(compact_targets)
    if not text or not link_ranges:
        return None
    try:
        source_index = int(getattr(line, "source_index", fallback_source_index))
    except (TypeError, ValueError):
        source_index = fallback_source_index
    return PDFTextLinkLine(
        bbox=line_bbox,
        text=text,
        link_ranges=link_ranges,
        source_index=source_index,
    )


def detect_pdf_text_link_lines(
    lines: Sequence[Any],
    annotations: Sequence[PDFLinkAnnotation],
) -> list[PDFTextLinkLine]:
    """从视觉文本 run 与 PDF Link 注解生成字符级超链接证据。"""

    if not annotations:
        return []
    payloads = [
        payload
        for line_index, line in enumerate(lines)
        if (
            payload := _build_link_line_payload(
                line,
                annotations,
                line_index,
            )
        )
        is not None
    ]
    return sorted(
        payloads,
        key=lambda line: (line.source_index, line.bbox[1], line.bbox[0]),
    )


def _block_bbox_to_page_bbox(value: Any, page_size: tuple[float, float]) -> BBox | None:
    """把 model-list 的归一化 bbox 转回页面 point，同时兼容已是绝对坐标的内部输入。"""

    bbox = _coerce_bbox(value)
    if bbox is None:
        return None
    if all(0.0 <= coordinate <= 1.0 for coordinate in bbox):
        return (
            bbox[0] * page_size[0],
            bbox[1] * page_size[1],
            bbox[2] * page_size[0],
            bbox[3] * page_size[1],
        )
    return bbox


def _bbox_overlap_ratio(first: BBox, second: BBox) -> float:
    """返回 first 面积中落入 second 的比例。"""

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = max(0.01, (first[2] - first[0]) * (first[3] - first[1]))
    return intersection_width * intersection_height / first_area


def _line_block_score(line_bbox: BBox, block_bbox: BBox) -> tuple[float, float, float]:
    """计算文本行归属 block 的中心包含、重叠率与紧致度评分。"""

    center_x = (line_bbox[0] + line_bbox[2]) / 2
    center_y = (line_bbox[1] + line_bbox[3]) / 2
    center_inside = float(block_bbox[0] <= center_x <= block_bbox[2] and block_bbox[1] <= center_y <= block_bbox[3])
    overlap_ratio = _bbox_overlap_ratio(line_bbox, block_bbox)
    block_area = (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])
    return center_inside, overlap_ratio, -block_area


def _assign_lines_to_blocks(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextEvidenceLine],
    page_size: tuple[float, float],
) -> dict[int, list[PDFTextEvidenceLine]]:
    """把每个视觉文本行唯一分配给最匹配的自然语言 block。"""

    target_bboxes = {
        block_index: block_bbox
        for block_index, block in enumerate(blocks)
        if block.get("type") in PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES
        and isinstance(block.get("content"), str)
        and (block_bbox := _block_bbox_to_page_bbox(block.get("bbox"), page_size)) is not None
    }
    assignments: dict[int, list[PDFTextEvidenceLine]] = {}
    for line in lines:
        matches = [
            (block_index, _line_block_score(line.bbox, block_bbox))
            for block_index, block_bbox in target_bboxes.items()
            if (
                block_bbox[0] <= (line.bbox[0] + line.bbox[2]) / 2 <= block_bbox[2]
                and block_bbox[1] <= (line.bbox[1] + line.bbox[3]) / 2 <= block_bbox[3]
            )
            or _bbox_overlap_ratio(line.bbox, block_bbox) >= 0.5
        ]
        if not matches:
            continue
        block_index, _score = max(matches, key=lambda item: (*item[1], -item[0]))
        assignments.setdefault(block_index, []).append(line)
    for block_lines in assignments.values():
        block_lines.sort(
            key=lambda line: (
                line.source_index,
                line.bbox[1],
                line.bbox[0],
            )
        )
    return assignments


def _filter_line_styles_for_block(
    lines: Sequence[PDFTextStyleLine],
    block_type: Any,
) -> list[PDFTextStyleLine]:
    """按目标 block 类型过滤样式区间，同时保留无样式物理行用于顺序对齐。"""

    output: list[PDFTextStyleLine] = []
    for line in lines:
        filtered_ranges: list[PDFTextStyleRange] = []
        for style_range in line.style_ranges:
            styles = _canonical_styles(
                style
                for style in style_range.styles
                if block_type
                in _PDF_TEXT_STYLE_TARGET_BLOCK_TYPES.get(
                    style,
                    frozenset(),
                )
            )
            if styles:
                filtered_ranges.append(
                    PDFTextStyleRange(
                        style_range.start,
                        style_range.end,
                        styles,
                    )
                )
        output.append(
            PDFTextStyleLine(
                bbox=line.bbox,
                text=line.text,
                style_ranges=tuple(filtered_ranges),
                source_index=line.source_index,
            )
        )
    return output


def _project_content_chars(content: str) -> list[_ProjectedChar]:
    """把原始文字投影为忽略空白和圆括号公式的可比较字符。"""

    projected: list[_ProjectedChar] = []
    pending_formula_gap = False
    cursor = 0
    while cursor < len(content):
        if content.startswith(r"\(", cursor):
            formula_end = content.find(r"\)", cursor + 2)
            if formula_end >= 0:
                cursor = formula_end + 2
                pending_formula_gap = True
                continue
        raw_char = content[cursor]
        fragment = _normalize_match_fragment(raw_char)
        for fragment_index, value in enumerate(fragment):
            projected.append(
                _ProjectedChar(
                    value=value,
                    raw_start=cursor,
                    raw_end=cursor + 1,
                    existing_styles=frozenset(),
                    formula_gap_before=pending_formula_gap and fragment_index == 0,
                    inside_hyperlink=False,
                )
            )
        if fragment:
            pending_formula_gap = False
        cursor += 1
    return projected


def _all_occurrences(content: str, target: str, start: int) -> list[int]:
    """返回 target 在 content 指定位置后的全部精确匹配起点。"""

    output: list[int] = []
    cursor = start
    while target and (match := content.find(target, cursor)) >= 0:
        output.append(match)
        cursor = match + 1
    return output


def _resolve_fallback_occurrence(
    content: str,
    line: PDFTextStyleLine,
    style_range: PDFTextStyleRange,
    start: int,
) -> int | None:
    """在整行无法对齐时，用唯一样式片段及两侧精确上下文选择位置。"""

    target = line.text[style_range.start : style_range.end]
    occurrences = _all_occurrences(content, target, start)
    if not occurrences:
        return None
    left_context = line.text[max(0, style_range.start - 12) : style_range.start]
    right_context = line.text[style_range.end : style_range.end + 12]
    scored = [
        (
            int(bool(left_context) and content[max(0, position - len(left_context)) : position] == left_context)
            + int(
                bool(right_context)
                and content[position + len(target) : position + len(target) + len(right_context)] == right_context
            ),
            position,
        )
        for position in occurrences
    ]
    has_geometric_style = bool(_PDF_GEOMETRIC_TEXT_STYLES.intersection(style_range.styles))
    if len(occurrences) == 1 and (has_geometric_style or len(target) >= 3):
        return occurrences[0]
    best_score = max(score for score, _position in scored)
    best_positions = [position for score, position in scored if score == best_score]
    if has_geometric_style:
        return best_positions[0] if best_score > 0 and len(best_positions) == 1 else None
    required_context_score = int(bool(left_context)) + int(bool(right_context))
    return (
        best_positions[0]
        if required_context_score > 0 and best_score == required_context_score and len(best_positions) == 1
        else None
    )


def _match_line_across_formula_gaps(
    line_text: str,
    projected: Sequence[_ProjectedChar],
    start: int,
) -> _LineProjectionMatch | None:
    """用精确字符序列跨过公式空洞，将一个物理行对齐到 block 文本。"""

    if not line_text or start >= len(projected) or not any(token.formula_gap_before for token in projected[start:]):
        return None
    for projected_start in range(max(0, start), len(projected)):
        first_token = projected[projected_start]
        if not first_token.formula_gap_before and first_token.value != line_text[0]:
            continue
        states: dict[int, tuple[int | None, ...]] = {0: ()}
        for projected_index in range(projected_start, len(projected)):
            token = projected[projected_index]
            next_states: dict[int, tuple[int | None, ...]] = {}
            for source_index, mapping in states.items():
                if source_index >= len(line_text):
                    continue
                if token.formula_gap_before:
                    candidate_source_indices = range(
                        source_index + 1,
                        len(line_text),
                    )
                elif line_text[source_index] == token.value:
                    candidate_source_indices = (source_index,)
                else:
                    continue
                for matched_source_index in candidate_source_indices:
                    if line_text[matched_source_index] != token.value:
                        continue
                    next_source_index = matched_source_index + 1
                    next_mapping = (
                        *mapping,
                        *([None] * (matched_source_index - source_index)),
                        projected_index,
                    )
                    next_states.setdefault(next_source_index, next_mapping)
            complete = next_states.get(len(line_text))
            if complete is not None:
                return _LineProjectionMatch(
                    start=projected_start,
                    end=projected_index + 1,
                    source_to_projected=complete,
                )
            if not next_states:
                break
            states = next_states
    return None


def _ranges_from_line_projection(
    line: PDFTextStyleLine,
    match: _LineProjectionMatch,
) -> list[PDFTextStyleRange]:
    """把物理行样式区间投影为公式字符被跳过后的 block 文本区间。"""

    output: list[PDFTextStyleRange] = []
    for style_range in line.style_ranges:
        current_start: int | None = None
        previous_index: int | None = None
        for projected_index in match.source_to_projected[style_range.start : style_range.end]:
            if projected_index is None:
                if current_start is not None and previous_index is not None:
                    output.append(
                        PDFTextStyleRange(
                            current_start,
                            previous_index + 1,
                            style_range.styles,
                        )
                    )
                current_start = None
                previous_index = None
                continue
            if current_start is not None and previous_index is not None and projected_index != previous_index + 1:
                output.append(
                    PDFTextStyleRange(
                        current_start,
                        previous_index + 1,
                        style_range.styles,
                    )
                )
                current_start = None
            if current_start is None:
                current_start = projected_index
            previous_index = projected_index
        if current_start is not None and previous_index is not None:
            output.append(
                PDFTextStyleRange(
                    current_start,
                    previous_index + 1,
                    style_range.styles,
                )
            )
    return output


def _lines_form_dehyphenated_continuation(
    line: PDFTextEvidenceLine,
    next_line: PDFTextEvidenceLine | None,
) -> bool:
    """判断相邻物理行是否符合正文回填使用的英文断词规则。"""

    return bool(
        next_line is not None
        and next_line.source_index == line.source_index + 1
        and line.text
        and next_line.text
        and is_hyphen_at_line_end(line.text)
        and next_line.text[0].islower()
    )


def _match_line_without_terminal_hyphen(
    projected_text: str,
    line: PDFTextEvidenceLine,
    next_line: PDFTextEvidenceLine | None,
    start: int,
) -> _LineProjectionMatch | None:
    """将 block 已删除的行末断词符映射为空洞，歧义时拒绝匹配。"""

    if not _lines_form_dehyphenated_continuation(line, next_line):
        return None
    candidate = line.text[:-1]
    next_first_char = next_line.text[0] if next_line is not None else ""
    occurrences = [
        position
        for position in _all_occurrences(projected_text, candidate, start)
        if (position + len(candidate) < len(projected_text) and projected_text[position + len(candidate)] == next_first_char)
    ]
    if len(occurrences) != 1:
        logger.debug(f"Skip ambiguous PDF text dehyphenation mapping: line={line.text!r}, occurrences={len(occurrences)}")
        return None
    position = occurrences[0]
    return _LineProjectionMatch(
        start=position,
        end=position + len(candidate),
        source_to_projected=(
            *range(position, position + len(candidate)),
            None,
        ),
    )


def _match_style_ranges(
    projected: Sequence[_ProjectedChar],
    lines: Sequence[PDFTextStyleLine],
) -> list[PDFTextStyleRange]:
    """按物理行顺序把字体与装饰线证据确定性对齐到 block 文本。"""

    projected_text = "".join(token.value for token in projected)
    output: list[PDFTextStyleRange] = []
    cursor = 0
    for line_index, line in enumerate(lines):
        next_line = lines[line_index + 1] if line_index + 1 < len(lines) else None
        line_start = projected_text.find(line.text, cursor)
        if line_start >= 0:
            output.extend(
                PDFTextStyleRange(
                    line_start + style_range.start,
                    line_start + style_range.end,
                    style_range.styles,
                )
                for style_range in line.style_ranges
            )
            cursor = line_start + len(line.text)
            continue
        formula_match = _match_line_across_formula_gaps(
            line.text,
            projected,
            cursor,
        )
        if formula_match is not None:
            output.extend(_ranges_from_line_projection(line, formula_match))
            cursor = formula_match.end
            continue
        dehyphenated_match = _match_line_without_terminal_hyphen(
            projected_text,
            line,
            next_line,
            cursor,
        )
        if dehyphenated_match is not None:
            output.extend(_ranges_from_line_projection(line, dehyphenated_match))
            cursor = dehyphenated_match.end
            continue
        skipped_ranges: list[PDFTextStyleRange] = []
        for style_range in line.style_ranges:
            position = _resolve_fallback_occurrence(
                projected_text,
                line,
                style_range,
                cursor,
            )
            if position is None:
                skipped_ranges.append(style_range)
                continue
            output.append(
                PDFTextStyleRange(
                    position,
                    position + style_range.end - style_range.start,
                    style_range.styles,
                )
            )
            cursor = position + style_range.end - style_range.start
        if skipped_ranges:
            skipped_samples = [
                (
                    line.text[style_range.start : style_range.end],
                    style_range.styles,
                )
                for style_range in skipped_ranges[:3]
            ]
            logger.debug(
                "Skip ambiguous PDF text style mapping: "
                f"line={line.text!r}, skipped={len(skipped_ranges)}, "
                f"samples={skipped_samples!r}"
            )
    return _merge_style_ranges(output)


def _merge_style_ranges(ranges: Sequence[PDFTextStyleRange]) -> list[PDFTextStyleRange]:
    """把重叠样式取并集，并合并相邻且样式集合相同的区间。"""

    events: dict[int, dict[PDFTextStyle, int]] = {}
    for style_range in ranges:
        styles = _canonical_styles(style_range.styles)
        if style_range.start >= style_range.end or not styles:
            continue
        for position, delta in (
            (style_range.start, 1),
            (style_range.end, -1),
        ):
            position_events = events.setdefault(position, {})
            for style in styles:
                position_events[style] = position_events.get(style, 0) + delta

    active_counts: dict[PDFTextStyle, int] = {}
    merged: list[PDFTextStyleRange] = []
    previous_position: int | None = None
    for position in sorted(events):
        active_styles = _canonical_styles(style for style, count in active_counts.items() if count > 0)
        if previous_position is not None and previous_position < position and active_styles:
            if merged and merged[-1].end == previous_position and merged[-1].styles == active_styles:
                merged[-1] = PDFTextStyleRange(
                    merged[-1].start,
                    position,
                    active_styles,
                )
            else:
                merged.append(
                    PDFTextStyleRange(
                        previous_position,
                        position,
                        active_styles,
                    )
                )
        for style, delta in events[position].items():
            active_counts[style] = active_counts.get(style, 0) + delta
        previous_position = position
    return merged


def _resolve_link_fallback_occurrence(
    content: str,
    line: PDFTextLinkLine,
    link_range: PDFTextLinkRange,
    start: int,
) -> int | None:
    """整行无法对齐时，用唯一标签或两侧精确上下文定位链接片段。"""

    target_text = line.text[link_range.start : link_range.end]
    occurrences = _all_occurrences(content, target_text, start)
    if not occurrences:
        return None
    if len(occurrences) == 1:
        return occurrences[0]

    left_context = line.text[max(0, link_range.start - 12) : link_range.start]
    right_context = line.text[link_range.end : link_range.end + 12]
    required_context_score = int(bool(left_context)) + int(bool(right_context))
    if required_context_score == 0:
        return None
    scored = [
        (
            int(bool(left_context) and content[max(0, position - len(left_context)) : position] == left_context)
            + int(
                bool(right_context)
                and content[position + len(target_text) : position + len(target_text) + len(right_context)] == right_context
            ),
            position,
        )
        for position in occurrences
    ]
    best_score = max(score for score, _position in scored)
    best_positions = [position for score, position in scored if score == best_score]
    return best_positions[0] if best_score == required_context_score and len(best_positions) == 1 else None


def _project_link_range_from_line_match(
    link_range: PDFTextLinkRange,
    match: _LineProjectionMatch,
    source_index: int,
) -> list[_MatchedLinkRange]:
    """按行字符投影映射链接区间，并在缺失字符处安全分段。"""

    output: list[_MatchedLinkRange] = []
    current_start: int | None = None
    previous_index: int | None = None
    for projected_index in match.source_to_projected[link_range.start : link_range.end]:
        if projected_index is None:
            if current_start is not None and previous_index is not None:
                output.append(
                    _MatchedLinkRange(
                        current_start,
                        previous_index + 1,
                        link_range.target,
                        source_index,
                    )
                )
            current_start = None
            previous_index = None
            continue
        if current_start is not None and previous_index is not None and projected_index != previous_index + 1:
            output.append(
                _MatchedLinkRange(
                    current_start,
                    previous_index + 1,
                    link_range.target,
                    source_index,
                )
            )
            current_start = None
        if current_start is None:
            current_start = projected_index
        previous_index = projected_index
    if current_start is not None and previous_index is not None:
        output.append(
            _MatchedLinkRange(
                current_start,
                previous_index + 1,
                link_range.target,
                source_index,
            )
        )
    return output


def _link_lines_form_dehyphenated_continuation(
    line: PDFTextLinkLine,
    next_line: PDFTextLinkLine | None,
) -> bool:
    """判断相邻同 href 链接行是否符合文本回填的英文断词规则。"""

    if not _lines_form_dehyphenated_continuation(line, next_line):
        return False
    tail_targets = {link_range.target for link_range in line.link_ranges if link_range.start < link_range.end == len(line.text)}
    head_targets = {link_range.target for link_range in next_line.link_ranges if link_range.start == 0 < link_range.end}
    return bool(tail_targets.intersection(head_targets))


def _match_link_line_without_terminal_hyphen(
    projected_text: str,
    line: PDFTextLinkLine,
    next_line: PDFTextLinkLine | None,
    start: int,
) -> _LineProjectionMatch | None:
    """在严格跨行条件下将已被 block 回填删除的行末断词符投影为空洞。"""

    if not _link_lines_form_dehyphenated_continuation(line, next_line):
        return None
    return _match_line_without_terminal_hyphen(
        projected_text,
        line,
        next_line,
        start,
    )


def _merge_matched_link_ranges(
    ranges: Sequence[_MatchedLinkRange],
) -> list[_MatchedLinkRange]:
    """删除不同目标重叠区，并合并同一物理行内的同目标相邻区间。"""

    valid_ranges = [link_range for link_range in ranges if link_range.start < link_range.end and link_range.target]
    if not valid_ranges:
        return []
    boundaries = sorted({position for link_range in valid_ranges for position in (link_range.start, link_range.end)})
    merged: list[_MatchedLinkRange] = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [link_range for link_range in valid_ranges if link_range.start < end and link_range.end > start]
        targets = {link_range.target for link_range in active}
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        source_index = min(link_range.source_index for link_range in active if link_range.target == target)
        if merged and merged[-1].end == start and merged[-1].target == target and merged[-1].source_index == source_index:
            merged[-1] = _MatchedLinkRange(
                merged[-1].start,
                end,
                target,
                source_index,
            )
        else:
            merged.append(
                _MatchedLinkRange(
                    start,
                    end,
                    target,
                    source_index,
                )
            )
    return merged


def _match_link_ranges(
    projected: Sequence[_ProjectedChar],
    lines: Sequence[PDFTextLinkLine],
) -> list[_MatchedLinkRange]:
    """按物理行顺序把 Link 几何证据确定性对齐到 block 文本。"""

    projected_text = "".join(token.value for token in projected)
    output: list[_MatchedLinkRange] = []
    cursor = 0
    for line_index, line in enumerate(lines):
        next_line = lines[line_index + 1] if line_index + 1 < len(lines) else None
        line_start = projected_text.find(line.text, cursor)
        if line_start >= 0:
            output.extend(
                _MatchedLinkRange(
                    line_start + link_range.start,
                    line_start + link_range.end,
                    link_range.target,
                    line.source_index,
                )
                for link_range in line.link_ranges
            )
            cursor = line_start + len(line.text)
            continue
        formula_match = _match_line_across_formula_gaps(
            line.text,
            projected,
            cursor,
        )
        if formula_match is not None:
            for link_range in line.link_ranges:
                output.extend(
                    _project_link_range_from_line_match(
                        link_range,
                        formula_match,
                        line.source_index,
                    )
                )
            cursor = formula_match.end
            continue

        dehyphenated_match = _match_link_line_without_terminal_hyphen(
            projected_text,
            line,
            next_line,
            cursor,
        )
        if dehyphenated_match is not None:
            for link_range in line.link_ranges:
                output.extend(
                    _project_link_range_from_line_match(
                        link_range,
                        dehyphenated_match,
                        line.source_index,
                    )
                )
            cursor = dehyphenated_match.end
            continue

        skipped_ranges: list[PDFTextLinkRange] = []
        for link_range in line.link_ranges:
            position = _resolve_link_fallback_occurrence(
                projected_text,
                line,
                link_range,
                cursor,
            )
            if position is None:
                skipped_ranges.append(link_range)
                continue
            output.append(
                _MatchedLinkRange(
                    position,
                    position + link_range.end - link_range.start,
                    link_range.target,
                    line.source_index,
                )
            )
            cursor = position + link_range.end - link_range.start
        if skipped_ranges:
            logger.debug(
                "Skip ambiguous PDF hyperlink mapping: "
                f"line={line.text!r}, skipped={len(skipped_ranges)}, "
                f"samples={[(line.text[item.start : item.end], item.target) for item in skipped_ranges[:3]]!r}"
            )
    return _merge_matched_link_ranges(output)


def _append_raw_link_interval(
    intervals: list[_RawLinkInterval],
    start: int | None,
    end: int,
    target: str,
    source_index: int,
) -> None:
    """向结果追加一个合法原字符串链接区间。"""

    if start is not None and start < end and target:
        intervals.append(
            _RawLinkInterval(
                start,
                end,
                target,
                source_index,
            )
        )


def _raw_link_intervals(
    content: str,
    projected: Sequence[_ProjectedChar],
    ranges: Sequence[_MatchedLinkRange],
) -> list[_RawLinkInterval]:
    """把链接区间转换为不跨公式或已有 hyperlink 的原字符串区间。"""

    intervals: list[_RawLinkInterval] = []
    for link_range in ranges:
        current_start: int | None = None
        current_end = 0
        for token in projected[link_range.start : link_range.end]:
            if token.inside_hyperlink or (token.formula_gap_before and current_start is not None):
                _append_raw_link_interval(
                    intervals,
                    current_start,
                    current_end,
                    link_range.target,
                    link_range.source_index,
                )
                current_start = None
            if token.inside_hyperlink:
                continue
            if current_start is None:
                current_start = token.raw_start
                current_end = token.raw_end
                continue
            gap = content[current_end : token.raw_start]
            if token.raw_start <= current_end or not gap or gap.isspace():
                current_end = max(current_end, token.raw_end)
            else:
                _append_raw_link_interval(
                    intervals,
                    current_start,
                    current_end,
                    link_range.target,
                    link_range.source_index,
                )
                current_start = token.raw_start
                current_end = token.raw_end
        _append_raw_link_interval(
            intervals,
            current_start,
            current_end,
            link_range.target,
            link_range.source_index,
        )
    return intervals


def _raw_link_gap_is_boundary_only(gap: str) -> bool:
    """判断两个跨行链接片段之间是否只包含空白或非正文边界符号。"""

    if not gap:
        return True
    if r"\(" in gap or r"\)" in gap:
        return False
    return not any(char.isalnum() for char in html.unescape(gap))


def _merge_raw_link_intervals(
    content: str,
    intervals: Sequence[_RawLinkInterval],
) -> list[_RawLinkInterval]:
    """合并相邻物理行中同 href 的首尾链接片段，不跨越正文或公式。"""

    merged: list[_RawLinkInterval] = []
    for interval in sorted(
        intervals,
        key=lambda item: (item.start, item.end, item.source_index, item.target),
    ):
        if interval.start >= interval.end or not interval.target:
            continue
        if (
            merged
            and merged[-1].target == interval.target
            and interval.source_index == merged[-1].source_index + 1
            and interval.start >= merged[-1].end
            and _raw_link_gap_is_boundary_only(content[merged[-1].end : interval.start])
        ):
            merged[-1] = _RawLinkInterval(
                merged[-1].start,
                interval.end,
                interval.target,
                interval.source_index,
            )
        else:
            merged.append(interval)
    return merged


def apply_pdf_text_links(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextLinkLine],
    page_size: tuple[float, float],
) -> None:
    """把页面 Link 几何证据写入自然语言 block，歧义时保持原文。"""

    assignments = _assign_lines_to_blocks(blocks, lines, page_size)
    for block_index, block_lines in assignments.items():
        content = blocks[block_index].get("content")
        if not isinstance(content, str) or not content:
            continue
        projected = _project_content_chars(content)
        link_ranges = _match_link_ranges(projected, block_lines)
        if not link_ranges:
            continue
        intervals = _raw_link_intervals(content, projected, link_ranges)
        if intervals:
            intervals = _merge_raw_link_intervals(content, intervals)
            blocks[block_index][_PDF_LINK_INTERVALS_KEY] = intervals


def _append_raw_style_interval(
    intervals: list[_RawStyleInterval],
    start: int | None,
    end: int,
    styles: tuple[PDFTextStyle, ...],
) -> None:
    """向结果追加一个合法原字符串样式区间。"""

    if start is not None and start < end and styles:
        intervals.append(_RawStyleInterval(start, end, styles))


def _merge_raw_style_intervals(
    content: str,
    intervals: Sequence[_RawStyleInterval],
) -> list[_RawStyleInterval]:
    """合并原字符串中相邻且样式一致、仅由普通空白隔开的区间。"""

    merged: list[_RawStyleInterval] = []
    for interval in sorted(intervals, key=lambda item: (item.start, item.end, item.styles)):
        if interval.start >= interval.end or not interval.styles:
            continue
        if (
            merged
            and merged[-1].styles == interval.styles
            and (interval.start <= merged[-1].end or content[merged[-1].end : interval.start].isspace())
        ):
            merged[-1] = _RawStyleInterval(
                merged[-1].start,
                max(merged[-1].end, interval.end),
                merged[-1].styles,
            )
        else:
            merged.append(interval)
    return merged


def _raw_style_intervals(
    content: str,
    projected: Sequence[_ProjectedChar],
    ranges: Sequence[PDFTextStyleRange],
) -> list[_RawStyleInterval]:
    """把样式区间转换为不跨公式或已有行内标签的原字符串区间。"""

    intervals: list[_RawStyleInterval] = []
    for style_range in ranges:
        current_start: int | None = None
        current_end = 0
        current_styles: tuple[PDFTextStyle, ...] = ()
        for token in projected[style_range.start : style_range.end]:
            # Link 注解已提供语义，链接范围内的几何下划线不重复输出为文本样式。
            missing_styles = _canonical_styles(
                style
                for style in style_range.styles
                if style not in token.existing_styles and not (style == "underline" and token.inside_hyperlink)
            )
            if not missing_styles:
                _append_raw_style_interval(
                    intervals,
                    current_start,
                    current_end,
                    current_styles,
                )
                current_start = None
                current_styles = ()
                continue
            if current_start is None:
                current_start = token.raw_start
                current_end = token.raw_end
                current_styles = missing_styles
                continue
            gap = content[current_end : token.raw_start]
            if missing_styles == current_styles and (token.raw_start <= current_end or not gap or gap.isspace()):
                current_end = max(current_end, token.raw_end)
            else:
                _append_raw_style_interval(
                    intervals,
                    current_start,
                    current_end,
                    current_styles,
                )
                current_start = token.raw_start
                current_end = token.raw_end
                current_styles = missing_styles
        _append_raw_style_interval(
            intervals,
            current_start,
            current_end,
            current_styles,
        )
    return _merge_raw_style_intervals(content, intervals)


def apply_pdf_text_styles(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextStyleLine],
    page_size: tuple[float, float],
) -> None:
    """把页面字体和装饰线证据写入自然语言 block，歧义时保持原文。"""

    assignments = _assign_lines_to_blocks(blocks, lines, page_size)
    for block_index, block_lines in assignments.items():
        block = blocks[block_index]
        block_lines = _filter_line_styles_for_block(
            block_lines,
            block.get("type"),
        )
        content = block.get("content")
        if not isinstance(content, str) or not content or not any(line.style_ranges for line in block_lines):
            continue
        projected = _project_content_chars(content)
        style_ranges = _match_style_ranges(projected, block_lines)
        if not style_ranges:
            continue
        intervals = _raw_style_intervals(content, projected, style_ranges)
        if intervals:
            block[_PDF_STYLE_INTERVALS_KEY] = intervals


def materialize_pdf_inline_spans(blocks: list[dict[str, Any]]) -> None:
    """把 PDF 原文、样式区间、链接区间和行内公式一次性物化为 Span。"""
    formula_pattern = re.compile(r"\\\((?P<latex>.*?)\\\)", re.DOTALL)
    for block in blocks:
        if block.get("type") not in _PDF_INLINE_SPAN_BLOCK_TYPES:
            continue
        content = block.get("content")
        link_intervals = block.pop(_PDF_LINK_INTERVALS_KEY, [])
        style_intervals = block.pop(_PDF_STYLE_INTERVALS_KEY, [])
        if not isinstance(content, str):
            continue
        formulas = list(formula_pattern.finditer(content))
        boundaries = {0, len(content)}
        for interval in [*link_intervals, *style_intervals]:
            boundaries.update((interval.start, interval.end))
        for formula in formulas:
            boundaries.update((formula.start(), formula.end()))
        ordered = sorted(value for value in boundaries if 0 <= value <= len(content))
        spans: list[dict[str, Any]] = []
        for start, end in zip(ordered, ordered[1:]):
            if start >= end:
                continue
            formula = next((item for item in formulas if item.start() == start and item.end() == end), None)
            if formula is not None:
                append_equation_span(spans, formula.group("latex"))
                continue
            text = content[start:end]
            if not text:
                continue
            styles = _canonical_styles(
                style
                for interval in style_intervals
                if interval.start <= start and end <= interval.end
                for style in interval.styles
            )
            link = next(
                (interval for interval in link_intervals if interval.start <= start and end <= interval.end),
                None,
            )
            if link is None:
                if styles and text.strip():
                    leading_length = len(text) - len(text.lstrip())
                    trailing_length = len(text) - len(text.rstrip())
                    core_end = len(text) - trailing_length if trailing_length else len(text)
                    append_text_span(spans, text[:leading_length])
                    append_text_span(spans, text[leading_length:core_end], styles)
                    append_text_span(spans, text[core_end:])
                else:
                    append_text_span(spans, text, styles)
                continue
            children: list[dict[str, Any]] = []
            append_text_span(children, text, (style for style in styles if style != "underline"))
            if spans and spans[-1].get("type") == "hyperlink" and spans[-1].get("url") == link.target:
                existing = spans[-1].get("content")
                if isinstance(existing, list):
                    extend_inline_spans(existing, children)
                    continue
            append_hyperlink_span(spans, children, link.target)
        block["content"] = normalize_span_dicts(spans)


__all__ = [
    "PDF_FONT_FORCE_BOLD_FLAG",
    "PDF_FONT_ITALIC_FLAG",
    "PDFTextLinkLine",
    "PDFTextLinkRange",
    "PDFTextStyle",
    "PDFTextStyleLine",
    "PDFTextStyleRange",
    "apply_pdf_text_links",
    "apply_pdf_text_styles",
    "detect_pdf_text_link_lines",
    "detect_pdf_text_style_lines",
    "materialize_pdf_inline_spans",
]
