# Copyright (c) Opendatalab. All rights reserved.
"""从字体、绘图线及链接注解提取原生行内证据。"""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

from .....types import BBox
from ..document import PDFLinkAnnotation
from .common import (
    _bbox_intersection_area,
    _canonical_styles,
    _coerce_bbox,
    _normalize_match_fragment,
    _ordered_line_chars,
    _style_line_reading_order_key,
)
from .types import (
    _PDF_BOLD_FONT_NAME_RE,
    _PDF_FONT_SUBSET_PREFIX_RE,
    _PDF_LIST_MARKER_CHARS,
    _PDF_TEXT_DECORATION_ORDER,
    PDF_BOLD_MIN_COMPARABLE_CHAR_COUNT,
    PDF_BOLD_MIN_WEIGHT,
    PDF_FONT_FORCE_BOLD_FLAG,
    PDF_LINK_CHAR_OVERLAP_THRESHOLD,
    STRIKETHROUGH_CENTER_TOLERANCE_HEIGHT_RATIO,
    TEXT_DECORATION_ENDPOINT_TOLERANCE_HEIGHT_RATIO,
    TEXT_DECORATION_MAX_WIDTH_HEIGHT_RATIO,
    TEXT_DECORATION_MIN_LENGTH_HEIGHT_RATIO,
    TEXT_DECORATION_MIN_TEXT_COVERAGE_RATIO,
    UNDERLINE_BOTTOM_TOLERANCE_HEIGHT_RATIO,
    UNDERLINE_FRACTION_MAX_GAP_HEIGHT_RATIO,
    UNDERLINE_FRACTION_MIN_LOWER_LINE_COVERAGE,
    PDFTextDecoration,
    PDFTextLinkLine,
    PDFTextLinkRange,
    PDFTextStyle,
    PDFTextStyleLine,
    PDFTextStyleRange,
    _DrawingMatch,
    _LineCandidate,
    _VisibleChar,
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


__all__ = [
    "_pdf_font_metadata",
    "_char_font_styles",
    "_has_list_marker_separator",
    "_filter_pdf_bold_runs",
    "_build_line_candidate",
    "_drawing_match_for_line",
    "_merge_source_ranges",
    "_line_style_payload",
    "_build_line_geometry_grids",
    "_is_fraction_bar_candidate",
    "detect_pdf_text_style_lines",
    "_link_region_hits_char",
    "_link_targets_for_char",
    "_compact_link_ranges",
    "_build_link_line_payload",
    "detect_pdf_text_link_lines",
]
