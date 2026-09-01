# Copyright (c) Opendatalab. All rights reserved.

"""将 PDF 原生字符重建为带排版信息的视觉文本行。"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Literal, Mapping, Sequence

from pdftext.schema import Char

from ....types import BBox
from .document import PDFDocument

from .models import (
    _AxisLine,
    _LineItem,
)
from .geometry import (
    _bbox_center_y,
    _bbox_union,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _horizontal_bbox_gap,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
)


_PDF_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_PDF_LINE_END_SOFT_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])[\x02\u00ad](?=[\t ]*(?:\n|$))")
_INLINE_REFERENCE_MARKER_RE = re.compile(
    r"^[\[（(［]\s*\d{1,4}\s*[\]）)］]$",
)
# Unicode Zs 空格在 model_list 中只承担分词作用，统一成可互操作的 ASCII 空格。
_PDF_SEPARATOR_SPACE_CHARS = "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
_PDF_UNICODE_TEXT_TRANSLATION = str.maketrans(
    {
        **dict.fromkeys(_PDF_SEPARATOR_SPACE_CHARS, " "),
        "\u0085": "\n",
        "\u2028": "\n",
        "\u2029": "\n",
        "\u200b": None,
        "\u2060": None,
        "\ufeff": None,
    }
)
_PDFTEXT_ROTATION_SPLIT_THRESHOLD_DEGREES = 44.9
_PDFTEXT_LINE_ANGLE_TOLERANCE_DEGREES = 0.1
_SUPPORTED_PDFTEXT_LINE_ANGLES = (0.0, 90.0, 270.0)
_PDFTEXT_SHEARED_HORIZONTAL_MAX_ANGLE_DEGREES = 30.0
_PDFTEXT_HORIZONTAL_BASELINE_MAX_ANGLE_DEGREES = 2.0
_PDFTEXT_HORIZONTAL_BASELINE_MAX_DISPERSION_RATIO = 0.75
_PDFTEXT_FORMULA_OPERATOR_CHARS = frozenset("=∑∫√±×÷")


def _build_native_line_items(
    pdf_lines: Sequence[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    page_rotation: int = 0,
    supported_angles: Sequence[float] = _SUPPORTED_PDFTEXT_LINE_ANGLES,
) -> list[_LineItem]:
    """按指定视觉方向将 pdftext 粗行精修成字符间隙分隔的视觉 run。"""

    normal_items: list[_LineItem] = []
    formula_items: list[_LineItem] = []
    supported_lines: list[tuple[dict[str, Any], int, bool]] = []
    for pdf_line in pdf_lines:
        for child_line in _split_pdftext_line_by_rotation(pdf_line):
            visual_angle = _resolve_pdftext_line_angle(
                child_line,
                page_rotation=page_rotation,
                supported_angles=supported_angles,
            )
            formula_candidate_only = False
            if visual_angle is None:
                visual_angle = _resolve_pdftext_formula_candidate_angle(
                    child_line,
                    page_rotation=page_rotation,
                    supported_angles=supported_angles,
                )
                formula_candidate_only = visual_angle is not None
            if visual_angle is not None:
                supported_lines.append((child_line, visual_angle, formula_candidate_only))
    for visual_row_id, (pdf_line, visual_angle, formula_candidate_only) in enumerate(supported_lines):
        bbox = _clip_bbox(_coerce_bbox(pdf_line.get("bbox")), page_size)
        if bbox is None:
            continue
        spans = pdf_line.get("spans") or []
        chars = [char for span in spans for char in (span.get("chars") or []) if isinstance(char, dict)]
        coarse_item = _LineItem(
            text="".join(str(span.get("text") or "") for span in spans),
            bbox=bbox,
            angle=visual_angle,
            source_index=-1,
            chars=chars,
            visual_row_id=visual_row_id,
            formula_candidate_only=formula_candidate_only,
        )
        target = formula_items if formula_candidate_only else normal_items
        target.extend(_split_native_visual_runs(coarse_item, page_size))

    stable_items = _merge_native_inline_scripts(normal_items, page_size)
    for source_index, item in enumerate(stable_items):
        # source_index 必须在页内唯一，表格投影和失败回滚都依赖该精确成员标识。
        item.source_index = source_index
    formula_items = _merge_native_inline_scripts(formula_items, page_size)
    next_source_index = len(stable_items)
    for item in formula_items:
        item.source_index = next_source_index
        next_source_index += 1
    output = [*stable_items, *formula_items]
    output.sort(key=lambda item: (item.visual_row_id if item.visual_row_id is not None else math.inf, item.run_index))
    return output


def _pdftext_angle_degrees(value: Any) -> float:
    """把 pdftext 弧度方向转换为 0 到 360 度，非法值按 0 度处理。"""

    try:
        angle_radians = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(angle_radians):
        return 0.0
    return math.degrees(angle_radians) % 360.0


def _circular_angle_distance(first: float, second: float) -> float:
    """返回两个方向之间不超过 180 度的最短圆周角差。"""

    return abs((first - second + 180.0) % 360.0 - 180.0)


def _span_has_visible_text(span: dict[str, Any]) -> bool:
    """判断 span 是否包含可见的非空白字符，换行与占位空格不参与方向拆分。"""

    return any(char.isprintable() and not char.isspace() for char in str(span.get("text") or ""))


def _build_pdftext_child_line(
    pdf_line: dict[str, Any],
    spans: list[dict[str, Any]],
    angle_degrees: float,
) -> dict[str, Any]:
    """使用同方向 span 重建子行，并收缩原粗行被异向内容扩大的 bbox。"""

    span_bboxes = [bbox for span in spans if (bbox := _coerce_bbox(span.get("bbox"))) is not None]
    child_line = dict(pdf_line)
    child_line["spans"] = spans
    child_line["bbox"] = _bbox_union_many(span_bboxes) if span_bboxes else pdf_line.get("bbox")
    child_line["rotation"] = math.radians(angle_degrees)
    return child_line


def _split_pdftext_line_by_rotation(pdf_line: dict[str, Any]) -> list[dict[str, Any]]:
    """按 45 度边界拆开 pdftext 误合并的异向 span，并保留小角度仿斜体。"""

    spans = [span for span in (pdf_line.get("spans") or []) if isinstance(span, dict)]
    if not spans:
        return [pdf_line]

    output: list[dict[str, Any]] = []
    current_spans: list[dict[str, Any]] = []
    current_angle = _pdftext_angle_degrees(pdf_line.get("rotation"))
    for span in spans:
        span_angle = _pdftext_angle_degrees(span.get("rotation"))
        if (
            current_spans
            and _span_has_visible_text(span)
            and _circular_angle_distance(span_angle, current_angle) >= _PDFTEXT_ROTATION_SPLIT_THRESHOLD_DEGREES
        ):
            output.append(_build_pdftext_child_line(pdf_line, current_spans, current_angle))
            current_spans = []
            current_angle = span_angle
        current_spans.append(span)

    if current_spans:
        output.append(_build_pdftext_child_line(pdf_line, current_spans, current_angle))
    return output


def _is_supported_pdftext_line_rotation(
    value: Any,
    *,
    page_rotation: int,
    supported_angles: Sequence[float] = _SUPPORTED_PDFTEXT_LINE_ANGLES,
) -> bool:
    """应用页面旋转后按调用方白名单筛选视觉文字方向。"""

    visual_angle = (_pdftext_angle_degrees(value) + int(page_rotation or 0)) % 360.0
    return any(
        _circular_angle_distance(visual_angle, supported_angle) <= _PDFTEXT_LINE_ANGLE_TOLERANCE_DEGREES
        for supported_angle in supported_angles
    )


def _resolve_pdftext_line_angle(
    pdf_line: dict[str, Any],
    *,
    page_rotation: int,
    supported_angles: Sequence[float] = _SUPPORTED_PDFTEXT_LINE_ANGLES,
) -> int | None:
    """解析视觉文字方向，并用字符基线纠正字体 shear 造成的伪斜向行。"""

    visual_angle = (_pdftext_angle_degrees(pdf_line.get("rotation")) + int(page_rotation or 0)) % 360.0
    for supported_angle in supported_angles:
        if _circular_angle_distance(visual_angle, supported_angle) <= _PDFTEXT_LINE_ANGLE_TOLERANCE_DEGREES:
            return int(supported_angle)
    supports_horizontal = any(
        _circular_angle_distance(0.0, supported_angle) <= _PDFTEXT_LINE_ANGLE_TOLERANCE_DEGREES
        for supported_angle in supported_angles
    )
    if (
        supports_horizontal
        and _circular_angle_distance(visual_angle, 0.0) <= _PDFTEXT_SHEARED_HORIZONTAL_MAX_ANGLE_DEGREES
        and _pdftext_line_has_horizontal_char_baseline(pdf_line)
    ):
        return 0
    return None


def _resolve_pdftext_formula_candidate_angle(
    pdf_line: dict[str, Any],
    *,
    page_rotation: int,
    supported_angles: Sequence[float],
) -> int | None:
    """保留小角度字体矩阵下的公式专用粗行，未被公式认领时不回流正文。"""
    visual_angle = (_pdftext_angle_degrees(pdf_line.get("rotation")) + int(page_rotation or 0)) % 360.0
    nearest = min(supported_angles, key=lambda angle: _circular_angle_distance(visual_angle, angle))
    if _circular_angle_distance(visual_angle, nearest) > _PDFTEXT_SHEARED_HORIZONTAL_MAX_ANGLE_DEGREES:
        return None
    spans = [span for span in (pdf_line.get("spans") or []) if isinstance(span, dict)]
    compact_text = "".join(str(span.get("text") or "") for span in spans)
    compact_text = "".join(char for char in compact_text if char.isprintable() and not char.isspace())
    if not compact_text:
        return None
    has_script_flag = any(span.get("superscript") is True or span.get("subscript") is True for span in spans)
    has_math_operator = any(char in _PDFTEXT_FORMULA_OPERATOR_CHARS for char in compact_text)
    font_sizes = [
        float(size)
        for span in spans
        if isinstance(span.get("font"), dict) and isinstance((size := span["font"].get("size")), (int, float)) and size > 0
    ]
    has_mixed_sizes = bool(font_sizes) and max(font_sizes) >= 1.35 * min(font_sizes)
    is_compact_identifier = len(compact_text) <= 3 and all(char.isalnum() for char in compact_text)
    return int(nearest) if has_script_flag or has_math_operator or has_mixed_sizes or is_compact_identifier else None


def _pdftext_line_has_horizontal_char_baseline(
    pdf_line: dict[str, Any],
) -> bool:
    """用字符中心的水平基线确认小角度只来自仿斜体变换，而非真实旋转。"""

    visible_bboxes: list[BBox] = []
    for span in pdf_line.get("spans") or []:
        if not isinstance(span, dict):
            continue
        for char in span.get("chars") or []:
            if not isinstance(char, dict):
                continue
            raw_char = str(char.get("char") or "")
            bbox = _coerce_bbox(char.get("bbox"))
            if bbox is not None and raw_char.isprintable() and not raw_char.isspace():
                visible_bboxes.append(bbox)
    if len(visible_bboxes) < 4:
        return False

    line_bbox = _bbox_union_many(visible_bboxes)
    line_width = line_bbox[2] - line_bbox[0]
    line_height = line_bbox[3] - line_bbox[1]
    glyph_heights = [bbox[3] - bbox[1] for bbox in visible_bboxes]
    median_height = statistics.median(glyph_heights)
    if line_height <= 0 or median_height <= 0 or line_width / line_height < 3.0:
        return False

    ordered = sorted(
        visible_bboxes,
        key=lambda bbox: ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
    )
    first_center = (
        (ordered[0][0] + ordered[0][2]) / 2.0,
        (ordered[0][1] + ordered[0][3]) / 2.0,
    )
    last_center = (
        (ordered[-1][0] + ordered[-1][2]) / 2.0,
        (ordered[-1][1] + ordered[-1][3]) / 2.0,
    )
    baseline_width = last_center[0] - first_center[0]
    if baseline_width <= 0:
        return False
    baseline_angle = abs(math.degrees(math.atan2(last_center[1] - first_center[1], baseline_width)))
    centers_y = [(bbox[1] + bbox[3]) / 2.0 for bbox in ordered]
    return (
        baseline_angle <= _PDFTEXT_HORIZONTAL_BASELINE_MAX_ANGLE_DEGREES
        and max(centers_y) - min(centers_y) <= _PDFTEXT_HORIZONTAL_BASELINE_MAX_DISPERSION_RATIO * median_height
    )


def _split_native_visual_runs(
    line: _LineItem,
    page_size: tuple[float, float],
    *,
    visual_bboxes: Mapping[int, BBox] | None = None,
    preserve_vertical_bbox: BBox | None = None,
) -> list[_LineItem]:
    """保留字符源顺序与空白信息，并按 canonical 字符框拆分远距视觉 run。"""

    tokens: list[tuple[Char, str, BBox | None, BBox | None]] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if raw_char in {"\r", "\n"}:
            continue
        char_idx = char.get("char_idx")
        visual_bbox = (
            visual_bboxes.get(char_idx)
            if visual_bboxes is not None
            and isinstance(char_idx, int)
            and raw_char.isprintable()
            and not raw_char.isspace()
            else None
        )
        bbox = _clip_bbox(
            _coerce_bbox(visual_bbox) or _coerce_bbox(char.get("bbox")),
            page_size,
        )
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle) if bbox is not None else None
        tokens.append((char, raw_char, bbox, local_bbox))

    visible_indices = [
        index
        for index, (_char, raw_char, _bbox, local_bbox) in enumerate(tokens)
        if raw_char.isprintable() and not raw_char.isspace() and local_bbox is not None
    ]
    if not visible_indices:
        text = _normalize_native_run_text(line.text)
        if not text:
            return []
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        line.text = text
        line.effective_height = max(0.1, local_bbox[3] - local_bbox[1])
        return [line]

    glyph_widths = [
        max(0.1, tokens[index][3][2] - tokens[index][3][0])  # type: ignore[index]
        for index in visible_indices
    ]
    median_glyph_width = statistics.median(glyph_widths)
    local_page_width = page_size[1] if line.angle in {90, 270} else page_size[0]
    hard_gap_threshold = max(15.0, 3.0 * median_glyph_width, 0.02 * local_page_width)
    adjacent_gaps: list[float] = []
    for previous, current in zip(visible_indices, visible_indices[1:]):
        previous_bbox = tokens[previous][3]
        current_bbox = tokens[current][3]
        if previous_bbox is None or current_bbox is None:
            continue
        gap = _horizontal_bbox_gap(previous_bbox, current_bbox)
        if gap < hard_gap_threshold:
            adjacent_gaps.append(gap)
    # 少于三个相邻样本无法可靠代表“常规”字距；零间隙也是真实的紧排字距，
    # 必须纳入统计，避免唯一的 15pt cell gap 反过来抬高软拆阈值。
    median_regular_gap = statistics.median(adjacent_gaps) if len(adjacent_gaps) >= 3 else 0.0

    split_indices: list[int] = []
    for previous, current in zip(visible_indices, visible_indices[1:]):
        previous_bbox = tokens[previous][3]
        current_bbox = tokens[current][3]
        if previous_bbox is None or current_bbox is None:
            continue
        gap = _horizontal_bbox_gap(previous_bbox, current_bbox)
        has_source_whitespace = any(tokens[index][1].isspace() for index in range(previous + 1, current))
        soft_gap_threshold = max(
            8.0,
            2.2 * median_glyph_width,
            3.0 * median_regular_gap,
        )
        if gap >= hard_gap_threshold or (has_source_whitespace and gap >= soft_gap_threshold):
            split_indices.append(current)

    ranges: list[tuple[int, int]] = []
    start = 0
    for split_index in split_indices:
        ranges.append((start, split_index))
        start = split_index
    ranges.append((start, len(tokens)))

    output: list[_LineItem] = []
    for run_index, (start, end) in enumerate(ranges):
        run_tokens = tokens[start:end]
        run_text = _normalize_native_run_text("".join(token[1] for token in run_tokens))
        run_bboxes = [
            token[2] for token in run_tokens if token[2] is not None and token[1].isprintable() and not token[1].isspace()
        ]
        if not run_text or not run_bboxes:
            continue
        run_bbox = _bbox_union_many(run_bboxes)
        if preserve_vertical_bbox is not None:
            local_run_bbox = _rotate_bbox_to_upright(
                run_bbox,
                page_size,
                line.angle,
            )
            local_vertical_bbox = _rotate_bbox_to_upright(
                preserve_vertical_bbox,
                page_size,
                line.angle,
            )
            run_bbox = _rotate_bbox_from_upright(
                (
                    local_run_bbox[0],
                    local_vertical_bbox[1],
                    local_run_bbox[2],
                    local_vertical_bbox[3],
                ),
                page_size,
                line.angle,
            )
        run_chars = [token[0] for token in run_tokens]
        run_item = _LineItem(
            text=run_text,
            bbox=run_bbox,
            angle=line.angle,
            source_index=-1,
            chars=run_chars,
            visual_row_id=line.visual_row_id,
            run_index=run_index,
            split_from_row=len(ranges) > 1,
            formula_candidate_only=line.formula_candidate_only,
        )
        _fill_native_typography(run_item, page_size)
        output.append(run_item)
    return output


def _resplit_native_visual_runs(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    visual_bboxes: Mapping[int, BBox],
    *,
    source_index_start: int | None = None,
) -> list[_LineItem]:
    """在字符几何修复后重切跨栏粗行，并保持页内 source identity 唯一。"""

    if not lines or not visual_bboxes:
        return list(lines)
    next_source_index = (
        source_index_start
        if source_index_start is not None
        else max(
            (line.source_index for line in lines),
            default=-1,
        )
        + 1
    )
    output: list[_LineItem] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(
            line.bbox,
            page_size,
            line.angle,
        )
        local_page_height = (
            page_size[0]
            if line.angle in {90, 270}
            else page_size[1]
        )
        local_page_width = (
            page_size[1]
            if line.angle in {90, 270}
            else page_size[0]
        )
        if (
            line.semantic_type is not None
            or local_bbox[2] - local_bbox[0]
            < 0.45 * local_page_width
            or _bbox_center_y(local_bbox) < 0.07 * local_page_height
            or _bbox_center_y(local_bbox) > 0.93 * local_page_height
        ):
            output.append(line)
            continue
        members = _split_native_visual_runs(
            line,
            page_size,
            visual_bboxes=visual_bboxes,
            preserve_vertical_bbox=line.bbox,
        )
        if len(members) <= 1:
            output.append(line)
            continue
        local_members = sorted(
            (
                _rotate_bbox_to_upright(
                    member.bbox,
                    page_size,
                    member.angle,
                )
                for member in members
            ),
            key=lambda bbox: bbox[0],
        )
        column_split = (
            len(local_members) == 2
            and all(
                bbox[2] - bbox[0]
                >= 0.15 * local_page_width
                for bbox in local_members
            )
            and local_members[1][0] - local_members[0][2]
            >= 0.02 * local_page_width
            and local_members[0][2] <= 0.52 * local_page_width
            and local_members[1][0] >= 0.48 * local_page_width
        )
        if not column_split:
            output.append(line)
            continue
        for member_index, member in enumerate(members):
            member.source_index = (
                line.source_index
                if member_index == 0
                else next_source_index
            )
            if member_index > 0:
                next_source_index += 1
            member.visual_row_id = line.visual_row_id
            member.run_index = line.run_index + member_index
            member.source_bbox = member.bbox
            member.baseline = line.baseline
            member.geometry_state = line.geometry_state
            member.geometry_confidence = line.geometry_confidence
            member.split_y_candidate = line.split_y_candidate
            member.em_height = line.em_height or member.effective_height
            member.split_from_row = True
            member.preserve_split_boundary = line.preserve_split_boundary
            member.semantic_type = line.semantic_type
            member.formula_candidate_only = line.formula_candidate_only
            member.style_scale_repaired = line.style_scale_repaired
            output.append(member)
    output.sort(
        key=lambda item: (
            item.visual_row_id
            if item.visual_row_id is not None
            else math.inf,
            item.run_index,
            item.source_index,
        )
    )
    return output


def _normalize_native_run_text(text: str) -> str:
    """清理原生 run 文本，并把字母后的 PDF 软断词标记转换成 ASCII hyphen。"""

    normalized = _sanitize_pdf_control_text(text, preserve_newlines=False)
    normalized = re.sub(r"[\t\f\v ]+", " ", normalized)
    return normalized.strip()


def _sanitize_pdf_control_text(text: str, *, preserve_newlines: bool) -> str:
    """规范 PDF 排版空白与控制字符，并按调用场景决定是否保留物理换行。"""

    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.translate(_PDF_UNICODE_TEXT_TRANSLATION)
    normalized = _PDF_LINE_END_SOFT_HYPHEN_RE.sub("-", normalized)
    normalized = normalized.replace("\u00ad", "")
    normalized = normalized.replace("\t", " ")
    if not preserve_newlines:
        normalized = normalized.replace("\n", "")
    return _PDF_CONTROL_CHAR_RE.sub("", normalized)


def _detect_leading_emphasis_width(
    glyphs: list[tuple[BBox, tuple[str, int] | None, float | None]],
) -> float | None:
    """从行首连续字体 run 中提取字重显著高于后续正文的几何宽度。"""

    if len(glyphs) < 4:
        return None
    first_signature = glyphs[0][1]
    if first_signature is None:
        return None

    prefix: list[tuple[BBox, tuple[str, int] | None, float | None]] = []
    body: list[tuple[BBox, tuple[str, int] | None, float | None]] = []
    reached_body = False
    for glyph in glyphs:
        if not reached_body and glyph[1] == first_signature:
            prefix.append(glyph)
            continue
        reached_body = True
        body.append(glyph)
    if len(prefix) < 2 or len(body) < 2:
        return None

    prefix_weights = [weight for _bbox, _signature, weight in prefix if weight is not None]
    body_weights = [weight for _bbox, _signature, weight in body if weight is not None]
    if not prefix_weights or not body_weights:
        return None
    prefix_weight = statistics.median(prefix_weights)
    body_weight = statistics.median(body_weights)
    if prefix_weight - body_weight < 100.0 or prefix_weight < 1.15 * max(1.0, body_weight):
        return None

    prefix_bbox = _bbox_union_many([bbox for bbox, _signature, _weight in prefix])
    return max(0.1, prefix_bbox[2] - prefix_bbox[0])


def _fill_native_typography(line: _LineItem, page_size: tuple[float, float]) -> None:
    """使用原始 bbox、PDF 字号和 dominant font 填充两套排版特征。"""

    canonical_em_height = line.em_height
    heights: list[float] = []
    glyph_widths: list[float] = []
    font_counts: dict[tuple[str, int], int] = {}
    font_weights: dict[tuple[str, int], list[float]] = {}
    glyph_typography: list[tuple[BBox, tuple[str, int] | None, float | None]] = []
    valid_font_chars = 0
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if not raw_char.isprintable() or raw_char.isspace():
            continue
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle)
        heights.append(max(0.1, local_bbox[3] - local_bbox[1]))
        glyph_widths.append(max(0.1, local_bbox[2] - local_bbox[0]))
        font = char.get("font") or {}
        font_name = str(font.get("name") or "")
        if not font_name:
            glyph_typography.append((local_bbox, None, None))
            continue
        try:
            font_flags = int(font.get("flags") or 0)
        except (TypeError, ValueError):
            font_flags = 0
        signature = (font_name, font_flags)
        font_counts[signature] = font_counts.get(signature, 0) + 1
        try:
            font_weight = float(font.get("weight"))
        except (TypeError, ValueError):
            font_weight = math.nan
        if math.isfinite(font_weight):
            font_weights.setdefault(signature, []).append(font_weight)
            glyph_typography.append((local_bbox, signature, font_weight))
        else:
            glyph_typography.append((local_bbox, signature, None))
        valid_font_chars += 1

    local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
    line.effective_height = statistics.median(heights) if heights else max(0.1, local_bbox[3] - local_bbox[1])
    line.em_height = canonical_em_height if canonical_em_height > 0 else line.effective_height
    line.median_glyph_width = statistics.median(glyph_widths) if glyph_widths else None
    if font_counts and valid_font_chars:
        line.font_signature, dominant_count = max(font_counts.items(), key=lambda item: item[1])
        line.font_coverage = dominant_count / valid_font_chars
        dominant_weights = font_weights.get(line.font_signature, [])
        line.dominant_font_weight = statistics.median(dominant_weights) if dominant_weights else None
    else:
        line.font_signature = None
        line.font_coverage = 0.0
        line.dominant_font_weight = None
    line.leading_emphasis_width = _detect_leading_emphasis_width(glyph_typography)


def _is_detached_inline_script_candidate(
    small_bbox: BBox,
    base_bbox: BBox,
    base_height: float,
) -> bool:
    """仅依据紧凑宽度、边缘邻接和垂直偏移确认低重叠外置上下标。"""

    small_width = max(0.0, small_bbox[2] - small_bbox[0])
    edge_distance = min(
        abs(base_bbox[0] - small_bbox[2]),
        abs(small_bbox[0] - base_bbox[2]),
    )
    vertical_gap = max(
        0.0,
        base_bbox[1] - small_bbox[3],
        small_bbox[1] - base_bbox[3],
    )
    center_offset = abs(_bbox_center_y(small_bbox) - _bbox_center_y(base_bbox))
    outside_offset = max(
        base_bbox[1] - small_bbox[1],
        small_bbox[3] - base_bbox[3],
    )
    return (
        small_width <= 0.75 * base_height
        and edge_distance <= max(1.0, 0.1 * base_height)
        and vertical_gap <= max(0.5, 0.15 * base_height)
        and center_offset >= max(0.5, 0.25 * base_height)
        and outside_offset >= max(0.5, 0.2 * base_height)
    )


def _native_typographic_scale(line: _LineItem) -> float:
    """返回原生行的字体尺度，禁止 loose 空间高度参与上下标字号比较。"""

    font_sizes: list[float] = []
    for char in line.chars:
        try:
            font_size = float((char.get("font") or {}).get("size") or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(font_size) and font_size > 0:
            font_sizes.append(font_size)
    return max(
        0.1,
        statistics.median(font_sizes) if font_sizes else line.em_height or line.effective_height,
    )


def _merge_native_inline_scripts(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """以 mutual-nearest 规则把跨粗行的小字号前后置标记合入主体视觉行。"""

    candidates: list[tuple[float, int, int, Literal["prefix", "suffix"]]] = []
    detached_candidate_pairs: set[tuple[int, int, Literal["prefix", "suffix"]]] = set()
    for small_index, small in enumerate(lines):
        compact_text = "".join(char for char in small.text if not char.isspace())
        if not compact_text:
            continue
        small_local_bbox = _rotate_bbox_to_upright(small.bbox, page_size, small.angle)
        for base_index, base in enumerate(lines):
            if small_index == base_index or small.angle != base.angle or small.visual_row_id == base.visual_row_id:
                continue
            if small.effective_height <= 0 or base.effective_height <= 0:
                continue
            canonical_small_scale = _native_typographic_scale(small)
            canonical_base_scale = _native_typographic_scale(base)
            legacy_small_scale = max(0.1, small.effective_height)
            legacy_base_scale = max(0.1, base.effective_height)
            legacy_ratio = legacy_small_scale / legacy_base_scale
            use_canonical_reference_scale = (
                _INLINE_REFERENCE_MARKER_RE.fullmatch(compact_text) is not None
                and not 0.35 <= legacy_ratio <= 0.8
                and 0.35 <= canonical_small_scale / canonical_base_scale <= 0.8
            )
            small_scale = canonical_small_scale if use_canonical_reference_scale else legacy_small_scale
            base_scale = canonical_base_scale if use_canonical_reference_scale else legacy_base_scale
            height_ratio = small_scale / base_scale
            if not 0.35 <= height_ratio <= 0.8:
                continue
            if len(compact_text) > 8 and small_local_bbox[2] - small_local_bbox[0] > 3.0 * base_scale:
                continue
            base_local_bbox = _rotate_bbox_to_upright(base.bbox, page_size, base.angle)
            vertical_overlap = max(
                0.0,
                min(small_local_bbox[3], base_local_bbox[3]) - max(small_local_bbox[1], base_local_bbox[1]),
            )
            small_height = max(0.1, small_local_bbox[3] - small_local_bbox[1])
            overlap_ratio = vertical_overlap / small_height
            detached_candidate = overlap_ratio < 0.5 and _is_detached_inline_script_candidate(
                small_local_bbox,
                base_local_bbox,
                base_scale,
            )
            if overlap_ratio < 0.5 and not detached_candidate:
                continue
            center_offset = abs(_bbox_center_y(small_local_bbox) - _bbox_center_y(base_local_bbox))
            if center_offset < max(0.5, 0.12 * base_scale):
                # 同基线居中的小字号文本更可能是表格相邻 cell，而不是上下标。
                continue
            gap_limit = max(1.5, 0.35 * base_scale)
            edge_options: list[tuple[float, Literal["prefix", "suffix"], float]] = []
            for position, gap in (
                ("prefix", base_local_bbox[0] - small_local_bbox[2]),
                ("suffix", small_local_bbox[0] - base_local_bbox[2]),
            ):
                if -0.35 * base_scale <= gap <= gap_limit:
                    edge_options.append((abs(gap), position, gap))
            if not edge_options:
                continue
            _edge_distance, position, gap = min(edge_options, key=lambda item: item[0])

            outside_offset = max(
                base_local_bbox[1] - small_local_bbox[1],
                small_local_bbox[3] - base_local_bbox[3],
            )
            tightly_attached = abs(gap) <= max(1.0, 0.1 * base_scale)
            if outside_offset < max(0.5, 0.08 * base_scale) and not tightly_attached:
                # 紧贴边缘的小字号上下标可能完全落入高字形 bbox；其余内嵌小字仍按普通 cell 排除。
                continue
            metric = abs(gap) + (1.0 - overlap_ratio) * base_scale
            candidates.append((metric, small_index, base_index, position))
            if detached_candidate:
                detached_candidate_pairs.add((small_index, base_index, position))

    best_base_for_small: dict[int, tuple[float, int, Literal["prefix", "suffix"]]] = {}
    best_small_for_base: dict[tuple[int, str], tuple[float, int]] = {}
    for metric, small_index, base_index, position in candidates:
        if small_index not in best_base_for_small or metric < best_base_for_small[small_index][0]:
            best_base_for_small[small_index] = (metric, base_index, position)
        base_key = (base_index, position)
        if base_key not in best_small_for_base or metric < best_small_for_base[base_key][0]:
            best_small_for_base[base_key] = (metric, small_index)

    matches: dict[int, dict[Literal["prefix", "suffix"], int]] = {}
    for small_index, (_metric, base_index, position) in best_base_for_small.items():
        if best_small_for_base.get((base_index, position), (math.inf, -1))[1] == small_index:
            matches.setdefault(base_index, {})[position] = small_index

    consumed_small_indices = {small_index for positions in matches.values() for small_index in positions.values()}
    merged_base_indices: set[int] = set()

    def merge_children(base_index: int, visiting: set[int]) -> None:
        """先合并更小的依赖标记，再把当前完整节点递归合入更大的主体行。"""

        if base_index in merged_base_indices or base_index in visiting:
            return
        visiting.add(base_index)
        positions = matches.get(base_index, {})
        for child_index in positions.values():
            merge_children(child_index, visiting)
        base = lines[base_index]
        stable_source_indices = [
            source_index
            for source_index in [base.source_index, *(lines[child_index].source_index for child_index in positions.values())]
            if source_index >= 0
        ]
        if stable_source_indices:
            base.source_index = min(stable_source_indices)
        formula_candidate_only = base.formula_candidate_only and all(
            lines[child_index].formula_candidate_only for child_index in positions.values()
        )
        merged_bbox = base.bbox
        merged_chars = list(base.chars)
        if "prefix" in positions:
            prefix_index = positions["prefix"]
            prefix = lines[prefix_index]
            base.text = f"{prefix.text.strip()} {base.text.lstrip()}"
            merged_bbox = _bbox_union(merged_bbox, prefix.bbox)
            merged_chars = [*prefix.chars, *merged_chars]
            base.split_from_row = base.split_from_row or prefix.split_from_row
            base.inline_math_regions.extend(prefix.inline_math_regions)
        if "suffix" in positions:
            suffix_index = positions["suffix"]
            suffix = lines[suffix_index]
            base.text = f"{base.text.rstrip()}{suffix.text.strip()}"
            merged_bbox = _bbox_union(merged_bbox, suffix.bbox)
            merged_chars.extend(suffix.chars)
            base.split_from_row = base.split_from_row or suffix.split_from_row
            base.inline_math_regions.extend(suffix.inline_math_regions)
        base.bbox = merged_bbox
        base.chars = merged_chars
        # 只有低重叠外置候选才需要按完整二维 bbox 计算后继行距；普通上下标保持原有基线行为。
        base.restored_inline_cluster = base.restored_inline_cluster or any(
            lines[child_index].restored_inline_cluster or (child_index, base_index, position) in detached_candidate_pairs
            for position, child_index in positions.items()
        )
        base.formula_candidate_only = formula_candidate_only
        _fill_native_typography(base, page_size)
        visiting.remove(base_index)
        merged_base_indices.add(base_index)

    # 从最终不会被消费的根主体开始，确保 small -> medium -> large 链不会丢失最小节点。
    root_base_indices = [base_index for base_index in matches if base_index not in consumed_small_indices]
    for base_index in root_base_indices:
        merge_children(base_index, set())
    for base_index in matches:
        merge_children(base_index, set())

    output = [line for index, line in enumerate(lines) if index not in consumed_small_indices]
    output.sort(key=lambda item: (item.visual_row_id if item.visual_row_id is not None else math.inf, item.run_index))
    return output


def _normalize_pdftext_angle(value: Any) -> int:
    """将 pdftext 弧度方向就近归一到四个标准角度。"""

    try:
        angle_radians = float(value or 0.0)
    except (TypeError, ValueError):
        return 0
    angle_degrees = math.degrees(angle_radians)
    normalized = int(round(angle_degrees / 90.0) * 90) % 360
    return normalized if normalized in {0, 90, 180, 270} else 0


def _get_pdf_drawing_lines(pdf_doc: PDFDocument, page_idx: int) -> list[_AxisLine]:
    """读取 PDFDocument 的公共绘图线结果，并隔离具体 PDFium 类型。"""

    output: list[_AxisLine] = []
    for drawing_line in pdf_doc.get_page_drawing_lines(page_idx):
        bbox = _coerce_bbox(drawing_line.bbox)
        if bbox is None:
            continue
        output.append(
            _AxisLine(
                bbox=bbox,
                width=max(0.0, float(drawing_line.width)),
                orientation=drawing_line.orientation,
            )
        )
    return output


def _median_native_glyph_width(line: _LineItem, page_size: tuple[float, float]) -> float | None:
    """返回单个原生 run 的可见字符中位宽度，缺少字符时返回空。"""

    if line.median_glyph_width is not None:
        return line.median_glyph_width
    widths: list[float] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if not raw_char.isprintable() or raw_char.isspace() or bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle)
        widths.append(max(0.1, local_bbox[2] - local_bbox[0]))
    line.median_glyph_width = statistics.median(widths) if widths else None
    return line.median_glyph_width
