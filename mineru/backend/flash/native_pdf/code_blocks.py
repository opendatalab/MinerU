# Copyright (c) Opendatalab. All rights reserved.

"""识别带填充背景的等宽代码区域并投影其空间文本。"""

from __future__ import annotations

import math
import statistics
import unicodedata
from typing import Any

from mineru.backend.hybrid.table_text import project_pdf_spatial_text
from mineru.types import BBox
from mineru.utils.pdf_document import PDFPathInfo

from .geometry import (
    _bbox_area,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_overlap_in_first,
    _bbox_overlap_in_smaller,
    _bbox_union_many,
)
from .models import _CodeCandidate, _LineItem, _PageSource
from .native_text import _sanitize_pdf_control_text


_MONOSPACE_FONT_HINTS = (
    "mono",
    "courier",
    "consolas",
    "menlo",
    "typewriter",
    "fixed",
    "code",
)


def _build_code_blocks(
    source: _PageSource,
    excluded_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """检测代码背景、唯一认领其文本，并输出页内 code block。"""

    candidates = _detect_code_candidates(
        source,
        excluded_bboxes,
        claimed_line_indices,
    )
    return _materialize_code_candidates(source, candidates)


def _build_rule_delimited_code_blocks(
    source: _PageSource,
    excluded_bboxes: list[BBox],
    claimed_line_indices: set[int] | None = None,
) -> tuple[list[dict[str, Any]], set[int]]:
    """在表格认领前物化外框或上下横线限定的代码清单。"""

    candidates = _detect_rule_delimited_code_candidates(
        source,
        excluded_bboxes,
        claimed_line_indices or set(),
    )
    return _materialize_code_candidates(source, candidates)


def _materialize_code_candidates(
    source: _PageSource,
    candidates: list[_CodeCandidate],
) -> tuple[list[dict[str, Any]], set[int]]:
    """统一投影代码候选，确保来源行只被一个 code block 认领。"""

    if not candidates:
        return [], set()
    lines_by_index = {line.source_index: line for line in source.lines}
    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for candidate in candidates:
        members = [
            lines_by_index[source_index]
            for source_index in sorted(candidate.line_indices)
            if source_index in lines_by_index
        ]
        if not members:
            continue
        content = project_pdf_spatial_text(
            _code_member_chars(members),
            candidate.bbox,
            candidate.angle,
            preserve_blank_rows=True,
        )
        if not content:
            content = _fallback_code_content(members)
        if not content:
            continue
        blocks.append(
            {
                "type": "code",
                "bbox": candidate.bbox,
                "angle": candidate.angle,
                "content": content,
            }
        )
        claimed.update(candidate.line_indices)
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _detect_rule_delimited_code_candidates(
    source: _PageSource,
    excluded_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> list[_CodeCandidate]:
    """按规则边界、稳定行距和缩进层次识别非等宽代码清单。"""

    page_width, page_height = source.page_size
    if page_width <= 0 or page_height <= 0:
        return []
    available = [
        line
        for line in source.lines
        if line.source_index not in claimed_line_indices and line.angle == 0
    ]
    if len(available) < 5:
        return []
    median_height = statistics.median(
        max(0.1, line.effective_height or line.bbox[3] - line.bbox[1])
        for line in available
    )
    horizontal_rules = sorted(
        [
            line
            for line in source.drawing_lines
            if line.orientation == "horizontal"
            and line.bbox[2] - line.bbox[0] >= 0.22 * page_width
        ],
        key=lambda line: (line.bbox[1], line.bbox[0]),
    )
    vertical_rules = [
        line
        for line in source.drawing_lines
        if line.orientation == "vertical"
    ]
    raw_candidates: list[_CodeCandidate] = []
    endpoint_tolerance = max(2.0, 0.75 * median_height)
    for top_index, top_rule in enumerate(horizontal_rules[:-1]):
        for bottom_rule in horizontal_rules[top_index + 1 :]:
            if (
                abs(top_rule.bbox[0] - bottom_rule.bbox[0]) > endpoint_tolerance
                or abs(top_rule.bbox[2] - bottom_rule.bbox[2]) > endpoint_tolerance
            ):
                continue
            candidate_bbox = _bbox_union_many(
                [top_rule.bbox, bottom_rule.bbox]
            )
            candidate_height = candidate_bbox[3] - candidate_bbox[1]
            if not 6.0 * median_height <= candidate_height <= 0.35 * page_height:
                continue
            if any(
                _bbox_overlap_in_smaller(candidate_bbox, excluded_bbox) >= 0.5
                for excluded_bbox in excluded_bboxes
            ):
                continue
            interior_rules = [
                rule
                for rule in horizontal_rules
                if top_rule.bbox[3] + 0.5 * median_height
                < rule.bbox[1]
                < bottom_rule.bbox[1] - 0.5 * median_height
                and _bbox_overlap_in_first(rule.bbox, candidate_bbox) >= 0.8
                and rule.bbox[2] - rule.bbox[0]
                >= 0.6 * (candidate_bbox[2] - candidate_bbox[0])
            ]
            if interior_rules:
                continue
            internal_vertical_rules = [
                rule
                for rule in vertical_rules
                if candidate_bbox[0] + median_height
                < _bbox_center_x(rule.bbox)
                < candidate_bbox[2] - median_height
                and rule.bbox[3] - rule.bbox[1] >= 0.6 * candidate_height
                and _bbox_overlap_in_first(rule.bbox, candidate_bbox) >= 0.8
            ]
            if internal_vertical_rules:
                continue
            members = [
                line
                for line in available
                if candidate_bbox[0] - 0.5 * median_height
                <= _bbox_center_x(line.bbox)
                <= candidate_bbox[2] + 0.5 * median_height
                and top_rule.bbox[3]
                <= _bbox_center_y(line.bbox)
                <= bottom_rule.bbox[1]
            ]
            if not _rule_delimited_code_members_are_structured(
                members,
                candidate_bbox,
                median_height,
            ):
                continue
            raw_candidates.append(
                _CodeCandidate(
                    bbox=candidate_bbox,
                    angle=0,
                    line_indices={line.source_index for line in members},
                )
            )

    accepted: list[_CodeCandidate] = []
    for candidate in sorted(raw_candidates, key=lambda item: _bbox_area(item.bbox)):
        if any(
            _bbox_overlap_in_smaller(candidate.bbox, existing.bbox) >= 0.85
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.bbox[1], item.bbox[0]))


def _rule_delimited_code_members_are_structured(
    members: list[_LineItem],
    candidate_bbox: BBox,
    median_height: float,
) -> bool:
    """验证候选具有稳定基线、代码缩进或窄行号槽，并排除规则多列表格。"""

    if len(members) < 5:
        return False
    rows: dict[int, list[_LineItem]] = {}
    fallback_row = 1_000_000
    for line in members:
        row_key = line.visual_row_id
        if row_key is None:
            row_key = fallback_row
            fallback_row += 1
        rows.setdefault(row_key, []).append(line)
    if len(rows) < 5:
        return False
    ordered_rows = sorted(
        rows.values(),
        key=lambda row: min(_bbox_center_y(line.bbox) for line in row),
    )
    row_centers = [
        statistics.median(_bbox_center_y(line.bbox) for line in row)
        for row in ordered_rows
    ]
    row_gaps = [
        current - previous
        for previous, current in zip(row_centers, row_centers[1:])
        if current > previous
    ]
    if not row_gaps:
        return False
    base_pitch = statistics.median(row_gaps)
    if base_pitch <= 0 or sum(
        0.65 * base_pitch <= gap <= 1.8 * base_pitch
        for gap in row_gaps
    ) / len(row_gaps) < 0.75:
        return False
    if sum(len(row) >= 3 for row in ordered_rows) / len(ordered_rows) >= 0.25:
        return False

    left_positions = sorted(line.bbox[0] for line in members)
    indent_clusters: list[list[float]] = []
    for position in left_positions:
        if (
            not indent_clusters
            or position - statistics.median(indent_clusters[-1])
            > 0.75 * median_height
        ):
            indent_clusters.append([position])
        else:
            indent_clusters[-1].append(position)
    narrow_gutter_rows = 0
    for row in ordered_rows:
        ordered = sorted(row, key=lambda line: line.bbox[0])
        if (
            len(ordered) >= 2
            and ordered[0].bbox[2] - ordered[0].bbox[0]
            <= 2.0 * median_height
            and ordered[1].bbox[0] - ordered[0].bbox[2]
            >= 0.5 * median_height
        ):
            narrow_gutter_rows += 1
    has_line_number_gutter = narrow_gutter_rows / len(ordered_rows) >= 0.35
    has_indent_hierarchy = (
        len(indent_clusters) >= 3
        and sum(len(cluster) >= 2 for cluster in indent_clusters) >= 2
    )
    if not has_line_number_gutter and not has_indent_hierarchy:
        return False

    occupied_width = max(line.bbox[2] for line in members) - min(
        line.bbox[0] for line in members
    )
    return occupied_width <= 0.95 * max(
        0.1,
        candidate_bbox[2] - candidate_bbox[0],
    )


def _code_member_chars(lines: list[_LineItem]) -> list[dict[str, Any]]:
    """按 char_idx 去重代码成员字符，避免区域内斜向水印混入空间投影。"""

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line in lines:
        for fallback_index, char in enumerate(line.chars):
            if not isinstance(char, dict):
                continue
            try:
                identity = ("source", int(char.get("char_idx")))
            except (TypeError, ValueError):
                identity = ("fallback", fallback_index + line.source_index * 1_000_000)
            if identity in seen:
                continue
            seen.add(identity)
            output.append(char)
    return output


def _detect_code_candidates(
    source: _PageSource,
    excluded_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> list[_CodeCandidate]:
    """以非白填充矩形、等宽字体和规则空间栅格筛选代码候选。"""

    page_width, page_height = source.page_size
    page_area = max(0.0, page_width) * max(0.0, page_height)
    if page_area <= 0 or not source.path_infos:
        return []
    raw_candidates: list[_CodeCandidate] = []
    for path_info in source.path_infos:
        bbox = path_info.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if (
            path_info.form_depth != 0
            or not path_info.fill_visible
            or path_info.segment_count < 4
            or not _path_has_visible_nonwhite_fill(path_info)
            or width < 0.5 * page_width
            or height < 0.008 * page_height
            or _bbox_area(bbox) >= 0.8 * page_area
            or any(
                _bbox_overlap_in_smaller(bbox, excluded_bbox) >= 0.5
                for excluded_bbox in excluded_bboxes
            )
        ):
            continue
        members = [
            line
            for line in source.lines
            if line.source_index not in claimed_line_indices
            and _bbox_overlap_in_first(line.bbox, bbox) >= 0.8
        ]
        if not members:
            continue
        dominant_angle = _dominant_code_angle(members)
        angle_members = [line for line in members if line.angle == dominant_angle]
        total_support = sum(_estimated_line_character_count(line) for line in members)
        angle_support = sum(
            _estimated_line_character_count(line)
            for line in angle_members
        )
        if total_support <= 0 or angle_support / total_support < 0.8:
            continue
        monospace_ratio, cell_widths = _monospace_character_support(angle_members)
        if monospace_ratio < 0.8 or not _monospace_advances_are_stable(cell_widths):
            continue
        median_cell_width = statistics.median(
            width_value
            for values in cell_widths.values()
            for width_value in values
        )
        if not _code_rows_have_spatial_structure(
            angle_members,
            bbox,
            median_cell_width,
        ):
            continue
        raw_candidates.append(
            _CodeCandidate(
                bbox=bbox,
                angle=dominant_angle,
                line_indices={line.source_index for line in angle_members},
            )
        )

    accepted: list[_CodeCandidate] = []
    for candidate in sorted(raw_candidates, key=lambda item: _bbox_area(item.bbox)):
        if any(
            _bbox_overlap_in_smaller(candidate.bbox, existing.bbox) >= 0.9
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.bbox[1], item.bbox[0]))


def _path_has_visible_nonwhite_fill(path_info: PDFPathInfo) -> bool:
    """检查填充色是否可见且与白色背景存在最小颜色差。"""

    if path_info.fill_rgba is None:
        return False
    red, green, blue, alpha = path_info.fill_rgba
    return alpha > 0 and max(255 - red, 255 - green, 255 - blue) >= 5


def _dominant_code_angle(lines: list[_LineItem]) -> int:
    """按估算字符数选择代码区域的主文本方向。"""

    support: dict[int, float] = {}
    for line in lines:
        support[line.angle] = support.get(line.angle, 0.0) + _estimated_line_character_count(line)
    return max(sorted(support), key=lambda angle: support[angle])


def _estimated_line_character_count(line: _LineItem) -> float:
    """优先按字符对象计数，缺失时用行宽和缓存字宽估算字符支持。"""

    valid_chars = [
        char
        for char in line.chars
        if isinstance(char, dict) and str(char.get("char") or "").strip()
    ]
    if valid_chars:
        return float(len(valid_chars))
    line_width = max(0.1, line.bbox[2] - line.bbox[0])
    glyph_width = max(0.1, line.median_glyph_width or line_width)
    return max(1.0, line_width / glyph_width)


def _font_name_looks_monospaced(name: str | None) -> bool:
    """按字体元数据中的通用等宽提示判断字体族，不匹配文档内容。"""

    normalized = (name or "").replace("-", "").replace("_", "").casefold()
    return any(hint in normalized for hint in _MONOSPACE_FONT_HINTS)


def _monospace_character_support(
    lines: list[_LineItem],
) -> tuple[float, dict[str, list[float]]]:
    """统计等宽字体字符占比，并按东西文宽度组收集字符 advance。"""

    supported = 0.0
    total = 0.0
    widths: dict[str, list[float]] = {"narrow": [], "wide": []}
    for line in lines:
        fallback_monospace = _font_name_looks_monospaced(
            line.font_signature[0] if line.font_signature is not None else None
        )
        for char in line.chars:
            if not isinstance(char, dict):
                continue
            value = str(char.get("char") or "")
            if not value.strip():
                continue
            total += 1.0
            font = char.get("font")
            font_name = font.get("name") if isinstance(font, dict) else None
            if _font_name_looks_monospaced(font_name) or fallback_monospace:
                supported += 1.0
            try:
                x0, _y0, x1, _y1 = [float(item) for item in char.get("bbox", [])]
            except (TypeError, ValueError):
                continue
            width = x1 - x0
            if not math.isfinite(width) or width <= 0.1:
                continue
            width_group = (
                "wide"
                if unicodedata.east_asian_width(value[0]) in {"W", "F"}
                else "narrow"
            )
            widths[width_group].append(width)

    if total <= 0:
        fallback_support = sum(_estimated_line_character_count(line) for line in lines)
        if fallback_support <= 0:
            return 0.0, widths
        supported_support = sum(
            _estimated_line_character_count(line)
            for line in lines
            if _font_name_looks_monospaced(
                line.font_signature[0]
                if line.font_signature is not None
                else None
            )
        )
        fallback_widths = [
            line.median_glyph_width
            for line in lines
            if line.median_glyph_width is not None
            and line.median_glyph_width > 0
        ]
        widths["narrow"].extend(fallback_widths)
        return supported_support / fallback_support, widths
    return supported / total, widths


def _monospace_advances_are_stable(
    widths: dict[str, list[float]],
) -> bool:
    """验证各字符宽度组的中位绝对偏差足够小，并校验中西文宽度关系。"""

    populated = [values for values in widths.values() if values]
    if not populated or sum(len(values) for values in populated) < 3:
        return False
    medians: dict[str, float] = {}
    for group_name, values in widths.items():
        if not values:
            continue
        median_width = statistics.median(values)
        mad = statistics.median(abs(value - median_width) for value in values)
        if mad / max(0.1, median_width) > 0.2:
            return False
        medians[group_name] = median_width
    if "narrow" in medians and "wide" in medians:
        ratio = medians["wide"] / medians["narrow"]
        if not 1.25 <= ratio <= 2.25:
            return False
    return True


def _code_rows_have_spatial_structure(
    lines: list[_LineItem],
    candidate_bbox: BBox,
    median_cell_width: float,
) -> bool:
    """验证代码行具有规则基线，并且左缘落在一致的等宽字符槽。"""

    rows: list[list[_LineItem]] = []
    for line in sorted(lines, key=lambda item: (_bbox_center_y(item.bbox), item.bbox[0])):
        target = next(
            (
                row
                for row in rows
                if any(
                    min(member.bbox[3], line.bbox[3])
                    - max(member.bbox[1], line.bbox[1])
                    >= 0.5
                    * min(
                        member.bbox[3] - member.bbox[1],
                        line.bbox[3] - line.bbox[1],
                    )
                    for member in row
                )
            ),
            None,
        )
        if target is None:
            rows.append([line])
        else:
            target.append(line)
    if not rows:
        return False

    residuals = []
    for line in lines:
        slot = (line.bbox[0] - candidate_bbox[0]) / max(0.1, median_cell_width)
        residuals.append(abs(slot - round(slot)))
    if sum(residual <= 0.35 for residual in residuals) / len(residuals) < 0.8:
        return False
    if len(rows) == 1:
        return True

    row_tops = sorted(min(line.bbox[1] for line in row) for row in rows)
    deltas = [
        current - previous
        for previous, current in zip(row_tops, row_tops[1:])
        if current > previous
    ]
    if not deltas:
        return False
    lower_count = max(1, math.ceil(0.6 * len(deltas)))
    base_pitch = statistics.median(sorted(deltas)[:lower_count])
    if base_pitch <= 0:
        return False
    return all(
        abs(delta / base_pitch - round(delta / base_pitch)) <= 0.35
        for delta in deltas
    )


def _fallback_code_content(lines: list[_LineItem]) -> str:
    """空间投影失败时按视觉行和水平位置保留代码文本的最小结构。"""

    ordered = sorted(
        lines,
        key=lambda line: (
            round(_bbox_center_y(line.bbox), 1),
            _bbox_center_x(line.bbox),
            line.source_index,
        ),
    )
    content = "\n".join(line.text for line in ordered if line.text)
    return _sanitize_pdf_control_text(content, preserve_newlines=True).strip()
