# Copyright (c) Opendatalab. All rights reserved.

"""基于排版差异分类文档标题和段落标题。"""

from __future__ import annotations

import statistics
from typing import Literal


from mineru.types import BBox

from .models import (
    _LaneBodyProfile,
    _LineItem,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _rotate_bbox_to_upright,
)
from .line_layout import (
    _effective_text_row_gap,
    _estimate_lane_gap,
    _infer_text_lanes,
    _line_effective_height,
    _should_connect_semantic_rows,
    _title_fonts_compatible,
)


def _classify_page_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    page_index: int,
    container_bboxes: list[BBox],
) -> None:
    """只用页面几何与字体排版标注首页文档标题和各页段落标题。"""

    for angle in sorted({line.angle for line in lines if line.semantic_type is None}):
        line_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.semantic_type is None
        ]
        if not line_geometry:
            continue
        median_height = statistics.median(
            _line_effective_height(line, bbox) for line, bbox in line_geometry
        )
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))

        document_title_bottom: float | None = None
        if page_index == 0:
            document_title_bottom = _classify_document_title(
                lanes,
                local_page_height,
            )
        preserve_front_matter_boundaries = _document_title_uses_page_fallback(
            lanes,
        )

        local_container_bboxes = [
            _rotate_bbox_to_upright(bbox, page_size, angle)
            for bbox in container_bboxes
        ]
        for lane in lanes:
            profile = _infer_lane_body_profile(lane)
            _classify_paragraph_titles_in_lane(
                lane,
                profile,
                local_page_height,
                local_container_bboxes,
                document_title_bottom=document_title_bottom,
                preserve_front_matter_boundaries=preserve_front_matter_boundaries,
            )


def _infer_lane_body_profile(lane: _TextLane) -> _LaneBodyProfile:
    """从栏带的长行主体估计正文行高、主字体、字重、常规行距和样式占比。"""

    available = [item for item in lane.lines if item[0].semantic_type is None]
    if not available:
        return _LaneBodyProfile(1.0, None, None, 0.35, {})
    lane_width = max(0.1, lane.right - lane.left)
    long_lines = [item for item in available if item[1][2] - item[1][0] >= 0.45 * lane_width]
    body_rows = long_lines or available
    body_line_ids = {id(line) for line, _bbox in body_rows}
    body_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in body_rows)

    font_support: dict[tuple[str, int], float] = {}
    style_support: dict[tuple[str, int], float] = {}
    total_style_width = 0.0
    for line, bbox in available:
        if line.font_signature is None or line.font_coverage < 0.75:
            continue
        line_width = max(0.1, bbox[2] - bbox[0])
        style_support[line.font_signature] = style_support.get(line.font_signature, 0.0) + line_width
        total_style_width += line_width
        if id(line) in body_line_ids and 0.75 <= _line_effective_height(line, bbox) / body_height <= 1.35:
            font_support[line.font_signature] = font_support.get(line.font_signature, 0.0) + line_width
    body_font = max(font_support, key=font_support.get) if font_support else None
    if total_style_width > 0:
        style_support = {signature: width / total_style_width for signature, width in style_support.items()}

    body_weights = [
        line.dominant_font_weight
        for line, bbox in body_rows
        if line.dominant_font_weight is not None
        and (body_font is None or line.font_signature == body_font)
        and 0.75 <= _line_effective_height(line, bbox) / body_height <= 1.35
    ]
    regular_gap, _gap_mad = _estimate_lane_gap(lane)
    return _LaneBodyProfile(
        body_height=max(0.1, body_height),
        body_font=body_font,
        body_weight=statistics.median(body_weights) if body_weights else None,
        regular_gap=regular_gap,
        style_support=style_support,
    )


def _classify_document_title(
    lanes: list[_TextLane],
    local_page_height: float,
) -> float | None:
    """从首页上部选取显著大字号锚点，并用同版式邻行扩展多行文档标题。"""

    candidates: list[tuple[float, _TextLane, int, tuple[_LineItem, BBox]]] = []
    lane_profiles = [(lane, _infer_lane_body_profile(lane)) for lane in lanes]
    column_body_heights = [profile.body_height for lane, profile in lane_profiles if not lane.is_span]
    document_body_height = (
        statistics.median(column_body_heights)
        if column_body_heights
        else min((profile.body_height for _lane, profile in lane_profiles), default=1.0)
    )
    for lane, profile in lane_profiles:
        lane_width = max(0.1, lane.right - lane.left)
        available = [item for item in lane.lines if item[0].semantic_type is None]
        for row_index, item in enumerate(available[:5]):
            line, bbox = item
            reference_height = min(profile.body_height, 1.25 * document_body_height)
            height_ratio = _line_effective_height(line, bbox) / max(0.1, reference_height)
            document_height_ratio = _line_effective_height(line, bbox) / max(
                0.1,
                document_body_height,
            )
            width_ratio = (bbox[2] - bbox[0]) / lane_width
            centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.15 * lane_width
            spans_columns_fallback = (
                lane.is_span
                and centered
                and width_ratio >= 0.65
                and document_height_ratio >= 1.3
            )
            if (
                _bbox_center_y(bbox) > 0.45 * local_page_height
                or (height_ratio < 1.4 and not spans_columns_fallback)
                or width_ratio < 0.2
                or (not centered and height_ratio < 1.7)
            ):
                continue
            top_preference = max(0.0, 0.45 - _bbox_center_y(bbox) / local_page_height)
            score = height_ratio + (0.75 if centered else 0.0) + top_preference - 0.05 * row_index
            candidates.append((score, lane, row_index, item))
    if not candidates:
        return None

    _score, lane, _row_index, anchor = max(candidates, key=lambda item: item[0])
    anchor_line, anchor_bbox = anchor
    anchor_height = _line_effective_height(anchor_line, anchor_bbox)
    anchor_line.semantic_type = "doc_title"
    selected = [anchor]
    ordered = lane.lines
    anchor_index = ordered.index(anchor)
    for direction in (-1, 1):
        index = anchor_index + direction
        previous_bbox = anchor_bbox
        while 0 <= index < len(ordered):
            candidate_line, candidate_bbox = ordered[index]
            if candidate_line.semantic_type is not None:
                break
            candidate_height = _line_effective_height(candidate_line, candidate_bbox)
            if not 0.8 <= candidate_height / anchor_height <= 1.25:
                break
            if not _title_fonts_compatible(anchor_line, candidate_line):
                break
            vertical_gap = max(candidate_bbox[1] - previous_bbox[3], previous_bbox[1] - candidate_bbox[3], 0.0)
            lane_width = max(0.1, lane.right - lane.left)
            centered = abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(anchor_bbox)) <= 0.18 * lane_width
            aligned = abs(candidate_bbox[0] - anchor_bbox[0]) <= 0.75 * anchor_height
            if vertical_gap > 1.1 * anchor_height or not (centered or aligned):
                break
            candidate_line.semantic_type = "doc_title"
            selected.append((candidate_line, candidate_bbox))
            previous_bbox = candidate_bbox
            index += direction
    return max(bbox[3] for _line, bbox in selected)


def _document_title_uses_page_fallback(lanes: list[_TextLane]) -> bool:
    """判断首页标题是否依赖跨栏 1.30 倍全文正文行高兜底。"""

    title_heights = [
        _line_effective_height(line, bbox)
        for lane in lanes
        for line, bbox in lane.lines
        if line.semantic_type == "doc_title"
    ]
    column_body_heights = [
        _infer_lane_body_profile(lane).body_height
        for lane in lanes
        if not lane.is_span
    ]
    if not title_heights or not column_body_heights:
        return False
    document_body_height = statistics.median(column_body_heights)
    return any(
        1.3 <= title_height / max(0.1, document_body_height) < 1.4
        for title_height in title_heights
    )


def _classify_paragraph_titles_in_lane(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    local_page_height: float,
    container_bboxes: list[BBox],
    *,
    document_title_bottom: float | None,
    preserve_front_matter_boundaries: bool,
) -> None:
    """以字号、样式、留白、对齐、栏宽和容器邻接判定段落标题。"""

    lane_width = max(0.1, lane.right - lane.left)
    rows = lane.lines
    selected_indices: set[int] = set()
    front_matter_boundary = _infer_front_matter_boundary(
        lane,
        profile,
        local_page_height,
        document_title_bottom=document_title_bottom,
    )
    for index, (line, bbox) in enumerate(rows):
        if line.semantic_type is not None:
            continue
        if not 0.07 * local_page_height <= _bbox_center_y(bbox) <= 0.93 * local_page_height:
            continue
        inside_front_matter = (
            front_matter_boundary is not None
            and _bbox_center_y(bbox) <= front_matter_boundary
        )
        if inside_front_matter and not preserve_front_matter_boundaries:
            continue
        if _line_near_visual_container(bbox, container_bboxes, profile.body_height):
            continue

        line_height = _line_effective_height(line, bbox)
        height_ratio = line_height / profile.body_height
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.12 * lane_width
        left_aligned = abs(bbox[0] - lane.left) <= 0.65 * profile.body_height
        style_differs = (
            profile.body_font is not None
            and line.font_signature is not None
            and line.font_coverage >= 0.75
            and line.font_signature != profile.body_font
        )
        style_support = profile.style_support.get(line.font_signature, 1.0) if line.font_signature is not None else 1.0
        weight_emphasized = (
            profile.body_weight is not None
            and line.dominant_font_weight is not None
            and line.dominant_font_weight >= max(profile.body_weight + 100.0, 1.15 * profile.body_weight)
        )
        gap_above = _normalized_title_gap(rows, index, direction=-1, body_height=profile.body_height)
        gap_below = _normalized_title_gap(rows, index, direction=1, body_height=profile.body_height)
        has_spacing_signal = gap_above >= 0.35

        if _visual_row_has_body_style_sibling(rows, index):
            continue
        if _is_near_full_mixed_inline_row(
            rows,
            index,
            lane_width,
            profile,
        ):
            continue
        if (
            index > 0
            and rows[index - 1][0].semantic_type == "paragraph_title"
            and width_ratio >= 0.85
            and gap_above < 1.0
            and not _title_fonts_compatible(rows[index - 1][0], line)
        ):
            continue
        compact_text_section = (
            height_ratio < 0.9
            and centered
            and width_ratio <= 0.7
            and gap_above >= 0.35
            and gap_below >= 0.2
            and _has_following_compact_text_section(
                rows,
                index,
                lane_width,
            )
        )
        if (
            height_ratio < 0.9
            and not inside_front_matter
            and not _has_following_body_row(
                rows,
                index,
                lane_width,
                profile,
            )
            and not compact_text_section
        ):
            continue
        if width_ratio >= 0.9 and height_ratio < 1.15 and not (
            style_differs and gap_above >= 0.65
        ):
            continue
        score = 0.0
        if style_differs:
            score += 2.0
            if style_support <= 0.2:
                score += 0.75
        if weight_emphasized:
            score += 1.0
        if height_ratio >= 1.18:
            score += 2.0
        elif height_ratio <= 0.9 and centered:
            score += 0.75
        if width_ratio <= 0.78:
            score += 0.75
        if centered and width_ratio <= 0.85:
            score += 1.25
        if gap_above >= 0.65:
            score += 1.25
        elif gap_above >= 0.35:
            score += 0.5
        if gap_below >= 0.35:
            score += 0.75
        if left_aligned:
            score += 0.25

        strong_layout_signal = (
            height_ratio >= 1.18
            or style_differs
            or weight_emphasized
            or (centered and width_ratio <= 0.7 and gap_above >= 0.35 and gap_below >= 0.2)
        )
        if score >= 4.0 and has_spacing_signal and strong_layout_signal:
            if _is_full_width_inline_heading(
                rows,
                index,
                lane_width,
                profile,
            ):
                continue
            line.semantic_type = "paragraph_title"
            selected_indices.add(index)

    _unify_visual_row_title_types(
        lane,
        selected_indices,
    )
    _expand_paragraph_title_neighbors(
        lane,
        selected_indices,
        profile,
    )
    if front_matter_boundary is not None and preserve_front_matter_boundaries:
        _protect_front_matter_title_types(
            lane,
            profile,
            front_matter_boundary,
        )


def _protect_front_matter_title_types(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    front_matter_boundary: float,
) -> None:
    """把作者区误命中的标题降为正文，并仅保留原本需要跨行聚合的内部标记。"""

    front_title_indices = [
        index
        for index, (line, bbox) in enumerate(lane.lines)
        if line.semantic_type == "paragraph_title"
        and _bbox_center_y(bbox) <= front_matter_boundary
    ]
    connected_indices: set[int] = set()
    for position, first_index in enumerate(front_title_indices):
        for second_index in front_title_indices[position + 1 :]:
            if _should_connect_semantic_rows(
                lane.lines[first_index],
                lane.lines[second_index],
                lane,
                profile.regular_gap,
                [],
                [],
            ):
                connected_indices.update((first_index, second_index))
    for index in front_title_indices:
        lane.lines[index][0].semantic_type = (
            "text" if index in connected_indices else None
        )


def _visual_row_has_body_style_sibling(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
) -> bool:
    """检查同一完整视觉行是否同时包含标题样式与正文样式 run。"""

    line, _bbox = rows[index]
    if line.visual_row_id is None:
        return False
    siblings = [
        sibling
        for sibling, _sibling_bbox in rows
        if sibling is not line and sibling.visual_row_id == line.visual_row_id
    ]
    return any(not _title_fonts_compatible(line, sibling) for sibling in siblings)


def _is_near_full_mixed_inline_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别近满栏混合字体行内强调，避免把粗体条目头单独标成标题。"""

    line, bbox = rows[index]
    line_height = _line_effective_height(line, bbox)
    if (
        line.font_coverage >= 0.95
        or (bbox[2] - bbox[0]) < 0.85 * lane_width
        or not 0.85 <= line_height / profile.body_height <= 1.15
        or index + 1 >= len(rows)
    ):
        return False
    next_line, next_bbox = rows[index + 1]
    next_height = _line_effective_height(next_line, next_bbox)
    gap_below = _effective_text_row_gap(rows[index], rows[index + 1])
    return (
        next_line.semantic_type is None
        and 0.8 <= next_height / profile.body_height <= 1.2
        and gap_below <= profile.regular_gap + 0.25 * profile.body_height
    )


def _is_full_width_inline_heading(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别后接常规正文续行的满栏正常字号行内标题。"""

    line, bbox = rows[index]
    line_height = _line_effective_height(line, bbox)
    if (
        not 0.85 <= line_height / profile.body_height <= 1.15
        or (bbox[2] - bbox[0]) < 0.9 * lane_width
        or index + 1 >= len(rows)
    ):
        return False

    next_line, next_bbox = rows[index + 1]
    next_height = _line_effective_height(next_line, next_bbox)
    next_uses_body_font = (
        profile.body_font is None
        or next_line.font_signature is None
        or next_line.font_coverage < 0.75
        or next_line.font_signature == profile.body_font
    )
    return (
        next_line.semantic_type is None
        and next_uses_body_font
        and 0.75 <= next_height / profile.body_height <= 1.15
        and (next_bbox[2] - next_bbox[0]) >= 0.75 * lane_width
        and _effective_text_row_gap(rows[index], rows[index + 1])
        <= profile.regular_gap + 0.25 * profile.body_height
    )


def _has_following_body_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """检查小字号候选下方是否仍有正常字号的正文续行。"""

    for line, bbox in rows[index + 1 :]:
        height_ratio = _line_effective_height(line, bbox) / profile.body_height
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        if 0.9 <= height_ratio <= 1.2 and width_ratio >= 0.45:
            return True
    return False


def _has_following_compact_text_section(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
) -> bool:
    """检查小字号候选后是否紧接三行同尺度的宽文本区段。"""

    candidate = rows[index]
    candidate_height = _line_effective_height(*candidate)
    following_rows = rows[index + 1 : index + 4]
    if len(following_rows) < 3:
        return False

    previous = candidate
    for current in following_rows:
        line, bbox = current
        height_ratio = _line_effective_height(line, bbox) / candidate_height
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        gap = _effective_text_row_gap(previous, current)
        if (
            line.semantic_type is not None
            or not 0.75 <= height_ratio <= 1.25
            or width_ratio < 0.45
            or not -0.25 * candidate_height <= gap <= 0.75 * candidate_height
        ):
            return False
        previous = current
    return True


def _unify_visual_row_title_types(
    lane: _TextLane,
    selected_indices: set[int],
) -> None:
    """按完整 visual_row_id 统一标题类型：同样式整行晋升，混合样式整行降级。"""

    row_indices: dict[int, list[int]] = {}
    for index, (line, _bbox) in enumerate(lane.lines):
        if line.visual_row_id is not None:
            row_indices.setdefault(line.visual_row_id, []).append(index)
    for indices in row_indices.values():
        title_indices = [
            index
            for index in indices
            if lane.lines[index][0].semantic_type == "paragraph_title"
        ]
        if not title_indices:
            continue
        anchor_line, anchor_bbox = lane.lines[title_indices[0]]
        anchor_height = _line_effective_height(anchor_line, anchor_bbox)
        compatible = all(
            0.75
            <= _line_effective_height(lane.lines[index][0], lane.lines[index][1])
            / anchor_height
            <= 1.25
            and _title_fonts_compatible(anchor_line, lane.lines[index][0])
            for index in indices
        )
        if compatible:
            for index in indices:
                lane.lines[index][0].semantic_type = "paragraph_title"
                selected_indices.add(index)
            continue
        for index in title_indices:
            lane.lines[index][0].semantic_type = None
            selected_indices.discard(index)


def _infer_front_matter_boundary(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    local_page_height: float,
    *,
    document_title_bottom: float | None,
) -> float | None:
    """用短行后衔接满栏正文的几何转折点界定首页作者等前置信息。"""

    if document_title_bottom is None:
        return None
    default_boundary = max(
        document_title_bottom + 2.5 * profile.body_height,
        0.39 * local_page_height,
    )
    lane_width = max(0.1, lane.right - lane.left)
    for index, (line, bbox) in enumerate(lane.lines[:-1]):
        if bbox[1] <= document_title_bottom:
            continue
        next_line, next_bbox = lane.lines[index + 1]
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        next_width_ratio = (next_bbox[2] - next_bbox[0]) / lane_width
        left_aligned = abs(bbox[0] - lane.left) <= 0.65 * profile.body_height
        centered = abs(
            _bbox_center_x(bbox) - (lane.left + lane.right) / 2.0
        ) <= 0.12 * lane_width
        next_height_ratio = _line_effective_height(next_line, next_bbox) / profile.body_height
        gap_below = max(0.0, _effective_text_row_gap((line, bbox), (next_line, next_bbox)))
        if (
            width_ratio <= 0.35
            and next_width_ratio >= 0.75
            and (left_aligned or centered)
            and 0.75 <= next_height_ratio <= 1.35
            and gap_below <= 1.5 * profile.body_height
        ):
            return min(default_boundary, bbox[1] - 0.1)
    return default_boundary


def _normalized_title_gap(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    *,
    direction: Literal[-1, 1],
    body_height: float,
) -> float:
    """返回标题候选与相邻物理行之间按正文行高归一化的有效净空。"""

    neighbor_index = index + direction
    if not 0 <= neighbor_index < len(rows):
        return 0.5
    if direction < 0:
        gap = _effective_text_row_gap(rows[neighbor_index], rows[index])
    else:
        gap = _effective_text_row_gap(rows[index], rows[neighbor_index])
    return max(0.0, gap) / max(0.1, body_height)


def _line_near_visual_container(
    line_bbox: BBox,
    container_bboxes: list[BBox],
    body_height: float,
) -> bool:
    """检查短行是否紧邻图、表等容器，以纯几何方式抑制 caption 误判。"""

    for container_bbox in container_bboxes:
        if _bbox_axis_overlap_ratio(line_bbox, container_bbox, axis="x") < 0.35:
            continue
        vertical_gap = max(line_bbox[1] - container_bbox[3], container_bbox[1] - line_bbox[3], 0.0)
        if vertical_gap <= 1.5 * body_height:
            return True
    return False


def _expand_paragraph_title_neighbors(
    lane: _TextLane,
    selected_indices: set[int],
    profile: _LaneBodyProfile,
) -> None:
    """用相邻行的字体、尺寸、对齐和紧凑净空补齐折行段落标题。"""

    if not selected_indices:
        return
    rows = lane.lines
    lane_width = max(0.1, lane.right - lane.left)
    pending = list(selected_indices)
    while pending:
        selected_index = pending.pop()
        selected_line, selected_bbox = rows[selected_index]
        selected_height = _line_effective_height(selected_line, selected_bbox)
        for candidate_index in (selected_index - 1, selected_index + 1):
            if not 0 <= candidate_index < len(rows) or candidate_index in selected_indices:
                continue
            candidate_line, candidate_bbox = rows[candidate_index]
            if candidate_line.semantic_type is not None:
                continue
            candidate_height = _line_effective_height(candidate_line, candidate_bbox)
            if not 0.75 <= candidate_height / selected_height <= 1.25:
                continue
            if candidate_bbox[2] - candidate_bbox[0] > 0.8 * lane_width:
                continue
            if not _title_fonts_compatible(selected_line, candidate_line):
                continue
            vertical_gap = max(
                candidate_bbox[1] - selected_bbox[3],
                selected_bbox[1] - candidate_bbox[3],
                0.0,
            )
            centers_align = abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(selected_bbox)) <= 0.15 * lane_width
            left_edges_align = abs(candidate_bbox[0] - selected_bbox[0]) <= 0.75 * profile.body_height
            wrapped_indent = abs(candidate_bbox[0] - selected_bbox[0]) <= 2.0 * profile.body_height
            if vertical_gap > 0.65 * profile.body_height or not (
                centers_align or left_edges_align or wrapped_indent
            ):
                continue
            candidate_line.semantic_type = "paragraph_title"
            selected_indices.add(candidate_index)
            pending.append(candidate_index)
