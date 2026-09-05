# Copyright (c) Opendatalab. All rights reserved.
"""在既有正文统计和结构证据下分类栏内标题。"""

from __future__ import annotations

import statistics
from typing import Literal

from .....types import BBox
from ..geometry import _bbox_axis_overlap_ratio, _bbox_center_x, _bbox_center_y, _bbox_union_many
from ..line_layout import (
    _effective_text_row_gap,
    _font_signatures_share_family,
    _font_weights_conflict,
    _line_effective_height,
    _should_connect_semantic_rows,
    _title_fonts_compatible,
)
from ..models import (
    _DocumentBodyProfile,
    _DocumentTitleProfile,
    _LaneBodyProfile,
    _LineItem,
    _TextLane,
)
from .body_profile import _line_uses_document_regular_font
from .common import _line_inside_visual_container, _line_near_visual_container
from .prototype import _line_conflicts_document_title_profile, _matching_document_title_prototype


def _classify_paragraph_titles_in_lane(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    local_page_width: float,
    local_page_height: float,
    container_bboxes: list[BBox],
    *,
    document_title_bottom: float | None,
    preserve_front_matter_boundaries: bool,
    physical_gaps: dict[int, tuple[float | None, float | None]],
    grid_title_suppressions: set[int],
    document_body_profile: _DocumentBodyProfile | None,
    document_title_profile: _DocumentTitleProfile | None,
    page_index: int,
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
        if line.source_index in grid_title_suppressions:
            continue
        if not 0.07 * local_page_height <= _bbox_center_y(bbox) <= 0.93 * local_page_height:
            continue
        inside_front_matter = front_matter_boundary is not None and _bbox_center_y(bbox) <= front_matter_boundary
        if inside_front_matter and not preserve_front_matter_boundaries:
            continue
        if (
            document_title_bottom is not None
            and document_title_bottom <= 0.6 * local_page_height
            and _bbox_center_y(bbox) >= 0.84 * local_page_height
        ):
            # 首页底部短行通常是版本、日期等封面元数据，不属于正文标题层级。
            continue
        line_height = _line_effective_height(line, bbox)
        title_prototype = _matching_document_title_prototype(
            line,
            bbox,
            lane,
            document_body_profile,
            document_title_profile,
        )
        title_profile_size_conflict = title_prototype is None and _line_conflicts_document_title_profile(
            line,
            bbox,
            lane,
            document_body_profile,
            document_title_profile,
        )
        inside_visual_container = _line_inside_visual_container(
            bbox,
            container_bboxes,
        )
        near_visual_container = _line_near_visual_container(
            bbox,
            container_bboxes,
            profile.body_height,
        )
        if inside_visual_container or (near_visual_container and title_prototype is None):
            continue

        recurrent_regular_font = _line_uses_document_regular_font(
            line,
            document_body_profile,
        )
        document_regular_body_candidate = (
            recurrent_regular_font
            and document_body_profile is not None
            and line_height >= 0.9 * document_body_profile.body_height
        )
        reference_body_height = profile.body_height
        if profile.body_row_count < 3 and document_body_profile is not None:
            reference_body_height = document_body_profile.body_height
        if document_regular_body_candidate and document_body_profile is not None:
            reference_body_height = max(
                reference_body_height,
                document_body_profile.body_height,
            )
        height_ratio = line_height / reference_body_height
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.12 * lane_width
        left_aligned = abs(bbox[0] - lane.left) <= 0.65 * profile.body_height
        if line.median_glyph_width is not None and bbox[2] - bbox[0] <= 1.25 * line.median_glyph_width and height_ratio < 1.18:
            continue
        style_differs = (
            not recurrent_regular_font
            and profile.body_font is not None
            and line.font_signature is not None
            and line.font_coverage >= 0.75
            and not _font_signatures_share_family(
                line.font_signature,
                profile.body_font,
            )
        )
        low_coverage_style_differs = (
            not recurrent_regular_font
            and profile.body_font is not None
            and line.font_signature is not None
            and 0.5 <= line.font_coverage < 0.75
            and not _font_signatures_share_family(
                line.font_signature,
                profile.body_font,
            )
        )
        style_support = profile.style_support.get(line.font_signature, 1.0) if line.font_signature is not None else 1.0
        weight_emphasized = (
            profile.body_weight is not None
            and line.dominant_font_weight is not None
            and line.dominant_font_weight >= max(profile.body_weight + 100.0, 1.15 * profile.body_weight)
        )
        gap_above = _normalized_title_gap(
            rows,
            index,
            direction=-1,
            body_height=profile.body_height,
            physical_gaps=physical_gaps,
        )
        gap_below = _normalized_title_gap(
            rows,
            index,
            direction=1,
            body_height=profile.body_height,
            physical_gaps=physical_gaps,
        )
        regular_gap_ratio = profile.regular_gap / max(0.1, profile.body_height)
        gap_above_excess = max(0.0, gap_above - regular_gap_ratio)
        gap_below_excess = max(0.0, gap_below - regular_gap_ratio)
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
        if _continues_local_body_row(
            rows,
            index,
            lane_width,
            profile,
        ):
            continue
        has_following_body_row = _has_following_body_row(
            rows,
            index,
            lane_width,
            lane.left,
            profile,
        )
        compact_local_transition = (
            low_coverage_style_differs
            and left_aligned
            and width_ratio <= 0.65
            and gap_above >= 0.1
            and gap_below >= 0.2
            and has_following_body_row
        )
        compact_text_section = (
            height_ratio < 0.9
            and centered
            and width_ratio <= 0.7
            and gap_above >= 0.35
            and gap_below >= 0.2
            and (page_index != 0 or style_differs or weight_emphasized or title_prototype is not None)
            and _has_following_compact_text_section(
                rows,
                index,
                lane_width,
            )
        )
        weak_regular_pitch_body_candidate = (
            document_body_profile is not None
            and 0.9 <= line_height / document_body_profile.body_height <= 1.1
            and title_prototype is None
            and (not centered or width_ratio >= 0.75)
            and not weight_emphasized
            and gap_above_excess < 0.1
            and gap_below_excess < 0.1
        )
        if weak_regular_pitch_body_candidate:
            continue
        if (
            title_profile_size_conflict
            and document_body_profile is not None
            and line_height <= 1.18 * document_body_profile.body_height
            and not weight_emphasized
        ):
            continue
        prototype_promotion = (
            title_prototype is not None
            and (width_ratio <= 0.9 or (bbox[2] - bbox[0]) / max(0.1, local_page_width) <= 0.8)
            and (centered or left_aligned)
            and (
                (document_body_profile is not None and line_height >= 1.18 * document_body_profile.body_height)
                or near_visual_container
                or (has_following_body_row and gap_above_excess >= 0.1)
                or gap_above_excess >= 0.35
            )
        )
        prototype_inline_heading = (
            document_body_profile is not None
            and line_height <= 1.15 * document_body_profile.body_height
            and _is_full_width_inline_heading(
                rows,
                index,
                lane_width,
                profile,
            )
        )
        if prototype_promotion and not prototype_inline_heading:
            line.semantic_type = "paragraph_title"
            selected_indices.add(index)
            continue
        if (
            document_body_profile is not None
            and 0.9 <= line_height / document_body_profile.body_height <= 1.1
            and title_prototype is None
            and centered
            and not has_following_body_row
            and not compact_text_section
        ):
            continue
        if height_ratio < 0.9 and not inside_front_matter and not has_following_body_row and not compact_text_section:
            continue
        if width_ratio >= 0.9 and height_ratio < 1.15 and not (style_differs and gap_above >= 0.65):
            continue
        score = 0.0
        if style_differs or compact_local_transition:
            score += 2.0
            if compact_local_transition or style_support <= 0.2:
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

        precisely_centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.05 * lane_width
        centered_structural_fallback = (
            0.75 <= height_ratio <= 1.15
            and precisely_centered
            and width_ratio <= 0.7
            and not inside_front_matter
            and not (page_index == 0 and _bbox_center_y(bbox) <= 0.2 * local_page_height)
            and 0.35 <= gap_above <= 1.5
            and 0.2 <= gap_below <= 1.5
            and has_following_body_row
            and not _is_continuous_field_row(
                rows,
                index,
                lane_width,
                profile,
            )
        )
        strong_layout_signal = (
            height_ratio >= 1.18
            or style_differs
            or compact_local_transition
            or weight_emphasized
            or centered_structural_fallback
            or compact_text_section
        )
        if score >= 4.0 and (has_spacing_signal or compact_local_transition) and strong_layout_signal:
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
        if line.semantic_type == "paragraph_title" and _bbox_center_y(bbox) <= front_matter_boundary
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
        lane.lines[index][0].semantic_type = "text" if index in connected_indices else None


def _visual_row_has_body_style_sibling(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
) -> bool:
    """检查同一完整视觉行是否同时包含标题样式与正文样式 run。"""

    line, _bbox = rows[index]
    if line.visual_row_id is None:
        return False
    siblings = [
        sibling for sibling, _sibling_bbox in rows if sibling is not line and sibling.visual_row_id == line.visual_row_id
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


def _continues_local_body_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别紧随同字体满行的正文尾行，阻止全局小字号基线造成标题误判。"""

    if index <= 0:
        return False
    previous_line, previous_bbox = rows[index - 1]
    line, bbox = rows[index]
    if previous_line.semantic_type is not None:
        return False
    previous_visual_row = (
        [item for item in rows[:index] if item[0].visual_row_id == previous_line.visual_row_id]
        if previous_line.visual_row_id is not None
        else [rows[index - 1]]
    )
    previous_style_members = [
        member
        for member, _member_bbox in previous_visual_row
        if member.font_signature is not None and member.font_coverage >= 0.5
    ]
    if (
        line.font_signature is not None
        and previous_style_members
        and not any(
            _font_signatures_share_family(member.font_signature, line.font_signature)
            and not _font_weights_conflict(member, line)
            for member in previous_style_members
        )
    ):
        return False
    previous_bbox = _bbox_union_many([member_bbox for _member, member_bbox in previous_visual_row])
    previous_height = statistics.median(
        _line_effective_height(member, member_bbox) for member, member_bbox in previous_visual_row
    )
    line_height = _line_effective_height(line, bbox)
    pair_height = max(previous_height, line_height)
    gap = _effective_text_row_gap(
        (previous_line, previous_bbox),
        rows[index],
    )
    return (
        0.8 <= line_height / previous_height <= 1.25
        and previous_bbox[2] - previous_bbox[0] >= 0.75 * lane_width
        and abs(bbox[0] - previous_bbox[0]) <= 2.0 * pair_height
        and -0.25 * pair_height <= gap <= profile.regular_gap + pair_height
    )


def _is_continuous_field_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别同缩进、同尺度且紧邻的连续字段行，阻止其借居中误差成为标题。"""

    _line, bbox = rows[index]
    line_height = _line_effective_height(*rows[index])
    if not 0.85 <= line_height / profile.body_height <= 1.15:
        return False
    for neighbor_index in (index - 1, index + 1):
        if not 0 <= neighbor_index < len(rows):
            continue
        neighbor_line, neighbor_bbox = rows[neighbor_index]
        neighbor_height = _line_effective_height(neighbor_line, neighbor_bbox)
        if (
            neighbor_line.semantic_type is None
            and 0.85 <= neighbor_height / profile.body_height <= 1.15
            and neighbor_bbox[2] - neighbor_bbox[0] >= 0.35 * lane_width
            and abs(neighbor_bbox[0] - bbox[0]) <= 0.75 * profile.body_height
            and max(
                _effective_text_row_gap(rows[neighbor_index], rows[index]),
                _effective_text_row_gap(rows[index], rows[neighbor_index]),
            )
            <= 1.5 * profile.body_height
        ):
            return True
    return False


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
        or _font_signatures_share_family(
            next_line.font_signature,
            profile.body_font,
        )
    )
    return (
        next_line.semantic_type is None
        and next_uses_body_font
        and 0.75 <= next_height / profile.body_height <= 1.15
        and (next_bbox[2] - next_bbox[0]) >= 0.75 * lane_width
        and _effective_text_row_gap(rows[index], rows[index + 1]) <= profile.regular_gap + 0.25 * profile.body_height
    )


def _has_following_body_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    lane_left: float,
    profile: _LaneBodyProfile,
) -> bool:
    """只检查紧邻且空间相关的下一行，禁止用远处正文支撑小字号标题。"""

    if index + 1 >= len(rows):
        return False
    current = rows[index]
    line, bbox = rows[index + 1]
    gap = _effective_text_row_gap(current, (line, bbox))
    horizontally_related = (
        _bbox_axis_overlap_ratio(current[1], bbox, axis="x") >= 0.35
        or abs(current[1][0] - bbox[0]) <= 0.75 * profile.body_height
    )
    current_centered = abs(_bbox_center_x(current[1]) - (lane_left + lane_width / 2.0)) <= 0.12 * lane_width
    compact_left_aligned = (
        current[1][2] - current[1][0] <= 0.35 * lane_width and abs(current[1][0] - lane_left) <= 0.65 * profile.body_height
    )
    maximum_gap = (1.5 if current_centered else 1.25 if compact_left_aligned else 0.75) * profile.body_height
    height_ratio = _line_effective_height(line, bbox) / profile.body_height
    width_ratio = (bbox[2] - bbox[0]) / lane_width
    return (
        line.semantic_type is None
        and horizontally_related
        and -0.25 * profile.body_height <= gap <= maximum_gap
        and 0.9 <= height_ratio <= 1.2
        and width_ratio >= 0.45
    )


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
        title_indices = [index for index in indices if lane.lines[index][0].semantic_type == "paragraph_title"]
        if not title_indices:
            continue
        anchor_line, anchor_bbox = lane.lines[title_indices[0]]
        anchor_height = _line_effective_height(anchor_line, anchor_bbox)
        compatible = all(
            0.75 <= _line_effective_height(lane.lines[index][0], lane.lines[index][1]) / anchor_height <= 1.25
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
        centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.12 * lane_width
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
    physical_gaps: dict[int, tuple[float | None, float | None]],
) -> float:
    """返回栏内邻行和页面最近物理邻行中较小的归一化净空。"""

    neighbor_index = index + direction
    if 0 <= neighbor_index < len(rows):
        if direction < 0:
            lane_gap = _effective_text_row_gap(rows[neighbor_index], rows[index])
        else:
            lane_gap = _effective_text_row_gap(rows[index], rows[neighbor_index])
    else:
        lane_gap = 0.5 * body_height
    physical_pair = physical_gaps.get(rows[index][0].source_index, (None, None))
    physical_gap = physical_pair[0 if direction < 0 else 1]
    gap = lane_gap if physical_gap is None else min(lane_gap, physical_gap)
    return max(0.0, gap) / max(0.1, body_height)


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
            if (
                selected_line.font_signature is not None
                and candidate_line.font_signature is not None
                and selected_line.font_signature != candidate_line.font_signature
            ):
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
            if vertical_gap > 0.65 * profile.body_height or not (centers_align or left_edges_align or wrapped_indent):
                continue
            candidate_line.semantic_type = "paragraph_title"
            selected_indices.add(candidate_index)
            pending.append(candidate_index)


__all__ = [
    "_classify_paragraph_titles_in_lane",
    "_protect_front_matter_title_types",
    "_visual_row_has_body_style_sibling",
    "_is_near_full_mixed_inline_row",
    "_continues_local_body_row",
    "_is_continuous_field_row",
    "_is_full_width_inline_heading",
    "_has_following_body_row",
    "_has_following_compact_text_section",
    "_unify_visual_row_title_types",
    "_infer_front_matter_boundary",
    "_normalized_title_gap",
    "_expand_paragraph_title_neighbors",
]
