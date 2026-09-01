# Copyright (c) Opendatalab. All rights reserved.

"""提供文本栏带、行距和行连接的共享布局判定。"""

from __future__ import annotations

import math
import re
import statistics


from ....utils.text import is_hyphen_at_line_end
from ....types import BBox

from .models import (
    _LineItem,
    _LocalAxisLine,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_intersects,
    _coerce_bbox,
)


def _title_fonts_compatible(first: _LineItem, second: _LineItem) -> bool:
    """检查标题字体和字重是否兼容；低字体覆盖率时仍保留可靠字重证据。"""

    font_conflicts = (
        first.font_signature is not None
        and second.font_signature is not None
        and first.font_coverage >= 0.75
        and second.font_coverage >= 0.75
        and first.font_signature != second.font_signature
    )
    weight_conflicts = _font_weights_conflict(first, second)
    return not (font_conflicts or weight_conflicts)


def _normalized_font_family(
    signature: tuple[str, int] | None,
) -> str | None:
    """移除 PDF 字体子集前缀并归一化字体族名称，供几何续行作软兼容判断。"""

    if signature is None:
        return None
    name = re.sub(r"^[A-Z]{6}\+", "", signature[0])
    return re.sub(r"[\s_-]+", "", name).casefold() or None


def _font_signatures_share_family(
    first: tuple[str, int] | None,
    second: tuple[str, int] | None,
) -> bool:
    """判断两个可靠字体签名是否仅因 PDF 子集前缀或描述标志不同。"""

    first_family = _normalized_font_family(first)
    second_family = _normalized_font_family(second)
    return first_family is not None and second_family is not None and first_family == second_family


def _font_weights_conflict(first: _LineItem, second: _LineItem) -> bool:
    """判断两行是否存在足以构成段落硬边界的显著字重差异。"""

    return (
        first.dominant_font_weight is not None
        and second.dominant_font_weight is not None
        and abs(first.dominant_font_weight - second.dominant_font_weight) >= 100.0
        and max(first.dominant_font_weight, second.dominant_font_weight)
        >= 1.15 * min(first.dominant_font_weight, second.dominant_font_weight)
    )


def _should_connect_semantic_rows(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
    lane: _TextLane,
    regular_gap: float,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """只用几何、字体和障碍连接同类型语义行，避免标题内容影响聚合。"""

    previous_line, previous_bbox = previous
    current_line, current_bbox = current
    if previous_line.semantic_type != current_line.semantic_type:
        return False
    if _connection_crosses_table(previous_line.bbox, current_line.bbox, table_bboxes):
        return False
    if _horizontal_rule_separates_rows(previous_bbox, current_bbox, lane, axis_lines):
        return False
    previous_height = _line_effective_height(previous_line, previous_bbox)
    current_height = _line_effective_height(current_line, current_bbox)
    pair_height = max(previous_height, current_height)
    if max(previous_height, current_height) / min(previous_height, current_height) > 1.35:
        return False
    vertical_gap = _effective_text_row_gap(previous, current)
    if not -0.25 * pair_height <= vertical_gap <= max(1.25 * pair_height, regular_gap + 0.75 * pair_height):
        return False
    if previous_line.semantic_type == "paragraph_title" and vertical_gap > 0.5 * pair_height:
        return False
    font_conflicts = (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature != current_line.font_signature
    )
    uncertain_document_title_font = (
        previous_line.semantic_type == "doc_title"
        and min(previous_line.font_coverage, current_line.font_coverage) < 0.85
        and (
            previous_line.dominant_font_weight is None
            or current_line.dominant_font_weight is None
            or abs(previous_line.dominant_font_weight - current_line.dominant_font_weight) < 100.0
        )
    )
    if font_conflicts and not uncertain_document_title_font:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    centered_pair = abs(_bbox_center_x(previous_bbox) - _bbox_center_x(current_bbox)) <= 0.15 * lane_width
    aligned_pair = abs(previous_bbox[0] - current_bbox[0]) <= 0.75 * pair_height
    overlapping_pair = _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") >= 0.35
    return centered_pair or aligned_pair or overlapping_pair


def _line_style_scale(line: _LineItem, local_bbox: BBox) -> float:
    """返回 canonical 字体尺度，缺失时兼容旧有效行高与局部 bbox。"""

    return max(
        0.1,
        line.em_height
        if line.style_scale_repaired and line.em_height > 0
        else line.effective_height or (local_bbox[3] - local_bbox[1]),
    )


def _line_canonical_style_scale(line: _LineItem, local_bbox: BBox) -> float:
    """忽略语义选择标记，直接返回 tight/origin 校准后的字体尺度。"""

    return max(
        0.1,
        line.em_height if line.em_height > 0 else line.effective_height or (local_bbox[3] - local_bbox[1]),
    )


def _line_effective_height(line: _LineItem, local_bbox: BBox) -> float:
    """兼容既有布局调用，并统一转发到 canonical 字体尺度。"""

    return _line_style_scale(line, local_bbox)


def _line_layout_height(_line: _LineItem, local_bbox: BBox) -> float:
    """返回 canonical 布局包络高度，供公式与视觉容器空间判断使用。"""

    return max(0.1, local_bbox[3] - local_bbox[1])


def _effective_text_row_gap(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
) -> float:
    """按前一行顶边与有效行高计算净空，避免高数学字形拉长 bbox 底边。"""

    previous_line, previous_bbox = previous
    _current_line, current_bbox = current
    if previous_line.restored_inline_cluster:
        # 二维文本簇的 bbox 底边是真实分母边界；同时截断深度重叠，避免相邻分式互相成为段落屏障。
        return max(
            current_bbox[1] - previous_bbox[3],
            -0.25 * _line_effective_height(previous_line, previous_bbox),
        )
    return current_bbox[1] - (previous_bbox[1] + _line_effective_height(previous_line, previous_bbox))


def _effective_body_text_row_gap(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
) -> float:
    """正文连接优先使用 origin 基线节奏，缺证据时回退既有 bbox 净空。"""

    previous_line, previous_bbox = previous
    current_line, _current_bbox = current
    if (
        previous_line.baseline is not None
        and current_line.baseline is not None
        and current_line.baseline > previous_line.baseline
    ):
        previous_scale = _line_effective_height(previous_line, previous_bbox)
        current_scale = _line_effective_height(*current)
        pitch = current_line.baseline - previous_line.baseline
        if (
            0.5 * min(previous_scale, current_scale)
            <= pitch
            <= 3.0
            * max(
                previous_scale,
                current_scale,
            )
        ):
            return pitch - previous_scale
    return _effective_text_row_gap(previous, current)


def _infer_text_lanes(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    median_height: float,
    *,
    recalculate_intervals: bool = True,
) -> list[_TextLane]:
    """从重复左右边缘推断稳定栏带，并按需用已分配成员重算边界。"""

    anchor_tolerance = max(3.0, 0.75 * median_height)
    anchor_geometry = [
        item for item in line_geometry if item[0].semantic_type not in {"header", "footer", "page_number", "aside_text"}
    ]
    regular_lines = [
        item
        for item in anchor_geometry
        if item[1][2] - item[1][0] >= max(4.0 * _line_effective_height(*item), 0.15 * local_page_width)
    ]
    left_clusters: list[list[tuple[_LineItem, BBox]]] = []
    for item in sorted(regular_lines, key=lambda value: value[1][0]):
        if not left_clusters:
            left_clusters.append([item])
            continue
        cluster_left = statistics.median(member[1][0] for member in left_clusters[-1])
        if abs(item[1][0] - cluster_left) <= anchor_tolerance:
            left_clusters[-1].append(item)
        else:
            left_clusters.append([item])

    supported_intervals = [
        (
            statistics.median(item[1][0] for item in cluster),
            statistics.median(item[1][2] for item in cluster),
            len(cluster),
        )
        for cluster in left_clusters
        if len(cluster) >= 3
    ]
    supported_intervals.sort(key=lambda interval: interval[0])
    filtered_intervals: list[tuple[float, float, int]] = []
    for interval in supported_intervals:
        if not filtered_intervals:
            filtered_intervals.append(interval)
            continue
        previous = filtered_intervals[-1]
        minimum_gutter = max(6.0, 0.75 * median_height)
        if interval[0] - previous[1] >= minimum_gutter:
            filtered_intervals.append(interval)
        elif interval[2] > previous[2]:
            filtered_intervals[-1] = interval

    if not filtered_intervals:
        source = regular_lines or line_geometry
        filtered_intervals = [
            (
                min(item[1][0] for item in source),
                max(item[1][2] for item in source),
                len(source),
            )
        ]

    nested_column_band = None
    if len(filtered_intervals) == 1 and anchor_geometry:
        nested_outer_interval = (
            (
                min(item[1][0] for item in anchor_geometry),
                max(item[1][2] for item in anchor_geometry),
            )
            if recalculate_intervals
            else filtered_intervals[0][:2]
        )
        nested_column_band = _infer_nested_column_band(
            anchor_geometry,
            local_page_width,
            median_height,
            nested_outer_interval,
            enhanced=recalculate_intervals,
        )
    if nested_column_band is not None:
        nested_lanes, band_top, band_bottom = nested_column_band
        fallback_lane = _TextLane(
            left=filtered_intervals[0][0],
            right=filtered_intervals[0][1],
        )
        span_lines: list[tuple[_LineItem, BBox]] = []
        for item in line_geometry:
            line, bbox = item
            center_y = _bbox_center_y(bbox)
            if (
                line.semantic_type in {"header", "footer", "page_number", "page_footnote", "aside_text"}
                or not band_top <= center_y <= band_bottom
            ):
                fallback_lane.lines.append(item)
                continue
            line_width = max(0.1, bbox[2] - bbox[0])
            scored_lanes = [
                (
                    max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left)) / line_width,
                    lane,
                )
                for lane in nested_lanes
            ]
            coverage_scores = sorted(
                (coverage for coverage, _lane in scored_lanes),
                reverse=True,
            )
            best_coverage, best_lane = max(scored_lanes, key=lambda value: value[0])
            fits_only_one_lane = _fits_only_one_lane(
                bbox,
                best_lane,
                nested_lanes,
                anchor_tolerance,
            )
            if len(coverage_scores) > 1 and coverage_scores[1] >= 0.2 and not fits_only_one_lane:
                span_lines.append(item)
                continue
            if best_coverage >= 0.5 or fits_only_one_lane:
                best_lane.lines.append(item)
            else:
                span_lines.append(item)
        lanes = [lane for lane in nested_lanes if lane.lines]
        if recalculate_intervals:
            _expand_nested_lane_intervals_from_members(
                lanes,
                anchor_tolerance,
            )
        if fallback_lane.lines:
            lanes.append(fallback_lane)
        if span_lines:
            lanes.append(
                _TextLane(
                    left=min(item[1][0] for item in span_lines),
                    right=max(item[1][2] for item in span_lines),
                    lines=span_lines,
                    is_span=True,
                )
            )
        _reattach_span_lane_continuations(lanes, median_height)
        _reattach_cross_lane_short_tails(lanes, median_height)
        return lanes

    lanes = [_TextLane(left=left, right=right) for left, right, _support in filtered_intervals]
    span_lines: list[tuple[_LineItem, BBox]] = []
    for item in line_geometry:
        bbox = item[1]
        line_width = max(0.1, bbox[2] - bbox[0])
        scored_lanes = [
            (
                max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left)) / line_width,
                lane,
            )
            for lane in lanes
        ]
        coverage_scores = sorted(
            (coverage for coverage, _lane in scored_lanes),
            reverse=True,
        )
        best_coverage, best_lane = max(scored_lanes, key=lambda value: value[0])
        fits_only_one_lane = _fits_only_one_lane(
            bbox,
            best_lane,
            lanes,
            anchor_tolerance,
        )
        # 同时覆盖两个稳定正文栏的行仍属于跨栏内容；只进入单侧栏且未越过栏沟的
        # 宽正文行则回到该栏，避免窄图注把正文错误挤入 span lane。
        if len(coverage_scores) > 1 and coverage_scores[1] >= 0.2 and not fits_only_one_lane:
            span_lines.append(item)
            continue
        if len(lanes) == 1 or fits_only_one_lane:
            best_lane.lines.append(item)
        else:
            span_lines.append(item)

    if recalculate_intervals:
        _expand_nested_lane_intervals_from_members(
            lanes,
            anchor_tolerance,
        )
    if span_lines:
        lanes.append(
            _TextLane(
                left=min(item[1][0] for item in span_lines),
                right=max(item[1][2] for item in span_lines),
                lines=span_lines,
                is_span=True,
            )
        )
    _reattach_span_lane_continuations(lanes, median_height)
    _reattach_cross_lane_short_tails(lanes, median_height)
    return lanes


def _fits_only_one_lane(
    bbox: BBox,
    best_lane: _TextLane,
    lanes: list[_TextLane],
    tolerance: float,
) -> bool:
    """判断宽行是否仍完整停留在某一栏及其栏沟边界以内。"""

    ordered = sorted(lanes, key=lambda lane: lane.left)
    lane_index = ordered.index(best_lane)
    if lane_index > 0 and bbox[0] < ordered[lane_index - 1].right - tolerance:
        return False
    if lane_index + 1 < len(ordered) and bbox[2] > ordered[lane_index + 1].left - 0.25 * tolerance:
        return False
    return best_lane.left - tolerance <= _bbox_center_x(bbox) <= best_lane.right + max(tolerance, bbox[2] - best_lane.right)


def _expand_nested_lane_intervals_from_members(
    lanes: list[_TextLane],
    tolerance: float,
) -> None:
    """按已归属成员扩展局部栏边界，并在相邻栏相交前保留稳定栏沟。"""

    for lane in lanes:
        alignment_tolerance = max(1.0, 0.5 * tolerance)
        body_members = [bbox for line, bbox in lane.lines if line.semantic_type is None]
        if not body_members:
            continue
        aligned_members = [
            bbox
            for bbox in body_members
            if abs(bbox[0] - lane.left) <= alignment_tolerance or abs(bbox[2] - lane.right) <= alignment_tolerance
        ]
        if not aligned_members:
            continue
        # 页眉、页码和标题不参与；同时只让至少一侧锚点稳定的正文扩张栏宽，
        # 避免页面后续另一版式区段把当前局部栏整体拉宽。
        lane.left = min(lane.left, min(bbox[0] for bbox in aligned_members))
        lane.right = max(lane.right, max(bbox[2] for bbox in aligned_members))
    ordered = sorted(lanes, key=lambda lane: lane.left)
    for left_lane, right_lane in zip(ordered, ordered[1:]):
        if left_lane.right < right_lane.left - tolerance:
            continue
        midpoint = (left_lane.right + right_lane.left) / 2.0
        left_lane.right = min(left_lane.right, midpoint)
        right_lane.left = max(right_lane.left, midpoint)


def _infer_nested_column_band(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    median_height: float,
    outer_interval: tuple[float, float],
    *,
    enhanced: bool = False,
) -> tuple[list[_TextLane], float, float] | None:
    """在全宽版心内查找仅占局部纵向区间的并列正文栏。"""

    outer_width = max(0.1, outer_interval[1] - outer_interval[0])
    candidates = [
        item
        for item in line_geometry
        if item[0].semantic_type is None
        and max(
            4.0 * _line_effective_height(*item),
            0.12 * local_page_width,
        )
        <= item[1][2] - item[1][0]
        <= 0.62 * outer_width
    ]
    if len(candidates) < 6:
        return None

    center_tolerance = max(2.0 * median_height, 0.06 * local_page_width)
    center_clusters: list[list[tuple[_LineItem, BBox]]] = []
    for item in sorted(candidates, key=lambda value: _bbox_center_x(value[1])):
        center = _bbox_center_x(item[1])
        target = next(
            (
                cluster
                for cluster in center_clusters
                if abs(center - statistics.median(_bbox_center_x(member[1]) for member in cluster)) <= center_tolerance
            ),
            None,
        )
        if target is None:
            center_clusters.append([item])
        else:
            target.append(item)

    supported = [cluster for cluster in center_clusters if len(cluster) >= 3]
    supported.sort(key=lambda cluster: statistics.median(_bbox_center_x(item[1]) for item in cluster))
    best_pair: (
        tuple[
            tuple[int, float, float],
            list[tuple[_LineItem, BBox]],
            list[tuple[_LineItem, BBox]],
            tuple[float, float],
            tuple[float, float],
        ]
        | None
    ) = None
    cluster_pairs = (
        [
            (left_cluster, right_cluster)
            for left_index, left_cluster in enumerate(supported[:-1])
            for right_cluster in supported[left_index + 1 :]
        ]
        if enhanced
        else list(zip(supported, supported[1:]))
    )
    for left_cluster, right_cluster in cluster_pairs:
        left_interval = (
            statistics.median(item[1][0] for item in left_cluster),
            statistics.median(item[1][2] for item in left_cluster),
        )
        right_interval = (
            statistics.median(item[1][0] for item in right_cluster),
            statistics.median(item[1][2] for item in right_cluster),
        )
        gutter = right_interval[0] - left_interval[1]
        common_top = max(
            min(item[1][1] for item in left_cluster),
            min(item[1][1] for item in right_cluster),
        )
        common_bottom = min(
            max(item[1][3] for item in left_cluster),
            max(item[1][3] for item in right_cluster),
        )
        combined_width = right_interval[1] - left_interval[0]
        if (
            gutter < max(6.0, 0.75 * median_height)
            or common_bottom - common_top < 2.0 * median_height
            or combined_width < 0.55 * outer_width
        ):
            continue
        score = (
            min(len(left_cluster), len(right_cluster)),
            common_bottom - common_top,
            gutter,
        )
        candidate_pair = (
            score,
            left_cluster,
            right_cluster,
            left_interval,
            right_interval,
        )
        if best_pair is None or candidate_pair[0] > best_pair[0]:
            best_pair = candidate_pair
    if best_pair is None:
        return None

    _score, left_cluster, right_cluster, left_interval, right_interval = best_pair
    band_top = min(item[1][1] for item in [*left_cluster, *right_cluster]) - 0.5 * median_height
    band_bottom = max(item[1][3] for item in [*left_cluster, *right_cluster]) + 0.5 * median_height
    return (
        [
            _TextLane(left=left_interval[0], right=left_interval[1]),
            _TextLane(left=right_interval[0], right=right_interval[1]),
        ],
        band_top,
        band_bottom,
    )


def _reattach_span_lane_continuations(
    lanes: list[_TextLane],
    median_height: float,
) -> None:
    """把紧随稳定跨栏多行之后的单栏宽短尾行迁回对应 span lane。"""

    regular_lanes = [lane for lane in lanes if not lane.is_span]
    span_lanes = [lane for lane in lanes if lane.is_span]
    if len(regular_lanes) < 2 or not span_lanes:
        return

    for span_lane in span_lanes:
        _reattach_repeated_indented_span_tails(
            span_lane,
            regular_lanes,
            median_height,
        )
        while True:
            span_lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            candidates: list[
                tuple[
                    float,
                    float,
                    _TextLane,
                    tuple[_LineItem, BBox],
                ]
            ] = []
            for regular_lane in regular_lanes:
                for candidate in regular_lane.lines:
                    candidate_line, candidate_bbox = candidate
                    preceding = [
                        item
                        for item in span_lane.lines
                        if item[0].semantic_type == candidate_line.semantic_type and item[1][1] < candidate_bbox[1]
                    ]
                    if len(preceding) < 2:
                        continue
                    previous, last = preceding[-2:]
                    previous_height = _line_effective_height(*previous)
                    last_height = _line_effective_height(*last)
                    if (
                        abs(previous[1][0] - last[1][0]) > 0.75 * median_height
                        or max(previous_height, last_height) / min(previous_height, last_height) > 1.35
                        or not _title_fonts_compatible(previous[0], last[0])
                        or not -0.25 * median_height <= _effective_text_row_gap(previous, last) <= 0.75 * median_height
                    ):
                        continue
                    gap = _effective_text_row_gap(last, candidate)
                    candidate_height = _line_effective_height(*candidate)
                    if (
                        not -0.25 * median_height <= gap <= 0.75 * median_height
                        or abs(candidate_bbox[0] - last[1][0]) > 0.75 * median_height
                        or max(last_height, candidate_height) / min(last_height, candidate_height) > 1.35
                        or not _title_fonts_compatible(last[0], candidate_line)
                    ):
                        continue
                    has_parallel_peer = any(
                        other_line is not candidate_line
                        and _bbox_axis_overlap_ratio(
                            candidate_bbox,
                            other_bbox,
                            axis="y",
                        )
                        >= 0.5
                        for lane in regular_lanes
                        for other_line, other_bbox in lane.lines
                    )
                    if has_parallel_peer:
                        continue
                    candidates.append(
                        (
                            candidate_bbox[1],
                            max(0.0, gap),
                            regular_lane,
                            candidate,
                        )
                    )
            if not candidates:
                break
            _top, _gap, regular_lane, candidate = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            regular_lane.lines.remove(candidate)
            span_lane.lines.append(candidate)
            span_lane.left = min(span_lane.left, candidate[1][0])
            span_lane.right = max(span_lane.right, candidate[1][2])
            span_lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))


def _reattach_cross_lane_short_tails(
    lanes: list[_TextLane],
    median_height: float,
) -> None:
    """把误入另一栏带、但完整落在唯一前序栏内的正文短尾迁回原栏。"""

    while True:
        moves: list[
            tuple[
                float,
                _TextLane,
                _TextLane,
                tuple[_LineItem, BBox],
            ]
        ] = []
        for source_lane in lanes:
            for candidate in source_lane.lines:
                candidate_line, candidate_bbox = candidate
                if candidate_line.semantic_type is not None:
                    continue
                matches: list[tuple[float, _TextLane]] = []
                for target_lane in lanes:
                    if target_lane is source_lane or not target_lane.lines:
                        continue
                    preceding = [
                        item
                        for item in target_lane.lines
                        if item[0].semantic_type == candidate_line.semantic_type and item[1][1] < candidate_bbox[1]
                    ]
                    if not preceding:
                        continue
                    previous = max(
                        preceding,
                        key=lambda item: (item[1][1], item[1][0]),
                    )
                    previous_line, previous_bbox = previous
                    pair_height = max(
                        _line_effective_height(*previous),
                        _line_effective_height(*candidate),
                        median_height,
                    )
                    lane_width = max(0.1, target_lane.right - target_lane.left)
                    if (
                        previous_bbox[2] - previous_bbox[0] < 0.65 * lane_width
                        or candidate_bbox[2] - candidate_bbox[0] > 0.85 * lane_width
                        or candidate_bbox[0] < target_lane.left - 0.75 * pair_height
                        or candidate_bbox[2] > target_lane.right + 0.75 * pair_height
                        or abs(candidate_bbox[0] - previous_bbox[0]) > 0.75 * pair_height
                        or not _title_fonts_compatible(previous_line, candidate_line)
                    ):
                        continue
                    gap = _effective_body_text_row_gap(previous, candidate)
                    if not -0.25 * pair_height <= gap <= 0.9 * pair_height:
                        continue
                    if (
                        previous_line.visual_row_id is not None
                        and candidate_line.visual_row_id is not None
                        and not 0 < candidate_line.visual_row_id - previous_line.visual_row_id <= 2
                    ):
                        continue
                    matches.append((max(0.0, gap), target_lane))
                if len(matches) == 1:
                    gap, target_lane = matches[0]
                    moves.append((candidate_bbox[1] + gap, source_lane, target_lane, candidate))
        if not moves:
            return
        _score, source_lane, target_lane, candidate = min(
            moves,
            key=lambda item: item[0],
        )
        if candidate not in source_lane.lines:
            continue
        source_lane.lines.remove(candidate)
        target_lane.lines.append(candidate)
        target_lane.lines.sort(
            key=lambda item: (item[1][1], item[1][0], item[0].source_index),
        )


def _reattach_repeated_indented_span_tails(
    span_lane: _TextLane,
    regular_lanes: list[_TextLane],
    median_height: float,
) -> None:
    """识别重复的跨栏首行与缩进短尾，并把短尾统一迁回跨栏栏带。"""

    span_rows = sorted(
        span_lane.lines,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    matches: list[tuple[_TextLane, tuple[_LineItem, BBox], tuple[_LineItem, BBox], float]] = []
    for span_index, span_row in enumerate(span_rows):
        span_line, span_bbox = span_row
        next_span_top = span_rows[span_index + 1][1][1] if span_index + 1 < len(span_rows) else float("inf")
        for regular_lane in regular_lanes:
            for candidate in regular_lane.lines:
                candidate_line, candidate_bbox = candidate
                if candidate_line.semantic_type != span_line.semantic_type:
                    continue
                gap = _effective_text_row_gap(span_row, candidate)
                indent = candidate_bbox[0] - span_bbox[0]
                span_height = _line_effective_height(*span_row)
                candidate_height = _line_effective_height(*candidate)
                if (
                    candidate_bbox[1] <= span_bbox[1]
                    or candidate_bbox[1] >= next_span_top
                    or not -0.25 * median_height <= gap <= 0.75 * median_height
                    or not 0.75 * median_height <= indent <= 6.0 * median_height
                    or max(span_height, candidate_height) / min(span_height, candidate_height) > 1.35
                    or not _title_fonts_compatible(span_line, candidate_line)
                ):
                    continue
                has_parallel_peer = any(
                    other_line is not candidate_line and _bbox_axis_overlap_ratio(candidate_bbox, other_bbox, axis="y") >= 0.5
                    for lane in regular_lanes
                    for other_line, other_bbox in lane.lines
                )
                if not has_parallel_peer:
                    matches.append((regular_lane, candidate, span_row, indent))

    if len(matches) < 2:
        return
    median_indent = statistics.median(match[3] for match in matches)
    supported = [match for match in matches if abs(match[3] - median_indent) <= max(0.75 * median_height, 0.25 * median_indent)]
    if len(supported) < 2:
        return
    for regular_lane, candidate, _span_row, _indent in supported:
        if candidate not in regular_lane.lines:
            continue
        regular_lane.lines.remove(candidate)
        span_lane.lines.append(candidate)
        span_lane.left = min(span_lane.left, candidate[1][0])
        span_lane.right = max(span_lane.right, candidate[1][2])
    span_lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))


def _estimate_lane_gap(lane: _TextLane) -> tuple[float, float]:
    """从栏带内兼容相邻行的较小间隙簇估计常规净空和 MAD。"""

    lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    heights = [_line_effective_height(line, bbox) for line, bbox in lane.lines]
    median_height = statistics.median(heights) if heights else 1.0
    gaps: list[float] = []
    for previous, current in zip(lane.lines, lane.lines[1:]):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        if previous_line.visual_row_id == current_line.visual_row_id and (
            previous_line.split_from_row or current_line.split_from_row
        ):
            continue
        previous_height = _line_effective_height(previous_line, previous_bbox)
        current_height = _line_effective_height(current_line, current_bbox)
        if max(previous_height, current_height) / min(previous_height, current_height) > 1.35:
            continue
        pair_height = max(previous_height, current_height)
        gap = _effective_text_row_gap(previous, current)
        if gap < -0.25 * pair_height or gap > 2.0 * pair_height:
            continue
        if (
            _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5
            and abs(previous_bbox[0] - current_bbox[0]) > 1.5 * median_height
        ):
            continue
        # PDF 字符框常在相邻基线间产生极小重叠；按零净空计入常规行距统计。
        gaps.append(max(0.0, gap))

    if not gaps:
        return 0.35 * median_height, 0.0
    sorted_gaps = sorted(gaps)
    lower_count = max(1, math.ceil(len(sorted_gaps) * 0.6))
    lower_gaps = sorted_gaps[:lower_count]
    regular_gap = statistics.median(lower_gaps)
    gap_mad = statistics.median(abs(gap - regular_gap) for gap in lower_gaps)
    return regular_gap, gap_mad


def _is_structural_typography_gap(
    previous_height: float,
    current_height: float,
    vertical_gap: float,
    regular_gap: float,
    gap_mad: float,
    *,
    reliable_style_change: bool = False,
) -> bool:
    """判断异常段间净空是否同时具有行高或可靠字体层级变化。"""

    pair_height = max(previous_height, current_height)
    minimum_height = max(0.1, min(previous_height, current_height))
    prominent_gap = vertical_gap > regular_gap + max(
        0.75 * pair_height,
        3.0 * gap_mad,
    )
    return prominent_gap and (pair_height / minimum_height >= 1.12 or reliable_style_change)


def _should_connect_text_rows(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """综合局部间距、首行缩进、字体和障碍判断两个相邻视觉行是否同段。"""

    previous_line, previous_bbox = previous
    current_line, current_bbox = current
    previous_height = _line_effective_height(previous_line, previous_bbox)
    current_height = _line_effective_height(current_line, current_bbox)
    pair_height = max(previous_height, current_height)
    lane_width = max(0.1, lane.right - lane.left)
    previous_width = previous_bbox[2] - previous_bbox[0]
    current_width = current_bbox[2] - current_bbox[0]
    vertical_gap = _effective_body_text_row_gap(previous, current)
    if previous_line.visual_row_id == current_line.visual_row_id and (
        previous_line.split_from_row or current_line.split_from_row
    ):
        return False
    if current_height < 0.88 * previous_height and vertical_gap > regular_gap + max(0.25 * previous_height, 3.0 * gap_mad):
        return False
    both_fill_lane = previous_width >= 0.8 * lane_width and current_width >= 0.8 * lane_width
    aligned_left_edges = abs(previous_bbox[0] - current_bbox[0]) <= 0.5 * pair_height
    current_returns_to_lane_left = (
        abs(current_bbox[0] - lane.left) <= 0.75 * pair_height
        and -0.5 * pair_height <= previous_bbox[0] - lane.left <= 2.0 * pair_height
    )
    reliable_font_match = (
        previous_line.font_signature is None
        or current_line.font_signature is None
        or previous_line.font_coverage < 0.75
        or current_line.font_coverage < 0.75
        or previous_line.font_signature == current_line.font_signature
        or _font_signatures_share_family(
            previous_line.font_signature,
            current_line.font_signature,
        )
        or (current_width <= 0.5 * lane_width and previous_line.font_signature[1] == current_line.font_signature[1])
    )
    previous_indent = previous_bbox[0] - lane.left
    repeated_indent_continuation = (
        previous_indent >= max(5.0, 1.8 * pair_height)
        and abs(current_bbox[0] - previous_bbox[0]) <= 0.5 * pair_height
        and reliable_font_match
        and -0.25 * pair_height <= vertical_gap <= regular_gap + max(0.75 * pair_height, 3.0 * gap_mad)
    )
    safe_short_tail = (
        previous_width >= 0.75 * lane_width
        and current_width <= 0.7 * lane_width
        and (aligned_left_edges or current_returns_to_lane_left)
        and reliable_font_match
        and not _font_weights_conflict(previous_line, current_line)
        and -0.25 * pair_height <= vertical_gap <= regular_gap + max(0.75 * pair_height, 3.0 * gap_mad)
    )
    height_ratio = max(previous_height, current_height) / min(previous_height, current_height)
    font_style_changed = (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_signature[1] != current_line.font_signature[1]
        and not _font_signatures_share_family(
            previous_line.font_signature,
            current_line.font_signature,
        )
    )
    reliable_style_conflict = (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and (
            (
                previous_line.font_signature != current_line.font_signature
                and not _font_signatures_share_family(
                    previous_line.font_signature,
                    current_line.font_signature,
                )
            )
            or _font_weights_conflict(previous_line, current_line)
        )
    )
    fallback_font_continuation = (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature[0] != current_line.font_signature[0]
        and previous_line.font_signature[1] == current_line.font_signature[1]
        and aligned_left_edges
        and height_ratio <= 1.25
        and not _font_weights_conflict(previous_line, current_line)
        and -0.25 * pair_height <= vertical_gap <= regular_gap + max(0.35 * pair_height, 3.0 * gap_mad)
    )
    if font_style_changed and _font_weights_conflict(previous_line, current_line) and not fallback_font_continuation:
        # 显式样式位与显著字重同时变化仍是硬边界，不能被满栏几何放宽。
        return False
    if (
        not is_hyphen_at_line_end(previous_line.text)
        and _is_structural_typography_gap(
            previous_height,
            current_height,
            vertical_gap,
            regular_gap,
            gap_mad,
            reliable_style_change=reliable_style_conflict,
        )
        and not fallback_font_continuation
    ):
        # 图注到正文等排版层级转换即使同栏满行，也不能被常规续行规则重新吸收。
        return False
    full_width_continuation = (
        both_fill_lane
        and aligned_left_edges
        and not font_style_changed
        and vertical_gap <= regular_gap + max(0.75 * min(previous_height, current_height), 3.0 * gap_mad)
    )
    if height_ratio > 1.35 and not safe_short_tail and not full_width_continuation:
        # 满栏混合字体可跨字号续接，但显式正体/斜体等样式边界仍保持原分段语义。
        if not both_fill_lane or not aligned_left_edges or font_style_changed:
            return False

    if vertical_gap < -0.25 * pair_height:
        return False
    if (
        _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5
        and abs(previous_bbox[0] - current_bbox[0]) > 1.5 * pair_height
        and not safe_short_tail
    ):
        return False
    if _connection_crosses_table(previous_line.bbox, current_line.bbox, table_bboxes):
        return False
    if _horizontal_rule_separates_rows(previous_bbox, current_bbox, lane, axis_lines):
        return False

    gap_limit = max(
        regular_gap + max(0.5 * pair_height, 3.0 * gap_mad),
        1.1 * pair_height,
    )
    # 排版断词可以跳过缩进、字体和短行规则，但仍须限制在邻近物理行内，
    # 避免页内远距离的 “cross-” 与后续标题被误拼为同一段。
    if is_hyphen_at_line_end(previous_line.text):
        return vertical_gap <= max(gap_limit, 1.8 * pair_height)
    if vertical_gap > gap_limit:
        return False

    terminal_previous = bool(re.search(r"[.!?。！？:：;；][\]\)}）】》”’'\"]*$", previous_line.text.rstrip()))
    sparse_lane = sum(line.semantic_type is None for line, _bbox in lane.lines) <= 6
    if (
        terminal_previous
        and not repeated_indent_continuation
        and ((sparse_lane and vertical_gap > 0.65 * pair_height) or vertical_gap > regular_gap + 0.5 * pair_height)
    ):
        return False

    # 局部版心可能比整栏推断边界更靠左，缩进需同时参考上一物理行。
    local_lane_left = min(lane.left, previous_bbox[0])
    local_lane_width = max(0.1, lane.right - local_lane_left)
    next_indent = current_bbox[0] - local_lane_left
    previous_fill = max(0.0, previous_bbox[2] - local_lane_left) / local_lane_width
    if (
        next_indent >= max(5.0, 0.65 * pair_height)
        and (previous_fill <= 0.8 or terminal_previous)
        and not safe_short_tail
        and not repeated_indent_continuation
    ):
        # 已确认的同左缘短尾优先于栏左缘缩进，避免参考文献冒号后的末行被切断。
        return False

    abnormal_gap = vertical_gap > regular_gap + max(0.25 * pair_height, 3.0 * gap_mad)
    if (
        reliable_style_conflict
        and (abnormal_gap or min(previous_width, current_width) <= 0.7 * lane_width)
        and not both_fill_lane
        and not safe_short_tail
        and not fallback_font_continuation
    ):
        return False
    if abnormal_gap and min(previous_width, current_width) <= 0.65 * lane_width and not safe_short_tail:
        return False
    return True


def _horizontal_rule_separates_rows(
    previous_bbox: BBox,
    current_bbox: BBox,
    lane: _TextLane,
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """检查两个相邻文本行之间是否存在覆盖当前栏带的长水平规则线。"""

    if current_bbox[1] <= previous_bbox[3]:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    for axis_line in axis_lines:
        if axis_line.orientation != "horizontal":
            continue
        line_y = _bbox_center_y(axis_line.bbox)
        if not previous_bbox[3] <= line_y <= current_bbox[1]:
            continue
        overlap = max(0.0, min(axis_line.bbox[2], lane.right) - max(axis_line.bbox[0], lane.left))
        if overlap / lane_width >= 0.6:
            return True
    return False


def _connection_crosses_table(
    first_bbox: BBox,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """检查两行中心连接区域是否穿过已确认表格。"""

    first_center = (_bbox_center_x(first_bbox), _bbox_center_y(first_bbox))
    second_center = (_bbox_center_x(second_bbox), _bbox_center_y(second_bbox))
    connector = _coerce_bbox(
        (
            min(first_center[0], second_center[0]) - 0.1,
            min(first_center[1], second_center[1]) - 0.1,
            max(first_center[0], second_center[0]) + 0.1,
            max(first_center[1], second_center[1]) + 0.1,
        )
    )
    return connector is not None and any(_bbox_intersects(connector, table_bbox) for table_bbox in table_bboxes)
