# Copyright (c) Opendatalab. All rights reserved.

"""提供文本栏带、行距和行连接的共享布局判定。"""

from __future__ import annotations

import math
import re
import statistics


from mineru.backend.utils.char_utils import is_hyphen_at_line_end
from mineru.types import BBox

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
    weight_conflicts = (
        first.dominant_font_weight is not None
        and second.dominant_font_weight is not None
        and abs(first.dominant_font_weight - second.dominant_font_weight) >= 100.0
        and max(first.dominant_font_weight, second.dominant_font_weight)
        >= 1.15 * min(first.dominant_font_weight, second.dominant_font_weight)
    )
    return not (font_conflicts or weight_conflicts)


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
    if (
        previous_line.semantic_type == "paragraph_title"
        and vertical_gap > 0.65 * pair_height
    ):
        return False
    if (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature != current_line.font_signature
    ):
        return False
    lane_width = max(0.1, lane.right - lane.left)
    centered_pair = (
        abs(_bbox_center_x(previous_bbox) - _bbox_center_x(current_bbox)) <= 0.15 * lane_width
    )
    aligned_pair = abs(previous_bbox[0] - current_bbox[0]) <= 0.75 * pair_height
    overlapping_pair = _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") >= 0.35
    return centered_pair or aligned_pair or overlapping_pair


def _line_effective_height(line: _LineItem, local_bbox: BBox) -> float:
    """返回字符统计得到的有效行高，缺失时回退到局部 bbox 高度。"""

    return max(0.1, line.effective_height or (local_bbox[3] - local_bbox[1]))


def _effective_text_row_gap(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
) -> float:
    """按前一行顶边与有效行高计算净空，避免高数学字形拉长 bbox 底边。"""

    previous_line, previous_bbox = previous
    _current_line, current_bbox = current
    return current_bbox[1] - (previous_bbox[1] + _line_effective_height(previous_line, previous_bbox))


def _infer_text_lanes(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    median_height: float,
) -> list[_TextLane]:
    """从重复左右边缘推断稳定栏带，并把跨栏行放入独立 span lane。"""

    anchor_tolerance = max(3.0, 0.75 * median_height)
    regular_lines = [
        item
        for item in line_geometry
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
        # 同时覆盖两个稳定正文栏的短尾行仍属于跨栏内容，不能按中心点落回单栏。
        if len(coverage_scores) > 1 and coverage_scores[1] >= 0.2:
            span_lines.append(item)
            continue
        best_coverage, best_lane = max(scored_lanes, key=lambda value: value[0])
        if len(lanes) == 1 or (
            best_coverage >= 0.5
            and best_lane.left - anchor_tolerance
            <= _bbox_center_x(bbox)
            <= best_lane.right + anchor_tolerance
        ):
            best_lane.lines.append(item)
        else:
            span_lines.append(item)

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
    return lanes


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
        span_lane.lines.sort(
            key=lambda item: (item[1][1], item[1][0], item[0].source_index)
        )
        if len(span_lane.lines) < 2:
            continue
        previous, last = span_lane.lines[-2:]
        previous_height = _line_effective_height(*previous)
        last_height = _line_effective_height(*last)
        if (
            previous[0].semantic_type != last[0].semantic_type
            or abs(previous[1][0] - last[1][0]) > 0.75 * median_height
            or max(previous_height, last_height) / min(previous_height, last_height) > 1.35
            or not _title_fonts_compatible(previous[0], last[0])
            or not -0.25 * median_height
            <= _effective_text_row_gap(previous, last)
            <= 0.75 * median_height
        ):
            continue

        while True:
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
                    if (
                        candidate_line.semantic_type != last[0].semantic_type
                        or candidate_bbox[1] <= last[1][1]
                    ):
                        continue
                    gap = _effective_text_row_gap(last, candidate)
                    candidate_height = _line_effective_height(*candidate)
                    if (
                        not -0.25 * median_height <= gap <= 0.75 * median_height
                        or abs(candidate_bbox[0] - last[1][0]) > 0.75 * median_height
                        or max(last_height, candidate_height)
                        / min(last_height, candidate_height)
                        > 1.35
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
                            max(0.0, gap),
                            candidate_bbox[1],
                            regular_lane,
                            candidate,
                        )
                    )
            if not candidates:
                break
            _gap, _top, regular_lane, candidate = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            regular_lane.lines.remove(candidate)
            span_lane.lines.append(candidate)
            span_lane.left = min(span_lane.left, candidate[1][0])
            span_lane.right = max(span_lane.right, candidate[1][2])
            span_lane.lines.sort(
                key=lambda item: (item[1][1], item[1][0], item[0].source_index)
            )
            last = candidate
            last_height = _line_effective_height(*last)


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
        if _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5 and abs(
            previous_bbox[0] - current_bbox[0]
        ) > 1.5 * median_height:
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
    vertical_gap = _effective_text_row_gap(previous, current)
    if previous_line.visual_row_id == current_line.visual_row_id and (
        previous_line.split_from_row or current_line.split_from_row
    ):
        return False
    if (
        current_height < 0.88 * previous_height
        and vertical_gap > regular_gap + 0.25 * previous_height
    ):
        return False
    height_ratio = max(previous_height, current_height) / min(previous_height, current_height)
    if height_ratio > 1.35:
        both_fill_lane = previous_width >= 0.8 * lane_width and current_width >= 0.8 * lane_width
        aligned_left_edges = abs(previous_bbox[0] - current_bbox[0]) <= 0.5 * pair_height
        font_style_changed = (
            previous_line.font_signature is not None
            and current_line.font_signature is not None
            and previous_line.font_signature[1] != current_line.font_signature[1]
        )
        # 满栏混合字体可跨字号续接，但显式正体/斜体等样式边界仍保持原分段语义。
        if not both_fill_lane or not aligned_left_edges or font_style_changed:
            return False

    if vertical_gap < -0.25 * pair_height:
        return False
    if _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5 and abs(
        previous_bbox[0] - current_bbox[0]
    ) > 1.5 * pair_height:
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
    if terminal_previous and vertical_gap > regular_gap + 0.5 * pair_height:
        return False

    # 局部版心可能比整栏推断边界更靠左，缩进需同时参考上一物理行。
    local_lane_left = min(lane.left, previous_bbox[0])
    local_lane_width = max(0.1, lane.right - local_lane_left)
    next_indent = current_bbox[0] - local_lane_left
    previous_fill = max(0.0, previous_bbox[2] - local_lane_left) / local_lane_width
    if next_indent >= max(5.0, 0.65 * pair_height) and (previous_fill <= 0.8 or terminal_previous):
        return False

    abnormal_gap = vertical_gap > regular_gap + 0.25 * pair_height
    if (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature != current_line.font_signature
        and (abnormal_gap or min(previous_width, current_width) <= 0.7 * lane_width)
    ):
        return False
    if abnormal_gap and min(previous_width, current_width) <= 0.65 * lane_width:
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

