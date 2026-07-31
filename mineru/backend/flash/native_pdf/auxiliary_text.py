# Copyright (c) Opendatalab. All rights reserved.

"""分类页眉、页脚、页码、侧栏和页脚注。"""

from __future__ import annotations

import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from typing import Literal


from mineru.types import BBox

from .models import (
    _AxisLine,
    _LineItem,
    _LocalAxisLine,
    _MarginalCandidate,
    _PageSource,
    _PreparedPage,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_intersects,
    _clip_bbox,
    _coerce_bbox,
    _expand_bbox,
    _horizontal_bbox_gap,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from .line_layout import (
    _effective_text_row_gap,
    _infer_text_lanes,
    _line_effective_height,
)
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?[\-\u2013\u2014\u00b7\u2022]*\s*(?:\u7b2c\s*)?"
    r"(?P<value>\d{1,4}|[ivxlcdm]+|[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24]+)"
    r"(?:\s*(?:/|of|\u5171)\s*(?:\d{1,4}|[ivxlcdm]+|[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24]+))?"
    r"\s*(?:\u9875)?\s*[\-\u2013\u2014\u00b7\u2022]*\s*$",
    re.IGNORECASE,
)


def _classify_page_auxiliary_text(prepared: _PreparedPage) -> None:
    """在容器认领后仅按空间关系标注侧栏文字和页脚注。"""

    _classify_aside_text(prepared.remaining_lines, prepared.page_size)
    _classify_image_footnotes(
        prepared.remaining_lines,
        [
            block["bbox"]
            for block in prepared.fixed_blocks
            if block.get("type") == "image"
        ],
        prepared.table_bboxes,
        prepared.drawing_lines,
        prepared.page_size,
    )
    prepared.page_footnote_groups = _classify_page_footnotes(
        prepared.remaining_lines,
        prepared.table_bboxes,
        prepared.drawing_lines,
        prepared.page_size,
        visual_bboxes=[
            block["bbox"]
            for block in prepared.fixed_blocks
            if block.get("type") == "image"
        ],
    )


def _classify_aside_text(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> None:
    """在横排正文占绝对多数时，以边缘带和物理尺寸识别垂直侧栏。"""

    available = [line for line in lines if line.semantic_type is None]
    upright_lines = [line for line in available if line.angle == 0]
    if len(upright_lines) < 4:
        return

    support_by_angle = _geometric_text_support_by_angle(available, page_size)
    total_support = sum(support_by_angle.values())
    if total_support <= 0 or support_by_angle.get(0, 0.0) / total_support < 0.8:
        return

    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return
    # 侧栏必须完整位于 12% 边缘带，且兼具不超过 8% 的窄宽和至少 15% 的物理高度。
    aside_source_indices = {
        line.source_index
        for line in available
        if line.angle in {90, 270}
        and line.bbox[2] - line.bbox[0] <= 0.08 * page_width
        and line.bbox[3] - line.bbox[1] >= 0.15 * page_height
        and (
            line.bbox[2] <= 0.12 * page_width
            or line.bbox[0] >= 0.88 * page_width
        )
    }
    for line in available:
        if line.source_index in aside_source_indices:
            line.semantic_type = "aside_text"


def _geometric_text_support_by_angle(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> dict[int, float]:
    """按局部行宽乘有效行高累计各文字方向的纯几何支持度。"""

    support_by_angle: dict[int, float] = {}
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        local_width = max(0.1, local_bbox[2] - local_bbox[0])
        support_by_angle[line.angle] = support_by_angle.get(line.angle, 0.0) + (
            local_width * _line_effective_height(line, local_bbox)
        )
    return support_by_angle


def _classify_image_footnotes(
    lines: list[_LineItem],
    image_bboxes: list[BBox],
    table_bboxes: list[BBox],
    drawing_lines: list[_AxisLine],
    page_size: tuple[float, float],
    *,
    reference_body_height: float | None = None,
) -> None:
    """用图片、下缘长横线和紧凑小字的联合关系识别图表脚注。"""

    available = [line for line in lines if line.semantic_type is None]
    if not available or not image_bboxes or not drawing_lines:
        return
    support_by_angle = _geometric_text_support_by_angle(available, page_size)
    if not support_by_angle:
        return
    dominant_angle = max(
        sorted(support_by_angle),
        key=lambda angle: support_by_angle[angle],
    )
    local_page_size = (
        (page_size[1], page_size[0])
        if dominant_angle in {90, 270}
        else page_size
    )
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return

    line_geometry = sorted(
        [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, dominant_angle))
            for line in available
            if line.angle == dominant_angle
        ],
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    if not line_geometry:
        return
    if reference_body_height is not None and reference_body_height > 0:
        # 图片占主导的稀疏页可能只剩图注和脚注，延迟复核时改用全文正文尺度。
        body_height = max(0.1, reference_body_height)
    else:
        body_samples = [
            _line_effective_height(line, bbox)
            for line, bbox in line_geometry
            if bbox[2] - bbox[0] >= 0.2 * local_page_width
        ]
        if not body_samples:
            body_samples = [
                _line_effective_height(line, bbox)
                for line, bbox in line_geometry
            ]
        body_height = max(0.1, statistics.median(body_samples))
    local_images = [
        _rotate_bbox_to_upright(bbox, page_size, dominant_angle)
        for bbox in image_bboxes
    ]
    local_axis_lines = _transform_axis_lines(
        drawing_lines,
        page_size,
        dominant_angle,
    )

    matched_source_indices: set[int] = set()
    for image_bbox in local_images:
        image_width = max(0.1, image_bbox[2] - image_bbox[0])
        # 同一视觉行的并排图可能高度略有差异；共享较低下缘可避免把留白误作远距。
        row_bottom = max(
            peer_bbox[3]
            for peer_bbox in local_images
            if _bbox_axis_overlap_ratio(image_bbox, peer_bbox, axis="y") >= 0.5
        )
        candidate_rules = [
            axis_line
            for axis_line in local_axis_lines
            if axis_line.orientation == "horizontal"
            and 0.75 * image_width
            <= axis_line.bbox[2] - axis_line.bbox[0]
            <= 1.3 * image_width
            and max(
                0.0,
                min(axis_line.bbox[2], image_bbox[2])
                - max(axis_line.bbox[0], image_bbox[0]),
            )
            >= 0.85 * image_width
            # 图片外框的底边属于图形本身，不能拿来证明下方文字是图表脚注。
            and 0.0
            <= axis_line.bbox[1] - row_bottom
            <= max(0.01 * local_page_height, 0.75 * body_height)
            and not _rule_belongs_to_confirmed_table(
                axis_line,
                local_axis_lines,
                table_bboxes,
                local_page_width,
            )
        ]
        if not candidate_rules:
            continue
        rule = min(
            candidate_rules,
            key=lambda item: (max(0.0, item.bbox[1] - row_bottom), item.bbox[1]),
        )
        matched_source_indices.update(
            _image_footnote_members(
                line_geometry,
                rule.bbox,
                body_height,
                local_page_height,
            )
        )

    for line in available:
        if line.source_index in matched_source_indices:
            line.semantic_type = "footnote"


def _classify_deferred_image_footnotes(
    prepared_pages: list[_PreparedPage],
    body_height: float,
) -> None:
    """在全文正文尺度确定后，仅重试仍未分类的图片脚注候选。"""

    if body_height <= 0:
        return
    for prepared in prepared_pages:
        _classify_image_footnotes(
            prepared.remaining_lines,
            [
                block["bbox"]
                for block in prepared.fixed_blocks
                if block.get("type") == "image"
            ],
            prepared.table_bboxes,
            prepared.drawing_lines,
            prepared.page_size,
            reference_body_height=body_height,
        )


def _image_footnote_members(
    line_geometry: list[tuple[_LineItem, BBox]],
    rule_bbox: BBox,
    body_height: float,
    local_page_height: float,
) -> set[int]:
    """返回长横线下方、位于同一水平走廊内的连续小字号文本行。"""

    first_gap_limit = max(0.025 * local_page_height, 2.0 * body_height)
    horizontal_tolerance = 0.5 * body_height
    candidates = [
        item
        for item in line_geometry
        if -0.25 * body_height <= item[1][1] - rule_bbox[3] <= first_gap_limit
        and item[1][0] >= rule_bbox[0] - horizontal_tolerance
        and item[1][2] <= rule_bbox[2] + horizontal_tolerance
        and _line_effective_height(*item) <= 0.9 * body_height
    ]
    if not candidates:
        return set()
    first = min(candidates, key=lambda item: (item[1][1], item[1][0]))
    members = [first]
    continuation_gap_limit = max(1.25 * _line_effective_height(*first), 0.01 * local_page_height)
    for current in line_geometry:
        if current[0] is first[0] or current[1][1] < first[1][1]:
            continue
        if current[1][0] < rule_bbox[0] - horizontal_tolerance:
            continue
        if current[1][2] > rule_bbox[2] + horizontal_tolerance:
            continue
        if _line_effective_height(*current) > 0.95 * body_height:
            continue
        if _effective_text_row_gap(members[-1], current) > continuation_gap_limit:
            break
        members.append(current)
    return {line.source_index for line, _bbox in members}


def _classify_page_footnotes(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    drawing_lines: list[_AxisLine],
    page_size: tuple[float, float],
    *,
    visual_bboxes: list[BBox] | None = None,
) -> list[set[int]]:
    """识别主方向页脚注，并按触发分隔线返回来源编号分组。"""

    available = [line for line in lines if line.semantic_type is None]
    if not available or not drawing_lines:
        return []
    support_by_angle = _geometric_text_support_by_angle(available, page_size)
    if not support_by_angle:
        return []
    dominant_angle = max(
        sorted(support_by_angle),
        key=lambda angle: support_by_angle[angle],
    )
    line_geometry = [
        (line, _rotate_bbox_to_upright(line.bbox, page_size, dominant_angle))
        for line in available
        if line.angle == dominant_angle
    ]
    if not line_geometry:
        return []

    local_page_size = (
        (page_size[1], page_size[0])
        if dominant_angle in {90, 270}
        else page_size
    )
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return []
    effective_heights = [
        _line_effective_height(line, bbox)
        for line, bbox in line_geometry
    ]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0
    lanes = _infer_text_lanes(
        line_geometry,
        local_page_width,
        median_height,
        # 脚注分隔线应对齐稳定栏锚点，不能被页眉或跨栏关键词的宽行扩张污染。
        recalculate_intervals=False,
    )
    local_axis_lines = _transform_axis_lines(
        drawing_lines,
        page_size,
        dominant_angle,
    )

    candidate_groups: list[set[int]] = []
    visual_bboxes = visual_bboxes or []
    for axis_line in local_axis_lines:
        if axis_line.orientation != "horizontal":
            continue
        # 常规短分隔线仍要求进入页面下方 30%；栏宽分隔线可在下方 45% 内
        # 依靠严格的单栏对齐和字号收缩证据提前触发。
        rule_center_y = _bbox_center_y(axis_line.bbox)
        if rule_center_y < 0.55 * local_page_height:
            continue
        # 表格边界会产生断裂横线；除框内线段外，也排除与其同高且近邻的框外线段。
        if _rule_belongs_to_confirmed_table(
            axis_line,
            local_axis_lines,
            table_bboxes,
            local_page_width,
        ):
            continue
        if any(
            _bbox_intersects(
                _expand_bbox(axis_line.original_bbox, max(0.5, axis_line.width)),
                visual_bbox,
            )
            for visual_bbox in visual_bboxes
        ):
            # 图形坐标轴和外框不能充当页面脚注分隔线。
            continue
        rule_source_indices: set[int] = set()
        for lane in lanes:
            following_rule_tops = [
                other.bbox[1]
                for other in local_axis_lines
                if other.orientation == "horizontal"
                and other.bbox[1] - axis_line.bbox[3] > 0.5 * median_height
                and _bbox_axis_overlap_ratio(
                    axis_line.bbox,
                    other.bbox,
                    axis="x",
                )
                >= 0.8
            ]
            rule_source_indices.update(
                _footnote_lane_members(
                    lane,
                    axis_line.bbox,
                    local_page_size,
                    page_median_height=median_height,
                    lane_width_reference=_footnote_lane_width_reference(
                        lane,
                        lanes,
                        median_height,
                    ),
                    allow_column_width_rule=(
                        rule_center_y >= 0.55 * local_page_height
                    ),
                    lower_barrier_y=(
                        min(following_rule_tops)
                        if following_rule_tops
                        else None
                    ),
                )
            )
        if rule_source_indices:
            candidate_groups.append(rule_source_indices)

    page_footnote_groups = _merge_overlapping_source_groups(candidate_groups)
    _augment_footnote_groups_with_edge_markers(
        page_footnote_groups,
        line_geometry,
        median_height,
    )
    footnote_source_indices = set().union(*page_footnote_groups) if page_footnote_groups else set()
    for line in available:
        if line.source_index in footnote_source_indices:
            line.semantic_type = "page_footnote"
    return page_footnote_groups


def _augment_footnote_groups_with_edge_markers(
    groups: list[set[int]],
    line_geometry: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> None:
    """把脚注正文左侧同高的窄编号标记补入对应分隔线分组。"""

    geometry_by_source = {
        line.source_index: (line, bbox)
        for line, bbox in line_geometry
    }
    for group in groups:
        members = [
            geometry_by_source[source_index]
            for source_index in group
            if source_index in geometry_by_source
        ]
        if not members:
            continue
        group_top = min(bbox[1] for _line, bbox in members)
        group_bottom = max(bbox[3] for _line, bbox in members)
        content_left = min(bbox[0] for _line, bbox in members)
        for line, bbox in line_geometry:
            if line.source_index in group:
                continue
            line_width = bbox[2] - bbox[0]
            center_y = _bbox_center_y(bbox)
            if (
                line_width <= 1.5 * median_height
                and content_left - 2.0 * median_height
                <= bbox[0]
                <= content_left
                and bbox[2] <= content_left + 0.5 * median_height
                and group_top - median_height
                <= center_y
                <= group_bottom + median_height
            ):
                group.add(line.source_index)


def _rule_belongs_to_confirmed_table(
    candidate: _LocalAxisLine,
    local_axis_lines: list[_LocalAxisLine],
    table_bboxes: list[BBox],
    local_page_width: float,
) -> bool:
    """把表格框内横线及其同高近邻断裂段一并排除，避免框外残段触发脚注。"""

    if not table_bboxes:
        return False
    maximum_segment_gap = 0.04 * local_page_width
    for table_line in local_axis_lines:
        if table_line.orientation != "horizontal":
            continue
        table_margin = max(0.5, table_line.width)
        if not any(
            _bbox_intersects(
                _expand_bbox(table_line.original_bbox, table_margin),
                table_bbox,
            )
            for table_bbox in table_bboxes
        ):
            continue
        center_tolerance = max(1.0, candidate.width, table_line.width)
        if abs(_bbox_center_y(candidate.bbox) - _bbox_center_y(table_line.bbox)) > center_tolerance:
            continue
        if _horizontal_bbox_gap(candidate.bbox, table_line.bbox) <= maximum_segment_gap:
            return True
    return False


def _merge_overlapping_source_groups(groups: list[set[int]]) -> list[set[int]]:
    """合并共享来源行的分隔线候选组，消除重复绘图线造成的重复分组。"""

    merged: list[set[int]] = []
    for group in groups:
        combined = set(group)
        index = 0
        while index < len(merged):
            if combined & merged[index]:
                combined.update(merged.pop(index))
                index = 0
                continue
            index += 1
        merged.append(combined)
    return sorted(merged, key=lambda group: min(group))


def _footnote_lane_members(
    lane: _TextLane,
    rule_bbox: BBox,
    local_page_size: tuple[float, float],
    *,
    page_median_height: float | None = None,
    lane_width_reference: float | None = None,
    allow_column_width_rule: bool = False,
    lower_barrier_y: float | None = None,
) -> set[int]:
    """验证横线与单个栏带的对齐关系，并返回其下连续脚注行的来源编号。"""

    lane_lines = [item for item in lane.lines if item[0].semantic_type is None]
    if not lane_lines:
        return set()
    lane_lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    local_page_width, local_page_height = local_page_size
    lane_width = max(
        0.1,
        lane.right - lane.left,
        lane_width_reference or 0.0,
    )
    lane_heights = [_line_effective_height(line, bbox) for line, bbox in lane_lines]
    median_height = statistics.median(lane_heights) if lane_heights else 1.0
    rule_width = max(0.0, rule_bbox[2] - rule_bbox[0])
    # 同时限制绝对短线、相对长线和左缘偏移，排除图标、公式线及跨栏正文分隔线。
    if rule_width < max(4.0 * median_height, 0.04 * local_page_width):
        return set()
    if abs(rule_bbox[0] - lane.left) > max(2.0 * median_height, 0.04 * lane_width):
        return set()

    rule_center_y = _bbox_center_y(rule_bbox)
    is_regular_short_rule = (
        rule_center_y >= 0.7 * local_page_height
        and rule_width <= 0.65 * lane_width
    )
    endpoint_tolerance = max(2.0 * median_height, 0.05 * lane_width)
    is_column_width_rule = (
        allow_column_width_rule
        and not lane.is_span
        and 0.65 * lane_width <= rule_width <= 1.05 * lane_width
        and abs(rule_bbox[2] - lane.right) <= endpoint_tolerance
    )
    if not is_regular_short_rule and not is_column_width_rule:
        return set()

    # 首行采用较宽的 3.5% 页高窗口；命中后仅按紧凑的连续净空向下扩展。
    first_gap_limit = max(3.0 * median_height, 0.035 * local_page_height)
    first_index: int | None = None
    for index, (_line, bbox) in enumerate(lane_lines):
        rule_gap = bbox[1] - rule_bbox[3]
        if rule_gap < -0.5 * median_height:
            continue
        if lower_barrier_y is not None and bbox[1] >= lower_barrier_y:
            break
        if rule_gap <= first_gap_limit:
            first_index = index
        break
    if first_index is None:
        return set()

    if is_column_width_rule:
        # 页面中段的栏宽横线只有在下方首行相对上方正文明显收缩时才可触发脚注，
        # 避免把章节分隔线或普通栏内横线误当成脚注边界。
        body_heights = [
            _line_effective_height(line, bbox)
            for line, bbox in lane_lines
            if bbox[3] <= rule_bbox[1] + 0.5 * median_height
        ]
        first_height = _line_effective_height(*lane_lines[first_index])
        body_reference_height = statistics.median(body_heights) if body_heights else 0.0
        if page_median_height is not None:
            body_reference_height = max(
                body_reference_height,
                page_median_height,
            )
        if (
            len(body_heights) < 3
            or first_height > 0.95 * body_reference_height
        ):
            return set()

    continuation_gap_limit = max(1.25 * median_height, 0.01 * local_page_height)
    members = [lane_lines[first_index]]
    for current in lane_lines[first_index + 1 :]:
        if lower_barrier_y is not None and current[1][1] >= lower_barrier_y:
            break
        if _effective_text_row_gap(members[-1], current) > continuation_gap_limit:
            break
        members.append(current)
    return {line.source_index for line, _bbox in members}


def _footnote_lane_width_reference(
    lane: _TextLane,
    lanes: list[_TextLane],
    median_height: float,
) -> float:
    """用下一稳定栏的左缘补偿当前栏因正文右缘参差造成的宽度低估。"""

    lane_width = max(0.1, lane.right - lane.left)
    stable_lanes = sorted(
        [
            candidate
            for candidate in lanes
            if not candidate.is_span and len(candidate.lines) >= 3
        ],
        key=lambda candidate: candidate.left,
    )
    if lane not in stable_lanes:
        return lane_width
    lane_index = stable_lanes.index(lane)
    if lane_index + 1 >= len(stable_lanes):
        return lane_width
    minimum_gutter = max(6.0, 0.75 * median_height)
    next_lane = stable_lanes[lane_index + 1]
    return max(
        lane_width,
        next_lane.left - lane.left - minimum_gutter,
    )


def _classify_rule_delimited_headers(pages: list[_PreparedPage]) -> None:
    """在页码完成跨页判定后，用页首长横线补标其上方未分类文本。"""

    for page in pages:
        available = [
            line
            for line in page.remaining_lines
            if line.semantic_type is None
        ]
        if not available or not page.drawing_lines:
            continue
        support_by_angle = _geometric_text_support_by_angle(
            page.remaining_lines,
            page.page_size,
        )
        if not support_by_angle:
            continue
        dominant_angle = max(
            sorted(support_by_angle),
            key=lambda angle: support_by_angle[angle],
        )
        local_page_size = (
            (page.page_size[1], page.page_size[0])
            if dominant_angle in {90, 270}
            else page.page_size
        )
        local_page_width, local_page_height = local_page_size
        if local_page_width <= 0 or local_page_height <= 0:
            continue
        local_axis_lines = _transform_axis_lines(
            page.drawing_lines,
            page.page_size,
            dominant_angle,
        )
        candidates = [
            axis_line
            for axis_line in local_axis_lines
            if axis_line.orientation == "horizontal"
            and _bbox_center_y(axis_line.bbox) <= 0.15 * local_page_height
            and axis_line.bbox[2] - axis_line.bbox[0]
            >= 0.6 * local_page_width
            and not _rule_belongs_to_confirmed_table(
                axis_line,
                local_axis_lines,
                page.table_bboxes,
                local_page_width,
            )
            and not _rule_overlaps_fixed_container(
                axis_line,
                page.fixed_blocks,
                page.page_size,
            )
        ]
        if not candidates:
            continue
        separator = max(candidates, key=lambda item: _bbox_center_y(item.bbox))
        separator_y = _bbox_center_y(separator.bbox)
        local_lines = [
            (
                line,
                _rotate_bbox_to_upright(
                    line.bbox,
                    page.page_size,
                    dominant_angle,
                ),
            )
            for line in available
            if line.angle == dominant_angle
        ]
        heights = [
            _line_effective_height(line, bbox)
            for line, bbox in local_lines
        ]
        median_height = statistics.median(heights) if heights else 1.0
        if not any(
            _bbox_center_y(bbox) >= separator_y + median_height
            for _line, bbox in local_lines
        ):
            continue
        for line, bbox in local_lines:
            if bbox[3] <= separator_y:
                line.semantic_type = "header"


def _classify_rule_delimited_footers(pages: list[_PreparedPage]) -> None:
    """用页面底部横线确认双线间页脚或单线下方的小字号栏内页脚。"""

    for page in pages:
        available = [
            line
            for line in page.remaining_lines
            if line.semantic_type is None
        ]
        if not available or not page.drawing_lines:
            continue
        support_by_angle = _geometric_text_support_by_angle(
            page.remaining_lines,
            page.page_size,
        )
        if not support_by_angle:
            continue
        dominant_angle = max(
            sorted(support_by_angle),
            key=lambda angle: support_by_angle[angle],
        )
        local_page_size = (
            (page.page_size[1], page.page_size[0])
            if dominant_angle in {90, 270}
            else page.page_size
        )
        local_page_width, local_page_height = local_page_size
        if local_page_width <= 0 or local_page_height <= 0:
            continue
        local_axis_lines = _transform_axis_lines(
            page.drawing_lines,
            page.page_size,
            dominant_angle,
        )
        rules = [
            rule
            for rule in local_axis_lines
            if rule.orientation == "horizontal"
            and _bbox_center_y(rule.bbox) >= 0.85 * local_page_height
            and rule.bbox[2] - rule.bbox[0] >= 0.2 * local_page_width
            and not _rule_belongs_to_confirmed_table(
                rule,
                local_axis_lines,
                page.table_bboxes,
                local_page_width,
            )
            and not _rule_overlaps_fixed_container(
                rule,
                page.fixed_blocks,
                page.page_size,
            )
        ]
        local_lines = [
            (
                line,
                _rotate_bbox_to_upright(
                    line.bbox,
                    page.page_size,
                    dominant_angle,
                ),
            )
            for line in available
            if line.angle == dominant_angle
        ]
        if not local_lines:
            continue
        median_height = statistics.median(
            _line_effective_height(line, bbox)
            for line, bbox in local_lines
        )
        lanes = [
            lane
            for lane in _infer_text_lanes(
                local_lines,
                local_page_width,
                median_height,
                recalculate_intervals=False,
            )
            if not lane.is_span
        ]
        for upper_index, upper in enumerate(rules[:-1]):
            for lower in rules[upper_index + 1 :]:
                vertical_gap = lower.bbox[1] - upper.bbox[3]
                if not 2.0 * median_height <= vertical_gap <= 6.0 * median_height:
                    continue
                if _bbox_axis_overlap_ratio(upper.bbox, lower.bbox, axis="x") < 0.9:
                    continue
                corridor_left = max(upper.bbox[0], lower.bbox[0])
                corridor_right = min(upper.bbox[2], lower.bbox[2])
                members = [
                    (line, bbox)
                    for line, bbox in local_lines
                    if bbox[1] >= upper.bbox[3]
                    and bbox[3] <= lower.bbox[1]
                    and bbox[0] >= corridor_left - 0.5 * median_height
                    and bbox[2] <= corridor_right + 0.5 * median_height
                ]
                if not 1 <= len(members) <= 3:
                    continue
                if any(
                    abs(_bbox_center_x(bbox) - 0.5 * (corridor_left + corridor_right))
                    > 0.15 * max(0.1, corridor_right - corridor_left)
                    for _line, bbox in members
                ):
                    continue
                for line, _bbox in members:
                    line.semantic_type = "footer"
                break
        for rule in rules:
            for line in _single_rule_footer_members(
                rule,
                local_lines,
                lanes,
                median_height,
            ):
                line.semantic_type = "footer"


def _single_rule_footer_members(
    rule: _LocalAxisLine,
    local_lines: list[tuple[_LineItem, BBox]],
    lanes: list[_TextLane],
    body_height: float,
) -> list[_LineItem]:
    """返回底部单横线下方、唯一栏内连续的小字号页脚行。"""

    rule_width = max(0.1, rule.bbox[2] - rule.bbox[0])
    rule_center_x = _bbox_center_x(rule.bbox)
    matching_lanes = []
    for lane in lanes:
        overlap = max(
            0.0,
            min(rule.bbox[2], lane.right) - max(rule.bbox[0], lane.left),
        )
        if overlap / rule_width >= 0.8 and lane.left <= rule_center_x <= lane.right:
            matching_lanes.append(lane)
    if len(matching_lanes) != 1:
        return []

    lane = matching_lanes[0]
    if (
        len(lanes) > 1
        and lane.left < max(candidate_lane.left for candidate_lane in lanes) - body_height
    ):
        return []
    tolerance = 0.5 * body_height
    rows_below = sorted(
        (
            (line, bbox)
            for line, bbox in local_lines
            if line.semantic_type is None
            and bbox[1] >= rule.bbox[3]
            and bbox[0] >= lane.left - tolerance
            and bbox[2] <= lane.right + tolerance
        ),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    if not rows_below:
        return []
    first_line, first_bbox = rows_below[0]
    if (
        first_bbox[1] - rule.bbox[3] > body_height
        or _line_effective_height(first_line, first_bbox) > 0.9 * body_height
    ):
        return []

    members = [(first_line, first_bbox)]
    for line, bbox in rows_below[1:]:
        previous_bbox = members[-1][1]
        if (
            bbox[1] - previous_bbox[3] > body_height
            or bbox[3] - first_bbox[1] > 5.0 * body_height
            or _line_effective_height(line, bbox) > 0.9 * body_height
        ):
            break
        members.append((line, bbox))
    if not 2 <= len(members) <= 8:
        return []
    member_left_edges = [bbox[0] for _line, bbox in members]
    if max(member_left_edges) - min(member_left_edges) > 0.75 * body_height:
        return []
    return [line for line, _bbox in members]


def _rule_overlaps_fixed_container(
    rule: _LocalAxisLine,
    fixed_blocks: list[dict[str, object]],
    page_size: tuple[float, float],
) -> bool:
    """排除落在表格、图片或公式容器内的页首横线。"""

    expanded_rule = _expand_bbox(
        rule.original_bbox,
        max(1.0, rule.width),
    )
    for block in fixed_blocks:
        if block.get("type") not in {"table", "image", "equation"}:
            continue
        bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page_size)
        if bbox is not None and _bbox_intersects(expanded_rule, bbox):
            return True
    return False


def _classify_page_number_outer_companions(
    pages: list[_PreparedPage],
) -> None:
    """把上下页码外侧的未分类文本和图片标为对应页眉或页脚。"""

    for page in pages:
        page_numbers = [
            line
            for line in page.remaining_lines
            if line.semantic_type == "page_number"
        ]
        for page_number in page_numbers:
            angle = page_number.angle
            local_page_size = (
                (page.page_size[1], page.page_size[0])
                if angle in {90, 270}
                else page.page_size
            )
            local_page_height = local_page_size[1]
            if local_page_height <= 0:
                continue
            page_number_bbox = _rotate_bbox_to_upright(
                page_number.bbox,
                page.page_size,
                angle,
            )
            normalized_center_y = (
                _bbox_center_y(page_number_bbox) / local_page_height
            )
            if normalized_center_y <= 0.3:
                target_type: Literal["header", "footer"] = "header"
                outward_limit = page_number_bbox[1]
            elif normalized_center_y >= 0.7:
                target_type = "footer"
                outward_limit = page_number_bbox[3]
            else:
                continue
            for line in page.remaining_lines:
                if line.semantic_type is not None or line.angle != angle:
                    continue
                local_bbox = _rotate_bbox_to_upright(
                    line.bbox,
                    page.page_size,
                    angle,
                )
                is_outward = (
                    local_bbox[3] <= outward_limit
                    if target_type == "header"
                    else local_bbox[1] >= outward_limit
                )
                same_marginal_row = (
                    _bbox_axis_overlap_ratio(
                        local_bbox,
                        page_number_bbox,
                        axis="y",
                    )
                    >= 0.5
                )
                if is_outward or same_marginal_row:
                    line.semantic_type = target_type
            for block in page.fixed_blocks:
                if block.get("type") != "image":
                    continue
                block_angle = int(block.get("angle", 0) or 0) % 360
                if block_angle != angle:
                    continue
                bbox = _clip_bbox(
                    _coerce_bbox(block.get("bbox")),
                    page.page_size,
                )
                if bbox is None:
                    continue
                local_bbox = _rotate_bbox_to_upright(
                    bbox,
                    page.page_size,
                    angle,
                )
                is_outward = (
                    local_bbox[3] <= outward_limit
                    if target_type == "header"
                    else local_bbox[1] >= outward_limit
                )
                same_marginal_row = (
                    _bbox_axis_overlap_ratio(
                        local_bbox,
                        page_number_bbox,
                        axis="y",
                    )
                    >= 0.5
                )
                if is_outward or same_marginal_row:
                    block["type"] = target_type


def _classify_split_marginal_row_companions(
    pages: list[_PreparedPage],
) -> None:
    """把页边缘同一拆分视觉行中的未分类碎片继承为页眉或页脚。"""

    for page in pages:
        row_groups: dict[tuple[int, int], list[_LineItem]] = {}
        for line in page.remaining_lines:
            if line.visual_row_id is None or not line.split_from_row:
                continue
            row_groups.setdefault((line.angle, line.visual_row_id), []).append(
                line
            )
        for (angle, _row_id), members in row_groups.items():
            local_page_height = (
                page.page_size[0]
                if angle in {90, 270}
                else page.page_size[1]
            )
            local_bboxes = [
                _rotate_bbox_to_upright(line.bbox, page.page_size, angle)
                for line in members
            ]
            row_center = statistics.fmean(
                _bbox_center_y(bbox)
                for bbox in local_bboxes
            )
            if row_center <= 0.1 * local_page_height:
                target_type: Literal["header", "footer"] = "header"
            elif row_center >= 0.9 * local_page_height:
                target_type = "footer"
            else:
                continue
            anchor_types = {
                line.semantic_type
                for line in members
                if line.semantic_type in {target_type, "page_number"}
            }
            if not anchor_types:
                continue
            for line in members:
                if line.semantic_type is None:
                    line.semantic_type = target_type


def _classify_raw_page_marginals(sources: list[_PageSource]) -> None:
    """在视觉容器认领前保护强跨页页码、页眉和页脚文本。"""

    if len(sources) < 2:
        return
    candidates = [
        candidate
        for page_index, source in enumerate(sources)
        for line in source.lines
        if (
            candidate := _build_marginal_candidate(
                page_index,
                line,
                source.page_size,
            )
        )
        is not None
        and (
            _bbox_center_y(candidate.local_bbox)
            / candidate.local_page_size[1]
            <= 0.08
            or _bbox_center_y(candidate.local_bbox)
            / candidate.local_page_size[1]
            >= 0.92
        )
    ]
    _classify_marginal_candidates(candidates)


def _classify_repeated_page_marginals(pages: list[_PreparedPage]) -> None:
    """仅用相邻或同奇偶页的重复证据标注页码、页眉和页脚。"""

    if len(pages) < 2:
        return
    candidates = [
        candidate
        for page_index, page in enumerate(pages)
        for line in page.remaining_lines
        if (candidate := _build_marginal_candidate(page_index, line, page.page_size)) is not None
    ]

    _classify_marginal_candidates(candidates)


def _classify_marginal_candidates(
    candidates: list[_MarginalCandidate],
) -> None:
    """复用跨页递增页码和稳定边缘文本的强证据匹配。"""

    for left_index, left in enumerate(candidates):
        left_value = _parse_page_number_value(left.line.text)
        if left_value is None:
            continue
        for right in candidates[left_index + 1 :]:
            page_delta = right.page_index - left.page_index
            if page_delta > 2:
                break
            right_value = _parse_page_number_value(right.line.text)
            if (
                page_delta > 0
                and right_value is not None
                and right_value - left_value == page_delta
                and _page_number_candidates_match(left, right)
            ):
                left.line.semantic_type = "page_number"
                right.line.semantic_type = "page_number"

    for left_index, left in enumerate(candidates):
        if left.line.semantic_type == "page_number":
            continue
        for right in candidates[left_index + 1 :]:
            page_delta = right.page_index - left.page_index
            if page_delta > 2:
                break
            if (
                page_delta > 0
                and left.region != "side"
                and right.region != "side"
                and right.line.semantic_type != "page_number"
                and _marginal_geometry_matches(left, right)
                and _marginal_text_matches(left.line.text, right.line.text)
            ):
                left.line.semantic_type = left.region
                right.line.semantic_type = right.region


def _classify_single_page_compound_headers(pages: list[_PreparedPage]) -> None:
    """以拆分同行、字号收缩和正文栏右缘共同确认单页复合页眉。"""

    if len(pages) != 1:
        return
    page = pages[0]
    page_width, page_height = page.page_size
    if page_width <= 0 or page_height <= 0:
        return

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in page.remaining_lines:
        if (
            line.semantic_type is None
            and line.visual_row_id is not None
            and line.split_from_row
        ):
            row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    for (angle, _row_id), members in row_groups.items():
        if len(members) < 2:
            continue
        local_page_width = page_height if angle in {90, 270} else page_width
        local_page_height = page_width if angle in {90, 270} else page_height
        local_members = [
            (line, _rotate_bbox_to_upright(line.bbox, page.page_size, angle))
            for line in members
        ]
        row_top = min(bbox[1] for _line, bbox in local_members)
        row_bottom = max(bbox[3] for _line, bbox in local_members)
        if row_top < 0 or row_bottom > 0.05 * local_page_height:
            continue

        row_left = min(bbox[0] for _line, bbox in local_members)
        row_right = max(bbox[2] for _line, bbox in local_members)
        related_body = [
            (line, local_bbox)
            for line in page.remaining_lines
            if line.semantic_type is None
            and line.angle == angle
            and line not in members
            and (
                local_bbox := _rotate_bbox_to_upright(
                    line.bbox,
                    page.page_size,
                    angle,
                )
            )[1]
            >= 0.05 * local_page_height
            and local_bbox[2] - local_bbox[0] >= 0.3 * local_page_width
            and _bbox_axis_overlap_ratio(
                (row_left, row_top, row_right, row_bottom),
                local_bbox,
                axis="x",
            )
            >= 0.2
        ]
        if len(related_body) < 3:
            continue
        body_height = statistics.median(
            _line_effective_height(line, bbox)
            for line, bbox in related_body
        )
        row_height = max(
            _line_effective_height(line, bbox)
            for line, bbox in local_members
        )
        if row_height > 0.85 * body_height:
            continue
        body_right = statistics.median(bbox[2] for _line, bbox in related_body)
        has_right_sidecar = any(
            bbox[2] - bbox[0] <= max(4.0 * row_height, 0.12 * local_page_width)
            and abs(bbox[2] - body_right) <= max(3.0, 0.02 * local_page_width)
            for _line, bbox in local_members
        )
        if has_right_sidecar:
            for line, _bbox in local_members:
                line.semantic_type = "header"


def _classify_isolated_first_page_footer(pages: list[_PreparedPage]) -> None:
    """用多页首页的极底位置、正文尺度和孤立净空补标唯一页脚。"""

    if len(pages) < 2:
        return
    page = pages[0]
    page_width, page_height = page.page_size
    if page_width <= 0 or page_height <= 0:
        return

    body_lines = [
        line
        for line in page.remaining_lines
        if line.semantic_type is None
        and line.angle == 0
        and line.bbox[2] - line.bbox[0] >= 0.3 * page_width
        and 0.1 * page_height <= _bbox_center_y(line.bbox) <= 0.94 * page_height
    ]
    if len(body_lines) < 4:
        return
    body_height = statistics.median(
        _line_effective_height(line, line.bbox)
        for line in body_lines
    )
    body_bottom = max(line.bbox[3] for line in body_lines)
    if body_bottom < 0.7 * page_height:
        return
    container_bboxes = [
        bbox
        for block in page.fixed_blocks
        if (bbox := _coerce_bbox(block.get("bbox"))) is not None
    ]
    candidates = [
        line
        for line in page.remaining_lines
        if line.semantic_type is None
        and line.angle == 0
        and line.bbox[1] >= 0.94 * page_height
        and line.bbox[2] - line.bbox[0] <= 0.6 * page_width
        and abs(_bbox_center_x(line.bbox) - 0.5 * page_width) <= 0.08 * page_width
        and _line_effective_height(line, line.bbox) <= 0.95 * body_height
        and line.bbox[1] - body_bottom >= 1.5 * body_height
        and not any(
            _bbox_intersects(line.bbox, container_bbox)
            for container_bbox in container_bboxes
        )
    ]
    if len(candidates) == 1:
        candidates[0].semantic_type = "footer"


def _classify_repeated_visual_headers(pages: list[_PreparedPage]) -> None:
    """仅按页首位置与跨页重复几何，把整体图片重标为视觉页眉。"""

    candidates: list[tuple[int, dict[str, object], BBox, int]] = []
    for page_index, page in enumerate(pages):
        # 首页常使用独立封面版式，不参与正文页视觉页眉聚类。
        if page_index == 0:
            continue
        page_width, page_height = page.page_size
        if page_width <= 0 or page_height <= 0:
            continue
        for block in page.fixed_blocks:
            if block.get("type") != "image":
                continue
            bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page.page_size)
            if bbox is None or bbox[3] > 0.12 * page_height:
                continue
            normalized_bbox = (
                bbox[0] / page_width,
                bbox[1] / page_height,
                bbox[2] / page_width,
                bbox[3] / page_height,
            )
            angle = int(block.get("angle", 0) or 0) % 360
            candidates.append((page_index, block, normalized_bbox, angle))

    if len(candidates) < 3:
        return

    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        """查找视觉页眉候选所属几何簇的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并跨页距离和归一化几何均匹配的两个候选。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, (
        first_page,
        _first_block,
        first_bbox,
        first_angle,
    ) in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second_page, _second_block, second_bbox, second_angle = candidates[
                second_index
            ]
            page_delta = second_page - first_page
            if page_delta > 2:
                break
            if (
                page_delta > 0
                and first_angle == second_angle
                and _visual_header_geometry_matches(first_bbox, second_bbox)
            ):
                union(first_index, second_index)

    clusters: dict[int, list[int]] = {}
    for candidate_index in range(len(candidates)):
        clusters.setdefault(find(candidate_index), []).append(candidate_index)
    for member_indices in clusters.values():
        page_indices = {candidates[index][0] for index in member_indices}
        if len(page_indices) < 3:
            continue
        for index in member_indices:
            candidates[index][1]["type"] = "header"


def _visual_header_geometry_matches(first: BBox, second: BBox) -> bool:
    """比较两个归一化页首图片的 IoU 与宽高尺度。"""

    first_width = first[2] - first[0]
    first_height = first[3] - first[1]
    second_width = second[2] - second[0]
    second_height = second[3] - second[1]
    if min(first_width, first_height, second_width, second_height) <= 0:
        return False
    if max(first_width, second_width) / min(first_width, second_width) > 1.1:
        return False
    if max(first_height, second_height) / min(first_height, second_height) > 1.1:
        return False

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union_area = first_width * first_height + second_width * second_height - intersection
    return union_area > 0 and intersection / union_area >= 0.9


def _build_marginal_candidate(
    page_index: int,
    line: _LineItem,
    page_size: tuple[float, float],
) -> _MarginalCandidate | None:
    """把页面上下百分之十五内的常规小行转换成跨页比较候选。"""

    if line.semantic_type not in {None, "page_footnote"}:
        return None
    local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
    local_page_size = (page_size[1], page_size[0]) if line.angle in {90, 270} else page_size
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return None
    normalized_center_y = _bbox_center_y(local_bbox) / local_page_height
    normalized_center_x = _bbox_center_x(local_bbox) / local_page_width
    if line.semantic_type == "page_footnote" and normalized_center_y < 0.94:
        # 只允许极底部脚注重新参加跨页强证据匹配，正文脚注继续保留原类型。
        return None
    if normalized_center_y <= 0.15:
        region: Literal["header", "footer", "side"] = "header"
    elif normalized_center_y >= 0.9:
        region = "footer"
    elif (
        normalized_center_y <= 0.18
        or normalized_center_y >= 0.82
        or (
            (normalized_center_x <= 0.15 or normalized_center_x >= 0.85)
            and (normalized_center_y <= 0.3 or normalized_center_y >= 0.7)
        )
    ):
        # 仅页码递增逻辑会消费 side；稳定文本不会被侧栏位置猜成页眉页脚。
        region = "side"
    else:
        return None
    if _line_effective_height(line, local_bbox) > 0.06 * local_page_height:
        return None
    return _MarginalCandidate(
        page_index=page_index,
        line=line,
        local_bbox=local_bbox,
        local_page_size=local_page_size,
        region=region,
    )


def _page_number_candidates_match(
    first: _MarginalCandidate,
    second: _MarginalCandidate,
) -> bool:
    """校验连续页码的同边缘几何，横竖版切换时允许边缘位置随版面改变。"""

    if _marginal_geometry_matches(first, second):
        return True
    first_landscape = first.local_page_size[0] > first.local_page_size[1]
    second_landscape = second.local_page_size[0] > second.local_page_size[1]
    if first_landscape == second_landscape or first.line.angle != second.line.angle:
        return False
    first_height = _line_effective_height(first.line, first.local_bbox) / first.local_page_size[1]
    second_height = _line_effective_height(second.line, second.local_bbox) / second.local_page_size[1]
    return min(first_height, second_height) > 0 and max(first_height, second_height) / min(
        first_height,
        second_height,
    ) <= 1.5


def _marginal_geometry_matches(
    first: _MarginalCandidate,
    second: _MarginalCandidate,
) -> bool:
    """比较边缘候选的方向、纵向带、字号以及同侧或镜像横向位置。"""

    if first.region != second.region or first.line.angle != second.line.angle:
        return False
    first_width, first_height = first.local_page_size
    second_width, second_height = second.local_page_size
    first_y = _bbox_center_y(first.local_bbox) / first_height
    second_y = _bbox_center_y(second.local_bbox) / second_height
    if abs(first_y - second_y) > 0.025:
        return False
    first_line_height = _line_effective_height(first.line, first.local_bbox) / first_height
    second_line_height = _line_effective_height(second.line, second.local_bbox) / second_height
    if min(first_line_height, second_line_height) <= 0 or max(first_line_height, second_line_height) / min(
        first_line_height,
        second_line_height,
    ) > 1.35:
        return False
    if (
        first.line.font_signature is not None
        and second.line.font_signature is not None
        and first.line.font_coverage >= 0.75
        and second.line.font_coverage >= 0.75
        and first.line.font_signature != second.line.font_signature
    ):
        return False

    first_normalized_bbox = (
        first.local_bbox[0] / first_width,
        first.local_bbox[1] / first_height,
        first.local_bbox[2] / first_width,
        first.local_bbox[3] / first_height,
    )
    second_normalized_bbox = (
        second.local_bbox[0] / second_width,
        second.local_bbox[1] / second_height,
        second.local_bbox[2] / second_width,
        second.local_bbox[3] / second_height,
    )
    same_side = (
        _bbox_axis_overlap_ratio(first_normalized_bbox, second_normalized_bbox, axis="x") >= 0.4
        or abs(_bbox_center_x(first_normalized_bbox) - _bbox_center_x(second_normalized_bbox)) <= 0.08
    )
    mirrored = abs(
        _bbox_center_x(first_normalized_bbox) + _bbox_center_x(second_normalized_bbox) - 1.0
    ) <= 0.12
    return same_side or mirrored


def _parse_page_number_value(text: str) -> int | None:
    """解析整行阿拉伯、罗马或中文页码；混有稳定正文的行不作为纯页码。"""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    match = _PAGE_NUMBER_RE.fullmatch(normalized)
    if match is None:
        return None
    value = match.group("value")
    if value.isdecimal():
        return int(value)
    if re.fullmatch(r"[ivxlcdm]+", value, re.IGNORECASE):
        return _roman_number_to_int(value)
    return _chinese_page_number_to_int(value)


def _roman_number_to_int(value: str) -> int | None:
    """把页码中的规范罗马数字转换成整数，非法组合返回空。"""

    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    normalized = value.upper()
    total = 0
    previous = 0
    for char in reversed(normalized):
        current = roman_values.get(char)
        if current is None:
            return None
        total += -current if current < previous else current
        previous = max(previous, current)
    if total <= 0 or total > 4999:
        return None
    return total


def _chinese_page_number_to_int(value: str) -> int | None:
    """把常见百位以内中文页码转换成整数，供跨页递增校验使用。"""

    digits = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if all(char in digits for char in value):
        try:
            return int("".join(str(digits[char]) for char in value))
        except ValueError:
            return None
    total = 0
    current_digit = 0
    for char in value:
        if char in digits:
            current_digit = digits[char]
        elif char == "十":
            total += (current_digit or 1) * 10
            current_digit = 0
        elif char == "百":
            total += (current_digit or 1) * 100
            current_digit = 0
        else:
            return None
    return total + current_digit if total + current_digit > 0 else None


def _marginal_text_matches(first_text: str, second_text: str) -> bool:
    """在屏蔽变化数字后比较边缘稳定文本，短文本只接受完全一致。"""

    first = _normalize_marginal_text(first_text)
    second = _normalize_marginal_text(second_text)
    if not first or not second:
        return False
    if first == second:
        return True
    if min(len(first), len(second)) < 8:
        return False
    return SequenceMatcher(a=first, b=second, autojunk=False).ratio() >= 0.92


def _normalize_marginal_text(text: str) -> str:
    """统一边缘重复文本的宽窄字符、大小写、空白和可变数字。"""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    return re.sub(r"\s+", "", normalized).strip()
