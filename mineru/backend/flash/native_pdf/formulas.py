# Copyright (c) Opendatalab. All rights reserved.

"""按空间关系检测并物化原生 PDF 公式块。"""

from __future__ import annotations

import re
import statistics
from dataclasses import replace
from typing import Any


from mineru.types import BBox

from .models import (
    _FormulaAnchor,
    _LineItem,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_union,
    _bbox_union_many,
    _rotate_bbox_to_upright,
)
from .native_text import _sanitize_pdf_control_text
from .line_layout import (
    _connection_crosses_table,
    _infer_text_lanes,
    _line_effective_height,
)
from .line_merging import _join_formula_visual_row


_FORMULA_NUMBER_SUFFIX_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<marker>[(（﹙][^()（）﹙﹚\r\n]+[)）﹚])\s*$"
)


def _build_formula_like_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[_LineItem]]:
    """仅依据栏带、右侧短锚点和空间连通关系聚合公式状区域。"""

    blocks: list[dict[str, Any]] = []
    claimed_source_indices: set[int] = set()
    for angle in sorted({line.angle for line in lines}):
        angle_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle
        ]
        if len(angle_geometry) < 2:
            continue
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in angle_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(angle_geometry, local_page_width, median_height)
        for lane in lanes:
            if lane.is_span or len(lane.lines) < 2:
                continue
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            dominant_body_font = _infer_formula_body_font(lane, median_height)
            anchors = _find_formula_spatial_anchors(
                lane,
                median_height,
                dominant_body_font,
            )
            if not anchors:
                continue
            anchor_centers = [_bbox_center_y(anchor.bbox) for anchor in anchors]
            lane_top = min(bbox[1] for _line, bbox in lane.lines)
            lane_bottom = max(bbox[3] for _line, bbox in lane.lines)
            for anchor_index, anchor in enumerate(anchors):
                anchor_line = anchor.line
                if anchor_line.source_index in claimed_source_indices:
                    continue
                band_top = lane_top
                band_bottom = lane_bottom
                if anchor_index > 0:
                    band_top = max(
                        band_top,
                        (anchor_centers[anchor_index - 1] + anchor_centers[anchor_index]) / 2.0,
                    )
                if anchor_index + 1 < len(anchors):
                    band_bottom = min(
                        band_bottom,
                        (anchor_centers[anchor_index] + anchor_centers[anchor_index + 1]) / 2.0,
                    )
                members = _grow_formula_spatial_component(
                    lane,
                    anchor,
                    band_top,
                    band_bottom,
                    claimed_source_indices,
                    table_bboxes,
                    dominant_body_font,
                    median_height,
                )
                if len(members) < 2:
                    continue
                if len(members) == 2 and _bbox_axis_overlap_ratio(
                    members[0][1],
                    members[1][1],
                    axis="y",
                ) < 0.2:
                    continue
                block = _formula_members_to_block(
                    members,
                    page_size,
                    angle,
                    anchor_source_index=anchor_line.source_index,
                )
                if block is None:
                    continue
                blocks.append(block)
                claimed_source_indices.update(line.source_index for line, _bbox in members)

    remaining_lines = [line for line in lines if line.source_index not in claimed_source_indices]
    return blocks, remaining_lines


def _find_formula_spatial_anchors(
    lane: _TextLane,
    median_height: float,
    dominant_body_font: tuple[str, int] | None = None,
) -> list[_FormulaAnchor]:
    """查找栏带右缘短块或带编号后缀的非正文字体公式锚点。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_interval = _formula_lane_body_interval(lane, median_height)
    if body_interval is None:
        return []
    body_top, body_bottom = body_interval
    anchors: list[_FormulaAnchor] = []
    for line, bbox in lane.lines:
        line_height = _line_effective_height(line, bbox)
        line_width = bbox[2] - bbox[0]
        is_short_right_anchor = line_width <= max(4.0 * line_height, 0.12 * lane_width)
        has_formula_number_suffix = _split_trailing_formula_number(line.text) is not None
        is_wide_numbered_anchor = (
            has_formula_number_suffix
            and line_width <= 0.75 * lane_width
            and dominant_body_font is not None
            and (
                line.font_signature != dominant_body_font
                or line.font_coverage < 0.75
            )
        )
        if not is_short_right_anchor and not is_wide_numbered_anchor:
            continue
        same_row_fragments = [
            other_line
            for other_line, _other_bbox in lane.lines
            if line.visual_row_id is not None
            and other_line.visual_row_id == line.visual_row_id
            and (line.split_from_row or other_line.split_from_row)
        ]
        if len(same_row_fragments) >= 3 and not any(
            other_line.font_coverage < 0.75
            or (
                dominant_body_font is not None
                and other_line.font_signature != dominant_body_font
            )
            for other_line in same_row_fragments
            if other_line.source_index != line.source_index
        ):
            # 一条粗行被多个大空格拆成密集词组时更像普通排版行，不能把末词当作公式编号锚点。
            continue
        if abs(lane.right - bbox[2]) > max(3.0, 0.02 * lane_width):
            continue
        center_y = _bbox_center_y(bbox)
        detached_below_body = body_bottom < center_y <= body_bottom + 6.0 * median_height
        if not body_top <= center_y <= body_bottom and not detached_below_body:
            continue
        has_left_peer = any(
            other_line.source_index != line.source_index
            and other_bbox[2] - other_bbox[0] <= 0.75 * lane_width
            and _bbox_center_x(other_bbox) < bbox[0]
            and (
                _formula_detached_seed_vertical_match(
                    bbox,
                    line_height,
                    other_bbox,
                    _line_effective_height(other_line, other_bbox),
                )
                if detached_below_body
                else _formula_seed_vertical_match(
                    bbox,
                    line_height,
                    other_bbox,
                    _line_effective_height(other_line, other_bbox),
                )
            )
            for other_line, other_bbox in lane.lines
        )
        if has_left_peer:
            anchors.append(
                _FormulaAnchor(
                    line=line,
                    bbox=bbox,
                    detached_below_body=detached_below_body,
                )
            )
    return _deduplicate_formula_anchors(anchors, median_height)


def _infer_formula_body_font(
    lane: _TextLane,
    median_height: float,
) -> tuple[str, int] | None:
    """从栏内常规宽正文行推断 dominant font，供公式扩张排除正文前缀。"""

    lane_width = max(0.1, lane.right - lane.left)
    font_counts: dict[tuple[str, int], int] = {}
    for line, bbox in lane.lines:
        line_height = _line_effective_height(line, bbox)
        if bbox[2] - bbox[0] < 0.35 * lane_width:
            continue
        if not 0.8 * median_height <= line_height <= 1.25 * median_height:
            continue
        if line.font_signature is None or line.font_coverage < 0.75:
            continue
        font_counts[line.font_signature] = font_counts.get(line.font_signature, 0) + 1
    if not font_counts:
        return None
    return max(font_counts.items(), key=lambda item: (item[1], item[0]))[0]


def _formula_lane_body_interval(
    lane: _TextLane,
    median_height: float,
) -> tuple[float, float] | None:
    """用连续出现的常规宽行确定栏带正文纵向范围，排除孤立页眉。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_lines = sorted(
        (
            item
            for item in lane.lines
            if item[1][2] - item[1][0]
            >= max(4.0 * _line_effective_height(*item), 0.35 * lane_width)
        ),
        key=lambda item: (item[1][1], item[1][0]),
    )
    if len(body_lines) < 3:
        return None
    dense_lines: list[tuple[_LineItem, BBox]] = []
    for index, item in enumerate(body_lines):
        has_close_previous = index > 0 and item[1][1] - body_lines[index - 1][1][3] <= 1.5 * median_height
        has_close_next = (
            index + 1 < len(body_lines)
            and body_lines[index + 1][1][1] - item[1][3] <= 1.5 * median_height
        )
        if has_close_previous or has_close_next:
            dense_lines.append(item)
    if len(dense_lines) < 3:
        return None
    return (
        min(bbox[1] for _line, bbox in dense_lines),
        max(bbox[3] for _line, bbox in dense_lines),
    )


def _deduplicate_formula_anchors(
    anchors: list[_FormulaAnchor],
    median_height: float,
) -> list[_FormulaAnchor]:
    """同一高度出现多个右缘短块时只保留最靠右的空间锚点。"""

    if not anchors:
        return []
    output: list[_FormulaAnchor] = []
    tolerance = max(1.5, 0.35 * median_height)
    for anchor in sorted(anchors, key=lambda item: (_bbox_center_y(item.bbox), -item.bbox[2])):
        if output and abs(_bbox_center_y(anchor.bbox) - _bbox_center_y(output[-1].bbox)) <= tolerance:
            if (anchor.bbox[2], -anchor.bbox[0]) > (output[-1].bbox[2], -output[-1].bbox[0]):
                output[-1] = anchor
            continue
        output.append(anchor)
    return output


def _grow_formula_spatial_component(
    lane: _TextLane,
    anchor: _FormulaAnchor,
    band_top: float,
    band_bottom: float,
    claimed_source_indices: set[int],
    table_bboxes: list[BBox],
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> list[tuple[_LineItem, BBox]]:
    """从右缘锚点的左侧首批成员出发，按二维邻接扩展公式分量。"""

    anchor_line, anchor_bbox = anchor.line, anchor.bbox
    anchor_geometry = (anchor_line, anchor_bbox)
    lane_width = max(0.1, lane.right - lane.left)
    candidates = [
        item
        for item in lane.lines
        if item[0].source_index not in claimed_source_indices
        and item[1][2] - item[1][0] <= 0.8 * lane_width
        and band_top <= _bbox_center_y(item[1]) <= band_bottom
        and not _is_formula_body_barrier(
            item,
            lane,
            dominant_body_font,
            median_height,
        )
        and not _is_formula_body_prefix(
            item,
            lane,
            anchor_geometry,
            dominant_body_font,
            median_height,
        )
    ]
    seeds = [
        item
        for item in candidates
        if item[0].source_index != anchor_line.source_index
        and _bbox_center_x(item[1]) < anchor_bbox[0]
        and (
            _formula_detached_seed_vertical_match(
                anchor_bbox,
                _line_effective_height(anchor_line, anchor_bbox),
                item[1],
                _line_effective_height(*item),
            )
            if anchor.detached_below_body
            else _formula_seed_vertical_match(
                anchor_bbox,
                _line_effective_height(anchor_line, anchor_bbox),
                item[1],
                _line_effective_height(*item),
            )
        )
        and not _connection_crosses_table(anchor_line.bbox, item[0].bbox, table_bboxes)
    ]
    if not seeds:
        return []

    members = [anchor_geometry, *seeds]
    member_sources = {line.source_index for line, _bbox in members}
    changed = True
    while changed:
        changed = False
        for candidate in candidates:
            candidate_line, candidate_bbox = candidate
            if candidate_line.source_index in member_sources:
                continue
            if any(
                _formula_lines_are_connected(
                    member_line,
                    member_bbox,
                    candidate_line,
                    candidate_bbox,
                    table_bboxes,
                )
                for member_line, member_bbox in members
            ):
                members.append(candidate)
                member_sources.add(candidate_line.source_index)
                changed = True
    return members


def _is_formula_body_barrier(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> bool:
    """识别近满栏、高置信正文行，阻止公式连通分量跨越正文边界。"""

    if dominant_body_font is None:
        return False
    line, bbox = candidate
    lane_width = max(0.1, lane.right - lane.left)
    line_height = _line_effective_height(line, bbox)
    return (
        line.font_signature == dominant_body_font
        and bbox[2] - bbox[0] >= 0.85 * lane_width
        and 0.8 * median_height <= line_height <= 1.25 * median_height
    )


def _is_formula_body_prefix(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    anchor: tuple[_LineItem, BBox],
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> bool:
    """识别锚点上方左对齐的常规正文行，防止公式空间扩张越界认领。"""

    if dominant_body_font is None:
        return False
    line, bbox = candidate
    anchor_line, anchor_bbox = anchor
    line_height = _line_effective_height(line, bbox)
    anchor_height = _line_effective_height(anchor_line, anchor_bbox)
    if _bbox_center_y(bbox) > _bbox_center_y(anchor_bbox) - 0.2 * max(line_height, anchor_height):
        return False
    if abs(bbox[0] - lane.left) > max(3.0, 0.75 * median_height):
        return False
    return (
        line.font_signature == dominant_body_font
        and line.font_coverage >= 0.75
        and 0.8 * median_height <= line_height <= 1.25 * median_height
    )


def _formula_detached_seed_vertical_match(
    anchor_bbox: BBox,
    anchor_height: float,
    candidate_bbox: BBox,
    candidate_height: float,
) -> bool:
    """放宽正文密集区下方锚点的同高匹配，以接纳多行分段公式底部。"""

    has_vertical_overlap = min(anchor_bbox[3], candidate_bbox[3]) > max(anchor_bbox[1], candidate_bbox[1])
    center_difference = abs(_bbox_center_y(anchor_bbox) - _bbox_center_y(candidate_bbox))
    return has_vertical_overlap or center_difference <= max(anchor_height, candidate_height)


def _formula_seed_vertical_match(
    anchor_bbox: BBox,
    anchor_height: float,
    candidate_bbox: BBox,
    candidate_height: float,
) -> bool:
    """判断左侧短行是否与右缘锚点处在同一公式高度带。"""

    overlap_ratio = _bbox_axis_overlap_ratio(anchor_bbox, candidate_bbox, axis="y")
    center_difference = abs(_bbox_center_y(anchor_bbox) - _bbox_center_y(candidate_bbox))
    return overlap_ratio >= 0.3 or center_difference <= 0.6 * max(anchor_height, candidate_height)


def _formula_lines_are_connected(
    first_line: _LineItem,
    first_bbox: BBox,
    second_line: _LineItem,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """按垂直接近和水平覆盖判断两个公式成员是否空间连通。"""

    if first_line.angle != second_line.angle:
        return False
    if _connection_crosses_table(first_line.bbox, second_line.bbox, table_bboxes):
        return False
    first_height = _line_effective_height(first_line, first_bbox)
    second_height = _line_effective_height(second_line, second_bbox)
    pair_height = max(first_height, second_height)
    vertical_overlap = _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="y")
    vertical_gap = max(first_bbox[1] - second_bbox[3], second_bbox[1] - first_bbox[3], 0.0)
    if vertical_overlap < 0.2 and vertical_gap > 0.6 * pair_height:
        return False
    horizontal_overlap = _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x")
    horizontal_gap = max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)
    return horizontal_overlap > 0.0 or horizontal_gap <= 1.5 * pair_height


def _is_detached_formula_sidecar(
    anchor: tuple[_LineItem, BBox],
    members: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> bool:
    """仅依据 bbox 判断右侧锚点是否为与公式主体分离的窄幅 sidecar。"""

    anchor_line, anchor_bbox = anchor
    body_bboxes = [
        bbox
        for line, bbox in members
        if line.source_index != anchor_line.source_index
    ]
    if not body_bboxes:
        return False

    body_bbox = _bbox_union_many(body_bboxes)
    component_bbox = _bbox_union(body_bbox, anchor_bbox)
    effective_height = max(0.1, median_height)
    anchor_width = max(0.0, anchor_bbox[2] - anchor_bbox[0])
    component_width = max(0.1, component_bbox[2] - component_bbox[0])
    horizontal_gap = anchor_bbox[0] - body_bbox[2]
    right_tolerance = max(0.5, 0.1 * effective_height)
    minimum_gap = max(2.5 * effective_height, 0.08 * component_width)

    return (
        anchor_bbox[0] >= body_bbox[2]
        and anchor_bbox[2] >= component_bbox[2] - right_tolerance
        and anchor_width <= 2.0 * effective_height
        and horizontal_gap > minimum_gap
    )


def _split_trailing_formula_number(text: str) -> tuple[str, str] | None:
    """拆出右缘文本末尾的圆括号公式序号，并保留序号前的标点或正文。"""

    match = _FORMULA_NUMBER_SUFFIX_RE.fullmatch(str(text or "").strip())
    if match is None:
        return None
    return match.group("prefix").rstrip(), match.group("marker").strip()


def _formula_members_to_block(
    members: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
    angle: int,
    *,
    anchor_source_index: int,
) -> dict[str, Any] | None:
    """把公式空间分量按视觉行聚类，并将非末视觉行的离散右侧 sidecar 后置。"""

    heights = [_line_effective_height(line, bbox) for line, bbox in members]
    median_height = statistics.median(heights) if heights else 1.0
    row_tolerance = max(1.5, 0.35 * median_height)
    rows: list[list[tuple[_LineItem, BBox]]] = []
    for member in sorted(members, key=lambda item: (_bbox_center_y(item[1]), item[1][0], item[0].source_index)):
        if not rows:
            rows.append([member])
            continue
        row_center = statistics.median(_bbox_center_y(bbox) for _line, bbox in rows[-1])
        if abs(_bbox_center_y(member[1]) - row_center) <= row_tolerance:
            rows[-1].append(member)
        else:
            rows.append([member])

    trailing_sidecar_content: str | None = None
    # 右侧 sidecar 按视觉 y 常落在分式中部；仅在其后仍有公式行时转为逻辑末行。
    for row_index, row in enumerate(rows[:-1]):
        anchor_member = next(
            (member for member in row if member[0].source_index == anchor_source_index),
            None,
        )
        if anchor_member is None:
            continue
        formula_number_parts = _split_trailing_formula_number(anchor_member[0].text)
        if formula_number_parts is not None:
            prefix, marker = formula_number_parts
            rows[row_index] = [
                (
                    (replace(member[0], text=prefix), member[1])
                    if member[0].source_index == anchor_source_index and prefix
                    else member
                )
                for member in row
                if member[0].source_index != anchor_source_index or prefix
            ]
            trailing_sidecar_content = marker
        elif _is_detached_formula_sidecar(anchor_member, members, median_height):
            rows[row_index] = [
                member for member in row if member[0].source_index != anchor_source_index
            ]
            trailing_sidecar_content = anchor_member[0].text.strip()
        break

    row_contents = [_join_formula_visual_row(row, page_size) for row in rows if row]
    if trailing_sidecar_content is not None:
        row_contents.append(trailing_sidecar_content)
    content = _sanitize_pdf_control_text("\n".join(filter(None, row_contents)), preserve_newlines=True)
    if not content.strip():
        return None
    return {
        "type": "equation",
        "bbox": _bbox_union_many([line.bbox for line, _bbox in members]),
        "angle": angle,
        "content": content,
    }

