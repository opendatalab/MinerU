# Copyright (c) Opendatalab. All rights reserved.

"""按空间关系检测并物化原生 PDF 公式块。"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, replace
from typing import Any


from ....types import BBox
from .document import PDFPathInfo
from ....utils.text import build_tagged_formula_content

from .models import (
    _FormulaAnchor,
    _LineItem,
    _PageSource,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_distance,
    _bbox_intersects,
    _bbox_overlap_in_first,
    _bbox_overlap_in_smaller,
    _bbox_union,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _expand_bbox,
    _rotate_bbox_to_upright,
)
from .native_text import _sanitize_pdf_control_text
from .line_layout import (
    _connection_crosses_table,
    _infer_text_lanes,
    _line_effective_height,
)
from .line_merging import _join_formula_visual_row


_FORMULA_NUMBER_SUFFIX_RE = re.compile(r"^(?P<prefix>.*?)(?P<marker>[(（﹙][^()（）﹙﹚\r\n]+[)）﹚])\s*$")


_FORMULA_PAGE_MARGIN_RATIO = 0.05
_VECTOR_FORMULA_COMPLEX_SEGMENTS = 8
_VECTOR_FORMULA_MIN_PATHS = 5
_VECTOR_FORMULA_MIN_COMPLEX_PATHS = 5
_VECTOR_FORMULA_MIN_COMPLEX_RATIO = 0.5
_VECTOR_FORMULA_NUMBER_MIN_PATHS = 3
_VECTOR_FORMULA_NUMBER_MAX_PATHS = 6


@dataclass(slots=True)
class _VectorPathComponent:
    """保存同栏邻接 Path 形成的矢量组件。"""

    lane_index: int
    path_infos: list[PDFPathInfo]
    bbox: BBox


@dataclass(slots=True)
class _VectorFormulaCandidate:
    """保存已通过主体校验、等待吸收横线和编号的矢量公式。"""

    lane_index: int
    bbox: BBox
    path_source_indices: set[int]
    has_number: bool = False


def _build_vector_formula_blocks(
    source: _PageSource,
    container_blocks: list[dict[str, Any]],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """从根层填充 Path 构建空内容公式，并唯一认领可提取的公式编号。"""

    available_lines = [line for line in source.lines if line.angle == 0 and line.source_index not in claimed_line_indices]
    if len(available_lines) < 3 or not source.path_infos:
        return [], set()

    line_geometry = [(line, line.bbox) for line in available_lines]
    effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
    median_height = statistics.median(effective_heights) if effective_heights else 0.0
    if median_height <= 0:
        return [], set()
    lanes = [
        lane
        for lane in _infer_text_lanes(line_geometry, source.page_size[0], median_height)
        if not lane.is_span and len(lane.lines) >= 3
    ]
    if not lanes:
        return [], set()

    components = _build_vector_path_components(
        source.path_infos,
        lanes,
        median_height,
    )
    container_bboxes = [bbox for block in container_blocks if (bbox := _coerce_bbox(block.get("bbox"))) is not None]
    candidates = [
        _VectorFormulaCandidate(
            lane_index=component.lane_index,
            bbox=component.bbox,
            path_source_indices={item.source_index for item in component.path_infos},
        )
        for component in components
        if _is_vector_formula_core(
            component,
            lanes[component.lane_index],
            median_height,
            source.page_size,
            container_bboxes,
        )
    ]
    if not candidates:
        return [], set()

    _attach_vector_formula_rules(candidates, components, median_height)
    _attach_vector_formula_path_numbers(candidates, components, lanes, median_height)
    claimed_number_indices = _attach_vector_formula_text_numbers(
        candidates,
        lanes,
        median_height,
        claimed_line_indices,
    )

    padding = min(1.5, 0.1 * median_height)
    blocks: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0])):
        padded_bbox = _clip_bbox(
            (
                candidate.bbox[0] - padding,
                candidate.bbox[1] - padding,
                candidate.bbox[2] + padding,
                candidate.bbox[3] + padding,
            ),
            source.page_size,
        )
        if padded_bbox is None:
            continue
        blocks.append(
            {
                "type": "equation",
                "bbox": padded_bbox,
                "angle": 0,
                "content": "",
            }
        )
    return blocks, claimed_number_indices


def _build_vector_path_components(
    path_infos: list[PDFPathInfo],
    lanes: list[_TextLane],
    median_height: float,
) -> list[_VectorPathComponent]:
    """按文本栏带筛选矢量字形，并用空间网格生成局部连通组件。"""

    members_by_lane: dict[int, list[PDFPathInfo]] = {}
    for path_info in path_infos:
        if path_info.form_depth != 0 or not path_info.fill_visible or path_info.stroke_visible:
            continue
        lane_index = _assign_vector_path_lane(path_info.bbox, lanes, median_height)
        if lane_index is None:
            continue
        if not _is_vector_formula_path_member(
            path_info.bbox,
            lanes[lane_index],
            median_height,
        ):
            continue
        members_by_lane.setdefault(lane_index, []).append(path_info)

    components: list[_VectorPathComponent] = []
    for lane_index, members in members_by_lane.items():
        components.extend(
            _connect_vector_path_members(
                members,
                lane_index,
                median_height,
            )
        )
    return sorted(
        components,
        key=lambda item: (item.bbox[1], item.bbox[0], item.path_infos[0].source_index),
    )


def _assign_vector_path_lane(
    bbox: BBox,
    lanes: list[_TextLane],
    median_height: float,
) -> int | None:
    """按中心点和水平覆盖率把 Path 唯一分配给一个正文栏带。"""

    center_x = _bbox_center_x(bbox)
    path_width = max(0.1, bbox[2] - bbox[0])
    tolerance = 0.75 * median_height
    matches: list[tuple[float, float, int]] = []
    for lane_index, lane in enumerate(lanes):
        if not lane.left - tolerance <= center_x <= lane.right + tolerance:
            continue
        overlap = max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left))
        coverage = overlap / path_width
        lane_center = (lane.left + lane.right) / 2.0
        matches.append((-coverage, abs(center_x - lane_center), lane_index))
    return min(matches)[2] if matches else None


def _is_vector_formula_path_member(
    bbox: BBox,
    lane: _TextLane,
    median_height: float,
) -> bool:
    """保留小字形轮廓和细横线，过滤跨栏或过大的普通矢量对象。"""

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    lane_width = max(0.1, lane.right - lane.left)
    is_glyph = width <= 3.0 * median_height and height <= 3.0 * median_height
    is_formula_rule = height <= 0.2 * median_height and width <= lane_width + median_height
    return is_glyph or is_formula_rule


def _connect_vector_path_members(
    members: list[PDFPathInfo],
    lane_index: int,
    median_height: float,
) -> list[_VectorPathComponent]:
    """用扩张 bbox 的网格邻接和并查集连接同栏 Path，避免全量两两比较。"""

    if not members:
        return []
    ordered = sorted(members, key=lambda item: item.source_index)
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        """查找并压缩一个 Path 的并查集根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(first: int, second: int) -> None:
        """合并两个相交扩张框所属的连通分量。"""

        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    margin = 0.5 * median_height
    cell_size = max(1.0, median_height)
    expanded_bboxes = [_expand_bbox(item.bbox, margin) for item in ordered]
    grid: dict[tuple[int, int], list[int]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for index, bbox in enumerate(expanded_bboxes):
        start_x = int(bbox[0] // cell_size)
        end_x = int(bbox[2] // cell_size)
        start_y = int(bbox[1] // cell_size)
        end_y = int(bbox[3] // cell_size)
        for cell_x in range(start_x, end_x + 1):
            for cell_y in range(start_y, end_y + 1):
                cell = (cell_x, cell_y)
                for other_index in grid.get(cell, []):
                    pair = (other_index, index)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    if _bbox_intersects(bbox, expanded_bboxes[other_index]):
                        merge(index, other_index)
                grid.setdefault(cell, []).append(index)

    grouped: dict[int, list[PDFPathInfo]] = {}
    for index, path_info in enumerate(ordered):
        grouped.setdefault(find(index), []).append(path_info)
    return [
        _VectorPathComponent(
            lane_index=lane_index,
            path_infos=group,
            bbox=_bbox_union_many([item.bbox for item in group]),
        )
        for group in grouped.values()
    ]


def _is_vector_formula_core(
    component: _VectorPathComponent,
    lane: _TextLane,
    median_height: float,
    page_size: tuple[float, float],
    container_bboxes: list[BBox],
) -> bool:
    """按复杂度、尺寸、正文碰撞和容器优先级校验公式主体组件。"""

    path_count = len(component.path_infos)
    complex_count = sum(item.segment_count >= _VECTOR_FORMULA_COMPLEX_SEGMENTS for item in component.path_infos)
    if (
        path_count < _VECTOR_FORMULA_MIN_PATHS
        or complex_count < _VECTOR_FORMULA_MIN_COMPLEX_PATHS
        or complex_count / path_count < _VECTOR_FORMULA_MIN_COMPLEX_RATIO
    ):
        return False

    bbox = component.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if not (width >= 2.5 * median_height and 0.9 * median_height <= height <= 8.0 * median_height and width >= 1.4 * height):
        return False
    if _is_formula_component_in_page_margin(bbox, page_size[1]):
        return False
    if any(_bbox_overlap_in_smaller(bbox, container_bbox) >= 0.5 for container_bbox in container_bboxes):
        return False
    return not any(_vector_formula_collides_with_text(bbox, line, line_bbox, median_height) for line, line_bbox in lane.lines)


def _is_formula_component_in_page_margin(bbox: BBox, page_height: float) -> bool:
    """仅当公式组件完全落在页面顶部或底部边缘带时排除。"""

    margin = _FORMULA_PAGE_MARGIN_RATIO * page_height
    return bbox[3] <= margin or bbox[1] >= page_height - margin


def _vector_formula_collides_with_text(
    formula_bbox: BBox,
    line: _LineItem,
    line_bbox: BBox,
    median_height: float,
) -> bool:
    """排除覆盖正文或紧贴正文同行的 Path 组件，独立公式编号除外。"""

    if _standalone_formula_number_marker(line.text) is not None:
        return False
    if _bbox_overlap_in_first(formula_bbox, line_bbox) >= 0.2:
        return True
    horizontal_gap = max(
        formula_bbox[0] - line_bbox[2],
        line_bbox[0] - formula_bbox[2],
        0.0,
    )
    return _bbox_axis_overlap_ratio(formula_bbox, line_bbox, axis="y") >= 0.5 and horizontal_gap <= median_height


def _attach_vector_formula_rules(
    candidates: list[_VectorFormulaCandidate],
    components: list[_VectorPathComponent],
    median_height: float,
) -> None:
    """把靠近公式主体且横向覆盖充分的孤立细横线唯一并入主体。"""

    used_sources = {source_index for candidate in candidates for source_index in candidate.path_source_indices}
    for component in components:
        component_sources = {item.source_index for item in component.path_infos}
        if component_sources & used_sources or not all(
            item.bbox[3] - item.bbox[1] <= 0.2 * median_height for item in component.path_infos
        ):
            continue
        matches = [
            (
                _bbox_distance(candidate.bbox, component.bbox),
                abs(_bbox_center_y(candidate.bbox) - _bbox_center_y(component.bbox)),
                candidate_index,
            )
            for candidate_index, candidate in enumerate(candidates)
            if candidate.lane_index == component.lane_index
            and _bbox_distance(candidate.bbox, component.bbox) <= 0.5 * median_height
            and _bbox_axis_overlap_ratio(candidate.bbox, component.bbox, axis="x") >= 0.5
        ]
        if not matches:
            continue
        candidate = candidates[min(matches)[2]]
        candidate.bbox = _bbox_union(candidate.bbox, component.bbox)
        candidate.path_source_indices.update(component_sources)
        used_sources.update(component_sources)


def _attach_vector_formula_path_numbers(
    candidates: list[_VectorFormulaCandidate],
    components: list[_VectorPathComponent],
    lanes: list[_TextLane],
    median_height: float,
) -> None:
    """把栏右缘的小型复杂 Path 组件作为公式编号并入唯一主体。"""

    used_sources = {source_index for candidate in candidates for source_index in candidate.path_source_indices}
    for component in components:
        component_sources = {item.source_index for item in component.path_infos}
        if component_sources & used_sources or not _is_vector_formula_number_component(
            component,
            lanes[component.lane_index],
            median_height,
        ):
            continue
        matches = _vector_formula_number_matches(
            component.bbox,
            component.lane_index,
            candidates,
            median_height,
        )
        if not matches:
            continue
        candidate = candidates[min(matches)[3]]
        candidate.bbox = _bbox_union(candidate.bbox, component.bbox)
        candidate.path_source_indices.update(component_sources)
        candidate.has_number = True
        used_sources.update(component_sources)


def _is_vector_formula_number_component(
    component: _VectorPathComponent,
    lane: _TextLane,
    median_height: float,
) -> bool:
    """识别位于栏右缘、尺寸接近正文行高的全复杂路径编号组件。"""

    path_count = len(component.path_infos)
    bbox = component.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (
        _VECTOR_FORMULA_NUMBER_MIN_PATHS <= path_count <= _VECTOR_FORMULA_NUMBER_MAX_PATHS
        and all(item.segment_count >= _VECTOR_FORMULA_COMPLEX_SEGMENTS for item in component.path_infos)
        and 0.5 * median_height <= width <= 2.0 * median_height
        and 0.6 * median_height <= height <= 1.4 * median_height
        and abs(lane.right - bbox[2]) <= 1.5 * median_height
    )


def _vector_formula_number_matches(
    number_bbox: BBox,
    lane_index: int,
    candidates: list[_VectorFormulaCandidate],
    median_height: float,
) -> list[tuple[float, float, float, int]]:
    """返回编号可关联的公式主体及稳定排序分值。"""

    matches: list[tuple[float, float, float, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        if candidate.has_number or candidate.lane_index != lane_index:
            continue
        vertical_overlap = _bbox_axis_overlap_ratio(candidate.bbox, number_bbox, axis="y")
        if number_bbox[0] < candidate.bbox[2] or vertical_overlap < 0.6:
            continue
        center_distance = abs(_bbox_center_y(candidate.bbox) - _bbox_center_y(number_bbox))
        horizontal_gap = max(0.0, number_bbox[0] - candidate.bbox[2])
        matches.append((-vertical_overlap, center_distance, horizontal_gap / max(0.1, median_height), candidate_index))
    return matches


def _attach_vector_formula_text_numbers(
    candidates: list[_VectorFormulaCandidate],
    lanes: list[_TextLane],
    median_height: float,
    claimed_line_indices: set[int],
) -> set[int]:
    """关联可提取的独立公式编号并认领其文本身份，防止重复输出。"""

    claimed: set[int] = set()
    for lane_index, lane in enumerate(lanes):
        for line, bbox in sorted(lane.lines, key=lambda item: (item[1][1], item[1][0])):
            if line.source_index in claimed_line_indices or _standalone_formula_number_marker(line.text) is None:
                continue
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if not (
                0.5 * median_height <= width <= 2.0 * median_height
                and 0.6 * median_height <= height <= 1.4 * median_height
                and abs(lane.right - bbox[2]) <= 1.5 * median_height
            ):
                continue
            matches = _vector_formula_number_matches(
                bbox,
                lane_index,
                candidates,
                median_height,
            )
            if not matches:
                continue
            candidate = candidates[min(matches)[3]]
            candidate.bbox = _bbox_union(candidate.bbox, bbox)
            candidate.has_number = True
            claimed.add(line.source_index)
    return claimed


def _standalone_formula_number_marker(text: str) -> str | None:
    """仅接受整行由圆括号公式编号构成的文本，不接纳带正文前缀的后缀。"""

    parts = _split_trailing_formula_number(text)
    if parts is None:
        return None
    prefix, marker = parts
    return marker if not prefix else None


def _build_formula_like_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[_LineItem]]:
    """仅依据栏带、右侧短锚点和空间连通关系聚合公式状区域。"""

    blocks, claimed_source_indices = _build_split_visual_row_formula_blocks(
        lines,
        table_bboxes,
        page_size,
    )
    for angle in sorted({line.angle for line in lines}):
        angle_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.source_index not in claimed_source_indices
        ]
        if len(angle_geometry) < 2:
            continue
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in angle_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        lanes = _infer_text_lanes(angle_geometry, local_page_width, median_height)
        for lane in lanes:
            if lane.is_span:
                continue
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            dominant_body_font = _infer_formula_body_font(
                lane,
                median_height,
            )
            for line, bbox in list(lane.lines):
                if (
                    (
                        line.compact_formula_cluster
                        and not _compact_cluster_has_nearby_number_anchor(
                            (line, bbox),
                            lane,
                            median_height,
                        )
                        and _is_isolated_compact_formula_cluster(
                            (line, bbox),
                            lane,
                            median_height,
                        )
                    )
                    or _is_isolated_unnumbered_formula_line(
                        (line, bbox),
                        lane,
                        median_height,
                        dominant_body_font,
                    )
                ) and not _is_formula_component_in_page_margin(
                    bbox,
                    local_page_height,
                ):
                    content = _sanitize_pdf_control_text(
                        line.text,
                        preserve_newlines=False,
                    ).strip()
                    if not content:
                        continue
                    blocks.append(
                        {
                            "type": "equation",
                            "bbox": line.bbox,
                            "angle": angle,
                            "content": content,
                        }
                    )
                    claimed_source_indices.add(line.source_index)
            lane.lines = [item for item in lane.lines if item[0].source_index not in claimed_source_indices]
            if len(lane.lines) < 2:
                continue
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
                if (
                    len(members) == 2
                    and _bbox_axis_overlap_ratio(
                        members[0][1],
                        members[1][1],
                        axis="y",
                    )
                    < 0.2
                ):
                    continue
                component_bbox = _bbox_union_many([member_bbox for _member_line, member_bbox in members])
                if _is_formula_component_in_page_margin(
                    component_bbox,
                    local_page_height,
                ):
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

    remaining_lines = [
        line for line in lines if line.source_index not in claimed_source_indices and not line.formula_candidate_only
    ]
    return blocks, remaining_lines


def _build_split_visual_row_formula_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], set[int]]:
    """在栏带推断前恢复同一视觉行中带右侧编号的多字体公式。"""

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None or not line.split_from_row:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for (angle, _row_id), members in row_groups.items():
        if len(members) < 3:
            continue
        markers = [member for member in members if _standalone_formula_number_marker(member.text) is not None]
        if len(markers) != 1:
            continue
        marker = markers[0]
        if any(_bbox_intersects(member.bbox, table_bbox) for member in members for table_bbox in table_bboxes):
            continue
        local_members = [
            (
                member,
                _rotate_bbox_to_upright(
                    member.bbox,
                    page_size,
                    angle,
                ),
            )
            for member in members
        ]
        marker_bbox = next(bbox for member, bbox in local_members if member is marker)
        body_members = [(member, bbox) for member, bbox in local_members if member is not marker]
        if not body_members or marker_bbox[0] <= max(_bbox_center_x(bbox) for _member, bbox in body_members):
            continue
        median_height = statistics.median(_line_effective_height(member, bbox) for member, bbox in local_members)
        row_center = statistics.median(_bbox_center_y(bbox) for _member, bbox in local_members)
        if any(abs(_bbox_center_y(bbox) - row_center) > 0.75 * median_height for _member, bbox in local_members):
            continue
        body_fonts = {member.font_signature for member, _bbox in body_members if member.font_signature is not None}
        has_math_typography = len(body_fonts) >= 2 or any(
            member.compact_formula_cluster or member.font_coverage < 0.8 for member, _bbox in body_members
        )
        if not has_math_typography:
            continue
        body_bbox = _bbox_union_many([bbox for _member, bbox in body_members])
        body_width = max(0.1, body_bbox[2] - body_bbox[0])
        # 同行成员可能只是分式尾部；窄尾部不能压低外部公式片段的宽度容差，
        # 否则会提前认领分母、右括号和编号，使左侧公式主体落回普通文本。
        nearby_fragment_width_limit = max(
            0.65 * body_width,
            3.0 * median_height,
        )
        member_ids = {id(member) for member in members}
        has_nearby_formula_fragment = False
        for other in lines:
            if id(other) in member_ids or other.angle != angle:
                continue
            other_bbox = _rotate_bbox_to_upright(
                other.bbox,
                page_size,
                angle,
            )
            vertical_gap = max(
                0.0,
                max(other_bbox[1], body_bbox[1]) - min(other_bbox[3], body_bbox[3]),
            )
            if (
                vertical_gap <= 0.75 * median_height
                and other_bbox[2] - other_bbox[0] <= nearby_fragment_width_limit
                and max(
                    0.0,
                    max(other_bbox[0], body_bbox[0]) - min(other_bbox[2], body_bbox[2]),
                )
                <= median_height
            ):
                has_nearby_formula_fragment = True
                break
        if has_nearby_formula_fragment:
            continue
        block = _formula_members_to_block(
            local_members,
            page_size,
            angle,
            anchor_source_index=marker.source_index,
        )
        if block is None:
            continue
        local_bbox = _bbox_union_many([bbox for _member, bbox in local_members])
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        if _is_formula_component_in_page_margin(
            local_bbox,
            local_page_height,
        ):
            continue
        blocks.append(block)
        claimed.update(member.source_index for member in members)
    return blocks, claimed


def _is_isolated_compact_formula_cluster(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """用上下正文邻行确认紧凑二维文本簇是独立行间公式。"""

    line, bbox = candidate
    if not line.compact_formula_cluster:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    if bbox[2] - bbox[0] > 0.6 * lane_width:
        return False
    if bbox[3] - bbox[1] > 3.0 * median_height:
        return False
    center_delta_ratio = abs(_bbox_center_x(bbox) - 0.5 * (lane.left + lane.right)) / lane_width
    left_indent_ratio = (bbox[0] - lane.left) / lane_width
    right_blank_ratio = (lane.right - bbox[2]) / lane_width
    # 部分期刊把独立公式按固定左缩进排版；同时要求右侧大留白，排除贴栏正文。
    deliberately_left_indented = 0.03 <= left_indent_ratio <= 0.25 and right_blank_ratio >= 0.35
    if center_delta_ratio > 0.2 and not deliberately_left_indented:
        return False

    candidate_center = _bbox_center_y(bbox)
    body_rows = [
        item
        for item in lane.lines
        if item[0].source_index != line.source_index
        and item[1][2] - item[1][0] >= 0.45 * lane_width
        and 0.8 * median_height <= _line_effective_height(*item) <= 1.25 * median_height
    ]
    rows_above = [item for item in body_rows if _bbox_center_y(item[1]) < candidate_center]
    rows_below = [item for item in body_rows if _bbox_center_y(item[1]) > candidate_center]
    if not rows_above or not rows_below:
        return False
    previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
    following = min(rows_below, key=lambda item: _bbox_center_y(item[1]))
    return (
        candidate_center - _bbox_center_y(previous[1]) <= 8.0 * median_height
        and _bbox_center_y(following[1]) - candidate_center <= 8.0 * median_height
    )


def _compact_cluster_has_nearby_number_anchor(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """检测紧凑公式右侧的独立编号，保留给既有空间锚点统一扩张。"""

    line, bbox = candidate
    return any(
        other_line.source_index != line.source_index
        and _standalone_formula_number_marker(other_line.text) is not None
        and other_bbox[0] > _bbox_center_x(bbox)
        and abs(_bbox_center_y(other_bbox) - _bbox_center_y(bbox)) <= 2.5 * median_height
        for other_line, other_bbox in lane.lines
    )


def _is_isolated_unnumbered_formula_line(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
    dominant_body_font: tuple[str, int] | None,
) -> bool:
    """用低正文覆盖的数学排版和上下正文邻接识别无编号行间公式。"""

    line, bbox = candidate
    if (
        line.compact_formula_cluster
        or dominant_body_font is None
        or line.font_signature is None
        or line.font_signature == dominant_body_font
        or line.font_coverage >= 0.75
    ):
        return False
    lane_width = max(0.1, lane.right - lane.left)
    line_width = bbox[2] - bbox[0]
    if not 0.15 * lane_width <= line_width <= 0.8 * lane_width:
        return False
    if abs(_bbox_center_x(bbox) - 0.5 * (lane.left + lane.right)) > 0.08 * lane_width:
        return False
    if bbox[3] - bbox[1] > 1.8 * median_height:
        return False
    if _is_hanging_indent_tail_line(candidate, lane, median_height):
        return False
    candidate_center = _bbox_center_y(bbox)
    if any(
        other_line.source_index != line.source_index
        and _standalone_formula_number_marker(other_line.text) is not None
        and abs(_bbox_center_y(other_bbox) - candidate_center) <= 4.0 * median_height
        for other_line, other_bbox in lane.lines
    ) or _has_nearby_punctuated_formula_number_anchor(
        candidate,
        lane,
        median_height,
    ):
        return False
    body_rows = [
        item
        for item in lane.lines
        if item[0].source_index != line.source_index
        and item[0].font_signature == dominant_body_font
        and item[0].font_coverage >= 0.75
        and item[1][2] - item[1][0] >= 0.45 * lane_width
        and 0.8 * median_height <= _line_effective_height(*item) <= 1.25 * median_height
    ]
    rows_above = [item for item in body_rows if _bbox_center_y(item[1]) < candidate_center]
    rows_below = [item for item in body_rows if _bbox_center_y(item[1]) > candidate_center]
    if not rows_above or not rows_below:
        return False
    previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
    following = min(rows_below, key=lambda item: _bbox_center_y(item[1]))
    return (
        candidate_center - _bbox_center_y(previous[1]) <= 4.0 * median_height
        and _bbox_center_y(following[1]) - candidate_center <= 4.0 * median_height
    )


def _is_hanging_indent_tail_line(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """用相邻行缩进、字体和节奏识别参考条目的悬挂缩进尾行。"""

    line, bbox = candidate
    if line.font_signature is None:
        return False
    candidate_center = _bbox_center_y(bbox)
    rows_above = [
        item for item in lane.lines if item[0].source_index != line.source_index and _bbox_center_y(item[1]) < candidate_center
    ]
    rows_below = [
        item for item in lane.lines if item[0].source_index != line.source_index and _bbox_center_y(item[1]) > candidate_center
    ]
    if not rows_above or not rows_below:
        return False
    previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
    following = min(rows_below, key=lambda item: _bbox_center_y(item[1]))
    previous_line, previous_bbox = previous
    _following_line, following_bbox = following
    previous_pitch = candidate_center - _bbox_center_y(previous_bbox)
    following_pitch = _bbox_center_y(following_bbox) - candidate_center
    lane_width = max(0.1, lane.right - lane.left)
    candidate_width = bbox[2] - bbox[0]
    previous_width = previous_bbox[2] - previous_bbox[0]
    return (
        previous_line.font_signature == line.font_signature
        and abs(previous_bbox[0] - bbox[0]) <= 0.5 * median_height
        and candidate_width <= 0.9 * previous_width
        and 0.65 * median_height <= previous_pitch <= 1.6 * median_height
        and 0.65 * median_height <= following_pitch <= 1.6 * median_height
        and following_bbox[0] <= bbox[0] - 1.5 * median_height
        and following_bbox[2] - following_bbox[0] >= 0.75 * lane_width
    )


def _has_nearby_punctuated_formula_number_anchor(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """识别同一公式带右侧仅带标点前缀的编号，避免分式上下行被提前认领。"""

    line, bbox = candidate
    for other_line, other_bbox in lane.lines:
        if other_line.source_index == line.source_index:
            continue
        parts = _split_trailing_formula_number(other_line.text)
        if parts is None:
            continue
        prefix, _marker = parts
        compact_prefix = prefix.strip()
        if not compact_prefix or len(compact_prefix) > 3 or any(character.isalnum() for character in compact_prefix):
            continue
        vertical_gap = max(
            0.0,
            max(other_bbox[1], bbox[1]) - min(other_bbox[3], bbox[3]),
        )
        horizontal_gap = max(0.0, other_bbox[0] - bbox[2])
        if (
            other_bbox[0] >= bbox[2] - 0.5 * median_height
            and vertical_gap <= 0.75 * median_height
            and horizontal_gap <= 4.0 * median_height
        ):
            return True
    return False


def _find_repeated_formula_number_anchors(
    lane: _TextLane,
    median_height: float,
    body_interval: tuple[float, float] | None,
) -> list[_FormulaAnchor]:
    """用栏右缘重复编号恢复正文区间之外的行间公式锚点。"""
    lane_width = max(0.1, lane.right - lane.left)
    markers = [
        (line, bbox)
        for line, bbox in lane.lines
        if (parts := _split_trailing_formula_number(line.text)) is not None
        and not parts[0]
        and abs(lane.right - bbox[2]) <= max(3.0, 0.02 * lane_width)
    ]
    output: list[_FormulaAnchor] = []
    for line, bbox in markers:
        if not any(
            other_line.source_index != line.source_index
            and abs(_bbox_center_y(other_bbox) - _bbox_center_y(bbox)) <= 6.0 * median_height
            for other_line, other_bbox in markers
        ):
            continue
        line_height = _line_effective_height(line, bbox)
        left_peers = [
            (other_line, other_bbox)
            for other_line, other_bbox in lane.lines
            if other_line.source_index != line.source_index
            and _bbox_center_x(other_bbox) < bbox[0]
            and _formula_detached_seed_vertical_match(
                bbox,
                line_height,
                other_bbox,
                _line_effective_height(other_line, other_bbox),
            )
        ]
        if len(left_peers) < 2 or not any(
            peer.formula_candidate_only or peer.compact_formula_cluster or peer.font_coverage < 0.75
            for peer, _peer_bbox in left_peers
        ):
            continue
        center_y = _bbox_center_y(bbox)
        detached_above = body_interval is None or center_y < body_interval[0]
        detached_below = body_interval is not None and center_y > body_interval[1]
        output.append(
            _FormulaAnchor(
                line=line,
                bbox=bbox,
                detached_below_body=detached_below,
                detached_above_body=detached_above,
                repeated_number_band=True,
            )
        )
    return output


def _find_formula_spatial_anchors(
    lane: _TextLane,
    median_height: float,
    dominant_body_font: tuple[str, int] | None = None,
) -> list[_FormulaAnchor]:
    """查找栏带右缘短块或带编号后缀的非正文字体公式锚点。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_interval = _formula_lane_body_interval(lane, median_height)
    repeated_anchors = _find_repeated_formula_number_anchors(
        lane,
        median_height,
        body_interval,
    )
    if body_interval is None:
        return _deduplicate_formula_anchors(repeated_anchors, median_height)
    body_top, body_bottom = body_interval
    anchors: list[_FormulaAnchor] = list(repeated_anchors)
    repeated_sources = {anchor.line.source_index for anchor in repeated_anchors}
    for line, bbox in lane.lines:
        if line.source_index in repeated_sources:
            continue
        line_height = _line_effective_height(line, bbox)
        line_width = bbox[2] - bbox[0]
        is_short_right_anchor = line_width <= max(4.0 * line_height, 0.12 * lane_width)
        has_formula_number_suffix = _split_trailing_formula_number(line.text) is not None
        is_wide_numbered_anchor = (
            has_formula_number_suffix
            and line_width <= 0.75 * lane_width
            and dominant_body_font is not None
            and (line.font_signature != dominant_body_font or line.font_coverage < 0.75)
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
            or (dominant_body_font is not None and other_line.font_signature != dominant_body_font)
            for other_line in same_row_fragments
            if other_line.source_index != line.source_index
        ):
            # 一条粗行被多个大空格拆成密集词组时更像普通排版行，不能把末词当作公式编号锚点。
            continue
        if _split_visual_row_has_prose_continuation(
            lane,
            line,
            same_row_fragments,
            median_height,
        ):
            continue
        if abs(lane.right - bbox[2]) > max(3.0, 0.02 * lane_width):
            continue
        center_y = _bbox_center_y(bbox)
        detached_below_body = body_bottom < center_y <= body_bottom + 6.0 * median_height
        detached_above_body = body_top - 6.0 * median_height <= center_y < body_top
        if not body_top <= center_y <= body_bottom and not detached_below_body and not detached_above_body:
            continue
        left_peers = [
            (other_line, other_bbox)
            for other_line, other_bbox in lane.lines
            if other_line.source_index != line.source_index
            and other_bbox[2] - other_bbox[0] <= 0.75 * lane_width
            and _bbox_center_x(other_bbox) < bbox[0]
            and (
                _formula_detached_seed_vertical_match(
                    bbox,
                    line_height,
                    other_bbox,
                    _line_effective_height(other_line, other_bbox),
                )
                if detached_below_body or detached_above_body
                else _formula_seed_vertical_match(
                    bbox,
                    line_height,
                    other_bbox,
                    _line_effective_height(other_line, other_bbox),
                )
            )
        ]
        if is_short_right_anchor and not has_formula_number_suffix:
            # 非编号短锚点必须与左侧主体真正分离；分母字符与正文横向重叠时不能扩张成公式。
            if any(_bbox_axis_overlap_ratio(bbox, other_bbox, axis="x") >= 0.5 for _other_line, other_bbox in left_peers):
                continue
            minimum_gap = max(0.5, 0.1 * line_height)
            if not any(bbox[0] - other_bbox[2] >= minimum_gap for _other_line, other_bbox in left_peers):
                continue
        if left_peers:
            anchors.append(
                _FormulaAnchor(
                    line=line,
                    bbox=bbox,
                    detached_below_body=detached_below_body,
                    detached_above_body=detached_above_body,
                )
            )
    return _deduplicate_formula_anchors(anchors, median_height)


def _split_visual_row_has_prose_continuation(
    lane: _TextLane,
    anchor_line: _LineItem,
    same_row_fragments: list[_LineItem],
    median_height: float,
) -> bool:
    """识别覆盖大部分栏宽且紧接下一正文行的同行拆分文本。"""

    if len(same_row_fragments) < 3 or anchor_line.visual_row_id is None:
        return False
    fragment_sources = {line.source_index for line in same_row_fragments}
    fragment_geometry = [(line, bbox) for line, bbox in lane.lines if line.source_index in fragment_sources]
    if len(fragment_geometry) < 3:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    row_bbox = _bbox_union_many([bbox for _line, bbox in fragment_geometry])
    if row_bbox[2] - row_bbox[0] < 0.75 * lane_width:
        return False
    row_center = statistics.median(_bbox_center_y(bbox) for _line, bbox in fragment_geometry)
    if any(
        abs(_bbox_center_y(bbox) - row_center) > 0.25 * median_height
        or not 0.7 * median_height <= _line_effective_height(line, bbox) <= 1.3 * median_height
        for line, bbox in fragment_geometry
    ):
        return False
    following_rows = [
        (line, bbox)
        for line, bbox in lane.lines
        if line.source_index not in fragment_sources and _bbox_center_y(bbox) > row_center + 0.5 * median_height
    ]
    if not following_rows:
        return False
    following_line, following_bbox = min(
        following_rows,
        key=lambda item: (_bbox_center_y(item[1]), item[1][0]),
    )
    following_height = _line_effective_height(
        following_line,
        following_bbox,
    )
    return (
        following_bbox[2] - following_bbox[0] >= 0.6 * lane_width
        and abs(following_bbox[0] - lane.left) <= 0.75 * median_height
        and 0.75 * median_height <= following_height <= 1.3 * median_height
        and following_bbox[1] - row_bbox[3] <= 1.25 * median_height
    )


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
        (item for item in lane.lines if item[1][2] - item[1][0] >= max(4.0 * _line_effective_height(*item), 0.35 * lane_width)),
        key=lambda item: (item[1][1], item[1][0]),
    )
    if len(body_lines) < 3:
        return None
    dense_lines: list[tuple[_LineItem, BBox]] = []
    for index, item in enumerate(body_lines):
        has_close_previous = index > 0 and item[1][1] - body_lines[index - 1][1][3] <= 1.5 * median_height
        has_close_next = index + 1 < len(body_lines) and body_lines[index + 1][1][1] - item[1][3] <= 1.5 * median_height
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
        and not _is_formula_title_barrier(
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
            minimum_font_coverage=0.5 if anchor.repeated_number_band else 0.75,
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
            if anchor.detached_below_body or anchor.detached_above_body
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
    """识别具有稳定正文排版的行，阻止公式分量吸收正文尾行。"""

    if dominant_body_font is None:
        return False
    line, bbox = candidate
    lane_width = max(0.1, lane.right - lane.left)
    line_height = _line_effective_height(line, bbox)
    return (
        line.font_signature == dominant_body_font
        and line.font_coverage >= 0.75
        and bbox[2] - bbox[0] >= 0.3 * lane_width
        and 0.8 * median_height <= line_height <= 1.25 * median_height
    )


def _is_formula_title_barrier(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> bool:
    """用左对齐、字号突变和字体变化隔离公式下方的章节标题。"""

    if dominant_body_font is None:
        return False
    line, bbox = candidate
    lane_width = max(0.1, lane.right - lane.left)
    line_height = _line_effective_height(line, bbox)
    return (
        line.font_signature is not None
        and line.font_signature != dominant_body_font
        and line.font_coverage >= 0.75
        and 1.1 * median_height <= line_height <= 1.6 * median_height
        and bbox[2] - bbox[0] >= 0.25 * lane_width
        and abs(bbox[0] - lane.left) <= median_height
    )


def _is_formula_body_prefix(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    anchor: tuple[_LineItem, BBox],
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
    *,
    minimum_font_coverage: float = 0.75,
) -> bool:
    """识别锚点上方左对齐的常规正文行，防止公式空间扩张越界认领。"""

    if dominant_body_font is None:
        return False
    line, bbox = candidate
    anchor_line, anchor_bbox = anchor
    if line.formula_candidate_only:
        return False
    line_height = _line_effective_height(line, bbox)
    anchor_height = _line_effective_height(anchor_line, anchor_bbox)
    if _bbox_center_y(bbox) > _bbox_center_y(anchor_bbox) - 0.2 * max(line_height, anchor_height):
        return False
    if abs(bbox[0] - lane.left) > max(3.0, 0.75 * median_height):
        return False
    return (
        line.font_signature == dominant_body_font
        and line.font_coverage >= minimum_font_coverage
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
    body_bboxes = [bbox for line, bbox in members if line.source_index != anchor_line.source_index]
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
    """把公式空间分量按视觉行聚类，将编号序列化为 tag 并后置其他 sidecar。"""

    anchor_line = next(
        (line for line, _bbox in members if line.source_index == anchor_source_index),
        None,
    )
    anchor_formula_number_parts = _split_trailing_formula_number(anchor_line.text) if anchor_line is not None else None
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
            rows[row_index] = [member for member in row if member[0].source_index != anchor_source_index]
            trailing_sidecar_content = anchor_member[0].text.strip()
        break

    row_contents = [_join_formula_visual_row(row, page_size) for row in rows if row]
    if trailing_sidecar_content is not None:
        row_contents.append(trailing_sidecar_content)
    content = _sanitize_pdf_control_text("\n".join(filter(None, row_contents)), preserve_newlines=True)
    if anchor_formula_number_parts is not None:
        _anchor_prefix, tag_content = anchor_formula_number_parts
        stripped_content = content.rstrip()
        if stripped_content.endswith(tag_content):
            formula_content = stripped_content[: -len(tag_content)].rstrip()
            tagged_content = build_tagged_formula_content(formula_content, tag_content)
            if tagged_content is not None:
                content = tagged_content
    if not content.strip():
        return None
    return {
        "type": "equation",
        "bbox": _bbox_union_many([line.bbox for line, _bbox in members]),
        "angle": angle,
        "content": content,
    }
