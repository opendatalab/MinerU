# Copyright (c) Opendatalab. All rights reserved.

"""将剩余文本行构造成正文和页脚注块。"""

from __future__ import annotations

import statistics
from typing import Any, Sequence


from mineru.backend.utils.char_utils import (
    is_hyphen_at_line_end,
    resolve_text_line_boundary,
)
from mineru.types import BBox
from mineru.utils.language import detect_lang

from .models import (
    _AxisLine,
    _LineItem,
    _LocalAxisLine,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_y,
    _bbox_union_many,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from .native_text import _normalize_native_run_text
from .line_layout import (
    _connection_crosses_table,
    _effective_text_row_gap,
    _estimate_lane_gap,
    _horizontal_rule_separates_rows,
    _infer_text_lanes,
    _line_effective_height,
    _should_connect_semantic_rows,
    _should_connect_text_rows,
)
def _build_hanging_indent_group_map(
    lane: _TextLane,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> dict[int, int]:
    """仅按重复的左突首行和稳定续行缩进识别悬挂缩进条目。"""

    if lane.is_span or len(lane.lines) < 4:
        return {}
    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1
                          ][1], item[1][0], item[0].source_index),
    )
    if len(rows) < 4:
        return {}
    median_height = statistics.median(
        _line_effective_height(line, bbox) for line, bbox in rows
    )
    start_tolerance = max(5.0, 0.65 * median_height)
    minimum_indent = max(7.0, 0.8 * median_height)
    continuation_tolerance = max(4.0, 0.55 * median_height)

    def rows_are_adjacent(
        previous: tuple[_LineItem, BBox],
        current: tuple[_LineItem, BBox],
    ) -> bool:
        """检查相邻行的净空和几何障碍是否允许组成同一缩进序列。"""

        effective_gap = _effective_text_row_gap(previous, current)
        if not -0.6 * median_height <= effective_gap <= 1.3 * median_height:
            return False
        if _connection_crosses_table(
            previous[0].bbox,
            current[0].bbox,
            table_bboxes,
        ):
            return False
        return not _horizontal_rule_separates_rows(
            previous[1],
            current[1],
            lane,
            axis_lines,
        )

    def consume_entry(
        start_index: int,
        start_left: float,
        expected_continuation_left: float | None,
        *,
        require_next_start: bool,
    ) -> tuple[int, float] | None:
        """消费一个左突首行及其续行，并返回下一条首行位置。"""

        if (
            start_index > 0
            and is_hyphen_at_line_end(rows[start_index - 1][0].text)
            and rows_are_adjacent(rows[start_index - 1], rows[start_index])
        ):
            # 排版断词后的下一物理行属于前文，不能被缩进几何误当成新条目首行。
            return None
        continuation_index = start_index + 1
        if continuation_index >= len(rows):
            return None
        first_continuation = rows[continuation_index]
        if not rows_are_adjacent(rows[start_index], first_continuation):
            return None
        continuation_left = first_continuation[1][0]
        if continuation_left < start_left + minimum_indent:
            return None
        if (
            expected_continuation_left is not None
            and abs(continuation_left - expected_continuation_left) > continuation_tolerance
        ):
            return None

        continuation_index += 1
        while continuation_index < len(rows):
            previous = rows[continuation_index - 1]
            current = rows[continuation_index]
            current_left = current[1][0]
            if not rows_are_adjacent(previous, current):
                break
            if current_left < start_left + minimum_indent:
                break
            if abs(current_left - continuation_left) > continuation_tolerance:
                break
            continuation_index += 1

        if not require_next_start:
            return continuation_index, continuation_left
        if continuation_index >= len(rows):
            return None
        if not rows_are_adjacent(rows[continuation_index - 1], rows[continuation_index]):
            return None
        if abs(rows[continuation_index][1][0] - start_left) > start_tolerance:
            return None
        return continuation_index, continuation_left

    group_map: dict[int, int] = {}
    group_index = 0
    row_index = 0
    while row_index < len(rows) - 3:
        start_left = rows[row_index][1][0]
        first_entry = consume_entry(
            row_index,
            start_left,
            None,
            require_next_start=True,
        )
        if first_entry is None:
            row_index += 1
            continue

        next_start_index, continuation_left = first_entry
        start_indices = [row_index, next_start_index]
        current_start_index = next_start_index
        while True:
            next_entry = consume_entry(
                current_start_index,
                start_left,
                continuation_left,
                require_next_start=True,
            )
            if next_entry is None:
                break
            next_start_index, _continuation_left = next_entry
            start_indices.append(next_start_index)
            current_start_index = next_start_index

        final_entry = consume_entry(
            start_indices[-1],
            start_left,
            continuation_left,
            require_next_start=False,
        )
        if final_entry is None:
            row_index += 1
            continue
        end_index, _continuation_left = final_entry

        entry_ranges = [
            (start, end)
            for start, end in zip(
                start_indices,
                [*start_indices[1:], end_index],
                strict=True,
            )
        ]
        for start, end in entry_ranges:
            for line, _bbox in rows[start:end]:
                group_map[line.source_index] = group_index
            group_index += 1
        row_index = end_index

    return group_map


def _build_grouped_page_footnote_blocks(
    lines: list[_LineItem],
    source_groups: Sequence[set[int]],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], set[int]]:
    """按分隔线组切分几何脚注条目，并返回已消费的来源编号。"""

    footnote_by_source = {
        line.source_index: line
        for line in lines
        if line.semantic_type == "page_footnote"
    }
    blocks: list[dict[str, Any]] = []
    consumed_source_indices: set[int] = set()
    for source_group in source_groups:
        group_lines = [
            footnote_by_source[source_index]
            for source_index in source_group
            if source_index in footnote_by_source
        ]
        for angle in sorted({line.angle for line in group_lines}):
            directional_lines = [line for line in group_lines if line.angle == angle]
            tight_bboxes = _tight_page_footnote_bboxes(
                directional_lines,
                page_size,
            )
            for entry_lines in _split_page_footnote_entries(
                directional_lines,
                page_size,
            ):
                ordered_lines = sorted(
                    entry_lines,
                    key=lambda line: (
                        _rotate_bbox_to_upright(line.bbox, page_size, angle)[1],
                        _rotate_bbox_to_upright(line.bbox, page_size, angle)[0],
                        line.source_index,
                    ),
                )
                content = _merge_text_line_content([line.text for line in ordered_lines])
                if not content:
                    continue
                visual_row_ids = {
                    line.visual_row_id
                    for line in ordered_lines
                    if line.visual_row_id is not None
                }
                single_run_row_id = (
                    ordered_lines[0].visual_row_id
                    if len(ordered_lines) == 1
                    and ordered_lines[0].split_from_row
                    and ordered_lines[0].visual_row_id is not None
                    else None
                )
                blocks.append(
                    {
                        "type": "page_footnote",
                        "bbox": _bbox_union_many(
                            [tight_bboxes[line.source_index] for line in ordered_lines]
                        ),
                        "angle": angle,
                        "content": content,
                        "_visual_row_ids": visual_row_ids,
                        "_single_run_row_id": single_run_row_id,
                    }
                )
                consumed_source_indices.update(
                    line.source_index for line in ordered_lines
                )
    return blocks, consumed_source_indices


def _split_page_footnote_entries(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[list[_LineItem]]:
    """按首行或续行缩进模式，把同一分隔线下的脚注行切成条目。"""

    if not lines:
        return []
    angle = lines[0].angle
    line_geometry = [
        (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
        for line in lines
    ]
    line_geometry.sort(
        key=lambda item: (item[1][1], item[1][0], item[0].source_index)
    )
    if len(line_geometry) == 1:
        return [[line_geometry[0][0]]]

    effective_heights = [
        _line_effective_height(line, bbox)
        for line, bbox in line_geometry
    ]
    median_height = statistics.median(effective_heights)
    glyph_widths = [
        line.median_glyph_width
        for line, _bbox in line_geometry
        if line.median_glyph_width is not None
    ]
    median_glyph_width = (
        statistics.median(glyph_widths)
        if glyph_widths
        else 0.5 * median_height
    )
    indent_threshold = max(1.5 * median_height, 2.0 * median_glyph_width)
    base_left = min(bbox[0] for _line, bbox in line_geometry)
    maximum_row_width = max(bbox[2] - bbox[0] for _line, bbox in line_geometry)
    first_line_is_indented = line_geometry[0][1][0] - base_left > indent_threshold

    entries: list[list[_LineItem]] = [[line_geometry[0][0]]]
    for previous, current in zip(line_geometry, line_geometry[1:]):
        previous_width = previous[1][2] - previous[1][0]
        previous_is_near_full = previous_width >= 0.8 * maximum_row_width
        current_is_indented = current[1][0] - base_left > indent_threshold
        # 满行后必然续接；其余行按首页观测到的首行/续行缩进模式决定边界。
        continues_previous = previous_is_near_full or (
            not current_is_indented
            if first_line_is_indented
            else current_is_indented
        )
        if continues_previous:
            entries[-1].append(current[0])
        else:
            entries.append([current[0]])
    return entries


def _tight_page_footnote_bboxes(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> dict[int, BBox]:
    """以有效行高和相邻行距收紧脚注框，避免异常字体框跨入相邻条目。"""

    if not lines:
        return {}
    angle = lines[0].angle
    line_geometry = [
        (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
        for line in lines
    ]
    line_geometry.sort(
        key=lambda item: (item[1][1], item[1][0], item[0].source_index)
    )
    median_height = statistics.median(
        _line_effective_height(line, bbox)
        for line, bbox in line_geometry
    )
    row_pitches = [
        current[1][1] - previous[1][1]
        for previous, current in zip(line_geometry, line_geometry[1:])
        if current[1][1] - previous[1][1] > 0.25 * median_height
    ]
    median_pitch = statistics.median(row_pitches) if row_pitches else None

    output: dict[int, BBox] = {}
    for line, bbox in line_geometry:
        tight_height = min(
            bbox[3] - bbox[1],
            _line_effective_height(line, bbox),
        )
        if median_pitch is not None:
            tight_height = min(tight_height, 0.9 * median_pitch)
        tight_height = max(0.1, tight_height)
        center_y = _bbox_center_y(bbox)
        tight_local_bbox = (
            bbox[0],
            center_y - 0.5 * tight_height,
            bbox[2],
            center_y + 0.5 * tight_height,
        )
        output[line.source_index] = _rotate_bbox_from_upright(
            tight_local_bbox,
            page_size,
            angle,
        )
    return output


def _build_text_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    drawing_lines: list[_AxisLine] | None = None,
    *,
    page_footnote_groups: Sequence[set[int]] | None = None,
) -> list[dict[str, Any]]:
    """先构建分组脚注，再按类型屏障、栏带和自然段边界聚合其余文本。"""

    blocks, grouped_footnote_indices = _build_grouped_page_footnote_blocks(
        lines,
        page_footnote_groups or [],
        page_size,
    )
    lines = [
        line
        for line in lines
        if line.source_index not in grouped_footnote_indices
    ]
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if not line_geometry:
            continue
        line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        local_axis_lines = _transform_axis_lines(drawing_lines or [], page_size, angle)

        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            if not lane.lines:
                continue
            regular_gap, gap_mad = _estimate_lane_gap(lane)
            hanging_indent_groups = _build_hanging_indent_group_map(
                lane,
                table_bboxes,
                local_axis_lines,
            )
            component: list[tuple[_LineItem, BBox]] = [lane.lines[0]]
            components: list[list[tuple[_LineItem, BBox]]] = []
            for previous, current in zip(lane.lines, lane.lines[1:]):
                previous_type = previous[0].semantic_type
                current_type = current[0].semantic_type
                if previous_type != current_type:
                    should_connect = False
                elif previous_type is not None:
                    should_connect = _should_connect_semantic_rows(
                        previous,
                        current,
                        lane,
                        regular_gap,
                        table_bboxes,
                        local_axis_lines,
                    )
                else:
                    previous_group = hanging_indent_groups.get(previous[0].source_index)
                    current_group = hanging_indent_groups.get(current[0].source_index)
                    if is_hyphen_at_line_end(previous[0].text):
                        # 断词续行优先于悬挂缩进分组，但仍复用正文连接中的距离和障碍限制。
                        should_connect = _should_connect_text_rows(
                            previous,
                            current,
                            lane,
                            regular_gap,
                            gap_mad,
                            table_bboxes,
                            local_axis_lines,
                        )
                    elif previous_group is not None or current_group is not None:
                        should_connect = previous_group is not None and previous_group == current_group
                    else:
                        should_connect = _should_connect_text_rows(
                            previous,
                            current,
                            lane,
                            regular_gap,
                            gap_mad,
                            table_bboxes,
                            local_axis_lines,
                        )
                if should_connect:
                    component.append(current)
                else:
                    components.append(component)
                    component = [current]
            components.append(component)

            for component_geometry in components:
                component_lines = [item[0] for item in component_geometry]
                content = _merge_text_line_content([line.text for line in component_lines])
                if not content:
                    continue
                visual_row_ids = {
                    line.visual_row_id for line in component_lines if line.visual_row_id is not None
                }
                single_run_row_id = (
                    component_lines[0].visual_row_id
                    if len(component_lines) == 1
                    and component_lines[0].split_from_row
                    and component_lines[0].visual_row_id is not None
                    else None
                )
                blocks.append(
                    {
                        "type": component_lines[0].semantic_type or "text",
                        "bbox": _bbox_union_many([line.bbox for line in component_lines]),
                        "angle": angle,
                        "content": content,
                        "_visual_row_ids": visual_row_ids,
                        "_single_run_row_id": single_run_row_id,
                        "_local_line_bboxes": [bbox for _line, bbox in component_geometry],
                        "_line_heights": [
                            _line_effective_height(line, bbox)
                            for line, bbox in component_geometry
                        ],
                        "_lane_interval": (lane.left, lane.right),
                        "_lane_is_span": lane.is_span,
                    }
                )
    return _merge_spatial_text_components(blocks, page_size)


def _merge_spatial_text_components(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """按短首行、紧邻续行和双栏递减尾行二次连接被栏带拆开的正文块。"""

    parents = list(range(len(blocks)))

    def find(index: int) -> int:
        """查找正文组件所属合并组的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并两个已经通过空间连续性校验的正文组件。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for angle in sorted(
        {
            int(block.get("angle", 0) or 0) % 360
            for block in blocks
            if block.get("type") == "text" and block.get("_local_line_bboxes")
        }
    ):
        candidate_indices = [
            index
            for index, block in enumerate(blocks)
            if block.get("type") == "text"
            and int(block.get("angle", 0) or 0) % 360 == angle
            and isinstance(block.get("_local_line_bboxes"), list)
            and block["_local_line_bboxes"]
        ]
        if len(candidate_indices) < 2:
            continue
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        all_heights = [
            float(height)
            for index in candidate_indices
            for height in blocks[index].get("_line_heights", [])
            if isinstance(height, (int, float)) and height > 0
        ]
        median_height = statistics.median(all_heights) if all_heights else 1.0
        section_starts = {
            index
            for index in candidate_indices
            if _block_starts_with_short_wide_rows(
                blocks[index],
                local_page_width,
            )
        }

        opener_pairs = _find_short_opener_pairs(
            blocks,
            candidate_indices,
            local_page_width,
            median_height,
        )
        for opener_index, body_index in opener_pairs:
            section_starts.add(opener_index)
            union(opener_index, body_index)

        for current_index in sorted(
            candidate_indices,
            key=lambda index: _text_component_sort_key(blocks[index]),
        ):
            section_roots = {find(index) for index in section_starts}
            if find(current_index) not in section_roots:
                continue
            next_index = _nearest_following_text_component(
                blocks,
                current_index,
                candidate_indices,
                maximum_gap=0.75 * median_height,
                section_starts=section_starts,
            )
            if next_index is not None:
                union(current_index, next_index)

        for current_index in candidate_indices:
            if not _has_parallel_text_component(
                blocks,
                current_index,
                candidate_indices,
            ):
                continue
            next_index = _nearest_tapered_tail_component(
                blocks,
                current_index,
                candidate_indices,
                median_height,
                section_starts,
            )
            if next_index is not None:
                union(current_index, next_index)

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(blocks)):
        grouped_indices.setdefault(find(index), []).append(index)

    output: list[dict[str, Any]] = []
    for indices in grouped_indices.values():
        if len(indices) == 1:
            output.append(blocks[indices[0]])
            continue
        ordered_indices = sorted(
            indices,
            key=lambda index: _text_component_sort_key(blocks[index]),
        )
        merged = dict(blocks[ordered_indices[0]])
        merged["bbox"] = _bbox_union_many([blocks[index]["bbox"] for index in ordered_indices])
        merged["content"] = _merge_text_line_content(
            [str(blocks[index].get("content", "")) for index in ordered_indices]
        )
        merged["_visual_row_ids"] = set().union(
            *[
                block_ids
                for index in ordered_indices
                if isinstance(
                    (block_ids := blocks[index].get("_visual_row_ids")),
                    set,
                )
            ]
        )
        merged["_single_run_row_id"] = None
        merged["_local_line_bboxes"] = [
            bbox
            for index in ordered_indices
            for bbox in blocks[index].get("_local_line_bboxes", [])
        ]
        merged["_line_heights"] = [
            height
            for index in ordered_indices
            for height in blocks[index].get("_line_heights", [])
        ]
        output.append(merged)
    return output


def _component_lane_interval(
    block: dict[str, Any],
) -> tuple[float, float] | None:
    """读取普通文本组件所属的有效非跨栏栏带区间。"""

    if block.get("_lane_is_span") is not False:
        return None
    interval = block.get("_lane_interval")
    if (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or not all(isinstance(value, (int, float)) for value in interval)
    ):
        return None
    left, right = float(interval[0]), float(interval[1])
    return (left, right) if right > left else None


def _component_reference_width(
    block: dict[str, Any],
    local_page_width: float,
) -> float:
    """优先返回组件的局部栏宽，缺少可靠栏带时回退页面宽度。"""

    interval = _component_lane_interval(block)
    return interval[1] - interval[0] if interval is not None else local_page_width


def _compatible_component_lane_width(
    first_block: dict[str, Any],
    second_block: dict[str, Any],
    local_page_width: float,
    median_height: float,
) -> float:
    """同一局部栏带的两个组件使用栏宽，否则继续使用页面宽度。"""

    first_interval = _component_lane_interval(first_block)
    second_interval = _component_lane_interval(second_block)
    if first_interval is None or second_interval is None:
        return local_page_width
    tolerance = 0.75 * median_height
    if (
        abs(first_interval[0] - second_interval[0]) > tolerance
        or abs(first_interval[1] - second_interval[1]) > tolerance
    ):
        return local_page_width
    return min(
        first_interval[1] - first_interval[0],
        second_interval[1] - second_interval[0],
    )


def _block_starts_with_short_wide_rows(
    block: dict[str, Any],
    local_page_width: float,
) -> bool:
    """判断组件内部是否以短首行和紧邻宽正文形成明确分组起点。"""

    line_bboxes = block.get("_local_line_bboxes")
    if not isinstance(line_bboxes, list) or len(line_bboxes) < 2:
        return False
    first_bbox, second_bbox = line_bboxes[:2]
    first_width = first_bbox[2] - first_bbox[0]
    second_width = second_bbox[2] - second_bbox[0]
    reference_width = _component_reference_width(block, local_page_width)
    return (
        first_width <= 0.35 * reference_width
        and second_width >= max(0.25 * reference_width, 1.8 * first_width)
    )


def _find_short_opener_pairs(
    blocks: list[dict[str, Any]],
    candidate_indices: list[int],
    local_page_width: float,
    median_height: float,
) -> list[tuple[int, int]]:
    """查找跨栏拆开的短首行与紧邻满宽正文组件。"""

    output: list[tuple[int, int]] = []
    for opener_index in candidate_indices:
        opener_bboxes = blocks[opener_index]["_local_line_bboxes"]
        if len(opener_bboxes) != 1:
            continue
        opener_bbox = opener_bboxes[0]
        opener_width = opener_bbox[2] - opener_bbox[0]
        matches: list[tuple[float, int]] = []
        for body_index in candidate_indices:
            if body_index == opener_index:
                continue
            body_bbox = blocks[body_index]["_local_line_bboxes"][0]
            gap = body_bbox[1] - opener_bbox[3]
            body_width = body_bbox[2] - body_bbox[0]
            reference_width = _compatible_component_lane_width(
                blocks[opener_index],
                blocks[body_index],
                local_page_width,
                median_height,
            )
            if (
                not 0.0 <= gap <= 0.75 * median_height
                or abs(body_bbox[0] - opener_bbox[0]) > 0.75 * median_height
                or opener_width > 0.35 * reference_width
                or body_width < 0.6 * reference_width
            ):
                continue
            matches.append((gap, body_index))
        if matches:
            output.append((opener_index, min(matches)[1]))
    return output


def _nearest_following_text_component(
    blocks: list[dict[str, Any]],
    current_index: int,
    candidate_indices: list[int],
    *,
    maximum_gap: float,
    section_starts: set[int],
) -> int | None:
    """返回同一水平流中紧邻且不是新分组起点的下一正文组件。"""

    current_bbox = blocks[current_index]["_local_line_bboxes"][-1]
    matches: list[tuple[float, float, int]] = []
    for candidate_index in candidate_indices:
        if candidate_index == current_index or candidate_index in section_starts:
            continue
        candidate_bbox = blocks[candidate_index]["_local_line_bboxes"][0]
        gap = candidate_bbox[1] - current_bbox[3]
        if not -0.25 * maximum_gap <= gap <= maximum_gap:
            continue
        overlap = _bbox_axis_overlap_ratio(current_bbox, candidate_bbox, axis="x")
        left_gap = abs(candidate_bbox[0] - current_bbox[0])
        if overlap < 0.5 and left_gap > 3.0 * maximum_gap:
            continue
        matches.append((max(0.0, gap), left_gap, candidate_index))
    return min(matches)[2] if matches else None


def _has_parallel_text_component(
    blocks: list[dict[str, Any]],
    current_index: int,
    candidate_indices: list[int],
) -> bool:
    """检查当前组件同一纵向带内是否存在水平分离的并栏组件。"""

    current_bbox = blocks[current_index]["bbox"]
    for candidate_index in candidate_indices:
        if candidate_index == current_index:
            continue
        candidate_bbox = blocks[candidate_index]["bbox"]
        vertical_overlap = max(
            0.0,
            min(current_bbox[3], candidate_bbox[3])
            - max(current_bbox[1], candidate_bbox[1]),
        )
        horizontally_separate = (
            candidate_bbox[0] >= current_bbox[2]
            or current_bbox[0] >= candidate_bbox[2]
        )
        if vertical_overlap > 0 and horizontally_separate:
            return True
    return False


def _nearest_tapered_tail_component(
    blocks: list[dict[str, Any]],
    current_index: int,
    candidate_indices: list[int],
    median_height: float,
    section_starts: set[int],
) -> int | None:
    """查找左对齐、行宽递减且行距略大的信息组尾行组件。"""

    previous_bbox = blocks[current_index]["_local_line_bboxes"][-1]
    previous_width = previous_bbox[2] - previous_bbox[0]
    matches: list[tuple[float, int]] = []
    for candidate_index in candidate_indices:
        if candidate_index == current_index or candidate_index in section_starts:
            continue
        candidate_bboxes = blocks[candidate_index]["_local_line_bboxes"]
        first_bbox = candidate_bboxes[0]
        last_bbox = candidate_bboxes[-1]
        gap = first_bbox[1] - previous_bbox[3]
        first_width = first_bbox[2] - first_bbox[0]
        last_width = last_bbox[2] - last_bbox[0]
        if (
            not 0.75 * median_height < gap <= 1.5 * median_height
            or len(candidate_bboxes) > 2
            or abs(first_bbox[0] - previous_bbox[0]) > 0.75 * median_height
            or first_width > 0.85 * previous_width
            or last_width > first_width + 0.5 * median_height
        ):
            continue
        matches.append((gap, candidate_index))
    return min(matches)[1] if matches else None


def _text_component_sort_key(block: dict[str, Any]) -> tuple[float, float]:
    """返回正文组件按局部首行位置排序的稳定键。"""

    first_bbox = block["_local_line_bboxes"][0]
    return first_bbox[1], first_bbox[0]


def _merge_text_line_content(line_texts: Sequence[str]) -> str:
    """按 Hybrid 语言与行末连字规则折叠普通文本行。"""

    normalized_lines = [_normalize_native_run_text(str(text or "")) for text in line_texts]
    normalized_lines = [text for text in normalized_lines if text]
    if not normalized_lines:
        return ""
    try:
        block_language = detect_lang("".join(normalized_lines))
    except Exception:
        block_language = ""
    content_parts = [normalized_lines[0]]
    for current_line in normalized_lines[1:]:
        processed_previous, separator = resolve_text_line_boundary(
            content_parts[-1],
            block_language=block_language,
            next_starts_with_lowercase=current_line[0].islower(),
        )
        content_parts[-1] = processed_previous
        content_parts.extend([separator, current_line])
    return "".join(content_parts).strip()
