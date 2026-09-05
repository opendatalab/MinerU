# Copyright (c) Opendatalab. All rights reserved.
"""按来源行和标记组装页面脚注。"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from .....types import BBox
from ..geometry import (
    _bbox_center_y,
    _bbox_union_many,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
)
from ..line_layout import (
    _effective_text_row_gap,
    _line_effective_height,
    _lines_tight_output_bbox,
    _title_fonts_compatible,
)
from ..models import _LineItem
from .common import _merge_text_line_content


def _build_grouped_page_footnote_blocks(
    lines: list[_LineItem],
    source_groups: Sequence[set[int]],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], set[int]]:
    """按分隔线组切分几何脚注条目，并返回已消费的来源编号。"""

    footnote_by_source = {line.source_index: line for line in lines if line.semantic_type == "page_footnote"}
    blocks: list[dict[str, Any]] = []
    consumed_source_indices: set[int] = set()
    for source_group in source_groups:
        group_lines = [footnote_by_source[source_index] for source_index in source_group if source_index in footnote_by_source]
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
                visual_row_ids = {line.visual_row_id for line in ordered_lines if line.visual_row_id is not None}
                single_run_row_id = (
                    ordered_lines[0].visual_row_id
                    if len(ordered_lines) == 1
                    and ordered_lines[0].split_from_row
                    and ordered_lines[0].visual_row_id is not None
                    else None
                )
                block = {
                    "type": "page_footnote",
                    "bbox": _bbox_union_many([tight_bboxes[line.source_index] for line in ordered_lines]),
                    "angle": angle,
                    "content": content,
                    "_visual_row_ids": visual_row_ids,
                    "_single_run_row_id": single_run_row_id,
                    "_inline_math_regions": [region for line in ordered_lines for region in line.inline_math_regions],
                }
                tight_output_bbox = _lines_tight_output_bbox(
                    ordered_lines,
                    page_size,
                )
                if tight_output_bbox is not None:
                    block["_tight_output_bbox"] = tight_output_bbox
                blocks.append(block)
                consumed_source_indices.update(line.source_index for line in ordered_lines)
    return blocks, consumed_source_indices


def _split_page_footnote_entries(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[list[_LineItem]]:
    """按首行或续行缩进模式，把同一分隔线下的脚注行切成条目。"""

    if not lines:
        return []
    angle = lines[0].angle
    line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines]
    line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    if len(line_geometry) == 1:
        return [[line_geometry[0][0]]]

    effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
    median_height = statistics.median(effective_heights)
    glyph_widths = [line.median_glyph_width for line, _bbox in line_geometry if line.median_glyph_width is not None]
    median_glyph_width = statistics.median(glyph_widths) if glyph_widths else 0.5 * median_height
    indent_threshold = max(1.5 * median_height, 2.0 * median_glyph_width)
    base_left = min(bbox[0] for _line, bbox in line_geometry)
    maximum_row_width = max(bbox[2] - bbox[0] for _line, bbox in line_geometry)
    marker_rows = _find_page_footnote_marker_rows(
        line_geometry,
        median_height,
        median_glyph_width,
        base_left,
    )
    if marker_rows:
        return _split_marked_page_footnote_entries(
            line_geometry,
            marker_rows,
            median_height,
            median_glyph_width,
            base_left,
            maximum_row_width,
            indent_threshold,
        )
    return _split_unmarked_page_footnote_entries(
        line_geometry,
        base_left,
        maximum_row_width,
        indent_threshold,
    )


def _find_page_footnote_marker_rows(
    line_geometry: list[tuple[_LineItem, BBox]],
    median_height: float,
    median_glyph_width: float,
    base_left: float,
) -> list[tuple[tuple[_LineItem, BBox], tuple[_LineItem, BBox]]]:
    """用同视觉行的窄左片段和右侧正文识别脚注编号锚点。"""

    by_visual_row: dict[int, list[tuple[_LineItem, BBox]]] = {}
    for item in line_geometry:
        line = item[0]
        if line.visual_row_id is None or not line.split_from_row:
            continue
        by_visual_row.setdefault(line.visual_row_id, []).append(item)

    marker_rows: list[tuple[tuple[_LineItem, BBox], tuple[_LineItem, BBox]]] = []
    marker_width_limit = max(1.5 * median_glyph_width, 0.75 * median_height)
    for members in by_visual_row.values():
        ordered = sorted(members, key=lambda item: item[1][0])
        if len(ordered) < 2:
            continue
        marker, body = ordered[0], ordered[1]
        marker_width = marker[1][2] - marker[1][0]
        body_width = body[1][2] - body[1][0]
        horizontal_gap = body[1][0] - marker[1][2]
        if (
            marker_width > marker_width_limit
            or marker[1][0] - base_left > 0.5 * median_height
            or not 0.0 <= horizontal_gap <= 1.5 * median_height
            or body[1][0] - base_left < max(1.5 * marker_width, 0.75 * median_height)
            or body_width < max(4.0 * marker_width, 4.0 * median_glyph_width)
            or abs(_bbox_center_y(marker[1]) - _bbox_center_y(body[1])) > 0.4 * median_height
        ):
            continue
        marker_rows.append((marker, body))
    marker_rows.sort(key=lambda pair: (min(pair[0][1][1], pair[1][1][1]), pair[0][1][0]))
    if not marker_rows:
        marker_rows = _find_geometric_page_footnote_marker_rows(
            line_geometry,
            median_height,
            median_glyph_width,
            base_left,
        )
    return marker_rows


def _find_geometric_page_footnote_marker_rows(
    line_geometry: list[tuple[_LineItem, BBox]],
    median_height: float,
    median_glyph_width: float,
    base_left: float,
) -> list[tuple[tuple[_LineItem, BBox], tuple[_LineItem, BBox]]]:
    """在 row id 缺失时用同基线窄标记和右侧正文恢复脚注首行。"""

    marker_width_limit = max(
        1.5 * median_glyph_width,
        0.75 * median_height,
    )
    candidates: list[
        tuple[
            float,
            tuple[_LineItem, BBox],
            tuple[_LineItem, BBox],
        ]
    ] = []
    for marker in line_geometry:
        marker_width = marker[1][2] - marker[1][0]
        if marker_width > marker_width_limit or marker[1][0] - base_left > 0.5 * median_height:
            continue
        for body in line_geometry:
            if body is marker:
                continue
            body_width = body[1][2] - body[1][0]
            horizontal_gap = body[1][0] - marker[1][2]
            if (
                not 0.0 <= horizontal_gap <= 1.5 * median_height
                or body[1][0] - base_left
                < max(
                    1.5 * marker_width,
                    0.4 * median_height,
                )
                or body_width
                < max(
                    4.0 * marker_width,
                    4.0 * median_glyph_width,
                )
                or abs(_bbox_center_y(marker[1]) - _bbox_center_y(body[1])) > 0.4 * median_height
                or not _title_fonts_compatible(
                    marker[0],
                    body[0],
                )
            ):
                continue
            candidates.append(
                (
                    horizontal_gap,
                    marker,
                    body,
                )
            )

    output = []
    consumed_sources: set[int] = set()
    for _gap, marker, body in sorted(
        candidates,
        key=lambda item: (
            min(item[1][1][1], item[2][1][1]),
            item[0],
            item[1][0].source_index,
            item[2][0].source_index,
        ),
    ):
        pair_sources = {
            marker[0].source_index,
            body[0].source_index,
        }
        if pair_sources & consumed_sources:
            continue
        output.append((marker, body))
        consumed_sources.update(pair_sources)
    return output


def _split_marked_page_footnote_entries(
    line_geometry: list[tuple[_LineItem, BBox]],
    marker_rows: list[tuple[tuple[_LineItem, BBox], tuple[_LineItem, BBox]]],
    median_height: float,
    median_glyph_width: float,
    base_left: float,
    maximum_row_width: float,
    indent_threshold: float,
) -> list[list[_LineItem]]:
    """把编号、右侧首行和对齐续行聚合，并保留编号区之前的独立脚注。"""

    first_marker_top = min(marker_rows[0][0][1][1], marker_rows[0][1][1][1])
    prefix_geometry = [item for item in line_geometry if item[1][1] < first_marker_top]
    entries = _split_unmarked_page_footnote_entries(
        prefix_geometry,
        base_left,
        maximum_row_width,
        indent_threshold,
    )
    consumed = {line.source_index for entry in entries for line in entry}
    continuation_tolerance = max(0.75 * median_height, 2.0 * median_glyph_width)
    for marker_index, (marker, body) in enumerate(marker_rows):
        next_marker_top = (
            min(marker_rows[marker_index + 1][0][1][1], marker_rows[marker_index + 1][1][1][1])
            if marker_index + 1 < len(marker_rows)
            else float("inf")
        )
        entry_items = [marker, body]
        consumed.update((marker[0].source_index, body[0].source_index))
        previous = body
        for candidate in line_geometry:
            if candidate[0].source_index in consumed:
                continue
            if candidate[1][1] < body[1][1] or candidate[1][1] >= next_marker_top:
                continue
            if abs(candidate[1][0] - body[1][0]) > continuation_tolerance:
                continue
            if _effective_text_row_gap(previous, candidate) > 1.5 * median_height:
                continue
            entry_items.append(candidate)
            consumed.add(candidate[0].source_index)
            previous = candidate
        entry_items.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        entries.append([line for line, _bbox in entry_items])

    leftovers = [item for item in line_geometry if item[0].source_index not in consumed]
    entries.extend(
        _split_unmarked_page_footnote_entries(
            leftovers,
            base_left,
            maximum_row_width,
            indent_threshold,
        )
    )
    entries.sort(
        key=lambda entry: min(
            bbox[1] for line in entry for candidate, bbox in line_geometry if candidate.source_index == line.source_index
        )
    )
    return entries


def _split_unmarked_page_footnote_entries(
    line_geometry: list[tuple[_LineItem, BBox]],
    base_left: float,
    maximum_row_width: float,
    indent_threshold: float,
) -> list[list[_LineItem]]:
    """沿用首行和续行缩进模式切分没有稳定编号锚点的脚注行。"""

    if not line_geometry:
        return []
    first_line_is_indented = line_geometry[0][1][0] - base_left > indent_threshold

    entries: list[list[_LineItem]] = [[line_geometry[0][0]]]
    for previous, current in zip(line_geometry, line_geometry[1:]):
        previous_width = previous[1][2] - previous[1][0]
        previous_is_near_full = previous_width >= 0.8 * maximum_row_width
        current_is_indented = current[1][0] - base_left > indent_threshold
        same_left_compact_continuation = (
            not first_line_is_indented
            and previous_width >= 0.7 * maximum_row_width
            and abs(current[1][0] - previous[1][0]) <= 0.5 * indent_threshold
            and _title_fonts_compatible(previous[0], current[0])
            and _effective_text_row_gap(previous, current) <= indent_threshold
        )
        # 满行后必然续接；其余行按首页观测到的首行/续行缩进模式决定边界。
        continues_previous = (
            previous_is_near_full
            or same_left_compact_continuation
            or (not current_is_indented if first_line_is_indented else current_is_indented)
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
    line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines]
    line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
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


__all__ = [
    "_build_grouped_page_footnote_blocks",
    "_split_page_footnote_entries",
    "_find_page_footnote_marker_rows",
    "_find_geometric_page_footnote_marker_rows",
    "_split_marked_page_footnote_entries",
    "_split_unmarked_page_footnote_entries",
    "_tight_page_footnote_bboxes",
]
