# Copyright (c) Opendatalab. All rights reserved.
"""在既有候选全部失败后恢复多行少线表格结构。"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .sparse_common import _LocalRule, _local_rules, cluster_members
from .candidate import GridCellSpec, build_candidate
from .contracts import (
    NativeTableCandidate,
    NativeTableGlyph,
    NativeTableInput,
    NativeTableText,
    NativeTableTextRow,
)
from .geometry import (
    covered_interval_ratio,
    normalize_angle,
    normalize_bbox,
    page_bbox_to_table_local,
    table_local_size,
)

MIN_MULTILINE_RELIABILITY = 0.98


@dataclass(frozen=True, slots=True)
class _LocalRectangle:
    """保存正向表格局部坐标中的矩形证据。"""

    bbox: tuple[float, float, float, float]
    fill_visible: bool
    stroke_visible: bool


@dataclass(frozen=True, slots=True)
class _ColumnHypothesis:
    """保存少线多行候选的一组列轨与来源证据。"""

    evidence: str
    x_tracks: tuple[float, ...]
    physical_boundaries: frozenset[int]
    filled_band_count: int


@dataclass(frozen=True, slots=True)
class _LogicalRow:
    """保存由一条或多条视觉基线组成的逻辑正文行。"""

    visual_indices: tuple[int, ...]
    top: float
    bottom: float


def _local_rectangles(
    table_input: NativeTableInput,
    width: float,
    height: float,
) -> tuple[_LocalRectangle, ...]:
    """把相交矩形转换为局部坐标并裁剪到表格范围。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return ()
    angle = normalize_angle(table_input.angle)
    output: list[_LocalRectangle] = []
    for rectangle in table_input.rectangles:
        if rectangle.segment_count != 5 or not (rectangle.fill_visible or rectangle.stroke_visible):
            continue
        bbox = normalize_bbox(rectangle.bbox)
        if bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(bbox, table_bbox, angle)
        if local_bbox is None:
            continue
        clipped = (
            max(0.0, local_bbox[0]),
            max(0.0, local_bbox[1]),
            min(width, local_bbox[2]),
            min(height, local_bbox[3]),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        output.append(
            _LocalRectangle(
                bbox=clipped,
                fill_visible=rectangle.fill_visible,
                stroke_visible=rectangle.stroke_visible,
            )
        )
    return tuple(output)


def _row_occupancy(
    row: NativeTableTextRow,
    glyph_by_id: dict[int, NativeTableGlyph],
    x_tracks: tuple[float, ...],
) -> set[int]:
    """按字符中心统计一条视觉行占用的叶子列。"""

    occupied: set[int] = set()
    for glyph_id in row.glyph_ids:
        glyph = glyph_by_id[glyph_id]
        center = (glyph.bbox[0] + glyph.bbox[2]) / 2.0
        for col, (left, right) in enumerate(zip(x_tracks, x_tracks[1:])):
            if left <= center <= right:
                occupied.add(col)
                break
    return occupied


def _infer_target_columns(text: NativeTableText) -> int | None:
    """从重复的最大 token 数推断叶子列数。"""

    counts = Counter(len(row.tokens) for row in text.rows if 2 <= len(row.tokens) <= 20)
    candidates = [count for count, occurrences in counts.items() if occurrences >= 2]
    return max(candidates) if candidates else None


def _infer_text_tracks(
    text: NativeTableText,
    width: float,
    target_cols: int,
) -> tuple[float, ...] | None:
    """从完整锚点行的相邻 token 空隙恢复文本列轨。"""

    anchor_rows = [row for row in text.rows if len(row.tokens) == target_cols]
    if len(anchor_rows) < 2:
        return None
    boundaries: list[float] = []
    for col in range(target_cols - 1):
        left_edges = [row.tokens[col].bbox[2] for row in anchor_rows]
        right_edges = [row.tokens[col + 1].bbox[0] for row in anchor_rows]
        global_left = max(left_edges)
        global_right = min(right_edges)
        if global_left < global_right:
            boundary = (global_left + global_right) / 2.0
        else:
            midpoints = [(left + right) / 2.0 for left, right in zip(left_edges, right_edges, strict=True)]
            boundary = float(statistics.median(midpoints))
        if not 0.0 < boundary < width:
            return None
        boundaries.append(boundary)
    tracks = _refine_text_tracks(
        text,
        (0.0, *boundaries, width),
    )
    minimum_width = max(1.0, 0.50 * text.median_glyph_width)
    if any(current - previous < minimum_width for previous, current in zip(tracks, tracks[1:])):
        return None
    return tracks


def _refine_text_tracks(
    text: NativeTableText,
    tracks: tuple[float, ...],
) -> tuple[float, ...]:
    """用全部简单 token 的外缘扩展文本列间空白走廊。"""

    refined = list(tracks)
    for boundary_index in range(1, len(tracks) - 1):
        boundary = refined[boundary_index]
        left_limits: list[float] = []
        right_limits: list[float] = []
        for row in text.rows:
            for token in row.tokens:
                crossed_boundaries = sum(token.bbox[0] < item < token.bbox[2] for item in tracks[1:-1])
                if crossed_boundaries > 1:
                    continue
                center = (token.bbox[0] + token.bbox[2]) / 2.0
                if center < boundary:
                    left_limits.append(token.bbox[2])
                elif center > boundary:
                    right_limits.append(token.bbox[0])
        if not left_limits or not right_limits:
            continue
        lower = max(left_limits)
        upper = min(right_limits)
        if refined[boundary_index - 1] < lower < upper < refined[boundary_index + 1]:
            refined[boundary_index] = (lower + upper) / 2.0
    return tuple(refined)


def _rectangle_tracks(
    rectangles: tuple[_LocalRectangle, ...],
    width: float,
    tolerance: float,
    outer_tolerance: float,
) -> tuple[float, ...]:
    """从重复矩形端点恢复列轨并去除一次性装饰边缘。"""

    edges = [coordinate for rectangle in rectangles for coordinate in (rectangle.bbox[0], rectangle.bbox[2])]
    positions: list[float] = [0.0, width]
    for coordinate, members in cluster_members(edges, tolerance):
        if len(members) < 2:
            continue
        snapped = 0.0 if coordinate <= outer_tolerance else width if width - coordinate <= outer_tolerance else coordinate
        positions.append(snapped)
    tracks = tuple(sorted(set(positions)))
    if any(current <= previous for previous, current in zip(tracks, tracks[1:])):
        return ()
    return tracks


def _filled_band_count(
    rectangles: tuple[_LocalRectangle, ...],
    width: float,
    median_height: float,
    tolerance: float,
) -> int:
    """统计能覆盖多列的重复填充行带数量。"""

    bands: list[tuple[float, float]] = []
    for rectangle in rectangles:
        if not rectangle.fill_visible:
            continue
        left, top, right, bottom = rectangle.bbox
        if bottom - top < 0.75 * median_height or right - left < 0.08 * width:
            continue
        bands.append((top, bottom))
    clustered = cluster_members(
        [(top + bottom) / 2.0 for top, bottom in bands],
        max(tolerance, 0.50 * median_height),
    )
    return len(clustered)


def _build_column_hypothesis(
    text: NativeTableText,
    width: float,
    rules: tuple[_LocalRule, ...],
    rectangles: tuple[_LocalRectangle, ...],
    diagnostics: dict[str, Any] | None,
) -> _ColumnHypothesis | None:
    """融合矩形端点与文本空白选择唯一列轨假设。"""

    target_cols = _infer_target_columns(text)
    if target_cols is None:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "column_count"
        return None
    text_tracks = _infer_text_tracks(text, width, target_cols)
    tolerance = max(1.0, 0.25 * text.median_glyph_height)
    rect_tracks = _rectangle_tracks(
        rectangles,
        width,
        tolerance,
        max(tolerance, 1.50 * text.median_glyph_height),
    )
    physical_boundaries: frozenset[int] = frozenset()
    tracks = text_tracks
    if len(rect_tracks) == target_cols + 1:
        tracks = rect_tracks
        physical_boundaries = frozenset(range(1, target_cols))
    if tracks is None:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "column_tracks"
        return None

    band_count = _filled_band_count(
        rectangles,
        width,
        text.median_glyph_height,
        tolerance,
    )
    internal_full_rules = [
        rule
        for rule in rules
        if rule.orientation == "horizontal"
        and rule.end - rule.start >= 0.90 * width
        and 0.01 * max((row.bbox[3] for row in text.rows), default=0.0)
        < rule.coordinate
        < 0.99 * max((row.bbox[3] for row in text.rows), default=0.0)
    ]
    if band_count >= 2 and physical_boundaries:
        evidence = "filled_record"
    elif internal_full_rules:
        evidence = "rule_band"
    else:
        evidence = "keyed_record"
    if diagnostics is not None:
        diagnostics.update(
            {
                "target_cols": target_cols,
                "x_tracks": list(tracks),
                "physical_boundaries": sorted(physical_boundaries),
                "filled_band_count": band_count,
                "evidence": evidence,
            }
        )
    return _ColumnHypothesis(
        evidence=evidence,
        x_tracks=tracks,
        physical_boundaries=physical_boundaries,
        filled_band_count=band_count,
    )


def _internal_full_rules(
    rules: tuple[_LocalRule, ...],
    width: float,
    height: float,
) -> tuple[_LocalRule, ...]:
    """返回排除上下外框后的长横线。"""

    return tuple(
        rule
        for rule in rules
        if rule.orientation == "horizontal"
        and rule.end - rule.start >= 0.90 * width
        and 0.01 * height < rule.coordinate < 0.99 * height
    )


def _infer_header_boundary(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    rules: tuple[_LocalRule, ...],
    width: float,
    height: float,
    evidence: str,
) -> float | None:
    """用首条正文锚点和最后一条表头长线确定表头底边。"""

    if len(text.rows) < 3:
        return None
    target_cols = len(x_tracks) - 1
    body_hint = next(
        (row for row in text.rows[1:] if len(row.tokens) == target_cols),
        text.rows[1],
    )
    body_center = (body_hint.bbox[1] + body_hint.bbox[3]) / 2.0
    candidates = [rule.coordinate for rule in _internal_full_rules(rules, width, height) if rule.coordinate < body_center]
    if candidates:
        boundary = max(candidates)
    else:
        if evidence in {"filled_record", "keyed_record"}:
            first = text.rows[0]
            second = text.rows[1]
            return ((first.bbox[1] + first.bbox[3]) / 2.0 + (second.bbox[1] + second.bbox[3]) / 2.0) / 2.0
        previous_rows = [row for row in text.rows if row.row_index < body_hint.row_index]
        if not previous_rows:
            return None
        previous = previous_rows[-1]
        boundary = ((previous.bbox[1] + previous.bbox[3]) / 2.0 + body_center) / 2.0
    if not text.rows[0].bbox[3] - 0.5 <= boundary <= text.rows[-1].bbox[1] + 0.5:
        return None
    return boundary


def _choose_key_column(
    occupancies: list[set[int]],
    cols: int,
) -> tuple[int, int]:
    """选择能重复标记逻辑记录起点的最左稳定关键列。"""

    stats: list[tuple[int, int, int]] = []
    for col in range(cols):
        flags = [col in occupancy for occupancy in occupancies]
        runs = 0
        previous = False
        for flag in flags:
            if flag and not previous:
                runs += 1
            previous = flag
        occupied = sum(flags)
        stats.append((col, runs, occupied))
    if stats[0][2] >= math.ceil(0.80 * len(occupancies)):
        return 0, stats[0][1]
    repeated = [item for item in stats if item[1] >= 3]
    if repeated:
        maximum_occupied = max(item[2] for item in stats)
        dense_repeated = [item for item in repeated if item[2] >= 0.60 * maximum_occupied]
        if dense_repeated:
            selected = min(dense_repeated, key=lambda item: item[0])
        else:
            selected = max(
                repeated,
                key=lambda item: (item[1], item[2], -item[0]),
            )
        return selected[0], selected[1]
    if stats[0][2] >= 2:
        return 0, stats[0][1]
    selected = max(stats, key=lambda item: (item[2], -item[0]))
    return selected[0], selected[1]


def _rule_bands(
    header_bottom: float,
    rules: tuple[_LocalRule, ...],
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    """用正文长横线切出有限物理行带。"""

    boundaries = [header_bottom]
    boundaries.extend(
        rule.coordinate for rule in _internal_full_rules(rules, width, height) if rule.coordinate > header_bottom + 0.5
    )
    boundaries.append(height)
    ordered = sorted(set(boundaries))
    return [(top, bottom) for top, bottom in zip(ordered, ordered[1:]) if bottom - top > 0.5]


def _split_rows_by_anchors(
    rows: list[NativeTableTextRow],
    key_col: int,
    occupancies_by_index: dict[int, set[int]],
    *,
    group_short_key_runs: bool,
) -> list[tuple[int, ...]]:
    """按关键列锚点把视觉基线拆成逻辑记录组。"""

    if not rows:
        return []
    anchor_positions = [index for index, row in enumerate(rows) if key_col in occupancies_by_index[row.row_index]]
    if not anchor_positions:
        return []
    if group_short_key_runs:
        runs: list[list[int]] = [[anchor_positions[0]]]
        for position in anchor_positions[1:]:
            if position == runs[-1][-1] + 1:
                runs[-1].append(position)
            else:
                runs.append([position])
        anchors = [run[0] for run in runs]
    else:
        anchors = anchor_positions

    groups: list[list[int]] = [[] for _ in anchors]
    for position, row in enumerate(rows):
        owner = 0
        for index, anchor in enumerate(anchors):
            if anchor <= position:
                owner = index
            else:
                break
        if position < anchors[0]:
            owner = 0
        groups[owner].append(row.row_index)
    return [tuple(group) for group in groups if group]


def _logical_body_rows(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    rules: tuple[_LocalRule, ...],
    header_bottom: float,
    width: float,
    height: float,
    evidence: str,
    diagnostics: dict[str, Any] | None,
) -> tuple[list[_LogicalRow], int, list[set[int]]] | None:
    """结合物理行带和关键列锚点构造正文逻辑行。"""

    glyph_by_id = {glyph.glyph_id: glyph for glyph in text.glyphs}
    body_rows = [row for row in text.rows if (row.bbox[1] + row.bbox[3]) / 2.0 > header_bottom]
    if len(body_rows) < 2:
        return None
    occupancies = [_row_occupancy(row, glyph_by_id, x_tracks) for row in body_rows]
    occupancies_by_index = {row.row_index: occupancy for row, occupancy in zip(body_rows, occupancies, strict=True)}
    body_internal_rules = [
        rule
        for rule in _internal_full_rules(rules, width, height)
        if header_bottom + 0.5 < rule.coordinate < (body_rows[-1].bbox[1] + body_rows[-1].bbox[3]) / 2.0
    ]
    if evidence == "rule_band" and not body_internal_rules:
        seen_first_col = False
        seen_first_col_gap = False
        for occupancy in occupancies:
            if 0 in occupancy:
                if seen_first_col and seen_first_col_gap:
                    if diagnostics is not None:
                        diagnostics["first_rejection_gate"] = "ambiguous_body_rowspan"
                    return None
                seen_first_col = True
            elif seen_first_col and occupancy != {len(x_tracks) - 2}:
                seen_first_col_gap = True
    key_col, key_runs = _choose_key_column(occupancies, len(x_tracks) - 1)

    flags = [key_col in occupancy for occupancy in occupancies]
    run_lengths: list[int] = []
    gap_lengths: list[int] = []
    index = 0
    while index < len(flags):
        if flags[index]:
            end = index
            while end + 1 < len(flags) and flags[end + 1]:
                end += 1
            run_lengths.append(end - index + 1)
            index = end + 1
        else:
            end = index
            while end + 1 < len(flags) and not flags[end + 1]:
                end += 1
            gap_lengths.append(end - index + 1)
            index = end + 1
    group_short_runs = (
        evidence == "keyed_record"
        and key_runs >= 3
        and run_lengths
        and statistics.median(run_lengths) <= 2
        and gap_lengths
        and statistics.median(gap_lengths) >= 2
    )
    if evidence == "keyed_record" and not group_short_runs:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "record_key_support"
        return None

    grouped_indices: list[tuple[int, ...]] = []
    for top, bottom in _rule_bands(header_bottom, rules, width, height):
        band_rows = [row for row in body_rows if top <= (row.bbox[1] + row.bbox[3]) / 2.0 <= bottom]
        if not band_rows:
            continue
        grouped_indices.extend(
            _split_rows_by_anchors(
                band_rows,
                key_col,
                occupancies_by_index,
                group_short_key_runs=group_short_runs,
            )
        )
    if len(grouped_indices) < 2:
        return None

    row_by_index = {row.row_index: row for row in body_rows}
    raw_extents = [
        (
            min(row_by_index[index].bbox[1] for index in indices),
            max(row_by_index[index].bbox[3] for index in indices),
        )
        for indices in grouped_indices
    ]
    boundaries = [header_bottom]
    for first, second in zip(raw_extents, raw_extents[1:]):
        if first[1] <= second[0]:
            boundary = (first[1] + second[0]) / 2.0
        else:
            boundary = ((first[0] + first[1]) / 2.0 + (second[0] + second[1]) / 2.0) / 2.0
        boundaries.append(boundary)
    boundaries.append(height)
    if any(current <= previous for previous, current in zip(boundaries, boundaries[1:])):
        return None
    logical_rows = [
        _LogicalRow(
            visual_indices=indices,
            top=boundaries[index],
            bottom=boundaries[index + 1],
        )
        for index, indices in enumerate(grouped_indices)
    ]
    logical_occupancies = [
        set().union(*(occupancies_by_index[index] for index in logical.visual_indices)) for logical in logical_rows
    ]
    if diagnostics is not None:
        diagnostics.update(
            {
                "key_col": key_col,
                "key_runs": key_runs,
                "group_short_key_runs": group_short_runs,
                "logical_body_rows": len(logical_rows),
                "body_groups": [list(row.visual_indices) for row in logical_rows],
            }
        )
    return logical_rows, key_col, logical_occupancies


def _separator_coverage(
    rules: tuple[_LocalRule, ...],
    y: float,
    left: float,
    right: float,
    tolerance: float,
) -> float:
    """计算一条表头局部分隔在指定列带的覆盖率。"""

    intervals = [
        (rule.start, rule.end) for rule in rules if rule.orientation == "horizontal" and abs(rule.coordinate - y) <= tolerance
    ]
    return covered_interval_ratio(intervals, left, right)


def _header_separator(
    text: NativeTableText,
    rules: tuple[_LocalRule, ...],
    header_bottom: float,
    width: float,
) -> float | None:
    """选择表头内部唯一的完整或局部分隔线。"""

    header_rows = [row for row in text.rows if (row.bbox[1] + row.bbox[3]) / 2.0 < header_bottom]
    if len(header_rows) < 2:
        return None
    candidates = [
        rule
        for rule in rules
        if rule.orientation == "horizontal"
        and rule.end - rule.start >= 0.20 * width
        and header_rows[0].bbox[3] - 0.5 < rule.coordinate < header_bottom - 0.5
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda rule: (
            rule.end - rule.start,
            -abs(rule.coordinate - header_bottom / 2.0),
        ),
        reverse=True,
    )
    return candidates[0].coordinate


def _header_layer_tokens(
    text: NativeTableText,
    top: float,
    bottom: float,
) -> list[tuple[float, float, float]]:
    """收集一个表头层中 token 的水平区间和中心。"""

    tokens: list[tuple[float, float, float]] = []
    for row in text.rows:
        center_y = (row.bbox[1] + row.bbox[3]) / 2.0
        if not top <= center_y <= bottom:
            continue
        tokens.extend(
            (
                token.bbox[0],
                token.bbox[2],
                (token.bbox[0] + token.bbox[2]) / 2.0,
            )
            for token in row.tokens
        )
    return tokens


def _two_layer_header_specs(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    header_bottom: float,
    separator: float,
    rules: tuple[_LocalRule, ...],
) -> tuple[GridCellSpec, ...] | None:
    """用表头局部分隔恢复两层表头合并格。"""

    cols = len(x_tracks) - 1
    tolerance = max(1.0, 0.25 * text.median_glyph_height)
    coverages = [
        _separator_coverage(
            rules,
            separator,
            x_tracks[col],
            x_tracks[col + 1],
            tolerance,
        )
        for col in range(cols)
    ]
    if any(0.20 < coverage < 0.80 for coverage in coverages):
        return None
    absent_cols = {col for col, coverage in enumerate(coverages) if coverage <= 0.20}
    present_cols = set(range(cols)) - absent_cols
    specs: list[GridCellSpec] = [
        GridCellSpec(
            row=0,
            col=col,
            rowspan=2,
            colspan=1,
            bbox=(x_tracks[col], 0.0, x_tracks[col + 1], header_bottom),
        )
        for col in sorted(absent_cols)
    ]
    glyph_by_id = {glyph.glyph_id: glyph for glyph in text.glyphs}
    bottom_occupied: set[int] = set()
    for row in text.rows:
        center_y = (row.bbox[1] + row.bbox[3]) / 2.0
        if not separator < center_y < header_bottom:
            continue
        bottom_occupied.update(_row_occupancy(row, glyph_by_id, x_tracks))
    top_tokens = _header_layer_tokens(text, 0.0, separator)
    assignments: dict[int, list[int]] = {index: [] for index in range(len(top_tokens))}
    for col in sorted(bottom_occupied.intersection(present_cols)):
        if not top_tokens:
            return None
        col_center = (x_tracks[col] + x_tracks[col + 1]) / 2.0
        owner = min(
            range(len(top_tokens)),
            key=lambda index: abs(col_center - top_tokens[index][2]),
        )
        assignments[owner].append(col)

    covered: set[int] = set()
    for index, (_left, _right, center) in enumerate(top_tokens):
        group = assignments[index]
        if not group:
            continue
        if group != list(range(group[0], group[-1] + 1)):
            return None
        if not x_tracks[group[0]] <= center <= x_tracks[group[-1] + 1]:
            return None
        specs.append(
            GridCellSpec(
                row=0,
                col=group[0],
                rowspan=1,
                colspan=group[-1] - group[0] + 1,
                bbox=(
                    x_tracks[group[0]],
                    0.0,
                    x_tracks[group[-1] + 1],
                    separator,
                ),
            )
        )
        covered.update(group)
    for col in sorted(present_cols - covered):
        specs.append(
            GridCellSpec(
                row=0,
                col=col,
                rowspan=1,
                colspan=1,
                bbox=(x_tracks[col], 0.0, x_tracks[col + 1], separator),
            )
        )
    for col in sorted(present_cols):
        specs.append(
            GridCellSpec(
                row=1,
                col=col,
                rowspan=1,
                colspan=1,
                bbox=(x_tracks[col], separator, x_tracks[col + 1], header_bottom),
            )
        )
    return tuple(specs)


def _logical_cell_has_glyph(
    text: NativeTableText,
    logical_row: _LogicalRow,
    col: int,
    x_tracks: tuple[float, ...],
) -> bool:
    """判断一个逻辑正文格是否含有字符中心。"""

    visual_indices = set(logical_row.visual_indices)
    for glyph in text.glyphs:
        if glyph.visual_row not in visual_indices:
            continue
        center_x = (glyph.bbox[0] + glyph.bbox[2]) / 2.0
        if x_tracks[col] <= center_x <= x_tracks[col + 1]:
            return True
    return False


def _body_specs(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    logical_rows: list[_LogicalRow],
    row_offset: int,
    evidence: str,
    key_col: int,
) -> tuple[GridCellSpec, ...]:
    """构造正文网格，并仅在填充记录表中推断首列 rowspan。"""

    cols = len(x_tracks) - 1
    specs: list[GridCellSpec] = []
    span_lengths: dict[int, int] = {}
    if evidence == "filled_record" and key_col > 0:
        occupied_rows = [
            index for index, logical in enumerate(logical_rows) if _logical_cell_has_glyph(text, logical, 0, x_tracks)
        ]
        for position, start in enumerate(occupied_rows):
            end = occupied_rows[position + 1] if position + 1 < len(occupied_rows) else len(logical_rows)
            span_lengths[start] = max(1, end - start)

    covered_first_col: set[int] = set()
    for body_row, logical in enumerate(logical_rows):
        if body_row in span_lengths:
            rowspan = span_lengths[body_row]
            specs.append(
                GridCellSpec(
                    row=row_offset + body_row,
                    col=0,
                    rowspan=rowspan,
                    colspan=1,
                    bbox=(
                        x_tracks[0],
                        logical.top,
                        x_tracks[1],
                        logical_rows[body_row + rowspan - 1].bottom,
                    ),
                )
            )
            covered_first_col.update(range(body_row, body_row + rowspan))
        elif body_row not in covered_first_col:
            specs.append(
                GridCellSpec(
                    row=row_offset + body_row,
                    col=0,
                    rowspan=1,
                    colspan=1,
                    bbox=(x_tracks[0], logical.top, x_tracks[1], logical.bottom),
                )
            )
        for col in range(1, cols):
            specs.append(
                GridCellSpec(
                    row=row_offset + body_row,
                    col=col,
                    rowspan=1,
                    colspan=1,
                    bbox=(
                        x_tracks[col],
                        logical.top,
                        x_tracks[col + 1],
                        logical.bottom,
                    ),
                )
            )
    return tuple(specs)


def _stable_gutters(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    physical_boundaries: frozenset[int],
    header_bottom: float,
    diagnostics: dict[str, Any] | None,
) -> bool:
    """校验每条文本列边界都由稳定空白走廊或物理边缘支持。"""

    edge_tolerance = max(0.15, 0.03 * text.median_glyph_width)
    supports: list[int] = []
    glyphs_by_row: dict[int, list[NativeTableGlyph]] = {}
    for glyph in text.glyphs:
        glyphs_by_row.setdefault(glyph.visual_row, []).append(glyph)
    body_glyphs = [
        glyph
        for glyph in text.glyphs
        if (text.rows[glyph.visual_row].bbox[1] + text.rows[glyph.visual_row].bbox[3]) / 2.0 > header_bottom
    ]
    for boundary_index, boundary in enumerate(x_tracks[1:-1], start=1):
        if any(glyph.bbox[0] + edge_tolerance < boundary < glyph.bbox[2] - edge_tolerance for glyph in body_glyphs):
            if diagnostics is not None:
                diagnostics.update(
                    {
                        "failed_gutter_boundary": boundary_index,
                        "failed_gutter_reason": "glyph_crossing",
                    }
                )
            return False
        if boundary_index in physical_boundaries:
            supports.append(len(text.rows))
            continue
        comparable = 0
        stable = 0
        for row in text.rows:
            if (row.bbox[1] + row.bbox[3]) / 2.0 <= header_bottom:
                continue
            row_glyphs = glyphs_by_row.get(row.row_index, [])
            left_glyphs = [glyph for glyph in row_glyphs if glyph.bbox[2] <= boundary]
            right_glyphs = [glyph for glyph in row_glyphs if glyph.bbox[0] >= boundary]
            if not left_glyphs or not right_glyphs:
                continue
            comparable += 1
            gap = min(glyph.bbox[0] for glyph in right_glyphs) - max(glyph.bbox[2] for glyph in left_glyphs)
            if gap >= max(0.25, 0.10 * text.median_glyph_width):
                stable += 1
        if comparable < 3 or stable / comparable < 0.80:
            if diagnostics is not None:
                diagnostics.update(
                    {
                        "failed_gutter_boundary": boundary_index,
                        "failed_gutter_reason": "support",
                        "failed_gutter_comparable": comparable,
                        "failed_gutter_stable": stable,
                    }
                )
            return False
        supports.append(stable)
    if diagnostics is not None:
        diagnostics["gutter_supports"] = supports
    return True


def _has_overlapping_formula_rows(
    text: NativeTableText,
    header_bottom: float,
) -> bool:
    """识别高公式字符框跨越相邻逻辑行的危险表格。"""

    body_rows = [row for row in text.rows if (row.bbox[1] + row.bbox[3]) / 2.0 > header_bottom]
    if any(row.bbox[3] - row.bbox[1] > 3.0 * text.median_glyph_height for row in body_rows):
        return True
    return False


def _has_ambiguous_body_descriptor(
    logical_occupancies: list[set[int]],
    evidence: str,
) -> bool:
    """识别无线正文中首列空缺后再次出现而无法唯一确定 rowspan 的情况。"""

    if evidence == "filled_record":
        return False
    seen_nonempty = False
    seen_gap = False
    for occupancy in logical_occupancies:
        if 0 in occupancy:
            if seen_nonempty and seen_gap:
                return True
            seen_nonempty = True
        elif seen_nonempty:
            seen_gap = True
    return False


def _build_candidate(
    table_input: NativeTableInput,
    text: NativeTableText,
    diagnostics: dict[str, Any] | None,
) -> NativeTableCandidate | None:
    """构造一个末级多行少线候选并执行高置信硬门。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return None
    width, height = table_local_size(
        table_bbox,
        normalize_angle(table_input.angle),
    )
    rules = _local_rules(table_input, width, height)
    rectangles = _local_rectangles(table_input, width, height)
    hypothesis = _build_column_hypothesis(
        text,
        width,
        rules,
        rectangles,
        diagnostics,
    )
    if hypothesis is None:
        return None
    header_bottom = _infer_header_boundary(
        text,
        hypothesis.x_tracks,
        rules,
        width,
        height,
        hypothesis.evidence,
    )
    if header_bottom is None:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "header_boundary"
        return None
    if _has_overlapping_formula_rows(text, header_bottom):
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "overlapping_formula_rows"
        return None
    body = _logical_body_rows(
        text,
        hypothesis.x_tracks,
        rules,
        header_bottom,
        width,
        height,
        hypothesis.evidence,
        diagnostics,
    )
    if body is None:
        if diagnostics is not None and diagnostics.get("first_rejection_gate") is None:
            diagnostics["first_rejection_gate"] = "logical_rows"
        return None
    logical_rows, key_col, logical_occupancies = body
    if _has_ambiguous_body_descriptor(
        logical_occupancies,
        hypothesis.evidence,
    ):
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "ambiguous_body_rowspan"
        return None
    if not _stable_gutters(
        text,
        hypothesis.x_tracks,
        hypothesis.physical_boundaries,
        header_bottom,
        diagnostics,
    ):
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "gutter_support"
        return None

    separator = _header_separator(text, rules, header_bottom, width)
    if separator is None:
        header_rows = 1
        header_specs = tuple(
            GridCellSpec(
                row=0,
                col=col,
                rowspan=1,
                colspan=1,
                bbox=(
                    hypothesis.x_tracks[col],
                    0.0,
                    hypothesis.x_tracks[col + 1],
                    header_bottom,
                ),
            )
            for col in range(len(hypothesis.x_tracks) - 1)
        )
    else:
        header_rows = 2
        header_specs = _two_layer_header_specs(
            text,
            hypothesis.x_tracks,
            header_bottom,
            separator,
            rules,
        )
        if header_specs is None:
            if diagnostics is not None:
                diagnostics["first_rejection_gate"] = "header_topology"
            return None
    body_specs = _body_specs(
        text,
        hypothesis.x_tracks,
        logical_rows,
        header_rows,
        hypothesis.evidence,
        key_col,
    )
    candidate_diagnostics: dict[str, object] = {}
    candidate = build_candidate(
        source="sparse_multiline",
        rows=header_rows + len(logical_rows),
        cols=len(hypothesis.x_tracks) - 1,
        specs=(*header_specs, *body_specs),
        text=text,
        structure_support=1.0,
        row_stability=1.0,
        column_stability=1.0,
        issues=(
            f"evidence={hypothesis.evidence}",
            f"header_rows={header_rows}",
            f"logical_body_rows={len(logical_rows)}",
            f"key_col={key_col}",
            f"filled_band_count={hypothesis.filled_band_count}",
        ),
        use_grid_index=True,
        diagnostics=candidate_diagnostics,
    )
    if candidate is None:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = candidate_diagnostics.get(
                "candidate_rejection_gate",
                "candidate_hard_gate",
            )
        return None
    if diagnostics is not None:
        diagnostics["candidate_diagnostics"] = dict(candidate_diagnostics)
        diagnostics["candidate_components"] = {
            "text_capture": candidate.text_capture,
            "order_consistency": candidate.order_consistency,
            "score": candidate.score,
            "ambiguous_glyph_ratio": candidate_diagnostics.get(
                "ambiguous_glyph_ratio",
                1.0,
            ),
        }
    ambiguous_ratio = float(candidate_diagnostics.get("ambiguous_glyph_ratio", 1.0))
    if (
        candidate.text_capture < 1.0
        or candidate.order_consistency < 1.0
        or ambiguous_ratio > 0.0
        or candidate.score < MIN_MULTILINE_RELIABILITY
    ):
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "verified_integrity"
        return None
    if hypothesis.evidence == "keyed_record" and len(logical_rows) < 3:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "record_count"
        return None
    if diagnostics is not None:
        diagnostics.update(
            {
                "first_rejection_gate": None,
                "grid": {"rows": candidate.rows, "cols": candidate.cols},
                "header_bottom": header_bottom,
                "header_rows": header_rows,
                "score": candidate.score,
                "token_split_count": candidate_diagnostics.get(
                    "token_split_count",
                    0,
                ),
            }
        )
    return candidate


def build_sparse_multiline_candidates(
    table_input: NativeTableInput,
    text: NativeTableText,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[NativeTableCandidate]:
    """生成仅在既有候选全部失败后运行的多行少线候选。"""

    record: dict[str, Any] | None = {"source": "sparse_multiline"} if diagnostics is not None else None
    candidate = _build_candidate(table_input, text, record)
    if diagnostics is not None and record is not None:
        diagnostics.append(record)
    return [candidate] if candidate is not None else []


def diagnose_sparse_multiline_candidate_builds(
    table_input: NativeTableInput,
    text: NativeTableText,
) -> tuple[dict[str, Any], ...]:
    """重放多行少线候选构造并返回私有诊断。"""

    diagnostics: list[dict[str, Any]] = []
    build_sparse_multiline_candidates(
        table_input,
        text,
        diagnostics=diagnostics,
    )
    return tuple(diagnostics)


__all__ = ["build_sparse_multiline_candidates"]
