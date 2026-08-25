# Copyright (c) Opendatalab. All rights reserved.
"""融合稀疏物理边界和文本对齐网络恢复少线表格结构。"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .candidate import GridCellSpec, build_candidate
from .contracts import (
    NativeTableCandidate,
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

MIN_COLUMN_SUPPORT = 0.60
MIN_OVERALL_ANCHOR_SUPPORT = 0.80
MIN_SPARSE_RELIABILITY = 0.98
MAX_SPARSE_HYPOTHESES = 8
MAX_HEADER_ROWS = 2


@dataclass(frozen=True, slots=True)
class _LocalRule:
    """保存转换到正向表格坐标后的细线区间。"""

    orientation: str
    coordinate: float
    start: float
    end: float
    width: float


@dataclass(frozen=True, slots=True)
class _TrackHypothesis:
    """保存一组少线表叶子列轨及其独立证据。"""

    evidence: str
    x_tracks: tuple[float, ...]
    physical_boundaries: frozenset[int]
    body_start: int
    reliability: float


@dataclass(frozen=True, slots=True)
class _DenseLayout:
    """保存正文稠密行推断出的列数和连续正文起点。"""

    target_cols: int
    body_start: int
    dense_row_indices: tuple[int, ...]


def _cluster_with_members(
    values: list[float],
    tolerance: float,
) -> list[tuple[float, tuple[float, ...]]]:
    """按一维距离聚类坐标并保留每簇原始成员。"""

    if not values:
        return []
    clusters: list[list[float]] = [[value] for value in sorted(values)[:1]]
    for value in sorted(values)[1:]:
        center = float(statistics.median(clusters[-1]))
        if abs(value - center) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [(float(statistics.median(cluster)), tuple(cluster)) for cluster in clusters]


def _local_rules(
    table_input: NativeTableInput,
    width: float,
    height: float,
) -> tuple[_LocalRule, ...]:
    """把页面 drawing 线转换为按局部长轴重新判向的规则区间。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return ()
    angle = normalize_angle(table_input.angle)
    output: list[_LocalRule] = []
    for rule in table_input.drawing_lines:
        bbox = normalize_bbox(rule.bbox)
        if bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(bbox, table_bbox, angle)
        if local_bbox is None:
            continue
        local_width = local_bbox[2] - local_bbox[0]
        local_height = local_bbox[3] - local_bbox[1]
        if local_width >= max(1.0, 3.0 * max(local_height, 0.1)):
            output.append(
                _LocalRule(
                    orientation="horizontal",
                    coordinate=(local_bbox[1] + local_bbox[3]) / 2.0,
                    start=max(0.0, local_bbox[0]),
                    end=min(width, local_bbox[2]),
                    width=max(rule.width, local_height),
                )
            )
        elif local_height >= max(1.0, 3.0 * max(local_width, 0.1)):
            output.append(
                _LocalRule(
                    orientation="vertical",
                    coordinate=(local_bbox[0] + local_bbox[2]) / 2.0,
                    start=max(0.0, local_bbox[1]),
                    end=min(height, local_bbox[3]),
                    width=max(rule.width, local_width),
                )
            )
    return tuple(output)


def _long_horizontal_rules(
    rules: tuple[_LocalRule, ...],
    width: float,
) -> tuple[_LocalRule, ...]:
    """筛选能够独立证明表带存在的长横线。"""

    return tuple(rule for rule in rules if rule.orientation == "horizontal" and rule.end - rule.start >= 0.50 * width)


def _vertical_track_evidence(
    rules: tuple[_LocalRule, ...],
    width: float,
    height: float,
    tolerance: float,
) -> tuple[tuple[float, ...], dict[float, float]]:
    """合并同 X 分段竖线并返回覆盖足够的物理列轨。"""

    vertical_rules = [rule for rule in rules if rule.orientation == "vertical"]
    clusters = _cluster_with_members(
        [rule.coordinate for rule in vertical_rules],
        tolerance,
    )
    positions: list[float] = []
    coverages: dict[float, float] = {}
    for coordinate, _members in clusters:
        intervals = [(rule.start, rule.end) for rule in vertical_rules if abs(rule.coordinate - coordinate) <= tolerance]
        coverage = covered_interval_ratio(intervals, 0.0, height)
        if coverage < 0.75:
            continue
        snapped = 0.0 if coordinate <= tolerance else width if width - coordinate <= tolerance else coordinate
        positions.append(snapped)
        coverages[snapped] = max(coverages.get(snapped, 0.0), coverage)
    positions.extend([0.0, width])
    return tuple(sorted(set(positions))), coverages


def _rectangle_edge_evidence(
    table_input: NativeTableInput,
    text: NativeTableText,
    width: float,
    height: float,
    tolerance: float,
) -> tuple[float, ...]:
    """从表头单元格矩形和上下细条中提取可复现的列边界。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return ()
    angle = normalize_angle(table_input.angle)
    raw_edges: list[float] = []
    header_limit = min(0.30 * height, 3.0 * text.median_glyph_height)
    thin_limit = max(1.5, 0.40 * text.median_glyph_height)
    for rectangle in table_input.rectangles:
        if rectangle.segment_count != 5 or not (rectangle.fill_visible or rectangle.stroke_visible):
            continue
        bbox = normalize_bbox(rectangle.bbox)
        if bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(bbox, table_bbox, angle)
        if local_bbox is None:
            continue
        rect_width = local_bbox[2] - local_bbox[0]
        rect_height = local_bbox[3] - local_bbox[1]
        is_thin_rule = rect_height <= thin_limit and rect_width >= 2.0 * text.median_glyph_width
        is_header_cell = (
            local_bbox[1] <= header_limit
            and rect_height <= 3.0 * text.median_glyph_height
            and rect_width >= 2.0 * text.median_glyph_width
        )
        if not (is_thin_rule or is_header_cell):
            continue
        raw_edges.extend([max(0.0, local_bbox[0]), min(width, local_bbox[2])])

    supported = [
        coordinate
        for coordinate, members in _cluster_with_members(raw_edges, tolerance)
        if len(members) >= 2 or coordinate <= tolerance or width - coordinate <= tolerance
    ]
    return tuple(sorted({0.0, width, *supported}))


def _canonical_edge_tracks(
    positions: tuple[float, ...],
    width: float,
    tolerance: float,
) -> tuple[float, ...]:
    """折叠靠近表格外缘的重复矩形端点并返回严格递增轨道。"""

    snapped = [0.0 if position <= tolerance else width if width - position <= tolerance else position for position in positions]
    tracks = tuple(sorted(set(snapped)))
    if any(current <= previous for previous, current in zip(tracks, tracks[1:])):
        return ()
    return tracks


def _longest_consecutive_run(indices: list[int]) -> tuple[int, ...]:
    """返回整数索引列表中最长的连续区间。"""

    if not indices:
        return ()
    runs: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        if index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return tuple(max(runs, key=lambda run: (len(run), run[-1])))


def _infer_dense_layout(text: NativeTableText) -> _DenseLayout | None:
    """从正文重复 token 数选择叶子列数和首条正文行。"""

    counts = Counter(len(row.tokens) for row in text.rows if len(row.tokens) >= 2)
    hypotheses: list[tuple[int, int, int, tuple[int, ...]]] = []
    for count, occurrences in counts.items():
        indices = [row.row_index for row in text.rows if len(row.tokens) == count]
        run = _longest_consecutive_run(indices)
        if len(run) < 2:
            continue
        hypotheses.append((len(run), occurrences, count, run))
    if not hypotheses:
        return None
    _run_length, _occurrences, target_cols, run = max(hypotheses)
    return _DenseLayout(
        target_cols=target_cols,
        body_start=run[0],
        dense_row_indices=run,
    )


def _infer_text_tracks(
    text: NativeTableText,
    width: float,
    layout: _DenseLayout,
) -> tuple[float, ...] | None:
    """用正文相邻 token 空隙的中位位置推断叶子列边界。"""

    dense_rows = [text.rows[index] for index in layout.dense_row_indices]
    boundaries: list[float] = []
    for col in range(layout.target_cols - 1):
        midpoints = [(row.tokens[col].bbox[2] + row.tokens[col + 1].bbox[0]) / 2.0 for row in dense_rows]
        boundary = float(statistics.median(midpoints))
        if not 0.0 < boundary < width:
            return None
        boundaries.append(boundary)
    tracks = (0.0, *boundaries, width)
    if any(current <= previous for previous, current in zip(tracks, tracks[1:])):
        return None
    return tracks


def _row_glyph_occupancy(
    row: NativeTableTextRow,
    text: NativeTableText,
    x_tracks: tuple[float, ...],
) -> set[int]:
    """按字符中心统计一条视觉行实际占用的叶子列。"""

    glyph_by_id = {glyph.glyph_id: glyph for glyph in text.glyphs}
    occupied: set[int] = set()
    for glyph_id in row.glyph_ids:
        glyph = glyph_by_id[glyph_id]
        center = (glyph.bbox[0] + glyph.bbox[2]) / 2.0
        col = next(
            (index for index, (left, right) in enumerate(zip(x_tracks, x_tracks[1:])) if left <= center <= right),
            None,
        )
        if col is not None:
            occupied.add(col)
    return occupied


def _track_support(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    body_start: int,
) -> tuple[float, float, tuple[set[int], ...]] | None:
    """校验正文行、关键列和各叶子列的重复占用支持。"""

    body_rows = text.rows[body_start:]
    cols = len(x_tracks) - 1
    if len(body_rows) < 2 or cols < 2:
        return None
    occupancies = tuple(_row_glyph_occupancy(row, text, x_tracks) for row in body_rows)
    minimum_dense_cols = max(2, math.ceil(MIN_COLUMN_SUPPORT * cols))
    if any(len(occupancy) < minimum_dense_cols for occupancy in occupancies):
        return None
    first_columns = [min(occupancy) for occupancy in occupancies if occupancy]
    if not first_columns:
        return None
    key_col = Counter(first_columns).most_common(1)[0][0]
    if sum(key_col in occupancy for occupancy in occupancies) / len(occupancies) < MIN_COLUMN_SUPPORT:
        return None
    supports = [sum(col in occupancy for occupancy in occupancies) / len(occupancies) for col in range(cols)]
    minimum_support = min(supports)
    overall_support = float(statistics.mean(supports))
    if minimum_support < MIN_COLUMN_SUPPORT or overall_support < MIN_OVERALL_ANCHOR_SUPPORT:
        return None
    return minimum_support, overall_support, occupancies


def _nearest_physical_boundaries(
    x_tracks: tuple[float, ...],
    physical_positions: tuple[float, ...],
    tolerance: float,
) -> frozenset[int]:
    """标记能被独立 drawing 或矩形边缘支持的内部列边界。"""

    return frozenset(
        index
        for index, coordinate in enumerate(x_tracks[1:-1], start=1)
        if any(abs(coordinate - physical) <= tolerance for physical in physical_positions)
    )


def _infer_y_tracks(
    text: NativeTableText,
    rules: tuple[_LocalRule, ...],
    height: float,
) -> tuple[float, ...] | None:
    """以视觉行中心中点为基础并优先吸附相邻行间横线。"""

    if len(text.rows) < 2:
        return None
    tracks: list[float] = [0.0]
    horizontal_rules = [rule for rule in rules if rule.orientation == "horizontal"]
    for previous, current in zip(text.rows, text.rows[1:]):
        candidates = [rule for rule in horizontal_rules if previous.bbox[3] - 0.5 <= rule.coordinate <= current.bbox[1] + 0.5]
        if candidates:
            boundary = max(candidates, key=lambda rule: rule.end - rule.start).coordinate
        else:
            previous_center = (previous.bbox[1] + previous.bbox[3]) / 2.0
            current_center = (current.bbox[1] + current.bbox[3]) / 2.0
            boundary = (previous_center + current_center) / 2.0
        tracks.append(boundary)
    tracks.append(height)
    if any(current <= previous for previous, current in zip(tracks, tracks[1:])):
        return None
    return tuple(tracks)


def _horizontal_separator_coverage(
    rules: tuple[_LocalRule, ...],
    boundary: float,
    left: float,
    right: float,
    tolerance: float,
) -> float:
    """计算指定表头行边界在一个叶子列范围内的横线覆盖率。"""

    intervals = [
        (rule.start, rule.end)
        for rule in rules
        if rule.orientation == "horizontal" and abs(rule.coordinate - boundary) <= tolerance
    ]
    return covered_interval_ratio(intervals, left, right)


def _row_token_columns(
    row: NativeTableTextRow,
    x_tracks: tuple[float, ...],
) -> list[int]:
    """把一行粗 token 的中心映射到叶子列。"""

    output: list[int] = []
    for token in row.tokens:
        center = (token.bbox[0] + token.bbox[2]) / 2.0
        col = next(
            (index for index, (left, right) in enumerate(zip(x_tracks, x_tracks[1:])) if left <= center <= right),
            None,
        )
        if col is not None:
            output.append(col)
    return output


def _two_level_header_specs(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    y_tracks: tuple[float, ...],
    rules: tuple[_LocalRule, ...],
    tolerance: float,
) -> tuple[GridCellSpec, ...] | None:
    """用局部横线和上下层文本恢复两层表头的 rowspan/colspan。"""

    cols = len(x_tracks) - 1
    boundary = y_tracks[1]
    coverages = [
        _horizontal_separator_coverage(
            rules,
            boundary,
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
            bbox=(x_tracks[col], y_tracks[0], x_tracks[col + 1], y_tracks[2]),
        )
        for col in sorted(absent_cols)
    ]

    child_occupied = _row_glyph_occupancy(text.rows[1], text, x_tracks).intersection(present_cols)
    top_tokens = [
        (token, col)
        for token, col in zip(
            text.rows[0].tokens,
            _row_token_columns(text.rows[0], x_tracks),
            strict=False,
        )
        if col in present_cols
    ]
    assigned_groups: dict[int, list[int]] = {index: [] for index in range(len(top_tokens))}
    for col in sorted(child_occupied):
        col_center = (x_tracks[col] + x_tracks[col + 1]) / 2.0
        if not top_tokens:
            return None
        owner = min(
            range(len(top_tokens)),
            key=lambda index: abs(col_center - (top_tokens[index][0].bbox[0] + top_tokens[index][0].bbox[2]) / 2.0),
        )
        assigned_groups[owner].append(col)

    top_covered: set[int] = set()
    for index, (token, _token_col) in enumerate(top_tokens):
        group = assigned_groups[index]
        if not group or group != list(range(group[0], group[-1] + 1)):
            return None
        token_center = (token.bbox[0] + token.bbox[2]) / 2.0
        if not x_tracks[group[0]] <= token_center <= x_tracks[group[-1] + 1]:
            return None
        specs.append(
            GridCellSpec(
                row=0,
                col=group[0],
                rowspan=1,
                colspan=group[-1] - group[0] + 1,
                bbox=(x_tracks[group[0]], y_tracks[0], x_tracks[group[-1] + 1], y_tracks[1]),
            )
        )
        top_covered.update(group)

    for col in sorted(present_cols - top_covered):
        specs.append(
            GridCellSpec(
                row=0,
                col=col,
                rowspan=1,
                colspan=1,
                bbox=(x_tracks[col], y_tracks[0], x_tracks[col + 1], y_tracks[1]),
            )
        )
    for col in sorted(present_cols):
        specs.append(
            GridCellSpec(
                row=1,
                col=col,
                rowspan=1,
                colspan=1,
                bbox=(x_tracks[col], y_tracks[1], x_tracks[col + 1], y_tracks[2]),
            )
        )
    return tuple(specs)


def _build_sparse_specs(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    y_tracks: tuple[float, ...],
    body_start: int,
    rules: tuple[_LocalRule, ...],
    tolerance: float,
) -> tuple[GridCellSpec, ...] | None:
    """构造完整少线网格，并仅在两层表头中推断合并格。"""

    rows = len(y_tracks) - 1
    cols = len(x_tracks) - 1
    if body_start > MAX_HEADER_ROWS:
        return None
    specs: list[GridCellSpec] = []
    if body_start == 2:
        header_specs = _two_level_header_specs(
            text,
            x_tracks,
            y_tracks,
            rules,
            tolerance,
        )
        if header_specs is None:
            return None
        specs.extend(header_specs)
    else:
        for row in range(body_start):
            for col in range(cols):
                specs.append(
                    GridCellSpec(
                        row=row,
                        col=col,
                        rowspan=1,
                        colspan=1,
                        bbox=(x_tracks[col], y_tracks[row], x_tracks[col + 1], y_tracks[row + 1]),
                    )
                )
    for row in range(body_start, rows):
        for col in range(cols):
            specs.append(
                GridCellSpec(
                    row=row,
                    col=col,
                    rowspan=1,
                    colspan=1,
                    bbox=(x_tracks[col], y_tracks[row], x_tracks[col + 1], y_tracks[row + 1]),
                )
            )
    return tuple(specs)


def _spec_owner_grid(
    rows: int,
    cols: int,
    specs: tuple[GridCellSpec, ...],
) -> list[list[int]]:
    """把逻辑单元格展开为原子格到 spec 下标的映射。"""

    owners = [[-1 for _ in range(cols)] for _ in range(rows)]
    for index, spec in enumerate(specs):
        for row in range(spec.row, spec.row + spec.rowspan):
            for col in range(spec.col, spec.col + spec.colspan):
                owners[row][col] = index
    return owners


def _validate_token_splits(
    text: NativeTableText,
    x_tracks: tuple[float, ...],
    specs: tuple[GridCellSpec, ...],
    physical_boundaries: frozenset[int],
) -> tuple[bool, int]:
    """只允许被强物理边界证明且不横切字符的粗 token 跨格。"""

    rows = len(text.rows)
    cols = len(x_tracks) - 1
    owners = _spec_owner_grid(rows, cols, specs)
    glyph_by_id = {glyph.glyph_id: glyph for glyph in text.glyphs}
    edge_tolerance = max(0.25, 0.05 * text.median_glyph_width)
    justified_splits = 0
    for row in text.rows:
        for token in row.tokens:
            for boundary_index, boundary in enumerate(x_tracks[1:-1], start=1):
                if owners[row.row_index][boundary_index - 1] == owners[row.row_index][boundary_index]:
                    continue
                if not token.bbox[0] + edge_tolerance < boundary < token.bbox[2] - edge_tolerance:
                    continue
                if boundary_index not in physical_boundaries:
                    return False, justified_splits
                if any(
                    glyph_by_id[glyph_id].bbox[0] + edge_tolerance < boundary < glyph_by_id[glyph_id].bbox[2] - edge_tolerance
                    for glyph_id in token.glyph_ids
                ):
                    return False, justified_splits
                justified_splits += 1
    return True, justified_splits


def _build_hypothesis_candidate(
    table_input: NativeTableInput,
    text: NativeTableText,
    rules: tuple[_LocalRule, ...],
    hypothesis: _TrackHypothesis,
    height: float,
    diagnostics: dict[str, Any] | None,
) -> NativeTableCandidate | None:
    """把一组少线轨道恢复为候选并执行全部高置信硬门。"""

    x_tracks = hypothesis.x_tracks
    y_tracks = _infer_y_tracks(text, rules, height)
    if y_tracks is None:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "row_tracks"
        return None
    support = _track_support(text, x_tracks, hypothesis.body_start)
    if support is None:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "anchor_support"
        return None
    minimum_support, overall_support, occupancies = support
    tolerance = max(1.0, 0.25 * text.median_glyph_height)
    specs = _build_sparse_specs(
        text,
        x_tracks,
        y_tracks,
        hypothesis.body_start,
        rules,
        tolerance,
    )
    if specs is None:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "header_topology"
        return None
    token_splits_valid, justified_splits = _validate_token_splits(
        text,
        x_tracks,
        specs,
        hypothesis.physical_boundaries,
    )
    if not token_splits_valid:
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "token_split"
        return None

    candidate_diagnostics: dict[str, object] = {}
    candidate = build_candidate(
        source="sparse_hybrid",
        rows=len(y_tracks) - 1,
        cols=len(x_tracks) - 1,
        specs=specs,
        text=text,
        structure_support=hypothesis.reliability,
        row_stability=1.0,
        column_stability=min(1.0, overall_support),
        issues=(
            f"evidence={hypothesis.evidence}",
            f"body_start={hypothesis.body_start}",
            f"minimum_column_support={minimum_support:.4f}",
            f"overall_anchor_support={overall_support:.4f}",
            f"physical_boundaries={len(hypothesis.physical_boundaries)}",
            f"physically_justified_token_splits={justified_splits}",
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
    ambiguous_ratio = float(candidate_diagnostics.get("ambiguous_glyph_ratio", 1.0))
    if (
        candidate.text_capture < 1.0
        or candidate.order_consistency < 1.0
        or ambiguous_ratio > 0.0
        or candidate.score < MIN_SPARSE_RELIABILITY
    ):
        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = "verified_integrity"
        return None
    if diagnostics is not None:
        diagnostics.update(
            {
                "first_rejection_gate": None,
                "grid": {"rows": candidate.rows, "cols": candidate.cols},
                "body_start": hypothesis.body_start,
                "x_tracks": list(x_tracks),
                "physical_boundaries": sorted(hypothesis.physical_boundaries),
                "minimum_column_support": minimum_support,
                "overall_anchor_support": overall_support,
                "body_occupancies": [sorted(occupancy) for occupancy in occupancies],
                "physically_justified_token_splits": justified_splits,
                "score": candidate.score,
            }
        )
    return candidate


def _build_track_hypotheses(
    table_input: NativeTableInput,
    text: NativeTableText,
    rules: tuple[_LocalRule, ...],
    width: float,
    height: float,
) -> tuple[_TrackHypothesis, ...]:
    """构造有限的文本轨和强竖线轨假设并消除同拓扑重复。"""

    layout = _infer_dense_layout(text)
    if layout is None:
        return ()
    tolerance = max(1.0, 0.25 * text.median_glyph_height)
    vertical_tracks, vertical_coverages = _vertical_track_evidence(
        rules,
        width,
        height,
        tolerance,
    )
    rect_positions = _rectangle_edge_evidence(
        table_input,
        text,
        width,
        height,
        tolerance,
    )
    physical_positions = tuple(sorted({*vertical_tracks, *rect_positions}))
    rect_tracks = _canonical_edge_tracks(rect_positions, width, tolerance)
    text_tracks = _infer_text_tracks(text, width, layout)
    hypotheses: list[_TrackHypothesis] = []

    physical_cols = len(vertical_tracks) - 1
    if (
        physical_cols >= 2
        and physical_cols in {layout.target_cols, layout.target_cols + 1}
        and _track_support(text, vertical_tracks, layout.body_start) is not None
    ):
        internal_coverages = [
            coverage for coordinate, coverage in vertical_coverages.items() if tolerance < coordinate < width - tolerance
        ]
        reliability = min(internal_coverages, default=1.0)
        if physical_cols == layout.target_cols:
            reliability = 1.0
        hypotheses.append(
            _TrackHypothesis(
                evidence="vertical_text",
                x_tracks=vertical_tracks,
                physical_boundaries=frozenset(range(1, len(vertical_tracks) - 1)),
                body_start=layout.body_start,
                reliability=min(1.0, reliability),
            )
        )

    rect_cols = len(rect_tracks) - 1
    if rect_cols == layout.target_cols and rect_cols >= 2 and _track_support(text, rect_tracks, layout.body_start) is not None:
        hypotheses.append(
            _TrackHypothesis(
                evidence="rect_text",
                x_tracks=rect_tracks,
                physical_boundaries=frozenset(range(1, len(rect_tracks) - 1)),
                body_start=layout.body_start,
                reliability=1.0,
            )
        )

    prefer_physical = bool(hypotheses)
    if text_tracks is not None and not prefer_physical:
        hypotheses.append(
            _TrackHypothesis(
                evidence="text_network",
                x_tracks=text_tracks,
                physical_boundaries=_nearest_physical_boundaries(
                    text_tracks,
                    physical_positions,
                    tolerance,
                ),
                body_start=layout.body_start,
                reliability=1.0,
            )
        )

    deduplicated: dict[tuple[int, tuple[int, ...]], _TrackHypothesis] = {}
    for hypothesis in hypotheses[:MAX_SPARSE_HYPOTHESES]:
        signature = (
            len(hypothesis.x_tracks),
            tuple(round(track / max(tolerance, 0.1)) for track in hypothesis.x_tracks),
        )
        existing = deduplicated.get(signature)
        if existing is None or hypothesis.reliability > existing.reliability:
            deduplicated[signature] = hypothesis
    return tuple(deduplicated.values())


def build_sparse_hybrid_candidates(
    table_input: NativeTableInput,
    text: NativeTableText,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[NativeTableCandidate]:
    """生成只在矢量网格失败后参与仲裁的高置信少线候选。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return []
    width, height = table_local_size(table_bbox, normalize_angle(table_input.angle))
    rules = _local_rules(table_input, width, height)
    long_rules = _long_horizontal_rules(rules, width)
    tolerance = max(1.0, 0.25 * text.median_glyph_height)
    vertical_tracks, _coverages = _vertical_track_evidence(
        rules,
        width,
        height,
        tolerance,
    )
    if len(long_rules) < 2 and len(vertical_tracks) < 3:
        if diagnostics is not None:
            diagnostics.append(
                {
                    "source": "sparse_hybrid",
                    "first_rejection_gate": "physical_sparse_evidence",
                    "long_horizontal_rules": len(long_rules),
                    "vertical_tracks": len(vertical_tracks),
                }
            )
        return []

    hypotheses = _build_track_hypotheses(
        table_input,
        text,
        rules,
        width,
        height,
    )
    candidates: list[NativeTableCandidate] = []
    for hypothesis in hypotheses:
        record: dict[str, Any] | None = (
            {
                "source": "sparse_hybrid",
                "evidence": hypothesis.evidence,
                "long_horizontal_rules": len(long_rules),
            }
            if diagnostics is not None
            else None
        )
        candidate = _build_hypothesis_candidate(
            table_input,
            text,
            rules,
            hypothesis,
            height,
            record,
        )
        if diagnostics is not None and record is not None:
            diagnostics.append(record)
        if candidate is not None:
            candidates.append(candidate)

    topologies = {candidate.topology for candidate in candidates}
    if len(topologies) > 1:
        if diagnostics is not None:
            diagnostics.append(
                {
                    "source": "sparse_hybrid",
                    "first_rejection_gate": "topology_ambiguity",
                    "topology_count": len(topologies),
                }
            )
        return []
    return candidates[:1]


def diagnose_sparse_hybrid_candidate_builds(
    table_input: NativeTableInput,
    text: NativeTableText,
) -> tuple[dict[str, Any], ...]:
    """重放少线候选构造并返回不进入用户结果的诊断。"""

    diagnostics: list[dict[str, Any]] = []
    build_sparse_hybrid_candidates(
        table_input,
        text,
        diagnostics=diagnostics,
    )
    return tuple(diagnostics)


__all__ = ["build_sparse_hybrid_candidates"]
