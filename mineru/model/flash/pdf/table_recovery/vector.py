# Copyright (c) Opendatalab. All rights reserved.
"""基于 PDF 横竖线与矩形路径恢复原子网格和合并单元格。"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from .candidate import GridCellSpec, build_candidate
from .contracts import (
    NativeTableCandidate,
    NativeTableInput,
    NativeTableText,
)
from .geometry import (
    bbox_area,
    bbox_intersection,
    clamp,
    cluster_positions,
    covered_interval_ratio,
    normalize_angle,
    normalize_bbox,
    page_bbox_to_table_local,
    rotate_local_bbox,
    table_local_size,
)

MAX_PRIMITIVES_PER_TABLE = 5000
MAX_TRACKS_PER_AXIS = 200
MAX_ATOMIC_CELLS = 10000
SEPARATOR_COVERAGE_THRESHOLD = 0.80
MAX_TRACK_HYPOTHESES = 8


@dataclass(frozen=True, slots=True)
class _MergedRule:
    """保存吸附并连接后的局部单轴线段。"""

    orientation: str
    coordinate: float
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class _CanonicalTrack:
    """保存折叠后的规范轨道及其全部原始坐标别名。"""

    coordinate: float
    aliases: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _PhysicalRowEvidence:
    """保存一个原子行由 drawing 独立验证的边界可靠度。"""

    row: int
    top_coverage: float
    bottom_coverage: float
    left_coverage: float
    right_coverage: float
    height_ratio: float
    glyph_crossing: bool

    @property
    def reliability(self) -> float:
        """返回该行所有强物理条件中的最小可靠度。"""

        if self.glyph_crossing:
            return 0.0
        return min(
            self.top_coverage,
            self.bottom_coverage,
            self.left_coverage,
            self.right_coverage,
            self.height_ratio,
        )

    @property
    def verified(self) -> bool:
        """判断该行能否脱离文本占用独立证明结构存在。"""

        return self.reliability >= SEPARATOR_COVERAGE_THRESHOLD


@dataclass(frozen=True, slots=True)
class _SingleRowEvidence:
    """保存单物理行网格的全部外框和纵向隔断可靠度。"""

    top_coverage: float
    bottom_coverage: float
    vertical_coverages: tuple[float, ...]
    height_ratio: float
    glyph_crossing: bool

    @property
    def reliability(self) -> float:
        """返回单行网格所有不可替代物理证据中的最小值。"""

        if self.glyph_crossing or not self.vertical_coverages:
            return 0.0
        return min(
            self.top_coverage,
            self.bottom_coverage,
            min(self.vertical_coverages),
            self.height_ratio,
        )

    @property
    def verified(self) -> bool:
        """判断单行网格是否可脱离文本对齐独立验证。"""

        return self.reliability >= SEPARATOR_COVERAGE_THRESHOLD

    @property
    def confidence(self) -> float:
        """把通过八成物理硬门的覆盖率校准到 verified 分数区间。"""

        if not self.verified:
            return 0.0
        return min(
            1.0,
            0.95 + 0.05 * (self.reliability - SEPARATOR_COVERAGE_THRESHOLD) / (1.0 - SEPARATOR_COVERAGE_THRESHOLD),
        )


@dataclass(frozen=True, slots=True)
class _SingleColumnEvidence:
    """保存多行单列表单的横向边界和左右外框可靠度。"""

    horizontal_coverages: tuple[float, ...]
    left_coverage: float
    right_coverage: float
    minimum_height_ratio: float
    glyph_crossing: bool

    @property
    def reliability(self) -> float:
        """返回单列表单所有强物理条件中的最小可靠度。"""

        if self.glyph_crossing or not self.horizontal_coverages:
            return 0.0
        return min(
            min(self.horizontal_coverages),
            self.left_coverage,
            self.right_coverage,
            self.minimum_height_ratio,
        )

    @property
    def verified(self) -> bool:
        """判断单列表单是否具备不依赖文本对齐的完整线框证据。"""

        return self.reliability >= SEPARATOR_COVERAGE_THRESHOLD

    @property
    def confidence(self) -> float:
        """把通过八成物理硬门的可靠度校准到 verified 分数区间。"""

        if not self.verified:
            return 0.0
        return min(
            1.0,
            0.95 + 0.05 * (self.reliability - SEPARATOR_COVERAGE_THRESHOLD) / (1.0 - SEPARATOR_COVERAGE_THRESHOLD),
        )


class _UnionFind:
    """维护缺失内部隔断连接的原子网格并查集。"""

    def __init__(self, size: int) -> None:
        """为固定数量原子格初始化各自独立的集合。"""

        self._parents = list(range(size))

    def find(self, index: int) -> int:
        """返回原子格根节点并执行路径压缩。"""

        parent = self._parents[index]
        if parent != index:
            self._parents[index] = self.find(parent)
        return self._parents[index]

    def union(self, first: int, second: int) -> None:
        """合并两个原子格所属集合。"""

        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self._parents[second_root] = first_root


def _drawing_bbox_to_table_local(
    rule_bbox: tuple[float, float, float, float],
    table_bbox: tuple[float, float, float, float],
    angle: int,
    evidence_halo: float,
) -> tuple[float, float, float, float] | None:
    """在不扩大字符区域的前提下，把邻近外框 drawing 转到表格局部坐标。"""

    clipped = bbox_intersection(rule_bbox, table_bbox)
    if clipped is None and evidence_halo > 0:
        expanded_bbox = (
            table_bbox[0] - evidence_halo,
            table_bbox[1] - evidence_halo,
            table_bbox[2] + evidence_halo,
            table_bbox[3] + evidence_halo,
        )
        clipped = bbox_intersection(rule_bbox, expanded_bbox)
    if clipped is None:
        return None
    width = table_bbox[2] - table_bbox[0]
    height = table_bbox[3] - table_bbox[1]
    relative = (
        clipped[0] - table_bbox[0],
        clipped[1] - table_bbox[1],
        clipped[2] - table_bbox[0],
        clipped[3] - table_bbox[1],
    )
    return rotate_local_bbox(relative, width, height, angle)


def _local_rule_fragments(
    table_input: NativeTableInput,
    snap_tolerance: float,
    *,
    include_drawing: bool = True,
    include_rectangles: bool = True,
    evidence_halo: float = 0.0,
) -> list[_MergedRule]:
    """按来源裁剪 drawing/矩形，并转换为局部轴线片段。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return []
    angle = normalize_angle(table_input.angle)
    fragments: list[_MergedRule] = []
    for rule in table_input.drawing_lines if include_drawing else ():
        rule_bbox = normalize_bbox(rule.bbox)
        if rule_bbox is None:
            continue
        local_bbox = _drawing_bbox_to_table_local(
            rule_bbox,
            table_bbox,
            angle,
            evidence_halo,
        )
        if local_bbox is None:
            continue
        width = local_bbox[2] - local_bbox[0]
        height = local_bbox[3] - local_bbox[1]
        orientation = "horizontal" if width >= height else "vertical"
        if orientation == "horizontal" and width >= max(1.0, 2.0 * snap_tolerance):
            fragments.append(
                _MergedRule(
                    orientation="horizontal",
                    coordinate=(local_bbox[1] + local_bbox[3]) / 2.0,
                    start=local_bbox[0],
                    end=local_bbox[2],
                )
            )
        elif orientation == "vertical" and height >= max(1.0, 2.0 * snap_tolerance):
            fragments.append(
                _MergedRule(
                    orientation="vertical",
                    coordinate=(local_bbox[0] + local_bbox[2]) / 2.0,
                    start=local_bbox[1],
                    end=local_bbox[3],
                )
            )

    local_width, local_height = table_local_size(table_bbox, angle)
    table_area = local_width * local_height
    for rectangle in table_input.rectangles if include_rectangles else ():
        if rectangle.segment_count != 5 or not (rectangle.fill_visible or rectangle.stroke_visible):
            continue
        rectangle_bbox = normalize_bbox(rectangle.bbox)
        if rectangle_bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(rectangle_bbox, table_bbox, angle)
        if local_bbox is None:
            continue
        rect_width = local_bbox[2] - local_bbox[0]
        rect_height = local_bbox[3] - local_bbox[1]
        area_ratio = bbox_area(local_bbox) / table_area if table_area > 0 else 0.0
        if rect_width <= snap_tolerance or rect_height <= snap_tolerance or area_ratio >= 0.85:
            continue
        fragments.extend(
            [
                _MergedRule("horizontal", local_bbox[1], local_bbox[0], local_bbox[2]),
                _MergedRule("horizontal", local_bbox[3], local_bbox[0], local_bbox[2]),
                _MergedRule("vertical", local_bbox[0], local_bbox[1], local_bbox[3]),
                _MergedRule("vertical", local_bbox[2], local_bbox[1], local_bbox[3]),
            ]
        )
    return fragments


def _merge_rule_fragments(
    fragments: list[_MergedRule],
    snap_tolerance: float,
    join_gap: float,
) -> list[_MergedRule]:
    """按方向和轴坐标吸附线段，再连接小间隙共线片段。"""

    output: list[_MergedRule] = []
    for orientation in ("horizontal", "vertical"):
        oriented = [fragment for fragment in fragments if fragment.orientation == orientation]
        coordinates = cluster_positions(
            (fragment.coordinate for fragment in oriented),
            snap_tolerance,
        )
        for coordinate in coordinates:
            intervals = sorted(
                (
                    min(fragment.start, fragment.end),
                    max(fragment.start, fragment.end),
                )
                for fragment in oriented
                if abs(fragment.coordinate - coordinate) <= snap_tolerance
            )
            if not intervals:
                continue
            current_start, current_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= current_end + join_gap:
                    current_end = max(current_end, end)
                    continue
                output.append(_MergedRule(orientation, coordinate, current_start, current_end))
                current_start, current_end = start, end
            output.append(_MergedRule(orientation, coordinate, current_start, current_end))
    return output


def _repeated_long_rule_endpoints(
    rules: list[_MergedRule],
    axis_extent: float,
    snap_tolerance: float,
) -> list[float]:
    """仅保留重复长线端点或接近表格外缘的外围轨道证据。"""

    if not rules:
        return []
    maximum_length = max(rule.end - rule.start for rule in rules)
    long_rules = [rule for rule in rules if rule.end - rule.start >= 0.80 * maximum_length]
    required_support = max(2, math.ceil(0.50 * len(long_rules)))
    positions = cluster_positions(
        (endpoint for rule in long_rules for endpoint in (rule.start, rule.end)),
        snap_tolerance,
    )
    output: list[float] = []
    for position in positions:
        support = sum(
            min(
                abs(rule.start - position),
                abs(rule.end - position),
            )
            <= snap_tolerance
            for rule in long_rules
        )
        if support >= required_support or position <= 2.0 * snap_tolerance or axis_extent - position <= 2.0 * snap_tolerance:
            output.append(position)
    return output


def _infer_grid_tracks(
    rules: list[_MergedRule],
    snap_tolerance: float,
    width: float,
    height: float,
    *,
    prune_unsupported_horizontal: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    """融合物理轴线与受重复门约束的端点，恢复开放外框轨道。"""

    horizontal = [rule for rule in rules if rule.orientation == "horizontal"]
    vertical = [rule for rule in rules if rule.orientation == "vertical"]
    x_values = [rule.coordinate for rule in vertical]
    x_values.extend(
        _repeated_long_rule_endpoints(
            horizontal,
            width,
            snap_tolerance,
        )
    )
    provisional_x_tracks = cluster_positions(x_values, snap_tolerance)
    removed_horizontal_tracks: list[float] = []
    supported_horizontal = horizontal
    if prune_unsupported_horizontal:
        supported_horizontal, removed_horizontal_tracks = _filter_horizontal_track_creators(
            horizontal,
            provisional_x_tracks,
            width,
            snap_tolerance,
        )
    y_values = [rule.coordinate for rule in supported_horizontal]
    y_values.extend(
        _repeated_long_rule_endpoints(
            vertical,
            height,
            snap_tolerance,
        )
    )
    return (
        provisional_x_tracks,
        cluster_positions(y_values, snap_tolerance),
        removed_horizontal_tracks,
    )


def _filter_horizontal_track_creators(
    rules: list[_MergedRule],
    x_tracks: list[float],
    width: float,
    snap_tolerance: float,
) -> tuple[list[_MergedRule], list[float]]:
    """剔除完全缩进在单元格内、不能形成真实横向轨道的装饰短线。"""

    if len(x_tracks) < 2:
        return rules, []
    supported_coordinates: set[float] = set()
    removed_coordinates: list[float] = []
    for coordinate in cluster_positions(
        (rule.coordinate for rule in rules),
        snap_tolerance,
    ):
        coordinate_rules = [rule for rule in rules if abs(rule.coordinate - coordinate) <= snap_tolerance]
        intervals = [(rule.start, rule.end) for rule in coordinate_rules]
        full_width = covered_interval_ratio(intervals, 0.0, width)
        band_supported = False
        for left, right in zip(x_tracks, x_tracks[1:]):
            if covered_interval_ratio(intervals, left, right) < SEPARATOR_COVERAGE_THRESHOLD:
                continue
            touches_left = any(
                rule.start <= left + snap_tolerance and rule.end >= left - snap_tolerance for rule in coordinate_rules
            )
            touches_right = any(
                rule.start <= right + snap_tolerance and rule.end >= right - snap_tolerance for rule in coordinate_rules
            )
            if touches_left and touches_right:
                band_supported = True
                break
        if full_width >= SEPARATOR_COVERAGE_THRESHOLD or band_supported:
            supported_coordinates.add(coordinate)
        else:
            removed_coordinates.append(coordinate)
    return (
        [
            rule
            for rule in rules
            if any(abs(rule.coordinate - coordinate) <= snap_tolerance for coordinate in supported_coordinates)
        ],
        removed_coordinates,
    )


def _rule_indices_for_track(
    tracks_rules: list[_MergedRule],
    orientation: str,
    track: _CanonicalTrack,
    snap_tolerance: float,
) -> set[int]:
    """返回能够归属指定轨道的全部物理线索引。"""

    return {
        index
        for index, rule in enumerate(tracks_rules)
        if rule.orientation == orientation and any(abs(rule.coordinate - alias) <= snap_tolerance for alias in track.aliases)
    }


def _collapse_outer_duplicate_tracks(
    tracks: list[_CanonicalTrack],
    glyph_centers: list[float],
    threshold: float,
    extent: float,
    rules: list[_MergedRule] | None,
    orientation: str,
    separator_extent: float,
    snap_tolerance: float,
    collapse_leading_edge: bool,
) -> tuple[list[_CanonicalTrack], int, bool]:
    """折叠外缘同一物理描边产生的重复轨，并保留独立双边界。"""

    if rules is None or not orientation or len(tracks) < 2:
        return tracks, 0, False
    collapsed_count = 0
    while len(tracks) >= 2:
        candidate_indices = [len(tracks) - 2]
        if collapse_leading_edge:
            candidate_indices.insert(0, 0)
        collapse_index: int | None = None
        for index in candidate_indices:
            left, right = tracks[index], tracks[index + 1]
            if right.coordinate - left.coordinate > threshold or any(
                left.coordinate < center < right.coordinate for center in glyph_centers
            ):
                continue
            near_left_edge = right.coordinate <= max(
                threshold,
                2.0 * snap_tolerance,
            )
            near_right_edge = extent - left.coordinate <= max(
                threshold,
                2.0 * snap_tolerance,
            )
            if not (near_left_edge or near_right_edge):
                continue
            left_rules = _rule_indices_for_track(
                rules,
                orientation,
                left,
                snap_tolerance,
            )
            right_rules = _rule_indices_for_track(
                rules,
                orientation,
                right,
                snap_tolerance,
            )
            if left_rules and right_rules and left_rules.isdisjoint(right_rules):
                continue
            combined = _CanonicalTrack(
                coordinate=float(
                    statistics.median(
                        (*left.aliases, *right.aliases),
                    )
                ),
                aliases=tuple(sorted({*left.aliases, *right.aliases})),
            )
            if (
                combined.aliases[-1] - combined.aliases[0] > threshold
                or _separator_coverage_for_track(
                    rules,
                    orientation,
                    combined,
                    0.0,
                    separator_extent,
                    snap_tolerance,
                )
                < SEPARATOR_COVERAGE_THRESHOLD
            ):
                continue
            collapse_index = index
            break
        if collapse_index is None:
            break
        left, right = tracks[collapse_index : collapse_index + 2]
        aliases = tuple(sorted({*left.aliases, *right.aliases}))
        if aliases[-1] - aliases[0] > threshold:
            return tracks, collapsed_count, True
        tracks[collapse_index : collapse_index + 2] = [
            _CanonicalTrack(
                coordinate=float(statistics.median(aliases)),
                aliases=aliases,
            )
        ]
        collapsed_count += 1
    return tracks, collapsed_count, False


def _canonicalize_axis_tracks(
    positions: list[float],
    glyph_centers: list[float],
    threshold: float,
    extent: float,
    evidence_halo: float,
    minimum_track_count: int,
    *,
    rules: list[_MergedRule] | None = None,
    orientation: str = "",
    separator_extent: float = 0.0,
    snap_tolerance: float = 0.0,
    preserve_double_boundary: bool = False,
    collapse_narrow_bands: bool = True,
    collapse_leading_edge: bool = True,
) -> tuple[list[_CanonicalTrack], bool, int]:
    """吸附外缘并折叠无字形占用的窄带，同时保留全部原始别名。"""

    tracks = [
        _CanonicalTrack(
            coordinate=coordinate,
            aliases=(coordinate,),
        )
        for coordinate in positions
    ]
    for edge in (0.0, extent):
        halo_indices = [index for index, track in enumerate(tracks) if abs(track.coordinate - edge) <= evidence_halo]
        if not halo_indices:
            continue
        closest_coordinate = min(
            (tracks[index].coordinate for index in halo_indices),
            key=lambda coordinate: abs(coordinate - edge),
        )
        edge_indices = [index for index in halo_indices if abs(tracks[index].coordinate - closest_coordinate) <= snap_tolerance]
        aliases = tuple(
            sorted(
                {
                    edge,
                    *(alias for index in edge_indices for alias in tracks[index].aliases),
                }
            )
        )
        first_index = edge_indices[0]
        tracks[first_index : edge_indices[-1] + 1] = [
            _CanonicalTrack(
                coordinate=edge,
                aliases=aliases,
            )
        ]

    tracks, outer_collapse_count, outer_alias_conflict = _collapse_outer_duplicate_tracks(
        tracks,
        glyph_centers,
        threshold,
        extent,
        rules,
        orientation,
        separator_extent,
        snap_tolerance,
        collapse_leading_edge,
    )
    if outer_alias_conflict:
        return tracks, True, outer_collapse_count

    while collapse_narrow_bands and len(tracks) > minimum_track_count:
        collapse_index = next(
            (
                index
                for index, (left, right) in enumerate(zip(tracks, tracks[1:]))
                if right.coordinate - left.coordinate <= threshold
                and not any(left.coordinate < center < right.coordinate for center in glyph_centers)
                and not (
                    preserve_double_boundary
                    and rules is not None
                    and _separator_coverage_for_track(
                        rules,
                        orientation,
                        left,
                        0.0,
                        separator_extent,
                        snap_tolerance,
                    )
                    > 0.0
                    and _separator_coverage_for_track(
                        rules,
                        orientation,
                        right,
                        0.0,
                        separator_extent,
                        snap_tolerance,
                    )
                    > 0.0
                )
            ),
            None,
        )
        if collapse_index is None:
            break
        aliases = tuple(
            sorted(
                {
                    *tracks[collapse_index].aliases,
                    *tracks[collapse_index + 1].aliases,
                }
            )
        )
        if aliases[-1] - aliases[0] > threshold:
            return tracks, True, outer_collapse_count
        tracks[collapse_index : collapse_index + 2] = [
            _CanonicalTrack(
                coordinate=float(statistics.median(aliases)),
                aliases=aliases,
            )
        ]
    return tracks, False, outer_collapse_count


def _canonical_track_coordinates(
    tracks: list[_CanonicalTrack],
) -> list[float]:
    """提取规范轨道数值坐标，供网格 bbox 和稳定度计算使用。"""

    return [track.coordinate for track in tracks]


def _canonical_tracks_are_unique(
    tracks: list[_CanonicalTrack],
    rules: list[_MergedRule],
    orientation: str,
    snap_tolerance: float,
) -> bool:
    """校验每条物理线最多只能归属一个规范轨道别名集合。"""

    for rule in rules:
        if rule.orientation != orientation:
            continue
        matching_tracks = sum(
            any(abs(rule.coordinate - alias) <= snap_tolerance for alias in track.aliases) for track in tracks
        )
        if matching_tracks > 1:
            return False
    return True


def _separator_coverage_for_track(
    rules: list[_MergedRule],
    orientation: str,
    track: _CanonicalTrack,
    start: float,
    end: float,
    snap_tolerance: float,
) -> float:
    """按规范轨道全部 alias 合并计算 separator 覆盖率。"""

    intervals = [
        (rule.start, rule.end)
        for rule in rules
        if rule.orientation == orientation and any(abs(rule.coordinate - alias) <= snap_tolerance for alias in track.aliases)
    ]
    return covered_interval_ratio(intervals, start, end)


def _rect_lattice_is_repeated(
    rules: list[_MergedRule],
    x_tracks: list[float],
    y_tracks: list[float],
    snap_tolerance: float,
) -> bool:
    """要求矩形边缘在二维晶格中重复且覆盖至少八成理论边界。"""

    rows = len(y_tracks) - 1
    cols = len(x_tracks) - 1
    if rows < 2 or cols < 2:
        return False
    segment_present: list[bool] = []
    vertical_repetitions: list[int] = []
    for boundary_index, coordinate in enumerate(x_tracks):
        repetitions = 0
        for top, bottom in zip(y_tracks, y_tracks[1:]):
            present = (
                _separator_coverage(
                    rules,
                    "vertical",
                    coordinate,
                    top,
                    bottom,
                    snap_tolerance,
                )
                >= SEPARATOR_COVERAGE_THRESHOLD
            )
            segment_present.append(present)
            repetitions += int(present)
        if 0 < boundary_index < len(x_tracks) - 1:
            vertical_repetitions.append(repetitions)
    horizontal_repetitions: list[int] = []
    for boundary_index, coordinate in enumerate(y_tracks):
        repetitions = 0
        for left, right in zip(x_tracks, x_tracks[1:]):
            present = (
                _separator_coverage(
                    rules,
                    "horizontal",
                    coordinate,
                    left,
                    right,
                    snap_tolerance,
                )
                >= SEPARATOR_COVERAGE_THRESHOLD
            )
            segment_present.append(present)
            repetitions += int(present)
        if 0 < boundary_index < len(y_tracks) - 1:
            horizontal_repetitions.append(repetitions)
    coverage = sum(segment_present) / len(segment_present) if segment_present else 0.0
    return (
        coverage >= SEPARATOR_COVERAGE_THRESHOLD
        and any(count >= 2 for count in vertical_repetitions)
        and any(count >= 2 for count in horizontal_repetitions)
    )


def _separator_coverage(
    rules: list[_MergedRule],
    orientation: str,
    coordinate: float,
    start: float,
    end: float,
    snap_tolerance: float,
) -> float:
    """计算指定潜在隔断被同轴物理线段覆盖的比例。"""

    intervals = [
        (rule.start, rule.end)
        for rule in rules
        if rule.orientation == orientation and abs(rule.coordinate - coordinate) <= snap_tolerance
    ]
    return covered_interval_ratio(intervals, start, end)


def _grid_index(row: int, col: int, cols: int) -> int:
    """把二维原子格坐标转换为并查集线性索引。"""

    return row * cols + col


def _build_component_specs(
    union_find: _UnionFind,
    rows: int,
    cols: int,
    x_tracks: list[float],
    y_tracks: list[float],
) -> tuple[GridCellSpec, ...] | None:
    """把原子格连通分量转成矩形逻辑单元格，非矩形分量整体拒绝。"""

    components: dict[int, list[tuple[int, int]]] = {}
    for row in range(rows):
        for col in range(cols):
            root = union_find.find(_grid_index(row, col, cols))
            components.setdefault(root, []).append((row, col))
    specs: list[GridCellSpec] = []
    for positions in components.values():
        row_values = [item[0] for item in positions]
        col_values = [item[1] for item in positions]
        min_row, max_row = min(row_values), max(row_values)
        min_col, max_col = min(col_values), max(col_values)
        expected_size = (max_row - min_row + 1) * (max_col - min_col + 1)
        if len(positions) != expected_size:
            return None
        specs.append(
            GridCellSpec(
                row=min_row,
                col=min_col,
                rowspan=max_row - min_row + 1,
                colspan=max_col - min_col + 1,
                bbox=(
                    x_tracks[min_col],
                    y_tracks[min_row],
                    x_tracks[max_col + 1],
                    y_tracks[max_row + 1],
                ),
            )
        )
    return tuple(sorted(specs, key=lambda item: (item.row, item.col)))


def _occupied_text_rows(
    text: NativeTableText,
    y_tracks: list[float],
) -> set[int]:
    """返回至少包含一个视觉文本行中心的物理行索引。"""

    occupied_rows: set[int] = set()
    for row in text.rows:
        center_y = (row.bbox[1] + row.bbox[3]) / 2.0
        for row_index, (top, bottom) in enumerate(zip(y_tracks, y_tracks[1:])):
            if top <= center_y <= bottom:
                occupied_rows.add(row_index)
                break
    return occupied_rows


def _line_grid_row_evidence(
    rules: list[_MergedRule],
    x_tracks: list[_CanonicalTrack],
    y_tracks: list[_CanonicalTrack],
    text: NativeTableText,
    snap_tolerance: float,
    minimum_row_height: float,
    local_width: float,
) -> tuple[_PhysicalRowEvidence, ...]:
    """计算 line-grid 每个原子行的独立物理封闭证据。"""

    left = x_tracks[0].coordinate
    right = x_tracks[-1].coordinate
    evidence: list[_PhysicalRowEvidence] = []
    for row_index, (top_track, bottom_track) in enumerate(zip(y_tracks, y_tracks[1:])):
        top = top_track.coordinate
        bottom = bottom_track.coordinate
        top_coverage = _separator_coverage_for_track(
            rules,
            "horizontal",
            top_track,
            left,
            right,
            snap_tolerance,
        )
        bottom_coverage = _separator_coverage_for_track(
            rules,
            "horizontal",
            bottom_track,
            left,
            right,
            snap_tolerance,
        )
        left_coverage = _separator_coverage_for_track(
            rules,
            "vertical",
            x_tracks[0],
            top,
            bottom,
            snap_tolerance,
        )
        right_coverage = _separator_coverage_for_track(
            rules,
            "vertical",
            x_tracks[-1],
            top,
            bottom,
            snap_tolerance,
        )
        endpoint_support = min(top_coverage, bottom_coverage)
        if left <= 2.0 * snap_tolerance:
            left_coverage = max(left_coverage, endpoint_support)
        if local_width - right <= 2.0 * snap_tolerance:
            right_coverage = max(right_coverage, endpoint_support)
        glyph_crossing = any(
            glyph.bbox[1] < bottom and glyph.bbox[3] > top and not (top <= (glyph.bbox[1] + glyph.bbox[3]) / 2.0 <= bottom)
            for glyph in text.glyphs
        )
        evidence.append(
            _PhysicalRowEvidence(
                row=row_index,
                top_coverage=top_coverage,
                bottom_coverage=bottom_coverage,
                left_coverage=left_coverage,
                right_coverage=right_coverage,
                height_ratio=min(
                    1.0,
                    (bottom - top) / max(minimum_row_height, 0.1),
                ),
                glyph_crossing=glyph_crossing,
            )
        )
    return tuple(evidence)


def _single_row_line_grid_evidence(
    rules: list[_MergedRule],
    x_tracks: list[_CanonicalTrack],
    y_tracks: list[_CanonicalTrack],
    text: NativeTableText,
    snap_tolerance: float,
    minimum_row_height: float,
) -> _SingleRowEvidence:
    """校验单物理行候选的上下外框及每一条纵向边界。"""

    top_track, bottom_track = y_tracks
    top = top_track.coordinate
    bottom = bottom_track.coordinate
    left = x_tracks[0].coordinate
    right = x_tracks[-1].coordinate
    top_coverage = _separator_coverage_for_track(
        rules,
        "horizontal",
        top_track,
        left,
        right,
        snap_tolerance,
    )
    bottom_coverage = _separator_coverage_for_track(
        rules,
        "horizontal",
        bottom_track,
        left,
        right,
        snap_tolerance,
    )
    vertical_coverages = tuple(
        _separator_coverage_for_track(
            rules,
            "vertical",
            track,
            top,
            bottom,
            snap_tolerance,
        )
        for track in x_tracks
    )
    glyph_crossing = any(glyph.bbox[0] < track.coordinate < glyph.bbox[2] for glyph in text.glyphs for track in x_tracks[1:-1])
    return _SingleRowEvidence(
        top_coverage=top_coverage,
        bottom_coverage=bottom_coverage,
        vertical_coverages=vertical_coverages,
        height_ratio=min(
            1.0,
            (bottom - top) / max(minimum_row_height, 0.1),
        ),
        glyph_crossing=glyph_crossing,
    )


def _single_column_line_grid_evidence(
    rules: list[_MergedRule],
    x_tracks: list[_CanonicalTrack],
    y_tracks: list[_CanonicalTrack],
    text: NativeTableText,
    snap_tolerance: float,
    minimum_row_height: float,
) -> _SingleColumnEvidence:
    """校验多行单列表单的全部横边和左右连续外框。"""

    left_track, right_track = x_tracks
    top = y_tracks[0].coordinate
    bottom = y_tracks[-1].coordinate
    horizontal_coverages = tuple(
        _separator_coverage_for_track(
            rules,
            "horizontal",
            track,
            left_track.coordinate,
            right_track.coordinate,
            snap_tolerance,
        )
        for track in y_tracks
    )
    left_coverage = _separator_coverage_for_track(
        rules,
        "vertical",
        left_track,
        top,
        bottom,
        snap_tolerance,
    )
    right_coverage = _separator_coverage_for_track(
        rules,
        "vertical",
        right_track,
        top,
        bottom,
        snap_tolerance,
    )
    minimum_height_ratio = min(
        (
            min(
                1.0,
                (current.coordinate - previous.coordinate) / max(minimum_row_height, 0.1),
            )
            for previous, current in zip(y_tracks, y_tracks[1:])
        ),
        default=0.0,
    )
    glyph_crossing = any(glyph.bbox[1] < track.coordinate < glyph.bbox[3] for glyph in text.glyphs for track in y_tracks[1:-1])
    return _SingleColumnEvidence(
        horizontal_coverages=horizontal_coverages,
        left_coverage=left_coverage,
        right_coverage=right_coverage,
        minimum_height_ratio=minimum_height_ratio,
        glyph_crossing=glyph_crossing,
    )


def _text_grid_stability(
    text: NativeTableText,
    x_tracks: list[float],
    y_tracks: list[float],
    physically_verified_rows: set[int] | None = None,
) -> tuple[float, float]:
    """衡量视觉文本行和文本项对推断行列轨道的占用稳定性。"""

    occupied_rows = _occupied_text_rows(text, y_tracks)
    supported_rows = occupied_rows | (physically_verified_rows or set())
    occupied_cols: set[int] = set()
    for row in text.rows:
        for token in row.tokens:
            center_x = (token.bbox[0] + token.bbox[2]) / 2.0
            for col_index, (left, right) in enumerate(zip(x_tracks, x_tracks[1:])):
                if left <= center_x <= right:
                    occupied_cols.add(col_index)
                    break
    row_denominator = min(
        len(y_tracks) - 1,
        max(1, len(text.rows)),
    )
    col_denominator = min(
        len(x_tracks) - 1,
        max((len(row.tokens) for row in text.rows), default=1),
    )
    return (
        min(1.0, len(supported_rows) / row_denominator),
        min(1.0, len(occupied_cols) / col_denominator),
    )


def _physical_row_dense_baseline_pairs(
    text: NativeTableText,
    x_tracks: list[float],
    y_tracks: list[float],
) -> tuple[dict[str, object], ...]:
    """识别同一物理行带内占用集合相同的多条稠密文本基线。"""

    cols = len(x_tracks) - 1
    dense_column_count = max(2, math.ceil(0.60 * cols))
    rows_by_band: dict[int, list[tuple[int, tuple[int, ...]]]] = {row: [] for row in range(len(y_tracks) - 1)}
    for text_row in text.rows:
        center_y = (text_row.bbox[1] + text_row.bbox[3]) / 2.0
        band = next(
            (row for row, (top, bottom) in enumerate(zip(y_tracks, y_tracks[1:])) if top <= center_y <= bottom),
            None,
        )
        if band is None:
            continue
        occupied_cols: set[int] = set()
        for token in text_row.tokens:
            center_x = (token.bbox[0] + token.bbox[2]) / 2.0
            col = next(
                (index for index, (left, right) in enumerate(zip(x_tracks, x_tracks[1:])) if left <= center_x <= right),
                None,
            )
            if col is not None:
                occupied_cols.add(col)
        rows_by_band[band].append(
            (
                text_row.row_index,
                tuple(sorted(occupied_cols)),
            )
        )

    ambiguous_pairs: list[dict[str, object]] = []
    for band, entries in rows_by_band.items():
        nonempty_entries = [entry for entry in entries if entry[1]]
        if (
            len(nonempty_entries) < 2
            or len(nonempty_entries[0][1]) < dense_column_count
            or any(entry[1] != nonempty_entries[0][1] for entry in nonempty_entries[1:])
        ):
            continue
        for previous, current in zip(nonempty_entries, nonempty_entries[1:]):
            ambiguous_pairs.append(
                {
                    "physical_row": band,
                    "visual_rows": [previous[0], current[0]],
                    "occupied_cols": list(previous[1]),
                }
            )
    return tuple(ambiguous_pairs)


def _looks_like_single_column_tracks(
    x_tracks: list[float],
    local_width: float,
    edge_band: float,
) -> bool:
    """判断初始 X 轨是否全部属于单列表单的左右外缘。"""

    return (
        len(x_tracks) >= 2
        and local_width > 0
        and all(min(abs(track), abs(local_width - track)) <= edge_band for track in x_tracks)
    )


@dataclass(frozen=True, slots=True)
class _VectorTracks:
    """保存已通过别名、尺寸和物理行数校验的规范轨道。"""

    snap_tolerance: float
    local_width: float
    rules: list[_MergedRule]
    canonical_x_tracks: list[_CanonicalTrack]
    canonical_y_tracks: list[_CanonicalTrack]
    x_tracks: list[float]
    y_tracks: list[float]
    narrow_empty_threshold: float
    is_line_grid: bool
    is_single_row_shape: bool
    is_single_column_shape: bool
    rows: int
    cols: int


@dataclass(frozen=True, slots=True)
class _VectorTopology:
    """保存隔断连接后的逻辑单元格及独立物理证据。"""

    specs: tuple[GridCellSpec, ...]
    separator_decisions: list[float]
    ambiguous_ratio: float
    alias_separator_recoveries: int
    y_alias_separator_recoveries: int
    alias_affected_rows: set[int]
    single_row_evidence: _SingleRowEvidence | None
    single_column_evidence: _SingleColumnEvidence | None


def _reject_vector_candidate(diagnostics: dict[str, Any] | None, gate: str) -> None:
    """记录当前假设的首个拒绝门，供各显式阶段保留统一诊断行为。"""

    if diagnostics is not None:
        diagnostics["first_rejection_gate"] = gate
    return None


def _build_vector_tracks(
    table_input: NativeTableInput,
    text: NativeTableText,
    *,
    include_drawing: bool,
    include_rectangles: bool,
    prune_unsupported_horizontal: bool,
    diagnostics: dict[str, Any] | None,
) -> _VectorTracks | None:
    """构造并规范化轨道，保持 halo、别名及物理行数的原有拒绝顺序。"""

    snap_tolerance = clamp(
        0.08 * text.median_glyph_height,
        0.5,
        2.5,
    )
    join_gap = clamp(
        0.40 * text.median_glyph_height,
        2.0,
        8.0,
    )
    configured_evidence_halo = (
        clamp(
            0.25 * text.median_glyph_height,
            1.0,
            3.0,
        )
        if include_drawing and not include_rectangles
        else 0.0
    )
    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return _reject_vector_candidate(diagnostics, "table_geometry")
    local_width, local_height = table_local_size(
        table_bbox,
        normalize_angle(table_input.angle),
    )
    exact_fragments = _local_rule_fragments(
        table_input,
        snap_tolerance,
        include_drawing=include_drawing,
        include_rectangles=include_rectangles,
        evidence_halo=0.0,
    )
    if not exact_fragments or len(exact_fragments) > MAX_PRIMITIVES_PER_TABLE:
        return _reject_vector_candidate(diagnostics, "raw_fragments")
    exact_rules = _merge_rule_fragments(
        exact_fragments,
        snap_tolerance,
        join_gap,
    )
    exact_x_tracks, exact_y_tracks, _removed = _infer_grid_tracks(
        exact_rules,
        snap_tolerance,
        local_width,
        local_height,
    )
    # 多行表格保持原始 bbox 裁剪；halo 只服务可能退化为单物理行的
    # 边界片段，避免吸入相邻行或页外端点改变既有拓扑。
    single_column_halo_hint = (
        include_drawing
        and not include_rectangles
        and len(exact_y_tracks) >= 3
        and _looks_like_single_column_tracks(
            exact_x_tracks,
            local_width,
            max(3.0 * configured_evidence_halo, 2.0),
        )
    )
    evidence_halo = (
        configured_evidence_halo
        if configured_evidence_halo > 0
        and (
            single_column_halo_hint
            or len(exact_y_tracks) == 2
            or (
                len(exact_y_tracks) == 3
                and min(
                    current - previous
                    for previous, current in zip(
                        exact_y_tracks,
                        exact_y_tracks[1:],
                    )
                )
                <= 0.75 * text.median_glyph_height
            )
        )
        else 0.0
    )
    if evidence_halo > 0:
        raw_fragments = _local_rule_fragments(
            table_input,
            snap_tolerance,
            include_drawing=include_drawing,
            include_rectangles=include_rectangles,
            evidence_halo=evidence_halo,
        )
        rules = _merge_rule_fragments(
            raw_fragments,
            snap_tolerance,
            join_gap,
        )
    else:
        raw_fragments = exact_fragments
        rules = exact_rules
    if diagnostics is not None:
        diagnostics["raw_fragment_count"] = len(raw_fragments)
    inferred_x_tracks, inferred_y_tracks, removed_horizontal_tracks = _infer_grid_tracks(
        rules,
        snap_tolerance,
        local_width,
        local_height,
        prune_unsupported_horizontal=prune_unsupported_horizontal,
    )
    if diagnostics is not None:
        diagnostics["track_hypothesis"] = "supported" if prune_unsupported_horizontal else "raw"
        diagnostics["evidence_halo"] = evidence_halo
        diagnostics["removed_horizontal_tracks"] = removed_horizontal_tracks
        diagnostics["inferred_tracks"] = {
            "x": len(inferred_x_tracks),
            "y": len(inferred_y_tracks),
        }
    if (
        include_rectangles
        and not include_drawing
        and not _rect_lattice_is_repeated(
            rules,
            inferred_x_tracks,
            inferred_y_tracks,
            snap_tolerance,
        )
    ):
        return _reject_vector_candidate(diagnostics, "rect_lattice")
    line_widths = [rule.width for rule in table_input.drawing_lines if rule.width > 0]
    median_line_width = float(statistics.median(line_widths)) if line_widths else 0.0
    narrow_empty_threshold = max(
        0.75 * text.median_glyph_height,
        3.0 * median_line_width,
    )
    glyph_centers_x = [(glyph.bbox[0] + glyph.bbox[2]) / 2.0 for glyph in text.glyphs]
    glyph_centers_y = [(glyph.bbox[1] + glyph.bbox[3]) / 2.0 for glyph in text.glyphs]
    canonical_x_tracks, x_alias_conflict, outer_x_collapses = _canonicalize_axis_tracks(
        inferred_x_tracks,
        glyph_centers_x,
        narrow_empty_threshold,
        local_width,
        evidence_halo,
        3,
        rules=rules,
        orientation="vertical",
        separator_extent=local_height,
        snap_tolerance=snap_tolerance,
    )
    canonical_y_tracks, y_alias_conflict, outer_y_collapses = _canonicalize_axis_tracks(
        inferred_y_tracks,
        glyph_centers_y,
        narrow_empty_threshold,
        local_height,
        evidence_halo,
        2,
        rules=rules,
        orientation="horizontal",
        separator_extent=local_width,
        snap_tolerance=snap_tolerance,
        preserve_double_boundary=True,
        collapse_narrow_bands=len(exact_y_tracks) <= 3,
        collapse_leading_edge=False,
    )
    if (
        x_alias_conflict
        or y_alias_conflict
        or not _canonical_tracks_are_unique(
            canonical_x_tracks,
            rules,
            "vertical",
            snap_tolerance,
        )
        or not _canonical_tracks_are_unique(
            canonical_y_tracks,
            rules,
            "horizontal",
            snap_tolerance,
        )
    ):
        return _reject_vector_candidate(diagnostics, "canonical_alias")
    x_tracks = _canonical_track_coordinates(canonical_x_tracks)
    y_tracks = _canonical_track_coordinates(canonical_y_tracks)
    if diagnostics is not None:
        diagnostics["canonical_tracks"] = [
            {
                "coordinate": track.coordinate,
                "aliases": list(track.aliases),
            }
            for track in canonical_x_tracks
        ]
        diagnostics["canonical_y_tracks"] = [
            {
                "coordinate": track.coordinate,
                "aliases": list(track.aliases),
            }
            for track in canonical_y_tracks
        ]
        diagnostics["outer_track_collapses"] = {
            "x": outer_x_collapses,
            "y": outer_y_collapses,
        }
    if any(
        right - left <= narrow_empty_threshold and not any(left < center < right for center in glyph_centers_x)
        for left, right in zip(x_tracks, x_tracks[1:])
    ):
        return _reject_vector_candidate(diagnostics, "remaining_narrow_track")
    if any(
        bottom_track.coordinate - top_track.coordinate <= narrow_empty_threshold
        and not any(top_track.coordinate < center < bottom_track.coordinate for center in glyph_centers_y)
        and _separator_coverage_for_track(
            rules,
            "horizontal",
            top_track,
            x_tracks[0],
            x_tracks[-1],
            snap_tolerance,
        )
        >= SEPARATOR_COVERAGE_THRESHOLD
        and _separator_coverage_for_track(
            rules,
            "horizontal",
            bottom_track,
            x_tracks[0],
            x_tracks[-1],
            snap_tolerance,
        )
        >= SEPARATOR_COVERAGE_THRESHOLD
        for top_track, bottom_track in zip(
            canonical_y_tracks,
            canonical_y_tracks[1:],
        )
    ):
        return _reject_vector_candidate(diagnostics, "remaining_narrow_track")
    is_line_grid = include_drawing and not include_rectangles
    is_single_row_shape = is_line_grid and len(y_tracks) == 2 and len(x_tracks) >= 3
    is_single_column_shape = is_line_grid and len(x_tracks) == 2 and len(y_tracks) >= 3
    if (
        len(x_tracks) < (2 if is_single_column_shape else 3)
        or len(y_tracks) < (2 if is_single_row_shape else 3)
        or len(x_tracks) > MAX_TRACKS_PER_AXIS
        or len(y_tracks) > MAX_TRACKS_PER_AXIS
    ):
        return _reject_vector_candidate(diagnostics, "track_count")
    rows = len(y_tracks) - 1
    cols = len(x_tracks) - 1
    if diagnostics is not None:
        diagnostics["grid"] = {"rows": rows, "cols": cols}
    if rows * cols > MAX_ATOMIC_CELLS:
        return _reject_vector_candidate(diagnostics, "atomic_cell_limit")
    dense_baseline_pairs = (
        _physical_row_dense_baseline_pairs(
            text,
            x_tracks,
            y_tracks,
        )
        if is_line_grid
        else ()
    )
    if diagnostics is not None:
        diagnostics["physical_row_dense_baseline_pairs"] = list(dense_baseline_pairs)
    if dense_baseline_pairs:
        return _reject_vector_candidate(diagnostics, "physical_row_undercount")

    return _VectorTracks(
        snap_tolerance=snap_tolerance,
        local_width=local_width,
        rules=rules,
        canonical_x_tracks=canonical_x_tracks,
        canonical_y_tracks=canonical_y_tracks,
        x_tracks=x_tracks,
        y_tracks=y_tracks,
        narrow_empty_threshold=narrow_empty_threshold,
        is_line_grid=is_line_grid,
        is_single_row_shape=is_single_row_shape,
        is_single_column_shape=is_single_column_shape,
        rows=rows,
        cols=cols,
    )


def _build_vector_topology(
    tracks: _VectorTracks,
    text: NativeTableText,
    diagnostics: dict[str, Any] | None,
) -> _VectorTopology | None:
    """连接原子格并验证矩形拓扑，保留单行和单列的独立物理证据。"""

    snap_tolerance = tracks.snap_tolerance
    rules = tracks.rules
    canonical_x_tracks = tracks.canonical_x_tracks
    canonical_y_tracks = tracks.canonical_y_tracks
    x_tracks = tracks.x_tracks
    y_tracks = tracks.y_tracks
    narrow_empty_threshold = tracks.narrow_empty_threshold
    is_single_row_shape = tracks.is_single_row_shape
    is_single_column_shape = tracks.is_single_column_shape
    rows = tracks.rows
    cols = tracks.cols

    union_find = _UnionFind(rows * cols)
    separator_decisions: list[float] = []
    ambiguous_separator_count = 0
    alias_separator_recoveries = 0
    y_alias_separator_recoveries = 0
    alias_affected_rows: set[int] = set()
    for row in range(rows):
        for boundary_index in range(1, len(canonical_x_tracks) - 1):
            track = canonical_x_tracks[boundary_index]
            strict_coverage = _separator_coverage(
                rules,
                "vertical",
                track.coordinate,
                y_tracks[row],
                y_tracks[row + 1],
                snap_tolerance,
            )
            coverage = _separator_coverage_for_track(
                rules,
                "vertical",
                track,
                y_tracks[row],
                y_tracks[row + 1],
                snap_tolerance,
            )
            if (
                len(track.aliases) > 1
                and strict_coverage <= 1.0 - SEPARATOR_COVERAGE_THRESHOLD
                and coverage >= SEPARATOR_COVERAGE_THRESHOLD
            ):
                alias_separator_recoveries += 1
                alias_affected_rows.add(row)
            separator_decisions.append(max(coverage, 1.0 - coverage))
            if coverage <= 1.0 - SEPARATOR_COVERAGE_THRESHOLD:
                union_find.union(
                    _grid_index(row, boundary_index - 1, cols),
                    _grid_index(row, boundary_index, cols),
                )
            elif coverage < SEPARATOR_COVERAGE_THRESHOLD:
                ambiguous_separator_count += 1
    for boundary_index in range(1, len(canonical_y_tracks) - 1):
        track = canonical_y_tracks[boundary_index]
        for col in range(cols):
            strict_coverage = _separator_coverage(
                rules,
                "horizontal",
                track.coordinate,
                x_tracks[col],
                x_tracks[col + 1],
                snap_tolerance,
            )
            coverage = _separator_coverage_for_track(
                rules,
                "horizontal",
                track,
                x_tracks[col],
                x_tracks[col + 1],
                snap_tolerance,
            )
            if (
                len(track.aliases) > 1
                and strict_coverage <= 1.0 - SEPARATOR_COVERAGE_THRESHOLD
                and coverage >= SEPARATOR_COVERAGE_THRESHOLD
            ):
                y_alias_separator_recoveries += 1
                if boundary_index > 0:
                    alias_affected_rows.add(boundary_index - 1)
                if boundary_index < rows:
                    alias_affected_rows.add(boundary_index)
            separator_decisions.append(max(coverage, 1.0 - coverage))
            if coverage <= 1.0 - SEPARATOR_COVERAGE_THRESHOLD:
                union_find.union(
                    _grid_index(boundary_index - 1, col, cols),
                    _grid_index(boundary_index, col, cols),
                )
            elif coverage < SEPARATOR_COVERAGE_THRESHOLD:
                ambiguous_separator_count += 1

    ambiguous_ratio = ambiguous_separator_count / len(separator_decisions) if separator_decisions else 0.0
    if diagnostics is not None:
        diagnostics["grid"] = {"rows": rows, "cols": cols}
        diagnostics["ambiguous_separator_ratio"] = ambiguous_ratio
        diagnostics["alias_separator_recoveries"] = alias_separator_recoveries
        diagnostics["y_alias_separator_recoveries"] = y_alias_separator_recoveries
        diagnostics["alias_affected_rows"] = sorted(alias_affected_rows)
    if ambiguous_ratio > 0.05:
        return _reject_vector_candidate(diagnostics, "ambiguous_separator")

    single_row_evidence: _SingleRowEvidence | None = None
    if is_single_row_shape:
        single_row_evidence = _single_row_line_grid_evidence(
            rules,
            canonical_x_tracks,
            canonical_y_tracks,
            text,
            snap_tolerance,
            narrow_empty_threshold,
        )
        if diagnostics is not None:
            diagnostics["single_row_evidence"] = {
                "reliability": single_row_evidence.reliability,
                "confidence": single_row_evidence.confidence,
                "verified": single_row_evidence.verified,
                "top": single_row_evidence.top_coverage,
                "bottom": single_row_evidence.bottom_coverage,
                "vertical": list(single_row_evidence.vertical_coverages),
                "height_ratio": single_row_evidence.height_ratio,
                "glyph_crossing": single_row_evidence.glyph_crossing,
            }
        if not single_row_evidence.verified:
            return _reject_vector_candidate(diagnostics, "single_row_physical_evidence")

    single_column_evidence: _SingleColumnEvidence | None = None
    if is_single_column_shape:
        single_column_evidence = _single_column_line_grid_evidence(
            rules,
            canonical_x_tracks,
            canonical_y_tracks,
            text,
            snap_tolerance,
            narrow_empty_threshold,
        )
        if diagnostics is not None:
            diagnostics["single_column_evidence"] = {
                "reliability": single_column_evidence.reliability,
                "confidence": single_column_evidence.confidence,
                "verified": single_column_evidence.verified,
                "horizontal": list(single_column_evidence.horizontal_coverages),
                "left": single_column_evidence.left_coverage,
                "right": single_column_evidence.right_coverage,
                "minimum_height_ratio": (single_column_evidence.minimum_height_ratio),
                "glyph_crossing": single_column_evidence.glyph_crossing,
            }
        if not single_column_evidence.verified:
            return _reject_vector_candidate(diagnostics, "single_column_physical_evidence")

    specs = _build_component_specs(
        union_find,
        rows,
        cols,
        x_tracks,
        y_tracks,
    )
    if specs is None:
        return _reject_vector_candidate(diagnostics, "nonrectangular_topology")
    maximum_row_cells = max(sum(spec.row <= row_index < spec.row + spec.rowspan for spec in specs) for row_index in range(rows))
    maximum_col_cells = max(sum(spec.col <= col_index < spec.col + spec.colspan for spec in specs) for col_index in range(cols))
    if (maximum_row_cells < 2 and not is_single_column_shape) or (maximum_col_cells < 2 and not is_single_row_shape):
        return _reject_vector_candidate(diagnostics, "degenerate_grid")
    return _VectorTopology(
        specs=specs,
        separator_decisions=separator_decisions,
        ambiguous_ratio=ambiguous_ratio,
        alias_separator_recoveries=alias_separator_recoveries,
        y_alias_separator_recoveries=y_alias_separator_recoveries,
        alias_affected_rows=alias_affected_rows,
        single_row_evidence=single_row_evidence,
        single_column_evidence=single_column_evidence,
    )


def _materialize_vector_candidate(
    tracks: _VectorTracks,
    topology: _VectorTopology,
    text: NativeTableText,
    evidence_label: str,
    diagnostics: dict[str, Any] | None,
) -> NativeTableCandidate | None:
    """将文本落格并评分，按原顺序执行完整性与空行发布门。"""

    snap_tolerance = tracks.snap_tolerance
    local_width = tracks.local_width
    rules = tracks.rules
    canonical_x_tracks = tracks.canonical_x_tracks
    canonical_y_tracks = tracks.canonical_y_tracks
    x_tracks = tracks.x_tracks
    y_tracks = tracks.y_tracks
    narrow_empty_threshold = tracks.narrow_empty_threshold
    is_line_grid = tracks.is_line_grid
    is_single_row_shape = tracks.is_single_row_shape
    is_single_column_shape = tracks.is_single_column_shape
    rows = tracks.rows
    cols = tracks.cols

    specs = topology.specs
    separator_decisions = topology.separator_decisions
    ambiguous_ratio = topology.ambiguous_ratio
    alias_separator_recoveries = topology.alias_separator_recoveries
    y_alias_separator_recoveries = topology.y_alias_separator_recoveries
    alias_affected_rows = topology.alias_affected_rows
    single_row_evidence = topology.single_row_evidence
    single_column_evidence = topology.single_column_evidence

    decisiveness = float(statistics.mean(separator_decisions)) if separator_decisions else 1.0
    if single_row_evidence is not None:
        decisiveness = max(
            decisiveness,
            single_row_evidence.confidence,
        )
    if single_column_evidence is not None:
        decisiveness = max(
            decisiveness,
            single_column_evidence.confidence,
        )
    evidence_ratio = min(1.0, len(rules) / max(1, rows + cols))
    structure_support = min(
        decisiveness,
        evidence_ratio,
        1.0 - ambiguous_ratio,
        (single_row_evidence.confidence if single_row_evidence is not None else 1.0),
        (single_column_evidence.confidence if single_column_evidence is not None else 1.0),
    )
    occupied_rows = _occupied_text_rows(text, y_tracks)
    line_row_evidence: tuple[_PhysicalRowEvidence, ...] = ()
    physically_verified_rows: set[int] = set()
    if is_line_grid:
        line_row_evidence = _line_grid_row_evidence(
            rules,
            canonical_x_tracks,
            canonical_y_tracks,
            text,
            snap_tolerance,
            narrow_empty_threshold,
            local_width,
        )
        physically_verified_rows = {evidence.row for evidence in line_row_evidence if evidence.verified}
        if diagnostics is not None:
            diagnostics["physical_rows"] = [
                {
                    "row": evidence.row,
                    "reliability": evidence.reliability,
                    "verified": evidence.verified,
                    "top": evidence.top_coverage,
                    "bottom": evidence.bottom_coverage,
                    "left": evidence.left_coverage,
                    "right": evidence.right_coverage,
                    "height_ratio": evidence.height_ratio,
                    "glyph_crossing": evidence.glyph_crossing,
                }
                for evidence in line_row_evidence
            ]
    row_stability, column_stability = _text_grid_stability(
        text,
        x_tracks,
        y_tracks,
        physically_verified_rows=(physically_verified_rows if is_line_grid else None),
    )
    collapsed_track_count = sum(len(track.aliases) > 1 for track in canonical_x_tracks)
    collapsed_y_track_count = sum(len(track.aliases) > 1 for track in canonical_y_tracks)
    maximum_alias_span = max(
        (
            track.aliases[-1] - track.aliases[0]
            for track in (*canonical_x_tracks, *canonical_y_tracks)
            if len(track.aliases) > 1
        ),
        default=0.0,
    )
    potential_blank_rows = sorted(set(range(rows)) - occupied_rows)
    candidate = build_candidate(
        source="vector_grid",
        rows=rows,
        cols=cols,
        specs=specs,
        text=text,
        structure_support=structure_support,
        row_stability=row_stability,
        column_stability=column_stability,
        issues=(
            f"evidence={evidence_label}",
            f"ambiguous_separator_ratio={ambiguous_ratio:.4f}",
            f"collapsed_x_tracks={collapsed_track_count}",
            f"collapsed_y_tracks={collapsed_y_track_count}",
            f"alias_max_span={maximum_alias_span:.4f}",
            f"alias_separator_recoveries={alias_separator_recoveries}",
            f"y_alias_separator_recoveries={y_alias_separator_recoveries}",
            f"single_row_line_grid={str(is_single_row_shape).lower()}",
            f"single_column_line_grid={str(is_single_column_shape).lower()}",
            "single_row_reliability="
            + (f"{single_row_evidence.reliability:.4f}" if single_row_evidence is not None else "n/a"),
            "single_row_confidence=" + (f"{single_row_evidence.confidence:.4f}" if single_row_evidence is not None else "n/a"),
            "single_column_reliability="
            + (f"{single_column_evidence.reliability:.4f}" if single_column_evidence is not None else "n/a"),
            "single_column_confidence="
            + (f"{single_column_evidence.confidence:.4f}" if single_column_evidence is not None else "n/a"),
            "physical_blank_rows=" + ",".join(str(row) for row in potential_blank_rows if row in physically_verified_rows),
        ),
        allow_single_row=is_single_row_shape,
        allow_single_column=is_single_column_shape,
        use_grid_index=True,
        diagnostics=diagnostics,
    )
    if candidate is None:
        candidate_gate = (
            str(diagnostics.get("candidate_rejection_gate"))
            if diagnostics is not None and diagnostics.get("candidate_rejection_gate")
            else "candidate_hard_gate"
        )
        return _reject_vector_candidate(diagnostics, candidate_gate)
    if is_single_row_shape and (candidate.text_capture < 1.0 or candidate.order_consistency < 1.0):
        return _reject_vector_candidate(diagnostics, "single_row_text_integrity")
    if is_single_column_shape and (candidate.text_capture < 1.0 or candidate.order_consistency < 1.0):
        return _reject_vector_candidate(diagnostics, "single_column_text_integrity")
    row_content_support = [
        sum(bool(cell.content.strip()) for cell in candidate.cells if cell.row <= row_index < cell.row + cell.rowspan)
        for row_index in range(candidate.rows)
    ]
    empty_rows = {row_index for row_index, support in enumerate(row_content_support) if support == 0}
    if empty_rows and len(text.rows) >= 2:
        if (
            not is_line_grid
            or not empty_rows.isdisjoint(alias_affected_rows)
            or not empty_rows.issubset(physically_verified_rows)
        ):
            if diagnostics is not None:
                diagnostics["empty_rows"] = sorted(empty_rows)
            return _reject_vector_candidate(diagnostics, "empty_row")
    if diagnostics is not None:
        diagnostics["first_rejection_gate"] = None
        diagnostics["score"] = candidate.score
        diagnostics["empty_rows"] = sorted(empty_rows)
    return candidate


def _build_vector_candidate(
    table_input: NativeTableInput,
    text: NativeTableText,
    *,
    include_drawing: bool,
    include_rectangles: bool,
    evidence_label: str,
    prune_unsupported_horizontal: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> NativeTableCandidate | None:
    """按轨道、拓扑、文本落格及评分的固定顺序构造矢量候选。"""

    if diagnostics is not None:
        diagnostics["evidence"] = evidence_label
    tracks = _build_vector_tracks(
        table_input,
        text,
        include_drawing=include_drawing,
        include_rectangles=include_rectangles,
        prune_unsupported_horizontal=prune_unsupported_horizontal,
        diagnostics=diagnostics,
    )
    if tracks is None:
        return None
    topology = _build_vector_topology(tracks, text, diagnostics)
    if topology is None:
        return None
    return _materialize_vector_candidate(tracks, topology, text, evidence_label, diagnostics)


def build_vector_candidates(
    table_input: NativeTableInput,
    text: NativeTableText,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[NativeTableCandidate]:
    """分别从 drawing 中心线和矩形晶格生成矢量网格候选。"""

    candidates: list[NativeTableCandidate] = []
    raw_line_diagnostics: dict[str, Any] | None = {} if diagnostics is not None else None
    line_candidate = _build_vector_candidate(
        table_input,
        text,
        include_drawing=True,
        include_rectangles=False,
        evidence_label="line_grid",
        diagnostics=raw_line_diagnostics,
    )
    line_hypotheses = [raw_line_diagnostics] if raw_line_diagnostics is not None else []
    selected_line_diagnostics = raw_line_diagnostics
    if line_candidate is None and len(line_hypotheses) < MAX_TRACK_HYPOTHESES:
        supported_line_diagnostics: dict[str, Any] | None = {} if diagnostics is not None else None
        supported_line_candidate = _build_vector_candidate(
            table_input,
            text,
            include_drawing=True,
            include_rectangles=False,
            evidence_label="line_grid",
            prune_unsupported_horizontal=True,
            diagnostics=supported_line_diagnostics,
        )
        if supported_line_diagnostics is not None:
            removed_tracks = supported_line_diagnostics.get(
                "removed_horizontal_tracks",
                [],
            )
            if removed_tracks:
                line_hypotheses.append(supported_line_diagnostics)
                selected_line_diagnostics = supported_line_diagnostics
        if supported_line_candidate is not None:
            line_candidate = supported_line_candidate
            selected_line_diagnostics = supported_line_diagnostics
    if diagnostics is not None and selected_line_diagnostics is not None:
        line_record = dict(selected_line_diagnostics)
        line_record["track_hypotheses"] = [dict(hypothesis) for hypothesis in line_hypotheses if hypothesis is not None]
        diagnostics.append(line_record)
    if line_candidate is not None:
        candidates.append(line_candidate)
    rect_diagnostics: dict[str, Any] | None = {} if diagnostics is not None else None
    rect_candidate = _build_vector_candidate(
        table_input,
        text,
        include_drawing=False,
        include_rectangles=True,
        evidence_label="rect_grid",
        diagnostics=rect_diagnostics,
    )
    if diagnostics is not None and rect_diagnostics is not None:
        diagnostics.append(rect_diagnostics)
    if rect_candidate is not None:
        candidates.append(rect_candidate)
    return candidates


def diagnose_vector_candidate_builds(
    table_input: NativeTableInput,
    text: NativeTableText,
) -> tuple[dict[str, Any], ...]:
    """返回 line/rect 假设真实首个拒绝门和物理证据。"""

    diagnostics: list[dict[str, Any]] = []
    build_vector_candidates(
        table_input,
        text,
        diagnostics=diagnostics,
    )
    return tuple(diagnostics)


__all__ = [
    "MAX_ATOMIC_CELLS",
    "MAX_PRIMITIVES_PER_TABLE",
    "MAX_TRACKS_PER_AXIS",
    "build_vector_candidates",
]
