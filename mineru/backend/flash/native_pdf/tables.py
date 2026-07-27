# Copyright (c) Opendatalab. All rights reserved.

"""检测、投影并物化 Flash 原生 PDF 表格。"""

from __future__ import annotations

import re
import statistics
from typing import Any

from loguru import logger

from mineru.backend.hybrid.table_text import project_pdf_table_text
from mineru.types import BBox
from mineru.utils.pdf_document import PDFPathInfo

from .models import (
    _Fragment,
    _LineItem,
    _LocalAxisLine,
    _PageSource,
    _TableCandidate,
    _VisualRow,
)
from .geometry import (
    _bbox_area,
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_overlap_in_smaller,
    _bbox_union,
    _bbox_union_many,
    _expand_bbox,
    _point_in_bbox,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from .line_layout import _line_effective_height
from .line_merging import _same_baseline_geometry


_TABLE_CAPTION_RE = re.compile(
    r"^(?:table|tab\.?|表格?)[\s:.–—-]*(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)\b(?P<suffix>.*)$",
    re.IGNORECASE,
)


_TABLE_NOTE_RE = re.compile(
    r"^(?:notes?|sources?)\b|^(?:注释?|说明)\s*[:：]?|^for\s+[*†‡]"
    r"|^(?:\d+|[*†‡])\s+\S"
    r"|^(?:[*†‡]|[a-z]|p|t|ns|na)\s+(?:indicates?|denotes?|rainfall\b|total\b|low\b|for\b)",
    re.IGNORECASE,
)


_TABLE_SPLIT_NUMBER_RE = re.compile(
    r"^(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)[.:：]?$",
    re.IGNORECASE,
)


_FILLED_GRID_MIN_PAGE_AREA_RATIO = 0.005
_FILLED_GRID_MAX_PAGE_AREA_RATIO = 0.25
_FILLED_GRID_MIN_PAGE_WIDTH_RATIO = 0.12
_FILLED_GRID_MIN_PAGE_HEIGHT_RATIO = 0.03


def _detect_table_candidates(
    source: _PageSource,
    excluded_bboxes: list[BBox] | None = None,
) -> list[_TableCandidate]:
    """按文本方向融合横线区间、规则文本和填充行带，生成表格候选。"""

    excluded_bboxes = excluded_bboxes or []
    filled_grid_candidates = _detect_filled_grid_table_candidates(
        source.path_infos,
        source.page_size,
        excluded_bboxes,
    )
    filled_grid_bboxes = [candidate.bbox for candidate in filled_grid_candidates]
    rule_candidates: list[_TableCandidate] = []
    angles = sorted({line.angle for line in source.lines})
    for angle in angles:
        angle_lines = [line for line in source.lines if line.angle == angle]
        if not angle_lines:
            continue
        fragments = _build_fragments(angle_lines, source.page_size)
        if not fragments:
            continue
        median_height = _median_fragment_height(fragments)
        rows = _cluster_fragment_rows(fragments, median_height)
        local_axis_lines = _transform_axis_lines(
            source.drawing_lines,
            source.page_size,
            angle,
        )
        local_excluded_bboxes = [
            _expand_bbox(
                _rotate_bbox_to_upright(bbox, source.page_size, angle),
                2.5 * median_height,
            )
            for bbox in [*excluded_bboxes, *filled_grid_bboxes]
        ]
        rule_candidates.extend(
            _build_rule_table_candidates(
                rows,
                angle_lines,
                source.page_size,
                angle,
                median_height,
                local_axis_lines,
                path_infos=source.path_infos,
                excluded_bboxes=local_excluded_bboxes,
            )
        )
    merged_rule_candidates = [
        candidate
        for candidate in _merge_table_candidates(rule_candidates)
        if not any(
            _bbox_overlap_in_smaller(candidate.bbox, filled_bbox) >= 0.2
            for filled_bbox in filled_grid_bboxes
        )
    ]
    return sorted(
        [*filled_grid_candidates, *merged_rule_candidates],
        key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]),
    )


def _detect_filled_grid_table_candidates(
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    excluded_bboxes: list[BBox] | None = None,
) -> list[_TableCandidate]:
    """仅按根层填充矩形的嵌套行带识别精确表格外框。"""

    page_width, page_height = page_size
    page_area = page_width * page_height
    if page_area <= 0:
        return []
    excluded_bboxes = excluded_bboxes or []
    rectangles = [
        path_info.bbox
        for path_info in path_infos
        if path_info.form_depth == 0
        and path_info.fill_visible
        and path_info.segment_count == 5
        and _bbox_area(path_info.bbox) > 0
    ]
    evidence: list[tuple[_TableCandidate, int, float, float]] = []
    for outer_bbox in rectangles:
        outer_width = outer_bbox[2] - outer_bbox[0]
        outer_height = outer_bbox[3] - outer_bbox[1]
        outer_area = _bbox_area(outer_bbox)
        if not (
            _FILLED_GRID_MIN_PAGE_AREA_RATIO
            <= outer_area / page_area
            <= _FILLED_GRID_MAX_PAGE_AREA_RATIO
            and outer_width >= _FILLED_GRID_MIN_PAGE_WIDTH_RATIO * page_width
            and outer_height >= _FILLED_GRID_MIN_PAGE_HEIGHT_RATIO * page_height
        ):
            continue
        if any(
            _bbox_overlap_in_smaller(outer_bbox, excluded_bbox) >= 0.5
            for excluded_bbox in excluded_bboxes
        ):
            continue

        cells = _select_maximal_filled_grid_cells(rectangles, outer_bbox)
        bands = _group_filled_grid_cells_into_bands(cells, outer_bbox)
        accepted_bands = [
            band
            for band in bands
            if _filled_grid_band_covers_outer_width(band, outer_bbox)
        ]
        if len(accepted_bands) < 4:
            continue
        accepted_bands.sort(
            key=lambda band: (
                min(cell[1] for cell in band),
                min(cell[0] for cell in band),
            )
        )
        covered_height = sum(
            max(cell[3] for cell in band) - min(cell[1] for cell in band)
            for band in accepted_bands
        )
        vertical_coverage = covered_height / outer_height
        if vertical_coverage < 0.75:
            continue
        if any(
            min(cell[1] for cell in next_band)
            - max(cell[3] for cell in current_band)
            > 0.05 * outer_height
            for current_band, next_band in zip(
                accepted_bands,
                accepted_bands[1:],
            )
        ):
            continue
        candidate = _TableCandidate(
            bbox=outer_bbox,
            local_bbox=outer_bbox,
            angle=0,
            score=float(100.0 + len(accepted_bands) + vertical_coverage),
            core_bbox=outer_bbox,
            line_indices=set(),
        )
        evidence.append(
            (
                candidate,
                len(accepted_bands),
                vertical_coverage,
                outer_area,
            )
        )

    output: list[_TableCandidate] = []
    for candidate, _band_count, _coverage, _outer_area in sorted(
        evidence,
        key=lambda item: (item[1], item[2], item[3]),
        reverse=True,
    ):
        if any(
            _bbox_overlap_in_smaller(candidate.bbox, accepted.bbox) >= 0.9
            for accepted in output
        ):
            continue
        output.append(candidate)
    return sorted(output, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]))


def _select_maximal_filled_grid_cells(
    rectangles: list[BBox],
    outer_bbox: BBox,
) -> list[BBox]:
    """移除边缘细条、重复 Path 和同跨度半行底纹，保留最大单元格。"""

    outer_width = outer_bbox[2] - outer_bbox[0]
    outer_height = outer_bbox[3] - outer_bbox[1]
    outer_area = _bbox_area(outer_bbox)
    x_tolerance = 0.01 * outer_width
    y_tolerance = 0.02 * outer_height
    nested_rectangles = [
        rectangle
        for rectangle in rectangles
        if rectangle != outer_bbox
        and _bbox_is_contained_with_tolerance(
            rectangle,
            outer_bbox,
            x_tolerance,
            y_tolerance,
        )
        and _bbox_area(rectangle) <= 0.8 * outer_area
        and rectangle[2] - rectangle[0] >= 0.1 * outer_width
        and rectangle[3] - rectangle[1] >= 0.06 * outer_height
    ]
    output: list[BBox] = []
    for rectangle in sorted(
        nested_rectangles,
        key=lambda bbox: (-_bbox_area(bbox), bbox[1], bbox[0]),
    ):
        if any(
            rectangle != other
            and _bbox_is_contained_with_tolerance(
                rectangle,
                other,
                x_tolerance,
                y_tolerance,
            )
            and _bbox_area(other) >= 1.08 * _bbox_area(rectangle)
            and _bbox_area(other) <= 0.8 * outer_area
            and abs(
                (rectangle[2] - rectangle[0])
                - (other[2] - other[0])
            )
            <= 0.01 * outer_width
            for other in nested_rectangles
        ):
            continue
        if any(
            _bbox_overlap_in_smaller(rectangle, accepted) >= 0.95
            for accepted in output
        ):
            continue
        output.append(rectangle)
    return sorted(output, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _bbox_is_contained_with_tolerance(
    inner_bbox: BBox,
    outer_bbox: BBox,
    x_tolerance: float,
    y_tolerance: float,
) -> bool:
    """按横纵独立容差判断一个矩形是否完整位于另一个矩形内。"""

    return (
        inner_bbox[0] >= outer_bbox[0] - x_tolerance
        and inner_bbox[1] >= outer_bbox[1] - y_tolerance
        and inner_bbox[2] <= outer_bbox[2] + x_tolerance
        and inner_bbox[3] <= outer_bbox[3] + y_tolerance
    )


def _group_filled_grid_cells_into_bands(
    cells: list[BBox],
    outer_bbox: BBox,
) -> list[list[BBox]]:
    """按相近上下边界把最大填充单元格聚成水平行带。"""

    y_tolerance = 0.02 * (outer_bbox[3] - outer_bbox[1])
    bands: list[list[BBox]] = []
    for cell in sorted(cells, key=lambda bbox: (bbox[1], bbox[0], bbox[3])):
        target = next(
            (
                band
                for band in bands
                if abs(cell[1] - band[0][1]) <= y_tolerance
                and abs(cell[3] - band[0][3]) <= y_tolerance
            ),
            None,
        )
        if target is None:
            bands.append([cell])
        else:
            target.append(cell)
    return bands


def _filled_grid_band_covers_outer_width(
    band: list[BBox],
    outer_bbox: BBox,
) -> bool:
    """校验单个填充行带的单元格数量、横向覆盖、间隙和端点。"""

    if len(band) < 2:
        return False
    outer_width = outer_bbox[2] - outer_bbox[0]
    segments = sorted((cell[0], cell[2]) for cell in band)
    current_left, current_right = segments[0]
    covered_width = 0.0
    maximum_gap = 0.0
    for left, right in segments[1:]:
        if left <= current_right:
            current_right = max(current_right, right)
            continue
        covered_width += current_right - current_left
        maximum_gap = max(maximum_gap, left - current_right)
        current_left, current_right = left, right
    covered_width += current_right - current_left
    return (
        covered_width >= 0.9 * outer_width
        and maximum_gap <= 0.01 * outer_width
        and abs(min(cell[0] for cell in band) - outer_bbox[0])
        <= 0.02 * outer_width
        and abs(max(cell[2] for cell in band) - outer_bbox[2])
        <= 0.02 * outer_width
    )


def _build_fragments(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_Fragment]:
    """将精修后的原生 run 转换成表格单元候选。"""

    fragments: list[_Fragment] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        fragments.append(
            _Fragment(
                text=line.text,
                bbox=line.bbox,
                local_bbox=local_bbox,
                line_index=line.source_index,
                # 复用原生粗行身份，避免同一字符行内不同 cell
                # 因轻微基线差异被误拆成多行。
                visual_row_id=line.visual_row_id,
            )
        )
    return fragments


def _cluster_fragment_rows(
    fragments: list[_Fragment],
    median_height: float,
) -> list[_VisualRow]:
    """优先复用原生视觉行身份，其余片段按中心线容差聚成表格行。"""

    tolerance = max(2.0, median_height * 0.5)
    native_groups: dict[int, list[_Fragment]] = {}
    geometric_fragments: list[_Fragment] = []
    for fragment in fragments:
        if fragment.visual_row_id is None:
            geometric_fragments.append(fragment)
        else:
            native_groups.setdefault(fragment.visual_row_id, []).append(fragment)

    # 先锁定同一原生粗行拆出的 run，再允许不同粗行按基线几何合并；
    # 旋转表格常把同一数据行的各 cell 分成多个 pdftext 粗行，不能只依赖 row id。
    seed_groups = [*native_groups.values(), *[[fragment] for fragment in geometric_fragments]]
    seed_groups.sort(
        key=lambda group: (
            statistics.fmean(_bbox_center_y(item.local_bbox) for item in group),
            min(item.local_bbox[0] for item in group),
        )
    )
    grouped: list[list[_Fragment]] = []
    for seed_group in seed_groups:
        center_y = statistics.fmean(_bbox_center_y(item.local_bbox) for item in seed_group)
        target_group: list[_Fragment] | None = None
        for group in grouped:
            group_center = statistics.fmean(_bbox_center_y(item.local_bbox) for item in group)
            if abs(center_y - group_center) <= tolerance:
                target_group = group
                break
        if target_group is None:
            grouped.append(list(seed_group))
        else:
            target_group.extend(seed_group)

    rows: list[_VisualRow] = []
    for group in grouped:
        group.sort(key=lambda item: item.local_bbox[0])
        bbox = _bbox_union_many([item.local_bbox for item in group])
        visual_row_ids = {item.visual_row_id for item in group if item.visual_row_id is not None}
        rows.append(
            _VisualRow(
                fragments=group,
                center_y=sum(_bbox_center_y(item.local_bbox) for item in group) / len(group),
                bbox=bbox,
                visual_row_id=next(iter(visual_row_ids)) if len(visual_row_ids) == 1 else None,
            )
        )
    rows.sort(key=lambda row: row.center_y)
    return rows


def _build_rule_table_candidates(
    rows: list[_VisualRow],
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
    *,
    path_infos: list[PDFPathInfo] | None = None,
    excluded_bboxes: list[BBox] | None = None,
) -> list[_TableCandidate]:
    """枚举同跨度横线边界区间，再以连续多列文本分布确认表格。"""

    candidates: list[_TableCandidate] = []
    path_infos = path_infos or []
    excluded_bboxes = excluded_bboxes or []
    for rule_group in _group_long_horizontal_rules(axis_lines, median_height):
        for first_index, top_rule in enumerate(rule_group[:-1]):
            for bottom_index in range(first_index + 1, len(rule_group)):
                bottom_rule = rule_group[bottom_index]
                boundary_rules = [top_rule, bottom_rule]
                rule_bbox = _bbox_union_many([line.bbox for line in boundary_rules])
                core_rows = _rows_inside_rule_interval(
                    rows,
                    rule_bbox,
                    excluded_bboxes,
                )
                if not _every_rule_interval_has_multi_cell_row(
                    core_rows,
                    rule_group[first_index : bottom_index + 1],
                    median_height,
                ):
                    continue
                if not _rule_intervals_are_column_compatible(
                    core_rows,
                    rule_group[first_index : bottom_index + 1],
                    median_height,
                ):
                    continue

                fill_band_count = _count_repeated_fill_bands(
                    path_infos,
                    rule_bbox,
                    page_size,
                    angle,
                    median_height,
                )
                aligned_vertical_count = _count_aligned_vertical_rules(
                    axis_lines,
                    rule_bbox,
                    median_height,
                )

                row_segments = _continuous_table_row_segments(core_rows, median_height)
                accepted: tuple[list[_VisualRow], list[_VisualRow], int, float] | None = None
                for row_segment in row_segments:
                    dense_rows = [row for row in row_segment if len(row.fragments) >= 2]
                    if len(dense_rows) < 3:
                        continue
                    # 真表格的多单元行会在整个数据带内反复出现；少数图题、图例和
                    # 坐标刻度偶然形成的多列行不能支撑一大片正文区域。
                    if len(dense_rows) / len(row_segment) < 0.2:
                        continue
                    stable_columns, column_coverage = _count_stable_columns(
                        dense_rows,
                        median_height,
                    )
                    if stable_columns < 2 or column_coverage < 0.5:
                        continue
                    if _looks_like_page_column_prose(
                        row_segment,
                        dense_rows,
                        stable_columns,
                        fill_band_count,
                        aligned_vertical_count,
                        rule_bbox,
                    ):
                        continue
                    if not _table_segment_reaches_boundaries(
                        row_segment,
                        rule_bbox,
                        median_height,
                    ):
                        continue
                    if not _table_rows_align_with_rule_span(
                        row_segment,
                        rule_bbox,
                        median_height,
                    ):
                        continue
                    result = (
                        row_segment,
                        dense_rows,
                        stable_columns,
                        column_coverage,
                    )
                    if accepted is None or (
                        len(row_segment),
                        len(dense_rows),
                        stable_columns,
                        column_coverage,
                    ) > (
                        len(accepted[0]),
                        len(accepted[1]),
                        accepted[2],
                        accepted[3],
                    ):
                        accepted = result
                if accepted is None:
                    continue

                accepted_rows, dense_rows, stable_columns, _coverage = accepted
                caption_line = _find_table_caption(
                    lines,
                    rule_bbox,
                    page_size,
                    angle,
                    median_height,
                )
                candidate = _expand_rule_table_candidate(
                    boundary_rules,
                    accepted_rows,
                    rows,
                    page_size,
                    angle,
                    median_height,
                    caption_line,
                )
                candidate.score = float(
                    2
                    + len(dense_rows)
                    + stable_columns
                    + min(fill_band_count, 8)
                )
                candidates.append(candidate)
    return candidates


def _group_long_horizontal_rules(
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[list[_LocalAxisLine]]:
    """按近似左右端点聚合长横线，并去除同位置重复路径。"""

    minimum_length = max(40.0, 10.0 * median_height)
    horizontal_lines = [
        line for line in axis_lines if line.orientation == "horizontal" and line.bbox[2] - line.bbox[0] >= minimum_length
    ]
    endpoint_tolerance = max(4.0, 2.0 * median_height)
    span_groups: list[list[_LocalAxisLine]] = []
    for line in sorted(horizontal_lines, key=lambda item: (item.bbox[0], item.bbox[2], item.bbox[1])):
        target = next(
            (
                group
                for group in span_groups
                if abs(line.bbox[0] - group[0].bbox[0]) <= endpoint_tolerance
                and abs(line.bbox[2] - group[0].bbox[2]) <= endpoint_tolerance
            ),
            None,
        )
        if target is None:
            span_groups.append([line])
        else:
            target.append(line)

    output: list[list[_LocalAxisLine]] = []
    for span_group in span_groups:
        unique_lines: list[_LocalAxisLine] = []
        for line in sorted(span_group, key=lambda item: _bbox_center_y(item.bbox)):
            if any(
                abs(_bbox_center_y(line.bbox) - _bbox_center_y(item.bbox)) <= 1.0
                for item in unique_lines
            ):
                continue
            unique_lines.append(line)
        if len(unique_lines) >= 2:
            output.append(unique_lines)
    return output


def _rows_inside_rule_interval(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    excluded_bboxes: list[BBox],
) -> list[_VisualRow]:
    """截取边界走廊内文本行，并移除已由强图形核心覆盖的片段。"""

    output: list[_VisualRow] = []
    for row in rows:
        clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=0.0)
        if clipped_row is None or not rule_bbox[1] <= clipped_row.center_y <= rule_bbox[3]:
            continue
        fragments = [
            fragment
            for fragment in clipped_row.fragments
            if not any(
                _point_in_bbox(
                    (
                        _bbox_center_x(fragment.local_bbox),
                        _bbox_center_y(fragment.local_bbox),
                    ),
                    excluded_bbox,
                )
                for excluded_bbox in excluded_bboxes
            )
        ]
        if not fragments:
            continue
        output.append(
            _VisualRow(
                fragments=fragments,
                center_y=sum(
                    _bbox_center_y(fragment.local_bbox) for fragment in fragments
                )
                / len(fragments),
                bbox=_bbox_union_many([fragment.local_bbox for fragment in fragments]),
                visual_row_id=clipped_row.visual_row_id,
            )
        )
    return output


def _every_rule_interval_has_multi_cell_row(
    rows: list[_VisualRow],
    rule_group: list[_LocalAxisLine],
    median_height: float,
) -> bool:
    """要求候选跨过的每个相邻横线区间都存在至少一行多单元文本。"""

    if len(rule_group) < 2:
        return False
    for interval_index, (top_rule, bottom_rule) in enumerate(
        zip(rule_group, rule_group[1:])
    ):
        top = _bbox_center_y(top_rule.bbox)
        bottom = _bbox_center_y(bottom_rule.bbox)
        interval_rows = [row for row in rows if top <= row.center_y <= bottom]
        if any(len(row.fragments) >= 2 for row in interval_rows):
            continue
        # 紧邻顶边界的合并表头可能由 pdftext 输出为一个短 fragment；
        # 只放宽高度很小的首区间，避免把远处章节标题接到表格上。
        if (
            interval_index == 0
            and interval_rows
            and bottom - top <= 2.5 * median_height
        ):
            continue
        return False
    return True


def _rule_intervals_are_column_compatible(
    rows: list[_VisualRow],
    rule_group: list[_LocalAxisLine],
    median_height: float,
) -> bool:
    """拒绝跨过长篇栏式正文、导致稳定列数明显塌缩的多表合并区间。"""

    profiles: list[tuple[int, float]] = []
    for top_rule, bottom_rule in zip(rule_group, rule_group[1:]):
        top = _bbox_center_y(top_rule.bbox)
        bottom = _bbox_center_y(bottom_rule.bbox)
        interval_rows = [
            row
            for row in rows
            if top <= row.center_y <= bottom and len(row.fragments) >= 2
        ]
        stable_columns, _coverage = _count_stable_columns(
            interval_rows,
            median_height,
        )
        profiles.append((stable_columns, bottom - top))
    maximum_columns = max((columns for columns, _height in profiles), default=0)
    if maximum_columns < 4:
        return True
    for interval_index, (columns, interval_height) in enumerate(profiles):
        # 首个区间可能只是跨列表头；后续长区间若退化成普通双栏正文，
        # 就不能把前后两张表合成一个候选。
        if interval_index == 0:
            continue
        if columns < max(2, int(0.5 * maximum_columns)) and interval_height > 4.0 * median_height:
            return False
    return True


def _continuous_table_row_segments(
    rows: list[_VisualRow],
    median_height: float,
) -> list[list[_VisualRow]]:
    """按物理行距切分边界区间，保留单元格换行参与连续性判断。"""

    segments: list[list[_VisualRow]] = []
    for row in sorted(rows, key=lambda item: item.center_y):
        if (
            not segments
            or max(0.0, row.bbox[1] - segments[-1][-1].bbox[3])
            > 3.0 * median_height
        ):
            segments.append([row])
        else:
            segments[-1].append(row)
    return segments


def _table_segment_reaches_boundaries(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    median_height: float,
) -> bool:
    """要求数据行链分别贴近最近的上下边界，排除页眉线和远处章节标题。"""

    if not rows:
        return False
    maximum_gap = 2.5 * median_height
    top_gap = max(0.0, rows[0].bbox[1] - rule_bbox[1])
    bottom_gap = max(0.0, rule_bbox[3] - rows[-1].bbox[3])
    return top_gap <= maximum_gap and bottom_gap <= maximum_gap


def _table_rows_align_with_rule_span(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    median_height: float,
) -> bool:
    """校验数据行总体跨度与横线走廊重叠，拒绝仅在边缘偶遇的多列文本。"""

    if not rows:
        return False
    rows_bbox = _bbox_union_many([row.bbox for row in rows])
    rule_width = max(0.1, rule_bbox[2] - rule_bbox[0])
    rows_width = max(0.1, rows_bbox[2] - rows_bbox[0])
    overlap = max(
        0.0,
        min(rows_bbox[2], rule_bbox[2]) - max(rows_bbox[0], rule_bbox[0]),
    )
    return (
        overlap / min(rule_width, rows_width) >= 0.9
        and rows_width >= max(8.0 * median_height, 0.25 * rule_width)
    )


def _count_aligned_vertical_rules(
    axis_lines: list[_LocalAxisLine],
    rule_bbox: BBox,
    median_height: float,
) -> int:
    """统计贯穿候选主要高度且位于横线跨度内的竖向分隔线。"""

    required_height = max(4.0 * median_height, 0.5 * (rule_bbox[3] - rule_bbox[1]))
    return sum(
        line.orientation == "vertical"
        and rule_bbox[0] - median_height <= _bbox_center_x(line.bbox) <= rule_bbox[2] + median_height
        and line.bbox[3] - line.bbox[1] >= required_height
        and _bbox_axis_overlap_ratio(line.bbox, rule_bbox, axis="y") >= 0.8
        for line in axis_lines
    )


def _looks_like_page_column_prose(
    rows: list[_VisualRow],
    dense_rows: list[_VisualRow],
    stable_columns: int,
    fill_band_count: int,
    aligned_vertical_count: int,
    rule_bbox: BBox,
) -> bool:
    """用双栏占宽率识别夹在远横线间的普通并排正文。"""

    if (
        stable_columns != 2
        or fill_band_count >= 2
        or aligned_vertical_count > 0
        or len(dense_rows) / len(rows) < 0.55
    ):
        return False
    corridor_width = max(0.1, rule_bbox[2] - rule_bbox[0])
    occupied_ratios = [
        sum(fragment.local_bbox[2] - fragment.local_bbox[0] for fragment in row.fragments)
        / corridor_width
        for row in dense_rows
    ]
    return statistics.median(occupied_ratios) >= 0.75


def _count_repeated_fill_bands(
    path_infos: list[PDFPathInfo],
    rule_bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
) -> int:
    """统计区间内左右端点和高度重复的填充行带，并对重叠 Path 去重。"""

    minimum_width = max(8.0 * median_height, 0.3 * (rule_bbox[2] - rule_bbox[0]))
    candidates: list[BBox] = []
    for path_info in path_infos:
        if path_info.form_depth != 0 or not path_info.fill_visible:
            continue
        bbox = _rotate_bbox_to_upright(path_info.bbox, page_size, angle)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if (
            width < minimum_width
            or not 0.25 * median_height <= height <= 3.0 * median_height
            or _bbox_center_y(bbox) < rule_bbox[1]
            or _bbox_center_y(bbox) > rule_bbox[3]
            or _bbox_axis_overlap_ratio(bbox, rule_bbox, axis="x") < 0.8
        ):
            continue
        if any(_bbox_overlap_in_smaller(bbox, item) >= 0.9 for item in candidates):
            continue
        candidates.append(bbox)

    endpoint_tolerance = max(3.0, median_height)
    groups: list[list[BBox]] = []
    for bbox in candidates:
        target = next(
            (
                group
                for group in groups
                if abs(bbox[0] - group[0][0]) <= endpoint_tolerance
                and abs(bbox[2] - group[0][2]) <= endpoint_tolerance
                and abs((bbox[3] - bbox[1]) - (group[0][3] - group[0][1]))
                <= endpoint_tolerance
            ),
            None,
        )
        if target is None:
            groups.append([bbox])
        else:
            target.append(bbox)
    return max((len(group) for group in groups), default=0)


def _clip_visual_row_to_corridor(
    row: _VisualRow,
    corridor_bbox: BBox,
    *,
    margin: float,
) -> _VisualRow | None:
    """仅保留横向走廊内的片段，避免同基线的另一栏文本污染表格区域。"""

    fragments = [
        fragment
        for fragment in row.fragments
        if corridor_bbox[0] - margin <= _bbox_center_x(fragment.local_bbox) <= corridor_bbox[2] + margin
    ]
    if not fragments:
        return None
    fragments.sort(key=lambda fragment: fragment.local_bbox[0])
    return _VisualRow(
        fragments=fragments,
        center_y=sum(_bbox_center_y(fragment.local_bbox) for fragment in fragments) / len(fragments),
        bbox=_bbox_union_many([fragment.local_bbox for fragment in fragments]),
        visual_row_id=row.visual_row_id,
    )


def _longest_dense_multi_cell_rows(
    rows: list[_VisualRow],
    median_height: float,
) -> list[_VisualRow]:
    """返回行距不超过四倍行高的最长连续多单元格文本段。"""

    segments: list[list[_VisualRow]] = []
    for row in (item for item in rows if len(item.fragments) >= 2):
        if not segments or row.center_y - segments[-1][-1].center_y > 4.0 * median_height:
            segments.append([row])
        else:
            segments[-1].append(row)
    return max(segments, key=len, default=[])


def _expand_rule_table_candidate(
    rule_group: list[_LocalAxisLine],
    core_rows: list[_VisualRow],
    all_rows: list[_VisualRow],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    caption_line: _LineItem | None,
) -> _TableCandidate:
    """合并横线核心、上方标题和下方脚注，构造统一投影候选。"""

    rule_bbox = _bbox_union_many([line.bbox for line in rule_group])
    core_line_indices = {fragment.line_index for row in core_rows for fragment in row.fragments}
    caption_rows = _collect_caption_rows(all_rows, caption_line, rule_bbox, median_height)
    footnote_rows = _collect_footnote_rows(
        all_rows,
        rule_bbox,
        median_height,
        core_line_indices,
    )
    included_rows = [*caption_rows, *core_rows, *footnote_rows]
    core_local_bbox = _bbox_union(rule_bbox, _bbox_union_many([row.bbox for row in core_rows]))
    local_bbox = _bbox_union(core_local_bbox, _bbox_union_many([row.bbox for row in included_rows]))
    return _TableCandidate(
        bbox=_rotate_bbox_from_upright(local_bbox, page_size, angle),
        local_bbox=local_bbox,
        angle=angle,
        score=0.0,
        core_bbox=_rotate_bbox_from_upright(core_local_bbox, page_size, angle),
        line_indices={fragment.line_index for row in included_rows for fragment in row.fragments},
    )


def _collect_caption_rows(
    rows: list[_VisualRow],
    caption_line: _LineItem | None,
    rule_bbox: BBox,
    median_height: float,
) -> list[_VisualRow]:
    """收集显式标题所在行及其到表格上边界之间的连续换行。"""

    if caption_line is None:
        return []
    caption_row_index = next(
        (
            index
            for index, row in enumerate(rows)
            if any(fragment.line_index == caption_line.source_index for fragment in row.fragments)
        ),
        None,
    )
    if caption_row_index is None:
        return []

    output: list[_VisualRow] = []
    previous_bbox: BBox | None = None
    margin = 2.0 * median_height
    for index, row in enumerate(rows[caption_row_index:], start=caption_row_index):
        clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=margin)
        if clipped_row is None:
            continue
        if index > caption_row_index and clipped_row.center_y >= rule_bbox[1]:
            break
        if previous_bbox is not None and max(0.0, clipped_row.bbox[1] - previous_bbox[3]) > 2.0 * median_height:
            break
        output.append(clipped_row)
        previous_bbox = clipped_row.bbox

    if not output or rule_bbox[1] - output[-1].bbox[3] > 2.0 * median_height:
        return []
    return output


def _collect_footnote_rows(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    median_height: float,
    core_line_indices: set[int],
) -> list[_VisualRow]:
    """从表格下边界开始吸收带通用标记的脚注及其连续换行。"""

    output: list[_VisualRow] = []
    bottom = rule_bbox[3]
    note_chain_started = False
    margin = 2.0 * median_height
    selected_line_indices = set(core_line_indices)
    for row in rows:
        clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=margin)
        if clipped_row is None or clipped_row.bbox[3] <= bottom:
            continue
        line_indices = {fragment.line_index for fragment in clipped_row.fragments}
        if line_indices.issubset(selected_line_indices):
            bottom = max(bottom, clipped_row.bbox[3])
            continue
        if max(0.0, clipped_row.bbox[1] - bottom) > 1.5 * median_height:
            break
        if not note_chain_started and not _is_table_note_text(_visual_row_text(clipped_row)):
            break
        output.append(clipped_row)
        selected_line_indices.update(line_indices)
        bottom = max(bottom, clipped_row.bbox[3])
        note_chain_started = True
    return output


def _visual_row_text(row: _VisualRow) -> str:
    """按局部 x 顺序拼接视觉行文本，供拆分脚注标记判断。"""

    return " ".join(fragment.text.strip() for fragment in row.fragments if fragment.text.strip())


def _is_table_note_text(text: str) -> bool:
    """判断表后首行是否具有明确的注释、来源或脚注标记。"""

    return bool(_TABLE_NOTE_RE.match(str(text or "").strip()))


def _find_table_caption(
    lines: list[_LineItem],
    core_bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
) -> _LineItem | None:
    """在核心表格上方最多十二倍行高内查找显式 Table/表标题。"""

    candidates: list[tuple[float, _LineItem]] = []
    for line in lines:
        text = line.text.strip()
        caption_match = _TABLE_CAPTION_RE.match(text)
        is_split_label = text.lower().rstrip(".") in {"table", "tab", "表", "表格"}
        if caption_match is None and not is_split_label:
            continue
        if caption_match is not None:
            suffix = caption_match.group("suffix").strip(" .:–—-")
            # 小写连续句通常是“Table 5 also ...”这类正文，不应作为标题。
            if suffix and suffix[0].islower():
                continue
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        if _bbox_axis_overlap_ratio(local_bbox, core_bbox, axis="x") < 0.05:
            continue
        if is_split_label:
            has_number_peer = bool(
                _find_caption_number_peers(
                    line,
                    lines,
                    page_size,
                    angle,
                    median_height,
                )
            )
            if not has_number_peer:
                continue
        gap = core_bbox[1] - local_bbox[3]
        if -median_height <= gap <= 12.0 * median_height:
            candidates.append((abs(gap), line))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _find_caption_number_peers(
    caption_line: _LineItem,
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
) -> list[_LineItem]:
    """查找与拆分 Table/表 标签同一视觉行的编号文本。"""

    caption_local_bbox = _rotate_bbox_to_upright(caption_line.bbox, page_size, angle)
    peers: list[_LineItem] = []
    for peer in lines:
        if peer.source_index == caption_line.source_index:
            continue
        if not _TABLE_SPLIT_NUMBER_RE.match(peer.text.strip()):
            continue
        peer_local_bbox = _rotate_bbox_to_upright(peer.bbox, page_size, angle)
        gap = peer_local_bbox[0] - caption_local_bbox[2]
        if _bbox_axis_overlap_ratio(caption_local_bbox, peer_local_bbox, axis="y") >= 0.5 and 0.0 <= gap <= 4.0 * median_height:
            peers.append(peer)
    return sorted(
        peers,
        key=lambda peer: _rotate_bbox_to_upright(peer.bbox, page_size, angle)[0],
    )


def _count_stable_columns(
    rows: list[_VisualRow],
    median_height: float,
) -> tuple[int, float]:
    """分别聚类片段左边界、中心和右边界，返回最稳定的列分布。"""

    tolerance = max(3.0, median_height * 0.75)
    best_result = (0, 0.0)
    # 三种对齐方式分别聚类，避免把同一片段的不同锚点混算为多列。
    for alignment in ("left", "center", "right"):
        clusters: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            for fragment in row.fragments:
                left, _top, right, _bottom = fragment.local_bbox
                if alignment == "left":
                    anchor = left
                elif alignment == "center":
                    anchor = (left + right) / 2
                else:
                    anchor = right
                cluster = next(
                    (item for item in clusters if abs(anchor - float(item["mean"])) <= tolerance),
                    None,
                )
                if cluster is None:
                    clusters.append({"mean": anchor, "values": [anchor], "rows": {row_index}})
                else:
                    cluster["values"].append(anchor)
                    cluster["rows"].add(row_index)
                    cluster["mean"] = sum(cluster["values"]) / len(cluster["values"])
        stable_coverages = [
            len(cluster["rows"]) / len(rows)
            for cluster in clusters
            if len(cluster["rows"]) / len(rows) >= 0.5
        ]
        result = (
            len(stable_coverages),
            min(stable_coverages) if stable_coverages else 0.0,
        )
        # 仅在结果严格更优时更新，平局时保留既有的左对齐优先级。
        if result > best_result:
            best_result = result
    return best_result


def _merge_table_candidates(candidates: list[_TableCandidate]) -> list[_TableCandidate]:
    """合并同方向且明显重叠的横线候选，避免同一表格重复输出。"""

    merged: list[_TableCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        target = next(
            (
                item
                for item in merged
                if item.angle == candidate.angle and _bbox_overlap_in_smaller(candidate.bbox, item.bbox) >= 0.2
            ),
            None,
        )
        if target is None:
            merged.append(candidate)
            continue
        target.bbox = _bbox_union(target.bbox, candidate.bbox)
        target.local_bbox = _bbox_union(target.local_bbox, candidate.local_bbox)
        if target.core_bbox is None:
            target.core_bbox = candidate.core_bbox
        elif candidate.core_bbox is not None:
            target.core_bbox = _bbox_union(target.core_bbox, candidate.core_bbox)
        target.line_indices.update(candidate.line_indices)
        target.score = max(target.score, candidate.score)
    return sorted(merged, key=lambda item: (item.bbox[1], item.bbox[0]))


def _materialize_table_blocks(
    source: _PageSource,
    candidates: list[_TableCandidate],
) -> tuple[list[dict[str, Any]], set[int]]:
    """为候选生成空间投影 content，仅认领投影成功的文本行。"""

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(_bbox_overlap_in_smaller(candidate.bbox, block["bbox"]) >= 0.5 for block in blocks):
            continue
        output_angle = candidate.angle
        projection_line_indices = _candidate_projection_line_indices(source, candidate)
        try:
            candidate_chars = [
                char for line in source.lines if line.source_index in projection_line_indices for char in line.chars
            ]
            content = project_pdf_table_text(
                candidate_chars,
                candidate.bbox,
                angle=candidate.angle,
            )
        except Exception as exc:
            # 单个表格的字符投影异常只撤销该候选，不能中止整页提取。
            logger.warning(f"Flash table projection failed and rolled back: bbox={candidate.bbox}, error={exc}")
            continue
        if not content or not content.strip():
            continue
        blocks.append(
            {
                "type": "table",
                "bbox": candidate.bbox,
                "angle": output_angle,
                "content": content,
            }
        )
        # 只认领候选明确接纳的视觉行，避免远标题与表格之间的正文被矩形 bbox 连带删除。
        claimed.update(projection_line_indices)
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _candidate_projection_line_indices(
    source: _PageSource,
    candidate: _TableCandidate,
) -> set[int]:
    """合并核心成员、同基线续段及非零角度表格的表头文本。"""

    line_indices = set(candidate.line_indices)
    if candidate.core_bbox is not None:
        for line in source.lines:
            if _point_in_bbox(
                (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                candidate.core_bbox,
            ):
                line_indices.add(line.source_index)
    if candidate.angle == 0:
        _expand_candidate_same_baseline_members(source, candidate, line_indices)
        return line_indices
    if candidate.core_bbox is None:
        return line_indices

    candidate_local_bbox = _rotate_bbox_to_upright(
        candidate.bbox,
        source.page_size,
        candidate.angle,
    )
    core_local_bbox = _rotate_bbox_to_upright(
        candidate.core_bbox,
        source.page_size,
        candidate.angle,
    )
    for line in source.lines:
        if line.angle != candidate.angle:
            continue
        page_center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
        if not _point_in_bbox(page_center, candidate.bbox):
            continue
        local_bbox = _rotate_bbox_to_upright(
            line.bbox,
            source.page_size,
            candidate.angle,
        )
        local_center_y = _bbox_center_y(local_bbox)
        if not candidate_local_bbox[1] <= local_center_y <= candidate_local_bbox[3]:
            continue
        if _bbox_axis_overlap_ratio(local_bbox, core_local_bbox, axis="x") < 0.05:
            continue
        line_indices.add(line.source_index)
    return line_indices


def _expand_candidate_same_baseline_members(
    source: _PageSource,
    candidate: _TableCandidate,
    line_indices: set[int],
) -> None:
    """迭代吸收完整候选框内与已认领成员同基线相邻的 angle=0 续段。"""

    local_bboxes = {
        line.source_index: _rotate_bbox_to_upright(
            line.bbox,
            source.page_size,
            candidate.angle,
        )
        for line in source.lines
        if line.angle == candidate.angle
    }
    changed = True
    while changed:
        changed = False
        selected_lines = [
            line
            for line in source.lines
            if line.angle == candidate.angle and line.source_index in line_indices
        ]
        for line in source.lines:
            if line.angle != candidate.angle or line.source_index in line_indices:
                continue
            if not _point_in_bbox(
                (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                candidate.bbox,
            ):
                continue
            line_bbox = local_bboxes[line.source_index]
            line_height = _line_effective_height(line, line_bbox)
            for selected in selected_lines:
                if (
                    line.font_signature is not None
                    and selected.font_signature is not None
                    and line.font_signature != selected.font_signature
                ):
                    continue
                selected_bbox = local_bboxes[selected.source_index]
                if not _same_baseline_geometry(
                    line_bbox,
                    line_height,
                    selected_bbox,
                    _line_effective_height(selected, selected_bbox),
                ):
                    continue
                line_indices.add(line.source_index)
                changed = True
                break


def _median_fragment_height(fragments: list[_Fragment]) -> float:
    """返回正向文本片段高度的中位数。"""

    heights = [
        fragment.local_bbox[3] - fragment.local_bbox[1]
        for fragment in fragments
        if fragment.local_bbox[3] > fragment.local_bbox[1]
    ]
    return max(0.1, float(statistics.median(heights)) if heights else 1.0)
