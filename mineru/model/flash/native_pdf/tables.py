# Copyright (c) Opendatalab. All rights reserved.

"""检测、投影并物化 Flash 原生 PDF 表格。"""

from __future__ import annotations

import re
import statistics
import unicodedata
from typing import Any, Literal

from loguru import logger
from pdftext.schema import Char

from mineru.utils.native_pdf_table import (
    NativeTableInput,
    NativeTableRectangle,
    NativeTableRule,
    coerce_native_table_rectangles,
    coerce_native_table_rules,
    recover_native_pdf_table,
)
from mineru.utils.text_utils import merge_text_line_contents
from mineru.utils.spatial_text import project_pdf_table_text
from mineru.types import BBox
from mineru.utils.pdf_document import PDFPathInfo
from mineru.utils.language import detect_lang

from .models import (
    _Fragment,
    _LineItem,
    _LocalAxisLine,
    _PageSource,
    _TableAnnotation,
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
    _coerce_bbox,
    _expand_bbox,
    _point_in_bbox,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from .line_layout import _line_effective_height
from .line_merging import _same_baseline_geometry
from .native_text import _normalize_native_run_text


_TABLE_CAPTION_RE = re.compile(
    r"^(?:table|tab\.?|表格?)[\s:.–—-]*(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)\b(?P<suffix>.*)$",
    re.IGNORECASE,
)


_TABLE_NOTE_RE = re.compile(
    r"^(?:notes?|sources?)\b|^(?:注释?|说明)\s*[:：]?|^for\s+[*†‡]"
    r"|^[*†‡]\s*\S",
    re.IGNORECASE,
)


_AUXILIARY_TABLE_NOTE_RE = re.compile(
    r"^\s*[([{（［【]?(?P<marker>[^\s)\]}）］】.:：、]{1,3})"
    r"[)\]}）］】]?[.:：、)]*\s+(?P<body>\S.*)$"
)


_TABLE_SPLIT_NUMBER_RE = re.compile(
    r"^(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)[.:：]?$",
    re.IGNORECASE,
)


_FILLED_GRID_MIN_PAGE_AREA_RATIO = 0.005
_FILLED_GRID_MAX_PAGE_AREA_RATIO = 0.25
_FILLED_GRID_MIN_PAGE_WIDTH_RATIO = 0.12
_FILLED_GRID_MIN_PAGE_HEIGHT_RATIO = 0.03


def _recover_native_table_html(
    source: _PageSource,
    table_bbox: BBox,
    angle: int,
    chars: tuple[Char, ...] | None = None,
    drawing_lines: tuple[NativeTableRule, ...] | None = None,
    rectangles: tuple[NativeTableRectangle, ...] | None = None,
) -> str:
    """使用共享原生字符与绘图原语恢复高置信表格 HTML。"""

    result = recover_native_pdf_table(
        NativeTableInput(
            table_bbox=table_bbox,
            page_size=source.page_size,
            angle=angle,
            chars=chars if chars is not None else tuple(source.chars),
            drawing_lines=(
                drawing_lines
                if drawing_lines is not None
                else coerce_native_table_rules(source.drawing_lines)
            ),
            rectangles=(
                rectangles
                if rectangles is not None
                else coerce_native_table_rectangles(source.path_infos)
            ),
        )
    )
    return result.html if result is not None else ""


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
        local_closed_grid_excluded_bboxes = [
            *local_excluded_bboxes,
            *[
                _rotate_bbox_to_upright(bbox, source.page_size, angle)
                for bbox in source.form_bboxes
            ],
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
        rule_candidates.extend(
            _build_closed_rule_grid_candidates(
                rows,
                angle_lines,
                source.page_size,
                angle,
                median_height,
                local_axis_lines,
                local_closed_grid_excluded_bboxes,
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
                interval_rules = rule_group[first_index : bottom_index + 1]
                boundary_rules = [top_rule, bottom_rule]
                rule_bbox = _bbox_union_many([line.bbox for line in boundary_rules])
                core_rows = _rows_inside_rule_interval(
                    rows,
                    rule_bbox,
                    excluded_bboxes,
                )
                if not _every_rule_interval_has_multi_cell_row(
                    core_rows,
                    interval_rules,
                    median_height,
                ):
                    continue
                if not _rule_intervals_are_column_compatible(
                    core_rows,
                    interval_rules,
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
                    compact_grid_columns = (
                        _compact_fully_ruled_grid_column_count(
                            row_segment,
                            dense_rows,
                            interval_rules,
                            axis_lines,
                            rule_bbox,
                            median_height,
                        )
                        if len(dense_rows) == 2
                        else 0
                    )
                    if len(dense_rows) < 3 and compact_grid_columns == 0:
                        continue
                    # 真表格的多单元行会在整个数据带内反复出现；少数图题、图例和
                    # 坐标刻度偶然形成的多列行不能支撑一大片正文区域。
                    if len(dense_rows) / len(row_segment) < 0.2:
                        continue
                    stable_columns, column_coverage = _count_stable_columns(
                        dense_rows,
                        median_height,
                    )
                    if compact_grid_columns > 0:
                        # 两行样本容易把左右/中心锚点误算成不同稳定列，使用物理网格列数。
                        stable_columns = compact_grid_columns
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
                    lines,
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
    return _expand_candidates_to_connected_rule_grids(
        candidates,
        rows,
        page_size,
        angle,
        median_height,
        axis_lines,
        excluded_bboxes,
    )


def _expand_candidates_to_connected_rule_grids(
    candidates: list[_TableCandidate],
    rows: list[_VisualRow],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
    excluded_bboxes: list[BBox],
) -> list[_TableCandidate]:
    """把已确认候选沿连续横边界和贯穿竖轨扩展到完整物理网格。"""

    grid_bboxes = [
        grid_bbox
        for grid_bbox in _connected_rule_grid_bboxes(axis_lines, median_height)
        if not any(
            _bbox_overlap_in_smaller(grid_bbox, excluded_bbox) >= 0.5
            for excluded_bbox in excluded_bboxes
        )
    ]
    if not grid_bboxes:
        return candidates

    tolerance = max(2.0, median_height)
    for candidate in candidates:
        if candidate.core_bbox is None:
            continue
        core_local_bbox = _rotate_bbox_to_upright(
            candidate.core_bbox,
            page_size,
            angle,
        )
        matches = [
            grid_bbox
            for grid_bbox in grid_bboxes
            if _bbox_axis_overlap_ratio(
                core_local_bbox,
                grid_bbox,
                axis="x",
            )
            >= 0.9
            and core_local_bbox[3] >= grid_bbox[1] - tolerance
            and core_local_bbox[1] <= grid_bbox[3] + tolerance
        ]
        if not matches:
            continue
        grid_bbox = max(
            matches,
            key=lambda bbox: (
                min(core_local_bbox[3], bbox[3])
                - max(core_local_bbox[1], bbox[1]),
                _bbox_area(bbox),
            ),
        )
        expanded_core_bbox = _bbox_union(core_local_bbox, grid_bbox)
        candidate.local_bbox = _bbox_union(candidate.local_bbox, grid_bbox)
        candidate.core_bbox = _rotate_bbox_from_upright(
            expanded_core_bbox,
            page_size,
            angle,
        )
        candidate.bbox = _rotate_bbox_from_upright(
            candidate.local_bbox,
            page_size,
            angle,
        )
        for row in rows:
            if not expanded_core_bbox[1] <= row.center_y <= expanded_core_bbox[3]:
                continue
            candidate.line_indices.update(
                fragment.line_index
                for fragment in row.fragments
                if _point_in_bbox(
                    (
                        _bbox_center_x(fragment.local_bbox),
                        _bbox_center_y(fragment.local_bbox),
                    ),
                    expanded_core_bbox,
                )
            )
        for annotation in candidate.annotations:
            candidate.line_indices.difference_update(annotation.line_indices)
    return candidates


def _build_closed_rule_grid_candidates(
    rows: list[_VisualRow],
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
    excluded_bboxes: list[BBox],
) -> list[_TableCandidate]:
    """用闭合物理网格接纳含空行或仅有表头文本的稀疏表格。"""

    candidates: list[_TableCandidate] = []
    for component in _connected_rule_grid_components(axis_lines, median_height):
        grid_bbox = _bbox_union_many([rule.bbox for rule in component])
        if any(
            _bbox_overlap_in_smaller(grid_bbox, excluded_bbox) >= 0.5
            for excluded_bbox in excluded_bboxes
        ):
            continue
        core_rows = _rows_inside_rule_interval(
            rows,
            grid_bbox,
            excluded_bboxes,
        )
        if not core_rows:
            continue

        vertical_positions = _closed_grid_vertical_track_positions(
            component,
            axis_lines,
            median_height,
        )
        if len(vertical_positions) < 2:
            continue
        edge_tolerance = max(2.0, 0.25 * median_height)
        if (
            abs(vertical_positions[0] - grid_bbox[0]) > edge_tolerance
            or abs(vertical_positions[-1] - grid_bbox[2]) > edge_tolerance
        ):
            continue

        if len(component) == 2:
            if len(vertical_positions) < 3:
                continue
            occupied_columns = _count_occupied_closed_grid_columns(
                core_rows,
                vertical_positions,
            )
            if occupied_columns < 2:
                continue

        caption_line = _find_table_caption(
            lines,
            grid_bbox,
            page_size,
            angle,
            median_height,
        )
        candidate = _expand_rule_table_candidate(
            [component[0], component[-1]],
            core_rows,
            rows,
            lines,
            page_size,
            angle,
            median_height,
            caption_line,
        )
        candidate.score = float(
            100 + len(component) + len(vertical_positions)
        )
        candidates.append(candidate)
    return candidates


def _closed_grid_vertical_track_positions(
    horizontal_rules: list[_LocalAxisLine],
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[float]:
    """收集覆盖首末横边界中心跨度至少九成的竖轨并合并重复路径。"""

    top = _bbox_center_y(horizontal_rules[0].bbox)
    bottom = _bbox_center_y(horizontal_rules[-1].bbox)
    grid_height = max(0.1, bottom - top)
    left = min(rule.bbox[0] for rule in horizontal_rules)
    right = max(rule.bbox[2] for rule in horizontal_rules)
    edge_tolerance = max(2.0, 0.25 * median_height)
    raw_positions = []
    for line in axis_lines:
        if line.orientation != "vertical":
            continue
        overlap = max(
            0.0,
            min(line.bbox[3], bottom) - max(line.bbox[1], top),
        )
        position = _bbox_center_x(line.bbox)
        if (
            overlap / grid_height >= 0.9
            and left - edge_tolerance <= position <= right + edge_tolerance
        ):
            raw_positions.append(position)

    position_tolerance = max(1.0, 0.1 * median_height)
    position_groups: list[list[float]] = []
    for position in sorted(raw_positions):
        if (
            position_groups
            and abs(position - statistics.mean(position_groups[-1]))
            <= position_tolerance
        ):
            position_groups[-1].append(position)
        else:
            position_groups.append([position])
    return [statistics.mean(group) for group in position_groups]


def _count_occupied_closed_grid_columns(
    rows: list[_VisualRow],
    vertical_positions: list[float],
) -> int:
    """按文本片段中心统计闭合网格中实际有文字的物理列数。"""

    occupied_columns: set[int] = set()
    for row in rows:
        for fragment in row.fragments:
            center_x = _bbox_center_x(fragment.local_bbox)
            matching_columns = [
                index
                for index, (left, right) in enumerate(
                    zip(vertical_positions, vertical_positions[1:])
                )
                if left < center_x < right
            ]
            if len(matching_columns) == 1:
                occupied_columns.add(matching_columns[0])
    return len(occupied_columns)


def _connected_rule_grid_bboxes(
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[BBox]:
    """把端点一致且由外轨或至少两条列轨贯穿的相邻横线组成网格框。"""

    return [
        _bbox_union_many([rule.bbox for rule in component])
        for component in _connected_rule_grid_components(
            axis_lines,
            median_height,
        )
    ]


def _connected_rule_grid_components(
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[list[_LocalAxisLine]]:
    """保留连续网格的横边界成员，供精确外轨和横边界数量校验。"""

    output: list[list[_LocalAxisLine]] = []
    for rule_group in _group_long_horizontal_rules(axis_lines, median_height):
        components: list[list[_LocalAxisLine]] = []
        for rule in rule_group:
            if not components or not _rule_bands_share_grid_tracks(
                components[-1][-1],
                rule,
                axis_lines,
                median_height,
            ):
                components.append([rule])
            else:
                components[-1].append(rule)
        output.extend(component for component in components if len(component) >= 2)
    return output


def _rule_bands_share_grid_tracks(
    top_rule: _LocalAxisLine,
    bottom_rule: _LocalAxisLine,
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> bool:
    """校验相邻横边界的跨度，并确认其间存在连续外框或稳定列分隔线。"""

    top_width = max(0.1, top_rule.bbox[2] - top_rule.bbox[0])
    bottom_width = max(0.1, bottom_rule.bbox[2] - bottom_rule.bbox[0])
    overlap_left = max(top_rule.bbox[0], bottom_rule.bbox[0])
    overlap_right = min(top_rule.bbox[2], bottom_rule.bbox[2])
    overlap_width = max(0.0, overlap_right - overlap_left)
    endpoint_tolerance = max(4.0, 2.0 * median_height)
    if (
        overlap_width / max(top_width, bottom_width) < 0.9
        or abs(top_rule.bbox[0] - bottom_rule.bbox[0]) > endpoint_tolerance
        or abs(top_rule.bbox[2] - bottom_rule.bbox[2]) > endpoint_tolerance
    ):
        return False

    top_y = _bbox_center_y(top_rule.bbox)
    bottom_y = _bbox_center_y(bottom_rule.bbox)
    track_tolerance = max(1.0, 0.25 * median_height)
    raw_positions = [
        _bbox_center_x(line.bbox)
        for line in axis_lines
        if line.orientation == "vertical"
        and line.bbox[1] <= top_y + track_tolerance
        and line.bbox[3] >= bottom_y - track_tolerance
        and overlap_left - track_tolerance
        <= _bbox_center_x(line.bbox)
        <= overlap_right + track_tolerance
    ]
    position_groups: list[list[float]] = []
    for position in sorted(raw_positions):
        if (
            position_groups
            and abs(position - statistics.mean(position_groups[-1]))
            <= track_tolerance
        ):
            position_groups[-1].append(position)
        else:
            position_groups.append([position])
    positions = [statistics.mean(group) for group in position_groups]
    has_outer_tracks = (
        any(abs(position - overlap_left) <= endpoint_tolerance for position in positions)
        and any(abs(position - overlap_right) <= endpoint_tolerance for position in positions)
    )
    interior_tracks = [
        position
        for position in positions
        if overlap_left + track_tolerance
        < position
        < overlap_right - track_tolerance
    ]
    return has_outer_tracks or len(interior_tracks) >= 2


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

    profiles: list[tuple[int, float, int, float]] = []
    for top_rule, bottom_rule in zip(rule_group, rule_group[1:]):
        top = _bbox_center_y(top_rule.bbox)
        bottom = _bbox_center_y(bottom_rule.bbox)
        interval_rows = [
            row
            for row in rows
            if top <= row.center_y <= bottom and len(row.fragments) >= 2
        ]
        stable_columns, column_coverage = _count_stable_columns(
            interval_rows,
            median_height,
        )
        profiles.append(
            (
                stable_columns,
                bottom - top,
                len(interval_rows),
                column_coverage,
            )
        )
    maximum_columns = max(
        (columns for columns, _height, _row_count, _coverage in profiles),
        default=0,
    )
    for interval_index, (columns, interval_height, row_count, coverage) in enumerate(profiles):
        # 紧凑首区间可能只是跨列表头；一旦区间明显高于普通表头，
        # 也必须具有连续多单元格行，不能无条件跨过正文连接两张表。
        if interval_index == 0 and interval_height <= 2.5 * median_height:
            continue
        if interval_height <= 6.0 * median_height:
            continue
        minimum_rows = max(2, int(interval_height / max(8.0 * median_height, 0.1)))
        if row_count < minimum_rows or coverage < 0.5:
            return False
        if columns < max(2, int(0.5 * maximum_columns)):
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


def _compact_fully_ruled_grid_column_count(
    row_segment: list[_VisualRow],
    dense_rows: list[_VisualRow],
    interval_rules: list[_LocalAxisLine],
    axis_lines: list[_LocalAxisLine],
    rule_bbox: BBox,
    median_height: float,
) -> int:
    """以完整横竖边界确认两行紧凑网格，并返回物理列数，失败时返回零。"""

    rule_height = max(0.1, rule_bbox[3] - rule_bbox[1])
    if (
        len(row_segment) != 2
        or len(dense_rows) != 2
        or len(interval_rules) < 3
        or rule_height > 6.0 * median_height
    ):
        return 0

    vertical_positions = _full_height_vertical_rule_positions(
        axis_lines,
        rule_bbox,
        median_height,
    )
    if len(vertical_positions) < 3:
        return 0

    edge_tolerance = max(1.5, 0.25 * median_height)
    left_boundary = min(
        vertical_positions,
        key=lambda position: abs(position - rule_bbox[0]),
    )
    right_boundary = min(
        vertical_positions,
        key=lambda position: abs(position - rule_bbox[2]),
    )
    if (
        abs(left_boundary - rule_bbox[0]) > edge_tolerance
        or abs(right_boundary - rule_bbox[2]) > edge_tolerance
        or right_boundary <= left_boundary
    ):
        return 0

    grid_boundaries = [
        position
        for position in vertical_positions
        if left_boundary <= position <= right_boundary
    ]
    if len(grid_boundaries) < 3:
        return 0
    grid_intervals = list(zip(grid_boundaries, grid_boundaries[1:]))

    occupied_columns: list[set[int]] = []
    for row in dense_rows:
        row_columns: list[int] = []
        for fragment in row.fragments:
            fragment_center = _bbox_center_x(fragment.local_bbox)
            matching_columns = [
                index
                for index, (left, right) in enumerate(grid_intervals)
                if left <= fragment_center <= right
            ]
            if len(matching_columns) != 1:
                return 0
            column_index = matching_columns[0]
            if column_index in row_columns:
                return 0
            row_columns.append(column_index)
        if len(row_columns) < 2:
            return 0
        occupied_columns.append(set(row_columns))

    if len(occupied_columns[0] & occupied_columns[1]) < 2:
        return 0
    return len(grid_intervals)


def _full_height_vertical_rule_positions(
    axis_lines: list[_LocalAxisLine],
    rule_bbox: BBox,
    median_height: float,
) -> list[float]:
    """收集覆盖紧凑候选主要高度的竖线中心，并合并同位置重复路径。"""

    rule_height = max(0.1, rule_bbox[3] - rule_bbox[1])
    raw_positions: list[float] = []
    for line in axis_lines:
        if line.orientation != "vertical":
            continue
        overlap = max(
            0.0,
            min(line.bbox[3], rule_bbox[3])
            - max(line.bbox[1], rule_bbox[1]),
        )
        if (
            overlap / rule_height < 0.8
            or line.bbox[3] - line.bbox[1] < 0.8 * rule_height
            or not rule_bbox[0] - median_height
            <= _bbox_center_x(line.bbox)
            <= rule_bbox[2] + median_height
        ):
            continue
        raw_positions.append(_bbox_center_x(line.bbox))

    deduplicated: list[list[float]] = []
    position_tolerance = max(1.0, 0.1 * median_height)
    for position in sorted(raw_positions):
        if deduplicated and abs(position - statistics.mean(deduplicated[-1])) <= position_tolerance:
            deduplicated[-1].append(position)
        else:
            deduplicated.append([position])
    return [statistics.mean(group) for group in deduplicated]


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
    all_lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    caption_line: _LineItem | None,
) -> _TableCandidate:
    """合并横线核心与上下注释，并保留注释的独立行身份。"""

    rule_bbox = _bbox_union_many([line.bbox for line in rule_group])
    core_line_indices = {fragment.line_index for row in core_rows for fragment in row.fragments}
    caption_rows = _collect_caption_rows(all_rows, caption_line, rule_bbox, median_height)
    footnote_rows = _collect_footnote_rows(
        all_rows,
        all_lines,
        rule_bbox,
        median_height,
        core_line_indices,
        page_size,
        angle,
    )
    core_local_bbox = _bbox_union(rule_bbox, _bbox_union_many([row.bbox for row in core_rows]))
    caption_annotation = _build_table_annotation(
        "caption",
        caption_rows,
        excluded_line_indices=core_line_indices,
        excluded_local_bbox=core_local_bbox,
    )
    footnote_annotation = _build_table_annotation(
        "footnote",
        footnote_rows,
        excluded_line_indices=core_line_indices,
    )
    annotations = [
        annotation
        for annotation in (caption_annotation, footnote_annotation)
        if annotation is not None
    ]
    annotation_line_indices = set().union(
        *(annotation.line_indices for annotation in annotations),
    ) if annotations else set()
    included_rows = [*caption_rows, *core_rows, *footnote_rows]
    local_bbox = _bbox_union(core_local_bbox, _bbox_union_many([row.bbox for row in included_rows]))
    return _TableCandidate(
        bbox=_rotate_bbox_from_upright(local_bbox, page_size, angle),
        local_bbox=local_bbox,
        angle=angle,
        score=0.0,
        core_bbox=_rotate_bbox_from_upright(core_local_bbox, page_size, angle),
        # 表体成员与注释成员保持互斥；物化失败时会显式把无效注释放回表体投影。
        line_indices=core_line_indices - annotation_line_indices,
        annotations=annotations,
    )


def _build_table_annotation(
    kind: Literal["caption", "footnote"],
    rows: list[_VisualRow],
    *,
    excluded_line_indices: set[int] | None = None,
    excluded_local_bbox: BBox | None = None,
) -> _TableAnnotation | None:
    """把已确认视觉行压缩成一个带精确来源行集合的表格注释记录。"""

    excluded_line_indices = excluded_line_indices or set()
    fragments = [
        fragment
        for row in rows
        for fragment in row.fragments
        if fragment.line_index not in excluded_line_indices
        and not (
            excluded_local_bbox is not None
            and fragment.local_bbox[3] > excluded_local_bbox[1]
            and _bbox_axis_overlap_ratio(
                fragment.local_bbox,
                excluded_local_bbox,
                axis="x",
            )
            >= 0.05
        )
    ]
    if not fragments:
        return None
    line_bboxes: dict[int, BBox] = {}
    for fragment in fragments:
        existing_bbox = line_bboxes.get(fragment.line_index)
        line_bboxes[fragment.line_index] = (
            fragment.bbox
            if existing_bbox is None
            else _bbox_union(existing_bbox, fragment.bbox)
        )
    return _TableAnnotation(
        kind=kind,
        bbox=_bbox_union_many(list(line_bboxes.values())),
        line_indices=set(line_bboxes),
        line_bboxes=line_bboxes,
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
    lines: list[_LineItem],
    rule_bbox: BBox,
    median_height: float,
    core_line_indices: set[int],
    page_size: tuple[float, float],
    angle: int,
) -> list[_VisualRow]:
    """从表格下边界吸收具有表内引用和版面证据的表注连续行。"""

    output: list[_VisualRow] = []
    bottom = rule_bbox[3]
    note_chain_started = False
    margin = 2.0 * median_height
    selected_line_indices = set(core_line_indices)
    line_by_index = {line.source_index: line for line in lines}
    core_lines = [line for line in lines if line.source_index in core_line_indices]
    body_reference_height = _table_note_body_reference_height(
        lines,
        rule_bbox,
        median_height,
        core_line_indices,
        page_size,
        angle,
    )
    note_left: float | None = None
    note_height: float | None = None
    note_fonts: set[tuple[str, int]] = set()
    for row in rows:
        clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=margin)
        if clipped_row is None or clipped_row.bbox[3] <= bottom:
            continue
        line_indices = {fragment.line_index for fragment in clipped_row.fragments}
        if line_indices.issubset(selected_line_indices):
            bottom = max(bottom, clipped_row.bbox[3])
            continue
        row_gap = max(0.0, clipped_row.bbox[1] - bottom)
        row_lines = [
            line_by_index[line_index]
            for line_index in line_indices
            if line_index in line_by_index
        ]
        row_heights = [
            _line_effective_height(
                line,
                _rotate_bbox_to_upright(line.bbox, page_size, angle),
            )
            for line in row_lines
        ]
        row_height = statistics.median(row_heights) if row_heights else clipped_row.bbox[3] - clipped_row.bbox[1]
        row_fonts = {
            line.font_signature
            for line in row_lines
            if line.font_signature is not None and line.font_coverage >= 0.75
        }
        if note_chain_started and clipped_row.bbox[3] - rule_bbox[3] > 10.0 * median_height:
            break
        row_text = _visual_row_text(clipped_row)
        explicit_note = _is_table_note_text(row_text)
        auxiliary_marker = _extract_auxiliary_table_note_marker(row_text)
        auxiliary_note = (
            auxiliary_marker is not None
            and _table_core_references_marker(
                auxiliary_marker,
                core_lines,
                page_size,
                angle,
            )
        )
        first_gap_limit = 0.75 if auxiliary_note and not explicit_note else 1.25
        if row_gap > (first_gap_limit if not note_chain_started else 1.0) * median_height:
            break
        if not note_chain_started:
            if not explicit_note and not auxiliary_note:
                break
            if explicit_note:
                spatially_compatible = (
                    _bbox_axis_overlap_ratio(clipped_row.bbox, rule_bbox, axis="x") >= 0.35
                    and abs(clipped_row.bbox[0] - rule_bbox[0]) <= 2.0 * median_height
                    and row_height <= 1.15 * median_height
                )
            else:
                spatially_compatible = (
                    _bbox_axis_overlap_ratio(clipped_row.bbox, rule_bbox, axis="x") >= 0.50
                    and abs(clipped_row.bbox[0] - rule_bbox[0]) <= 3.0 * median_height
                    and row_height <= 1.05 * median_height
                    and row_height <= 0.90 * body_reference_height
                )
            if not spatially_compatible:
                break
            note_left = clipped_row.bbox[0]
            note_height = max(0.1, row_height)
            note_fonts = row_fonts
        elif (
            note_left is None
            or note_height is None
            or abs(clipped_row.bbox[0] - note_left) > 1.5 * median_height
            or not 0.75 <= row_height / note_height <= 1.25
            or (
                note_fonts
                and row_fonts
                and note_fonts.isdisjoint(row_fonts)
            )
            or _bbox_axis_overlap_ratio(clipped_row.bbox, rule_bbox, axis="x") < 0.35
        ):
            # 字号、字体或缩进突变表明已进入标题/正文，表注链必须立即终止。
            break
        output.append(clipped_row)
        selected_line_indices.update(line_indices)
        bottom = max(bottom, clipped_row.bbox[3])
        note_chain_started = True
    return output


def _extract_auxiliary_table_note_marker(text: str) -> str | None:
    """提取行首一至三个通用 Unicode 标记，不解释任何具体标记含义。"""

    match = _AUXILIARY_TABLE_NOTE_RE.match(str(text or ""))
    if match is None:
        return None
    marker = unicodedata.normalize("NFKC", match.group("marker")).casefold()
    if not marker or not all(unicodedata.category(char)[0] in {"L", "N", "S"} for char in marker):
        return None
    return marker


def _table_core_references_marker(
    marker: str,
    core_lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
) -> bool:
    """要求通用短标记在表格核心中具有上标或紧凑单元格引用。"""

    return any(
        _line_has_superscript_marker(line, marker, page_size, angle)
        or _line_has_compact_marker_token(line.text, marker)
        for line in core_lines
    )


def _line_has_compact_marker_token(text: str, marker: str) -> bool:
    """仅在短小单元格文本中确认独立标记 token，避免普通句子偶然命中。"""

    normalized_text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    if sum(not char.isspace() for char in normalized_text) > 12:
        return False
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized_text:
        if unicodedata.category(char)[0] in {"L", "N", "S"}:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return len(tokens) <= 4 and marker in tokens


def _line_has_superscript_marker(
    line: _LineItem,
    marker: str,
    page_size: tuple[float, float],
    angle: int,
) -> bool:
    """在正向局部坐标中检查标记字形是否同时更小并明显上移。"""

    glyphs: list[tuple[str, BBox]] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if not raw_char.isprintable() or raw_char.isspace():
            continue
        bbox = _coerce_bbox(char.get("bbox"))
        if bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, angle)
        glyphs.append((unicodedata.normalize("NFKC", raw_char).casefold(), local_bbox))
    if len(glyphs) < 2:
        return False

    for start_index in range(len(glyphs)):
        combined = ""
        for end_index in range(start_index, len(glyphs)):
            combined += glyphs[end_index][0]
            if not marker.startswith(combined):
                break
            if combined != marker:
                continue
            marker_indices = set(range(start_index, end_index + 1))
            ordinary_bboxes = [bbox for index, (_char, bbox) in enumerate(glyphs) if index not in marker_indices]
            if not ordinary_bboxes:
                continue
            normal_height = statistics.median(bbox[3] - bbox[1] for bbox in ordinary_bboxes)
            if normal_height <= 0:
                continue
            baseline_bboxes = [
                bbox
                for bbox in ordinary_bboxes
                if bbox[3] - bbox[1] >= 0.90 * normal_height
            ]
            marker_bboxes = [glyphs[index][1] for index in marker_indices]
            marker_height = statistics.median(bbox[3] - bbox[1] for bbox in marker_bboxes)
            marker_center = statistics.median(_bbox_center_y(bbox) for bbox in marker_bboxes)
            normal_center = statistics.median(_bbox_center_y(bbox) for bbox in baseline_bboxes)
            if (
                marker_height <= 0.85 * normal_height
                and normal_center - marker_center >= 0.12 * normal_height
            ):
                return True
    return False


def _table_note_body_reference_height(
    lines: list[_LineItem],
    rule_bbox: BBox,
    median_height: float,
    core_line_indices: set[int],
    page_size: tuple[float, float],
    angle: int,
) -> float:
    """以同方向非表格行的最高四分位估计正文高度，样本不足时稳健回退。"""

    exclusion_top = rule_bbox[1] - 3.0 * median_height
    exclusion_bottom = rule_bbox[3] + 10.0 * median_height
    heights: list[float] = []
    for line in lines:
        if line.angle != angle or line.source_index in core_line_indices:
            continue
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        if exclusion_top <= _bbox_center_y(local_bbox) <= exclusion_bottom:
            continue
        heights.append(_line_effective_height(line, local_bbox))
    if len(heights) < 4:
        return 1.25 * median_height
    heights.sort()
    upper_quartile_count = max(1, (len(heights) + 3) // 4)
    return statistics.median(heights[-upper_quartile_count:])


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
        _merge_table_candidate_annotations(target, candidate)
        target.score = max(target.score, candidate.score)
    return sorted(merged, key=lambda item: (item.bbox[1], item.bbox[0]))


def _merge_table_candidate_annotations(
    target: _TableCandidate,
    candidate: _TableCandidate,
) -> None:
    """按类型合并重复候选注释，并以表体优先消解来源行角色冲突。"""

    for annotation in candidate.annotations:
        existing = next(
            (
                item
                for item in target.annotations
                if item.kind == annotation.kind
            ),
            None,
        )
        if existing is None:
            target.annotations.append(
                _TableAnnotation(
                    kind=annotation.kind,
                    bbox=annotation.bbox,
                    line_indices=set(annotation.line_indices),
                    line_bboxes=dict(annotation.line_bboxes),
                )
            )
            continue
        existing.bbox = _bbox_union(existing.bbox, annotation.bbox)
        existing.line_indices.update(annotation.line_indices)
        for line_index, bbox in annotation.line_bboxes.items():
            existing_bbox = existing.line_bboxes.get(line_index)
            existing.line_bboxes[line_index] = (
                bbox
                if existing_bbox is None
                else _bbox_union(existing_bbox, bbox)
            )

    # 重复候选发生角色冲突时以任一候选确认的表体成员为准，避免表头被并入 caption。
    retained_annotations: list[_TableAnnotation] = []
    for annotation in target.annotations:
        annotation.line_indices.difference_update(target.line_indices)
        annotation.line_bboxes = {
            line_index: bbox
            for line_index, bbox in annotation.line_bboxes.items()
            if line_index in annotation.line_indices
        }
        if not annotation.line_indices:
            continue
        if annotation.line_bboxes:
            annotation.bbox = _bbox_union_many(
                list(annotation.line_bboxes.values()),
            )
        retained_annotations.append(annotation)
    target.annotations = retained_annotations


def _materialize_table_blocks(
    source: _PageSource,
    candidates: list[_TableCandidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    """原子物化表体及其独立注释，仅认领整组成功输出的文本行。"""

    table_blocks: list[dict[str, Any]] = []
    annotation_blocks: list[dict[str, Any]] = []
    accepted_candidate_bboxes: list[BBox] = []
    claimed: set[int] = set()
    native_chars = tuple(source.chars)
    native_rules = coerce_native_table_rules(source.drawing_lines)
    native_rectangles = coerce_native_table_rectangles(source.path_infos)
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        # 候选去重仍使用包含注释的完整框，不能因输出表体收缩而放行重复表格。
        if any(
            _bbox_overlap_in_smaller(candidate.bbox, bbox) >= 0.5
            for bbox in accepted_candidate_bboxes
        ):
            continue
        output_angle = candidate.angle
        candidate_annotation_blocks, externalized_line_indices, failed_annotations = (
            _materialize_table_annotations(source, candidate)
        )
        projection_line_indices = _candidate_projection_line_indices(source, candidate)
        for annotation in candidate.annotations:
            projection_line_indices.update(annotation.line_indices)
        projection_line_indices.difference_update(externalized_line_indices)
        body_bbox = _table_body_materialization_bbox(
            candidate,
            failed_annotations,
        )
        content = ""
        try:
            content = _recover_native_table_html(
                source,
                body_bbox,
                candidate.angle,
                native_chars,
                native_rules,
                native_rectangles,
            )
        except Exception as exc:
            logger.warning(
                "Flash native table recovery failed and fell back to projection: "
                f"bbox={candidate.bbox}, error={exc}"
            )
        if content:
            logger.debug(
                "Flash native table recovery accepted: "
                f"bbox={body_bbox}, angle={candidate.angle}"
            )
        try:
            if not content:
                # 使用完整原始字符流保留 PDF 物理换行；行索引仅负责表体与注释的所有权认领。
                content = project_pdf_table_text(
                    source.chars,
                    body_bbox,
                    angle=candidate.angle,
                )
        except Exception as exc:
            # 表体投影异常时同时撤销预构造注释，保持整组不输出、不认领。
            logger.warning(f"Flash table projection failed and rolled back: bbox={candidate.bbox}, error={exc}")
            continue
        if not content or not content.strip():
            continue
        table_blocks.append(
            {
                "type": "table",
                "bbox": body_bbox,
                "angle": output_angle,
                "content": content,
            }
        )
        annotation_blocks.extend(candidate_annotation_blocks)
        accepted_candidate_bboxes.append(candidate.bbox)
        # 表体和成功外置的注释行在同一事务中各认领一次。
        claimed.update(projection_line_indices | externalized_line_indices)
    table_blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    annotation_blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return table_blocks, annotation_blocks, claimed


def _materialize_table_annotations(
    source: _PageSource,
    candidate: _TableCandidate,
) -> tuple[list[dict[str, Any]], set[int], list[_TableAnnotation]]:
    """构造候选注释块，并返回成功外置行与需回退到表体的注释记录。"""

    blocks: list[dict[str, Any]] = []
    externalized_line_indices: set[int] = set()
    failed_annotations: list[_TableAnnotation] = []
    annotations = sorted(
        candidate.annotations,
        key=lambda annotation: (
            _rotate_bbox_to_upright(
                annotation.bbox,
                source.page_size,
                candidate.angle,
            )[1],
            _rotate_bbox_to_upright(
                annotation.bbox,
                source.page_size,
                candidate.angle,
            )[0],
            annotation.kind,
        ),
    )
    for annotation in annotations:
        block = _build_table_annotation_block(
            source,
            candidate,
            annotation,
        )
        if block is None:
            failed_annotations.append(annotation)
            continue
        blocks.append(block)
        externalized_line_indices.update(annotation.line_indices)
    return blocks, externalized_line_indices, failed_annotations


def _build_table_annotation_block(
    source: _PageSource,
    candidate: _TableCandidate,
    annotation: _TableAnnotation,
) -> dict[str, Any] | None:
    """按父表格局部方向整理原生行，并生成保留排版元数据的独立注释块。"""

    line_geometry = [
        (
            line,
            _rotate_bbox_to_upright(
                line.bbox,
                source.page_size,
                candidate.angle,
            ),
        )
        for line in source.lines
        if line.source_index in annotation.line_indices
    ]
    # 原生来源顺序可抵抗旋转文字同一基线上的字形顶边抖动。
    line_geometry.sort(key=lambda item: item[0].source_index)
    content = _merge_table_annotation_content(
        [line.text for line, _local_bbox in line_geometry],
    )
    if not line_geometry or not content:
        return None
    return {
        "type": annotation.kind,
        "bbox": annotation.bbox,
        "angle": candidate.angle,
        "content": content,
        "_local_line_bboxes": [bbox for _line, bbox in line_geometry],
        "_line_heights": [
            _line_effective_height(line, bbox)
            for line, bbox in line_geometry
        ],
        "_font_signatures": {
            line.font_signature
            for line, _bbox in line_geometry
            if line.font_signature is not None and line.font_coverage >= 0.5
        },
        # 已由表格检测器给出完整边界，禁止通用表注规则继续向下扩张。
        "_table_annotation_complete": True,
    }


def _merge_table_annotation_content(line_texts: list[str]) -> str:
    """按正文一致的语言与行末断词规则折叠表格注释原生行。"""

    normalized_lines = [
        normalized
        for text in line_texts
        if (normalized := _normalize_native_run_text(str(text or "")))
    ]
    if not normalized_lines:
        return ""
    try:
        block_language = detect_lang("".join(normalized_lines))
    except Exception:
        block_language = ""
    return merge_text_line_contents(
        normalized_lines,
        block_language=block_language,
    )


def _table_body_materialization_bbox(
    candidate: _TableCandidate,
    failed_annotations: list[_TableAnnotation],
) -> BBox:
    """返回排除有效注释后的表体框，并把无效注释边界保守并回表体。"""

    if not candidate.annotations or len(failed_annotations) == len(candidate.annotations):
        return candidate.bbox
    body_bbox = candidate.core_bbox or candidate.bbox
    for annotation in failed_annotations:
        body_bbox = _bbox_union(body_bbox, annotation.bbox)
    return body_bbox


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
