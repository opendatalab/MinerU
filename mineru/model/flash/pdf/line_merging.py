# Copyright (c) Opendatalab. All rights reserved.

"""提供同基线文本和拆分视觉行的几何合并。"""

from __future__ import annotations

import statistics


from ....types import BBox

from .models import (
    _LineItem,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_y,
    _bbox_intersects,
    _bbox_union_many,
    _horizontal_bbox_gap,
    _rotate_bbox_to_upright,
)
from .native_text import (
    _fill_native_typography,
    _median_native_glyph_width,
)
from .line_layout import (
    _connection_crosses_table,
    _infer_text_lanes,
    _line_effective_height,
)


def _merge_same_baseline_text_lines(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在表格认领后合并同基线、同字体且水平邻近的正文 run。"""

    if len(lines) < 2:
        return list(lines)
    local_bboxes = [_rotate_bbox_to_upright(line.bbox, page_size, line.angle) for line in lines]
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找同行合并并查集的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        """合并两个满足同行条件的文本 run。"""

        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_line in enumerate(lines):
        for right_index in range(left_index + 1, len(lines)):
            right_line = lines[right_index]
            if _can_merge_same_baseline_pair(
                left_line,
                local_bboxes[left_index],
                right_line,
                local_bboxes[right_index],
                table_bboxes,
            ):
                union(left_index, right_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)

    output: list[_LineItem] = []
    for indices in groups.values():
        if len(indices) == 1:
            output.append(lines[indices[0]])
            continue
        indices.sort(key=lambda index: (local_bboxes[index][0], local_bboxes[index][1], lines[index].source_index))
        output.append(_merge_same_baseline_group(indices, lines, local_bboxes, page_size))
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _merge_overlapping_inline_text_clusters(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在容器认领后恢复由分子、分母和上下标拆成的二维物理文本行。"""

    if len(lines) < 2:
        return list(lines)

    consumed_source_indices: set[int] = set()
    merged_lines: list[_LineItem] = []
    for angle in sorted({line.angle for line in lines}):
        angle_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if len(angle_geometry) < 2:
            continue
        angle_median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in angle_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(
            angle_geometry,
            local_page_width,
            angle_median_height,
        )
        for lane in lanes:
            if lane.is_span or len(lane.lines) < 2:
                continue
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            lane_median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in lane.lines)
            parents = list(range(len(lane.lines)))

            def find(index: int) -> int:
                """查找当前栏二维文本簇并查集的根节点。"""

                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            def union(first_index: int, second_index: int) -> None:
                """合并两个满足二维物理行邻接条件的成员。"""

                first_root = find(first_index)
                second_root = find(second_index)
                if first_root != second_root:
                    parents[second_root] = first_root

            for first_index, first in enumerate(lane.lines):
                for second_index in range(first_index + 1, len(lane.lines)):
                    second = lane.lines[second_index]
                    if _overlapping_inline_cluster_pair_is_connected(
                        first,
                        second,
                        lane_median_height,
                        table_bboxes,
                        local_page_width=local_page_width,
                    ):
                        union(first_index, second_index)

            groups: dict[int, list[tuple[_LineItem, BBox]]] = {}
            for index, item in enumerate(lane.lines):
                groups.setdefault(find(index), []).append(item)
            for members in groups.values():
                cluster_kind = _classify_overlapping_inline_cluster(
                    members,
                    lane,
                    lane_median_height,
                    table_bboxes,
                )
                if cluster_kind is None:
                    continue
                merged_lines.append(
                    _merge_overlapping_inline_cluster(
                        members,
                        page_size,
                        lane_median_height,
                        compact_formula_cluster=cluster_kind == "formula",
                    )
                )
                consumed_source_indices.update(line.source_index for line, _bbox in members)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(merged_lines)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(
                line.bbox,
                page_size,
                line.angle,
            )[1],
            _rotate_bbox_to_upright(
                line.bbox,
                page_size,
                line.angle,
            )[0],
            line.source_index,
        )
    )
    return output


def _overlapping_inline_cluster_pair_is_connected(
    first: tuple[_LineItem, BBox],
    second: tuple[_LineItem, BBox],
    median_height: float,
    table_bboxes: list[BBox],
    *,
    local_page_width: float | None = None,
) -> bool:
    """判断两个同栏成员是否为同一二维物理行中的重叠片段。"""

    first_line, first_bbox = first
    second_line, second_bbox = second
    if first_line.angle != second_line.angle:
        return False
    if first_line.formula_candidate_only != second_line.formula_candidate_only:
        return False
    if first_line.semantic_type != second_line.semantic_type:
        return False
    if _connection_crosses_table(
        first_line.bbox,
        second_line.bbox,
        table_bboxes,
    ):
        return False
    if _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="y") < 0.55:
        return False

    first_guard_bbox = first_line.source_bbox or first_bbox
    second_guard_bbox = second_line.source_bbox or second_bbox
    left_bbox, right_bbox = sorted(
        (first_guard_bbox, second_guard_bbox),
        key=lambda bbox: bbox[0],
    )
    if (
        local_page_width is not None
        and (
            first_line.document_style_anomaly
            or second_line.document_style_anomaly
        )
        and left_bbox[2] <= 0.5 * local_page_width
        and right_bbox[0] >= 0.5 * local_page_width
        and right_bbox[0] - left_bbox[2]
        >= 0.02 * local_page_width
    ):
        return False

    horizontal_gap = _horizontal_bbox_gap(first_bbox, second_bbox)
    if first_line.visual_row_id == second_line.visual_row_id and (first_line.split_from_row or second_line.split_from_row):
        return horizontal_gap <= 3.0 * median_height
    pair_height = max(
        _line_effective_height(first_line, first_bbox),
        _line_effective_height(second_line, second_bbox),
    )
    return _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x") > 0.0 or horizontal_gap <= 1.5 * pair_height


def _classify_overlapping_inline_cluster(
    members: list[tuple[_LineItem, BBox]],
    lane: _TextLane,
    median_height: float,
    table_bboxes: list[BBox],
) -> str | None:
    """按正文宿主和紧凑程度区分行内文本簇与独立公式簇。"""

    if len(members) < 2:
        return None
    visual_row_ids = {line.visual_row_id for line, _bbox in members if line.visual_row_id is not None}
    if len(visual_row_ids) < 2:
        return None
    if any(_bbox_intersects(line.bbox, table_bbox) for line, _bbox in members for table_bbox in table_bboxes):
        return None

    union_bbox = _bbox_union_many([bbox for _line, bbox in members])
    if union_bbox[3] - union_bbox[1] > 3.0 * median_height:
        return None
    lane_width = max(0.1, lane.right - lane.left)
    has_fragment = any(
        _line_effective_height(line, bbox) <= 0.88 * median_height or line.font_coverage < 0.75 for line, bbox in members
    )
    if not has_fragment:
        return None

    has_body_host = any(
        bbox[2] - bbox[0] >= max(4.0 * median_height, 0.35 * lane_width)
        and _line_effective_height(line, bbox) >= 0.8 * median_height
        and line.font_coverage >= 0.75
        for line, bbox in members
    )
    if has_body_host:
        return "inline"
    if len(members) >= 3 and len(visual_row_ids) >= 3 and union_bbox[2] - union_bbox[0] <= 0.6 * lane_width:
        return "formula"
    return None


def _select_overlapping_inline_cluster_host(
    members: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> _LineItem:
    """选择与其他成员纵向重叠最多且最接近正文尺度的宿主行。"""

    union_bbox = _bbox_union_many([bbox for _line, bbox in members])
    union_center_y = _bbox_center_y(union_bbox)

    def host_score(item: tuple[_LineItem, BBox]) -> tuple[float, ...]:
        """生成宿主候选的正文尺度、同行支持和中心距离评分。"""

        line, bbox = item
        vertical_support = sum(
            max(0.0, min(bbox[3], other_bbox[3]) - max(bbox[1], other_bbox[1]))
            for other_line, other_bbox in members
            if other_line is not line
        )
        same_row_support = sum(
            line.visual_row_id is not None and line.visual_row_id == other_line.visual_row_id
            for other_line, _other_bbox in members
            if other_line is not line
        )
        body_like = float(_line_effective_height(line, bbox) >= 0.8 * median_height and line.font_coverage >= 0.75)
        return (
            body_like,
            float(same_row_support),
            vertical_support,
            bbox[2] - bbox[0],
            -abs(_bbox_center_y(bbox) - union_center_y),
            -float(line.source_index),
        )

    return max(members, key=host_score)[0]


def _merge_overlapping_inline_cluster(
    members: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
    median_height: float,
    *,
    compact_formula_cluster: bool,
) -> _LineItem:
    """按来源顺序合并二维文本簇，并保留字符与宿主排版信息。"""

    ordered_members = sorted(
        (line for line, _bbox in members),
        key=lambda line: line.source_index,
    )
    host = _select_overlapping_inline_cluster_host(members, median_height)
    detected_regions = (
        [line.bbox for line in ordered_members]
        if compact_formula_cluster
        else [line.bbox for line in ordered_members if line is not host]
    )
    merged = _LineItem(
        text=" ".join(text for line in ordered_members if (text := line.text.strip())),
        bbox=_bbox_union_many([line.bbox for line in ordered_members]),
        angle=host.angle,
        source_index=min(line.source_index for line in ordered_members),
        chars=[char for line in ordered_members for char in line.chars],
        visual_row_id=host.visual_row_id,
        run_index=host.run_index,
        effective_height=host.effective_height,
        em_height=host.em_height or host.effective_height,
        font_signature=host.font_signature,
        font_coverage=host.font_coverage,
        dominant_font_weight=host.dominant_font_weight,
        median_glyph_width=host.median_glyph_width,
        leading_emphasis_width=ordered_members[0].leading_emphasis_width,
        split_from_row=any(line.split_from_row for line in ordered_members),
        preserve_split_boundary=any(line.preserve_split_boundary for line in ordered_members),
        semantic_type=host.semantic_type,
        restored_inline_cluster=True,
        compact_formula_cluster=compact_formula_cluster,
        formula_candidate_only=all(line.formula_candidate_only for line in ordered_members),
        document_style_anomaly=any(
            line.document_style_anomaly
            for line in ordered_members
        ),
        inline_math_regions=[
            *(region for line in ordered_members for region in line.inline_math_regions),
            *detected_regions,
        ],
    )
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def _merge_post_semantic_text_runs(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在容器、公式和标题结束后合并紧贴同基线的普通混合字体 run。"""

    if len(lines) < 2:
        return list(lines)
    local_bboxes = [_rotate_bbox_to_upright(line.bbox, page_size, line.angle) for line in lines]
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找后处理同行合并分量的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并两个后处理同行分量。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first_line in enumerate(lines):
        if first_line.semantic_type is not None:
            continue
        first_bbox = local_bboxes[first_index]
        first_height = _line_effective_height(first_line, first_bbox)
        for second_index in range(first_index + 1, len(lines)):
            second_line = lines[second_index]
            if second_line.semantic_type is not None or first_line.angle != second_line.angle:
                continue
            if first_line.formula_candidate_only != second_line.formula_candidate_only:
                continue
            if _connection_crosses_table(
                first_line.bbox,
                second_line.bbox,
                table_bboxes,
            ):
                continue
            second_bbox = local_bboxes[second_index]
            second_height = _line_effective_height(second_line, second_bbox)
            if _post_semantic_same_baseline_geometry(
                first_bbox,
                first_height,
                second_bbox,
                second_height,
            ):
                union(first_index, second_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)
    output: list[_LineItem] = []
    for indices in groups.values():
        if len(indices) == 1:
            output.append(lines[indices[0]])
            continue
        indices.sort(
            key=lambda index: (
                local_bboxes[index][0],
                local_bboxes[index][1],
                lines[index].source_index,
            )
        )
        output.append(
            _merge_same_baseline_group(
                indices,
                lines,
                local_bboxes,
                page_size,
            )
        )
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _post_semantic_same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
) -> bool:
    """放宽上下标字号差异，仅合并水平紧贴且垂直充分交叠的普通 run。"""

    pair_height = max(first_height, second_height)
    if (
        min(first_height, second_height) <= 0
        or pair_height
        / min(
            first_height,
            second_height,
        )
        > 1.6
    ):
        return False
    y_overlap = max(
        0.0,
        min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]),
    )
    shorter_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / shorter_bbox_height < 0.7:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    horizontal_gap = right_bbox[0] - left_bbox[2]
    return -0.2 * pair_height <= horizontal_gap <= max(2.0, 0.35 * pair_height)


def _can_merge_same_baseline_pair(
    first: _LineItem,
    first_bbox: BBox,
    second: _LineItem,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """判断两个剩余文本 run 是否属于同一条物理基线。"""

    if first.angle != second.angle:
        return False
    if first.formula_candidate_only != second.formula_candidate_only:
        return False
    if first.semantic_type != second.semantic_type:
        return False
    if first.visual_row_id == second.visual_row_id and (first.split_from_row or second.split_from_row):
        return False
    if _connection_crosses_table(first.bbox, second.bbox, table_bboxes):
        return False
    first_height = _line_effective_height(first, first_bbox)
    second_height = _line_effective_height(second, second_bbox)
    has_compatible_dominant_font = not (
        first.font_signature is None
        or second.font_signature is None
        or first.font_coverage < 0.75
        or second.font_coverage < 0.75
        or first.font_signature != second.font_signature
    )
    if has_compatible_dominant_font and _same_baseline_geometry(
        first_bbox,
        first_height,
        second_bbox,
        second_height,
    ):
        return True
    return _touching_same_baseline_geometry(
        first_bbox,
        first_height,
        second_bbox,
        second_height,
    )


def _touching_same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
) -> bool:
    """为低字体覆盖率 run 提供严格的紧贴同基线几何兜底。"""

    pair_height = max(first_height, second_height)
    y_overlap = max(0.0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    smaller_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / smaller_bbox_height < 0.7:
        return False
    if abs(_bbox_center_y(first_bbox) - _bbox_center_y(second_bbox)) > 0.5 * pair_height:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    signed_gap = right_bbox[0] - left_bbox[2]
    return -0.15 * pair_height <= signed_gap <= 0.75


def _same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
    *,
    maximum_gap: float | None = None,
) -> bool:
    """仅依据行高、垂直交叠和水平净空判断两个局部 bbox 是否同基线相邻。"""

    pair_height = max(first_height, second_height)
    if min(first_height, second_height) <= 0 or pair_height / min(first_height, second_height) > 1.35:
        return False
    y_overlap = max(0.0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    shorter_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / shorter_bbox_height < 0.7 and abs(first_bbox[3] - second_bbox[3]) > 0.25 * pair_height:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    signed_gap = right_bbox[0] - left_bbox[2]
    gap_limit = max(3.0, 0.75 * pair_height) if maximum_gap is None else maximum_gap
    return -0.25 * pair_height <= signed_gap <= gap_limit


def _merge_same_baseline_group(
    indices: list[int],
    lines: list[_LineItem],
    local_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> _LineItem:
    """按局部 x 顺序合并一个同基线分量，并保留全部字符与几何信息。"""

    members = [lines[index] for index in indices]
    content_parts = [members[0].text.strip()]
    for previous_index, current_index in zip(indices, indices[1:]):
        previous_bbox = local_bboxes[previous_index]
        current_bbox = local_bboxes[current_index]
        signed_gap = current_bbox[0] - previous_bbox[2]
        glyph_width = statistics.median(
            [
                width
                for member in (lines[previous_index], lines[current_index])
                if (width := _median_native_glyph_width(member, page_size)) is not None
            ]
            or [1.0]
        )
        separator = "" if signed_gap <= max(0.5, 0.25 * glyph_width) else " "
        content_parts.extend([separator, lines[current_index].text.strip()])

    merged = _LineItem(
        text="".join(content_parts).strip(),
        bbox=_bbox_union_many([member.bbox for member in members]),
        angle=members[0].angle,
        source_index=min(member.source_index for member in members),
        chars=[char for member in members for char in member.chars],
        visual_row_id=min(
            (member.visual_row_id for member in members if member.visual_row_id is not None),
            default=None,
        ),
        run_index=min(member.run_index for member in members),
        effective_height=statistics.median(member.effective_height for member in members),
        em_height=statistics.median(
            member.em_height or member.effective_height
            for member in members
        ),
        font_signature=members[0].font_signature,
        font_coverage=min(member.font_coverage for member in members),
        dominant_font_weight=statistics.median(
            member.dominant_font_weight for member in members if member.dominant_font_weight is not None
        )
        if any(member.dominant_font_weight is not None for member in members)
        else None,
        median_glyph_width=statistics.median(
            member.median_glyph_width for member in members if member.median_glyph_width is not None
        )
        if any(member.median_glyph_width is not None for member in members)
        else None,
        leading_emphasis_width=members[0].leading_emphasis_width,
        split_from_row=any(member.split_from_row for member in members),
        preserve_split_boundary=any(member.preserve_split_boundary for member in members),
        semantic_type=members[0].semantic_type,
        restored_inline_cluster=any(member.restored_inline_cluster for member in members),
        compact_formula_cluster=any(member.compact_formula_cluster for member in members),
        formula_candidate_only=all(member.formula_candidate_only for member in members),
        document_style_anomaly=any(
            member.document_style_anomaly
            for member in members
        ),
        inline_math_regions=[region for member in members for region in member.inline_math_regions],
    )
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def _join_formula_visual_row(
    row: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
) -> str:
    """将一个公式视觉行按局部 x 排序，并按字宽估计几何空格。"""

    ordered = sorted(row, key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    if not ordered:
        return ""
    parts = [ordered[0][0].text.strip()]
    for previous, current in zip(ordered, ordered[1:]):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        gap = current_bbox[0] - previous_bbox[2]
        pair_height = max(
            _line_effective_height(previous_line, previous_bbox),
            _line_effective_height(current_line, current_bbox),
        )
        glyph_widths = [
            width
            for line in (previous_line, current_line)
            if (width := _median_native_glyph_width(line, page_size)) is not None
        ]
        glyph_width = statistics.median(glyph_widths) if glyph_widths else max(1.0, 0.5 * pair_height)
        if gap <= max(0.5, 0.2 * pair_height):
            separator = ""
        else:
            separator = " " * max(1, min(8, int(round(gap / glyph_width))))
        parts.extend([separator, current_line.text.strip()])
    return "".join(parts).strip()


def _restore_dense_split_visual_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在公式认领后恢复同一栏带内被均匀大空格拆开的密集原生视觉行。"""

    if len(lines) < 3:
        return list(lines)
    lane_keys: dict[int, tuple[int, int]] = {}
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if not line_geometry:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        for lane_index, lane in enumerate(lanes):
            if lane.is_span:
                continue
            for line, _bbox in lane.lines:
                lane_keys[line.source_index] = (angle, lane_index)

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    consumed_source_indices: set[int] = set()
    restored_lines: list[_LineItem] = []
    for members in row_groups.values():
        if not _can_restore_dense_split_visual_row(
            members,
            page_size,
            table_bboxes,
            lane_keys,
        ):
            continue
        restored_lines.append(_merge_dense_split_visual_row(members, page_size))
        consumed_source_indices.update(member.source_index for member in members)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(restored_lines)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _merge_title_resolved_visual_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """在标题判定后合并同行标题或已降级的混合字体正文 run。"""

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    consumed_source_indices: set[int] = set()
    merged_rows: list[_LineItem] = []
    for members in row_groups.values():
        if len(members) < 2 or not all(member.split_from_row for member in members):
            continue
        if any(member.preserve_split_boundary for member in members):
            continue
        semantic_types = {member.semantic_type for member in members}
        if len(semantic_types) != 1:
            continue
        semantic_type = next(iter(semantic_types))
        font_signatures = {member.font_signature for member in members}
        dense_same_font_text = _is_dense_same_font_two_run_row(
            members,
            page_size,
        )
        if semantic_type != "paragraph_title" and not (
            semantic_type is None and (len(font_signatures) > 1 or dense_same_font_text)
        ):
            continue
        local_geometry = [
            (
                member,
                _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
            )
            for member in members
        ]
        local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
        if any(
            not _same_baseline_geometry(
                previous[1],
                _line_effective_height(*previous),
                current[1],
                _line_effective_height(*current),
                maximum_gap=3.0
                * max(
                    _line_effective_height(*previous),
                    _line_effective_height(*current),
                ),
            )
            for previous, current in zip(local_geometry, local_geometry[1:])
        ):
            continue
        merged_rows.append(_merge_dense_split_visual_row(members, page_size))
        consumed_source_indices.update(member.source_index for member in members)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(merged_rows)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _is_dense_same_font_two_run_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> bool:
    """检查两个普通文本 run 是否为同字体且占用充分的完整视觉行。"""

    if (
        len(members) != 2
        or not all(member.split_from_row for member in members)
        or any(member.preserve_split_boundary for member in members)
        or any(member.semantic_type is not None for member in members)
        or any(member.font_signature is None for member in members)
        or any(member.font_coverage < 0.75 for member in members)
    ):
        return False
    ordered = sorted(members, key=lambda member: member.run_index)
    if [member.run_index for member in ordered] != [0, 1]:
        return False
    if ordered[0].font_signature != ordered[1].font_signature:
        return False

    local_geometry = [
        (
            member,
            _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
        )
        for member in ordered
    ]
    local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    first, second = local_geometry
    pair_height = max(
        _line_effective_height(*first),
        _line_effective_height(*second),
    )
    if not _same_baseline_geometry(
        first[1],
        _line_effective_height(*first),
        second[1],
        _line_effective_height(*second),
        maximum_gap=3.0 * pair_height,
    ):
        return False

    union_bbox = _bbox_union_many([bbox for _member, bbox in local_geometry])
    occupied_width = sum(bbox[2] - bbox[0] for _member, bbox in local_geometry)
    return occupied_width / max(0.1, union_bbox[2] - union_bbox[0]) >= 0.85


def _can_restore_dense_split_visual_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
    lane_keys: dict[int, tuple[int, int]],
) -> bool:
    """检查 hard-split run 是否构成同字体且占用充分的完整视觉行。"""

    if (
        len(members) < 2
        or not all(member.split_from_row for member in members)
        or any(member.preserve_split_boundary for member in members)
    ):
        return False
    if len({member.semantic_type for member in members}) != 1:
        return False
    if len({member.formula_candidate_only for member in members}) != 1:
        return False
    ordered = sorted(members, key=lambda member: member.run_index)
    if [member.run_index for member in ordered] != list(range(len(ordered))):
        return False
    member_lane_keys = {lane_keys.get(member.source_index) for member in ordered}
    same_inferred_lane = None not in member_lane_keys and len(member_lane_keys) == 1
    font_signatures = {member.font_signature for member in ordered}
    if len(font_signatures) != 1:
        return False
    if any(_bbox_intersects(member.bbox, table_bbox) for member in ordered for table_bbox in table_bboxes):
        return False

    local_geometry = [
        (
            member,
            _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
        )
        for member in ordered
    ]
    local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    if not same_inferred_lane:
        member_widths = [bbox[2] - bbox[0] for _member, bbox in local_geometry]
        if len(members) == 2 and min(member_widths) > 0.35 * max(member_widths):
            return False
        if 3 <= len(members) <= 6:
            return False
    heights = [_line_effective_height(member, bbox) for member, bbox in local_geometry]
    median_height = statistics.median(heights)
    glyph_widths = [
        width for member, _bbox in local_geometry if (width := _median_native_glyph_width(member, page_size)) is not None
    ]
    median_glyph_width = statistics.median(glyph_widths) if glyph_widths else 0.0
    gap_limit = (
        max(12.0, 1.75 * median_height, 3.0 * median_glyph_width)
        if same_inferred_lane
        else max(8.0, 2.0 * median_height, 2.5 * median_glyph_width)
    )
    for previous, current in zip(local_geometry, local_geometry[1:]):
        if not _same_baseline_geometry(
            previous[1],
            _line_effective_height(*previous),
            current[1],
            _line_effective_height(*current),
            maximum_gap=gap_limit,
        ):
            return False

    union_bbox = _bbox_union_many([bbox for _member, bbox in local_geometry])
    occupied_width = sum(bbox[2] - bbox[0] for _member, bbox in local_geometry)
    minimum_occupancy = (
        0.8
        if len(members) == 2 and same_inferred_lane
        else 0.85
        if len(members) == 2
        else 0.65
        if not same_inferred_lane
        else 0.65
    )
    return occupied_width / max(0.1, union_bbox[2] - union_bbox[0]) >= minimum_occupancy


def _merge_dense_split_visual_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> _LineItem:
    """按局部 x 顺序恢复密集视觉行，正净空使用单空格，重叠片段直接连接。"""

    ordered_geometry = sorted(
        (
            (
                member,
                _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
            )
            for member in members
        ),
        key=lambda item: (item[1][0], item[1][1], item[0].source_index),
    )
    content_parts = [ordered_geometry[0][0].text.strip()]
    for previous, current in zip(ordered_geometry, ordered_geometry[1:]):
        separator = "" if current[1][0] <= previous[1][2] else " "
        content_parts.extend([separator, current[0].text.strip()])

    ordered_members = [member for member, _bbox in ordered_geometry]
    merged = _LineItem(
        text="".join(content_parts).strip(),
        bbox=_bbox_union_many([member.bbox for member in ordered_members]),
        angle=ordered_members[0].angle,
        source_index=min(member.source_index for member in ordered_members),
        chars=[char for member in ordered_members for char in member.chars],
        visual_row_id=ordered_members[0].visual_row_id,
        run_index=0,
        effective_height=statistics.median(_line_effective_height(member, bbox) for member, bbox in ordered_geometry),
        em_height=statistics.median(
            _line_effective_height(member, bbox)
            for member, bbox in ordered_geometry
        ),
        font_signature=ordered_members[0].font_signature,
        font_coverage=min(member.font_coverage for member in ordered_members),
        dominant_font_weight=statistics.median(
            member.dominant_font_weight for member in ordered_members if member.dominant_font_weight is not None
        )
        if any(member.dominant_font_weight is not None for member in ordered_members)
        else None,
        median_glyph_width=statistics.median(
            member.median_glyph_width for member in ordered_members if member.median_glyph_width is not None
        )
        if any(member.median_glyph_width is not None for member in ordered_members)
        else None,
        leading_emphasis_width=ordered_members[0].leading_emphasis_width,
        split_from_row=False,
        preserve_split_boundary=any(member.preserve_split_boundary for member in ordered_members),
        semantic_type=ordered_members[0].semantic_type,
        restored_inline_cluster=any(member.restored_inline_cluster for member in ordered_members),
        compact_formula_cluster=any(member.compact_formula_cluster for member in ordered_members),
        formula_candidate_only=all(member.formula_candidate_only for member in ordered_members),
        document_style_anomaly=any(
            member.document_style_anomaly
            for member in ordered_members
        ),
        inline_math_regions=[region for member in ordered_members for region in member.inline_math_regions],
    )
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged
