# Copyright (c) Opendatalab. All rights reserved.

"""检测、投影并物化 Flash 原生 PDF 表格。"""

from __future__ import annotations

import re
import statistics
from typing import Any

from loguru import logger

from mineru.backend.hybrid.table_text import project_pdf_table_text
from mineru.types import BBox

from .models import (
    _Fragment,
    _LineItem,
    _LocalAxisLine,
    _PageSource,
    _TableCandidate,
    _VisualRow,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_overlap_in_smaller,
    _bbox_union,
    _bbox_union_many,
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


def _detect_table_candidates(source: _PageSource) -> list[_TableCandidate]:
    """按文本方向融合三条长横线与规则文本分布，生成表格候选。"""

    candidates: list[_TableCandidate] = []
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
        candidates.extend(
            _build_rule_table_candidates(
                rows,
                angle_lines,
                source.page_size,
                angle,
                median_height,
                local_axis_lines,
            )
        )
    return _merge_table_candidates(candidates)


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
) -> list[_TableCandidate]:
    """用三条长横线确定区域，再以连续多列文本分布确认表格。"""

    candidates: list[_TableCandidate] = []
    for rule_group in _group_long_horizontal_rules(axis_lines, median_height):
        rule_bbox = _bbox_union_many([line.bbox for line in rule_group])
        core_rows: list[_VisualRow] = []
        for row in rows:
            clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=0.0)
            if clipped_row is not None and rule_bbox[1] <= clipped_row.center_y <= rule_bbox[3]:
                core_rows.append(clipped_row)

        dense_rows = _longest_dense_multi_cell_rows(core_rows, median_height)
        if len(dense_rows) < 3:
            continue
        stable_columns, column_coverage = _count_stable_columns(dense_rows, median_height)
        if stable_columns < 2 or column_coverage < 0.5:
            continue

        caption_line = _find_table_caption(lines, rule_bbox, page_size, angle, median_height)
        candidate = _expand_rule_table_candidate(
            rule_group,
            core_rows,
            rows,
            page_size,
            angle,
            median_height,
            caption_line,
        )
        candidate.score = float(len(rule_group) + len(dense_rows) + stable_columns)
        candidates.append(candidate)
    return candidates


def _group_long_horizontal_rules(
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[list[_LocalAxisLine]]:
    """按近似左右端点和纵向距离聚合长横线，并去除同位置重复路径。"""

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
    maximum_vertical_gap = 24.0 * median_height
    for span_group in span_groups:
        vertical_groups: list[list[_LocalAxisLine]] = []
        for line in sorted(span_group, key=lambda item: _bbox_center_y(item.bbox)):
            if (
                not vertical_groups
                or _bbox_center_y(line.bbox) - _bbox_center_y(vertical_groups[-1][-1].bbox) > maximum_vertical_gap
            ):
                vertical_groups.append([line])
            else:
                vertical_groups[-1].append(line)

        for vertical_group in vertical_groups:
            unique_lines: list[_LocalAxisLine] = []
            for line in vertical_group:
                if any(abs(_bbox_center_y(line.bbox) - _bbox_center_y(item.bbox)) <= 1.0 for item in unique_lines):
                    continue
                unique_lines.append(line)
            if len(unique_lines) >= 3:
                output.append(unique_lines)
    return output


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
