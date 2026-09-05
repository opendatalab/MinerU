# Copyright (c) Opendatalab. All rights reserved.
"""合并正文、公式上下文和列表引导块的空间组件。"""

from __future__ import annotations

import re
import statistics
from typing import Any

from .....types import BBox
from ..geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_y,
    _bbox_union_many,
    _rotate_bbox_from_upright,
)
from .common import (
    _FIGURE_CAPTION_MARKER_RE,
    _INLINE_MATH_RECOVERY_MARKER,
    _LABELLED_METADATA_RE,
    _LIST_ITEM_RE,
    _PARAGRAPH_FORMULA_CONTEXT_MARKER,
    _SHORT_SAME_BASELINE_PREFIX_RE,
    _URL_LINE_RE,
    _block_starts_with_short_wide_rows,
    _compatible_component_lane_width,
    _component_connection_skips_block,
    _component_declared_lane_interval,
    _component_lane_interval,
    _components_share_lane_role,
    _find_short_opener_pairs,
    _has_parallel_text_component,
    _merge_internal_text_block_group,
    _merge_text_line_content,
    _nearest_following_text_component,
    _nearest_tapered_tail_component,
    _text_component_sort_key,
)


def _merge_short_same_baseline_prefix_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """合并括号序号或时刻等短前缀与右侧同基线正文。"""

    replacements: dict[int, dict[str, Any]] = {}
    consumed: set[int] = set()
    for prefix_index, prefix in enumerate(blocks):
        prefix_rows = prefix.get("_local_line_bboxes")
        prefix_content = str(prefix.get("content") or "").strip()
        if (
            prefix_index in consumed
            or prefix.get("type") != "text"
            or not isinstance(prefix_rows, list)
            or len(prefix_rows) != 1
            or _SHORT_SAME_BASELINE_PREFIX_RE.match(prefix_content) is None
        ):
            continue
        prefix_bbox = prefix_rows[0]
        prefix_heights = [
            float(height)
            for height in prefix.get("_line_heights", [])
            if isinstance(height, (int, float)) and float(height) > 0
        ]
        prefix_height = statistics.median(prefix_heights) if prefix_heights else max(0.1, prefix_bbox[3] - prefix_bbox[1])
        angle = int(prefix.get("angle", 0) or 0) % 360
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        matches: list[tuple[float, int]] = []
        for host_index, host in enumerate(blocks):
            host_rows = host.get("_local_line_bboxes")
            if (
                host_index == prefix_index
                or host_index in consumed
                or host.get("type") != "text"
                or int(host.get("angle", 0) or 0) % 360 != angle
                or not isinstance(host_rows, list)
                or not host_rows
            ):
                continue
            host_bbox = host_rows[0]
            host_width = host_bbox[2] - host_bbox[0]
            horizontal_gap = host_bbox[0] - prefix_bbox[2]
            if (
                host_bbox[0] < prefix_bbox[2]
                or horizontal_gap > 1.25 * prefix_height
                or host_width < 0.15 * local_page_width
                or _bbox_axis_overlap_ratio(
                    prefix_bbox,
                    host_bbox,
                    axis="y",
                )
                < 0.5
            ):
                continue
            matches.append((horizontal_gap, host_index))
        if not matches:
            continue
        _gap, host_index = min(matches)
        replacement_index = min(prefix_index, host_index)
        replacement = _merge_internal_text_block_group(
            blocks,
            [prefix_index, host_index],
            preserve_visual_spaces=True,
        )
        replacement["type"] = "text"
        replacements[replacement_index] = replacement
        consumed.update({prefix_index, host_index})
    return [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]


def _blocks_share_boundary_visual_row(
    first: dict[str, Any],
    second: dict[str, Any],
    pair_height: float,
) -> bool:
    """检查前块末行与后块首行是否为被错误切开的同一视觉行。"""

    first_rows = first.get("_local_line_bboxes")
    second_rows = second.get("_local_line_bboxes")
    if (
        not isinstance(first_rows, list)
        or not isinstance(second_rows, list)
        or max(len(first_rows), len(second_rows)) < 3
        or len(first_rows) + len(second_rows) > 6
        or not _components_share_lane_role(first, second, pair_height)
    ):
        return False
    if first["bbox"][1] <= second["bbox"][1]:
        upper_rows, lower_rows = first_rows, second_rows
    else:
        upper_rows, lower_rows = second_rows, first_rows
    upper_boundary = max(upper_rows, key=lambda bbox: (_bbox_center_y(bbox), bbox[0]))
    lower_boundary = min(lower_rows, key=lambda bbox: (_bbox_center_y(bbox), bbox[0]))
    vertical_overlap = max(
        0.0,
        min(upper_boundary[3], lower_boundary[3]) - max(upper_boundary[1], lower_boundary[1]),
    )
    shorter_height = max(
        0.1,
        min(
            upper_boundary[3] - upper_boundary[1],
            lower_boundary[3] - lower_boundary[1],
        ),
    )
    horizontal_gap = lower_boundary[0] - upper_boundary[2]
    union_bbox = _bbox_union_many([first["bbox"], second["bbox"]])
    return (
        vertical_overlap / shorter_height >= 0.7
        and -0.2 * pair_height <= horizontal_gap <= 0.75 * pair_height
        and union_bbox[3] - union_bbox[1] <= 6.0 * pair_height
    )


def _merge_overlapping_same_line_text_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """合并块体或边界视觉行重叠的宽正文块，修复错误分栏。"""

    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for first_index, first in enumerate(blocks):
        first_bbox = first.get("bbox")
        first_rows = first.get("_local_line_bboxes")
        angle = int(first.get("angle", 0) or 0) % 360
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        if (
            first_index in consumed
            or first.get("type") != "text"
            or not isinstance(first_bbox, (list, tuple))
            or not isinstance(first_rows, list)
            or not 1 <= len(first_rows) <= 5
            or first_bbox[2] - first_bbox[0] < 0.3 * local_page_width
        ):
            continue
        first_heights = [
            float(height) for height in first.get("_line_heights", []) if isinstance(height, (int, float)) and height > 0
        ]
        first_height = statistics.median(first_heights) if first_heights else first_bbox[3] - first_bbox[1]
        for second_index in range(first_index + 1, len(blocks)):
            second = blocks[second_index]
            second_bbox = second.get("bbox")
            second_rows = second.get("_local_line_bboxes")
            if (
                second_index in consumed
                or second.get("type") != "text"
                or int(second.get("angle", 0) or 0) % 360 != angle
                or not isinstance(second_bbox, (list, tuple))
                or not isinstance(second_rows, list)
                or not 1 <= len(second_rows) <= 5
                or second_bbox[2] - second_bbox[0] < 0.3 * local_page_width
            ):
                continue
            later_block = max(
                (first, second),
                key=_text_component_sort_key,
            )
            if later_block.get("_hard_break_before") is True:
                continue
            second_heights = [
                float(height) for height in second.get("_line_heights", []) if isinstance(height, (int, float)) and height > 0
            ]
            second_height = statistics.median(second_heights) if second_heights else second_bbox[3] - second_bbox[1]
            pair_height = max(first_height, second_height, 0.1)
            vertical_overlap = max(
                0.0,
                min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]),
            )
            minimum_box_height = max(
                0.1,
                min(
                    first_bbox[3] - first_bbox[1],
                    second_bbox[3] - second_bbox[1],
                ),
            )
            first_fonts = first.get("_font_signatures")
            second_fonts = second.get("_font_signatures")
            fonts_conflict = (
                isinstance(first_fonts, set)
                and isinstance(second_fonts, set)
                and first_fonts
                and second_fonts
                and first_fonts.isdisjoint(second_fonts)
            )
            compact_overlap = (
                len(first_rows) <= 2
                and len(second_rows) <= 2
                and min(len(first_rows), len(second_rows)) == 1
                and abs(first_bbox[0] - second_bbox[0]) <= pair_height
                and vertical_overlap / minimum_box_height >= 0.5
                and _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x") >= 0.75
            )
            boundary_overlap = _blocks_share_boundary_visual_row(
                first,
                second,
                pair_height,
            )
            if fonts_conflict or not (compact_overlap or boundary_overlap):
                continue
            union_bbox = _bbox_union_many([first_bbox, second_bbox])
            if not boundary_overlap and union_bbox[3] - union_bbox[1] > 4.0 * pair_height:
                continue
            replacement_index = min(first_index, second_index)
            replacements[replacement_index] = _merge_internal_text_block_group(
                blocks,
                [first_index, second_index],
            )
            consumed.update({first_index, second_index})
            break
    return [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]


def _merge_inline_math_fragment_text_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """把同一宽正文行上下叠放的多个小数学碎片收回一个文本块。"""

    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for host_index, host in enumerate(blocks):
        host_bbox = host.get("bbox")
        host_heights = host.get("_line_heights")
        host_rows = host.get("_local_line_bboxes")
        angle = int(host.get("angle", 0) or 0) % 360
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        if (
            host_index in consumed
            or host.get("type") != "text"
            or (host.get("_single_run_row_id") is None and (not isinstance(host_rows, list) or len(host_rows) > 2))
            or not isinstance(host_bbox, (list, tuple))
            or host_bbox[2] - host_bbox[0] < 0.35 * local_page_width
            or not isinstance(host_heights, list)
            or not host_heights
        ):
            continue
        host_height = statistics.median(
            float(height) for height in host_heights if isinstance(height, (int, float)) and height > 0
        )
        fragment_indices: list[int] = []
        for candidate_index, candidate in enumerate(blocks):
            candidate_bbox = candidate.get("bbox")
            candidate_rows = candidate.get("_local_line_bboxes")
            if (
                candidate_index == host_index
                or candidate_index in consumed
                or candidate.get("type") != "text"
                or int(candidate.get("angle", 0) or 0) % 360 != angle
                or not isinstance(candidate_bbox, (list, tuple))
                or not isinstance(candidate_rows, list)
                or len(candidate_rows) != 1
                or candidate_bbox[2] - candidate_bbox[0] > 0.25 * (host_bbox[2] - host_bbox[0])
                or _bbox_axis_overlap_ratio(
                    host_bbox,
                    candidate_bbox,
                    axis="x",
                )
                <= 0.0
            ):
                continue
            union_bbox = _bbox_union_many([host_bbox, candidate_bbox])
            vertical_gap = max(
                0.0,
                max(host_bbox[1], candidate_bbox[1]) - min(host_bbox[3], candidate_bbox[3]),
            )
            if vertical_gap <= 0.75 * host_height and union_bbox[3] - union_bbox[1] <= 3.5 * host_height:
                fragment_indices.append(candidate_index)
        if len(fragment_indices) < 2:
            continue
        group_indices = [host_index, *fragment_indices]
        replacement_index = min(group_indices)
        replacements[replacement_index] = _merge_inline_math_recovery_group(
            blocks,
            group_indices,
        )
        consumed.update(group_indices)
    output = [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]
    output = _merge_hostless_inline_math_fragment_blocks(output, page_size)
    output = _merge_residual_narrow_math_text_blocks(output, page_size)
    return _merge_inline_math_paragraph_continuations(output, page_size)


def _component_local_union_bbox(
    block: dict[str, Any],
) -> BBox | None:
    """合并正文组件持有的正向行框，非法或缺失元数据时返回空。"""

    rows = block.get("_local_line_bboxes")
    if not isinstance(rows, list):
        return None
    bboxes: list[BBox] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            continue
        try:
            bbox = tuple(float(value) for value in row)
        except (TypeError, ValueError):
            continue
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            bboxes.append(bbox)  # type: ignore[arg-type]
    return _bbox_union_many(bboxes) if bboxes else None


def _merge_paragraph_formula_context_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """把误似行间公式的复杂行内分式与同栏前后正文恢复成一个块。"""

    terminal_re = re.compile(
        r"[.!?。！？][\]\)}）】》”’'\"]*$",
    )
    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    seed_indices = [
        index
        for index, block in enumerate(blocks)
        if block.get("type") == "text" and block.get(_PARAGRAPH_FORMULA_CONTEXT_MARKER) is True
    ]
    for seed_index in seed_indices:
        if seed_index in consumed:
            continue
        group = {seed_index}
        changed = True
        while changed:
            changed = False
            for candidate_index, candidate in enumerate(blocks):
                if candidate_index in group or candidate_index in consumed or candidate.get("type") != "text":
                    continue
                for member_index in group:
                    member = blocks[member_index]
                    if int(candidate.get("angle", 0) or 0) % 360 != int(member.get("angle", 0) or 0) % 360:
                        continue
                    pair_heights = [
                        float(height)
                        for block in (candidate, member)
                        for height in block.get("_line_heights", [])
                        if isinstance(height, (int, float)) and float(height) > 0
                    ]
                    if not pair_heights:
                        continue
                    pair_height = statistics.median(pair_heights)
                    if not _components_share_lane_role(
                        candidate,
                        member,
                        pair_height,
                    ):
                        continue
                    candidate_bbox = _component_local_union_bbox(candidate)
                    member_bbox = _component_local_union_bbox(member)
                    if candidate_bbox is None or member_bbox is None:
                        continue
                    vertical_gap = max(
                        candidate_bbox[1] - member_bbox[3],
                        member_bbox[1] - candidate_bbox[3],
                        0.0,
                    )
                    if vertical_gap > 1.5 * pair_height:
                        continue
                    candidate_below = candidate_bbox[1] >= member_bbox[3]
                    member_below = member_bbox[1] >= candidate_bbox[3]
                    if candidate_below and (
                        candidate.get("_hard_break_before") is True
                        or terminal_re.search(str(member.get("content") or "").rstrip()) is not None
                    ):
                        continue
                    if member_below and (
                        member.get("_hard_break_before") is True
                        or terminal_re.search(str(candidate.get("content") or "").rstrip()) is not None
                    ):
                        continue
                    group.add(candidate_index)
                    changed = True
                    break
                if changed:
                    break
        if len(group) < 2:
            continue
        ordered_group = sorted(group)
        replacement_index = min(ordered_group)
        merged = _merge_internal_text_block_group(
            blocks,
            ordered_group,
        )
        local_rows = [
            bbox for bbox in merged.get("_local_line_bboxes", []) if isinstance(bbox, (list, tuple)) and len(bbox) == 4
        ]
        maximum_width = max(
            (bbox[2] - bbox[0] for bbox in local_rows),
            default=0.0,
        )
        body_rows = [bbox for bbox in local_rows if bbox[2] - bbox[0] >= 0.75 * maximum_width]
        if len(body_rows) >= 2:
            # 复杂分式可能比正文左缘多探出少量 glyph；公开框按重复满行边界稳定收口。
            local_merged_bbox = _bbox_union_many(local_rows)
            local_output_bbox = (
                min(bbox[0] for bbox in body_rows),
                local_merged_bbox[1],
                max(bbox[2] for bbox in body_rows),
                local_merged_bbox[3],
            )
            merged["bbox"] = _rotate_bbox_from_upright(
                local_output_bbox,
                page_size,
                int(merged.get("angle", 0) or 0) % 360,
            )
        replacements[replacement_index] = merged
        consumed.update(ordered_group)
    return [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]


def _merge_residual_narrow_math_text_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """把仍嵌在宽正文行范围内的单个窄数学碎片吸收到唯一宿主块。"""

    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(blocks):
        candidate_bbox = candidate.get("bbox")
        candidate_rows = candidate.get("_local_line_bboxes")
        angle = int(candidate.get("angle", 0) or 0) % 360
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        if (
            candidate.get("type") != "text"
            or not isinstance(candidate_bbox, (list, tuple))
            or not isinstance(candidate_rows, list)
            or len(candidate_rows) != 1
            or candidate_bbox[2] - candidate_bbox[0] > 0.05 * local_page_width
        ):
            continue
        candidate_heights = [
            float(height) for height in candidate.get("_line_heights", []) if isinstance(height, (int, float)) and height > 0
        ]
        candidate_height = statistics.median(candidate_heights) if candidate_heights else candidate_bbox[3] - candidate_bbox[1]
        hosts: list[tuple[float, float, int]] = []
        for host_index, host in enumerate(blocks):
            host_bbox = host.get("bbox")
            host_heights = [
                float(height) for height in host.get("_line_heights", []) if isinstance(height, (int, float)) and height > 0
            ]
            if (
                host_index == candidate_index
                or host_index in consumed
                or host.get("type") != "text"
                or int(host.get("angle", 0) or 0) % 360 != angle
                or not isinstance(host_bbox, (list, tuple))
                or host_bbox[2] - host_bbox[0] < 0.3 * local_page_width
                or candidate_bbox[0] < host_bbox[0]
                or candidate_bbox[2] > host_bbox[2]
            ):
                continue
            host_height = statistics.median(host_heights) if host_heights else host_bbox[3] - host_bbox[1]
            vertical_overlap = max(
                0.0,
                min(candidate_bbox[3], host_bbox[3]) - max(candidate_bbox[1], host_bbox[1]),
            )
            vertical_gap = max(
                0.0,
                max(candidate_bbox[1], host_bbox[1]) - min(candidate_bbox[3], host_bbox[3]),
            )
            pair_height = max(candidate_height, host_height, 0.1)
            union_bbox = _bbox_union_many([candidate_bbox, host_bbox])
            if (
                vertical_overlap < 0.35 * min(candidate_height, host_height) and vertical_gap > 0.5 * pair_height
            ) or union_bbox[3] - union_bbox[1] > 3.0 * pair_height:
                continue
            hosts.append(
                (
                    -vertical_overlap,
                    abs(_bbox_center_y(candidate_bbox) - _bbox_center_y(host_bbox)),
                    host_index,
                )
            )
        if len(hosts) != 1:
            continue
        host_index = hosts[0][2]
        replacement_index = min(candidate_index, host_index)
        replacements[replacement_index] = _merge_inline_math_recovery_group(
            blocks,
            [candidate_index, host_index],
        )
        consumed.update({candidate_index, host_index})
    return [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]


def _merge_hostless_inline_math_fragment_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """把没有单一宽宿主但在一栏内二维密集排列的数学碎片合成文本块。"""

    grouped_indices: list[list[int]] = []
    for angle in sorted({int(block.get("angle", 0) or 0) % 360 for block in blocks if block.get("type") == "text"}):
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        candidates = {
            index
            for index, block in enumerate(blocks)
            if block.get("type") == "text"
            and int(block.get("angle", 0) or 0) % 360 == angle
            and isinstance(block.get("bbox"), (list, tuple))
            and isinstance(block.get("_local_line_bboxes"), list)
            and len(block["_local_line_bboxes"]) == 1
            and block["bbox"][2] - block["bbox"][0] <= 0.25 * local_page_width
        }
        while candidates:
            component = {candidates.pop()}
            changed = True
            while changed:
                changed = False
                for candidate_index in list(candidates):
                    candidate_bbox = blocks[candidate_index]["bbox"]
                    candidate_heights = blocks[candidate_index].get(
                        "_line_heights",
                        [],
                    )
                    candidate_height = (
                        statistics.median(candidate_heights) if candidate_heights else candidate_bbox[3] - candidate_bbox[1]
                    )
                    if any(
                        (
                            max(
                                0.0,
                                max(candidate_bbox[1], blocks[index]["bbox"][1])
                                - min(candidate_bbox[3], blocks[index]["bbox"][3]),
                            )
                            <= 0.75
                            * max(
                                candidate_height,
                                statistics.median(blocks[index].get("_line_heights", []))
                                if blocks[index].get("_line_heights")
                                else blocks[index]["bbox"][3] - blocks[index]["bbox"][1],
                            )
                            and max(
                                0.0,
                                max(candidate_bbox[0], blocks[index]["bbox"][0])
                                - min(candidate_bbox[2], blocks[index]["bbox"][2]),
                            )
                            <= 1.5
                            * max(
                                candidate_height,
                                statistics.median(blocks[index].get("_line_heights", []))
                                if blocks[index].get("_line_heights")
                                else blocks[index]["bbox"][3] - blocks[index]["bbox"][1],
                            )
                        )
                        for index in component
                    ):
                        component.add(candidate_index)
                        candidates.remove(candidate_index)
                        changed = True
            if len(component) < 4:
                continue
            component_heights = [
                float(height)
                for index in component
                for height in blocks[index].get("_line_heights", [])
                if isinstance(height, (int, float)) and height > 0
            ]
            if not component_heights:
                continue
            median_height = statistics.median(component_heights)
            union_bbox = _bbox_union_many([blocks[index]["bbox"] for index in component])
            font_signatures = set().union(
                *[
                    signatures
                    for index in component
                    if isinstance(
                        (signatures := blocks[index].get("_font_signatures")),
                        set,
                    )
                ]
            )
            if (
                len(font_signatures) < 2
                or not 0.25 * local_page_width <= union_bbox[2] - union_bbox[0] <= 0.5 * local_page_width
                or union_bbox[3] - union_bbox[1] > 3.5 * median_height
                or sum(blocks[index]["bbox"][2] - blocks[index]["bbox"][0] >= 3.5 * median_height for index in component) < 2
            ):
                continue
            grouped_indices.append(sorted(component))

    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for group in grouped_indices:
        if any(index in consumed for index in group):
            continue
        replacement_index = min(group)
        replacements[replacement_index] = _merge_inline_math_recovery_group(
            blocks,
            group,
        )
        consumed.update(group)
    return [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]


def _merge_inline_math_recovery_group(
    blocks: list[dict[str, Any]],
    indices: list[int],
) -> dict[str, Any]:
    """合并数学碎片并保留仅供后续段落闭合使用的内部标记。"""
    member_bboxes = [blocks[index]["bbox"] for index in indices]
    widths = [bbox[2] - bbox[0] for bbox in member_bboxes]
    maximum_width = max(widths, default=0.0)
    detected_regions = [
        bbox for bbox, width in zip(member_bboxes, widths) if maximum_width <= 0 or width <= 0.25 * maximum_width
    ]
    if not detected_regions:
        detected_regions = list(member_bboxes)
    merged = _merge_internal_text_block_group(blocks, indices)
    merged[_INLINE_MATH_RECOVERY_MARKER] = True
    merged["_inline_math_regions"] = [
        *merged.get("_inline_math_regions", []),
        *detected_regions,
    ]
    return merged


def _merge_inline_math_paragraph_continuations(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """在数学碎片恢复后，合并同栏连续且足够宽的正文段落块。"""

    if sum(block.get(_INLINE_MATH_RECOVERY_MARKER) is True for block in blocks) < 2:
        return blocks

    lane_groups: list[list[int]] = []
    for index, block in sorted(
        enumerate(blocks),
        key=lambda item: (
            int(item[1].get("angle", 0) or 0) % 360,
            _text_component_sort_key(item[1])
            if isinstance(item[1].get("_local_line_bboxes"), list) and item[1]["_local_line_bboxes"]
            else (float("inf"), float("inf")),
        ),
    ):
        local_rows = block.get("_local_line_bboxes")
        if (
            block.get("type") != "text"
            or not isinstance(block.get("_lane_is_span"), bool)
            or not isinstance(local_rows, list)
            or not local_rows
        ):
            continue
        block_heights = [
            float(height) for height in block.get("_line_heights", []) if isinstance(height, (int, float)) and height > 0
        ]
        if not block_heights:
            continue
        for lane_group in lane_groups:
            representative = blocks[lane_group[0]]
            if int(representative.get("angle", 0) or 0) % 360 != int(block.get("angle", 0) or 0) % 360:
                continue
            representative_heights = [
                float(height)
                for height in representative.get("_line_heights", [])
                if isinstance(height, (int, float)) and height > 0
            ]
            pair_height = statistics.median([*representative_heights, *block_heights])
            if _components_share_lane_role(representative, block, pair_height):
                lane_group.append(index)
                break
        else:
            lane_groups.append([index])

    candidate_chains: list[list[int]] = []
    for lane_group in lane_groups:
        ordered_indices = sorted(
            lane_group,
            key=lambda index: _text_component_sort_key(blocks[index]),
        )
        chain = [ordered_indices[0]]
        for current_index in ordered_indices[1:]:
            previous_index = chain[-1]
            previous = blocks[previous_index]
            current = blocks[current_index]
            previous_rows = previous["_local_line_bboxes"]
            current_rows = current["_local_line_bboxes"]
            previous_local_bbox = _bbox_union_many(previous_rows)
            current_local_bbox = _bbox_union_many(current_rows)
            pair_heights = [
                float(height)
                for block in (previous, current)
                for height in block.get("_line_heights", [])
                if isinstance(height, (int, float)) and height > 0
            ]
            pair_height = statistics.median(pair_heights)
            angle = int(previous.get("angle", 0) or 0) % 360
            local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
            lane_width = _compatible_component_lane_width(
                previous,
                current,
                local_page_width,
                pair_height,
            )
            previous_fonts = previous.get("_font_signatures")
            current_fonts = current.get("_font_signatures")
            fonts_conflict = (
                isinstance(previous_fonts, set)
                and isinstance(current_fonts, set)
                and previous_fonts
                and current_fonts
                and previous_fonts.isdisjoint(current_fonts)
            )
            vertical_gap = current_local_bbox[1] - previous_local_bbox[3]
            connects = (
                _components_share_lane_role(previous, current, pair_height)
                and previous_local_bbox[2] - previous_local_bbox[0] >= 0.8 * lane_width
                and current_local_bbox[2] - current_local_bbox[0] >= 0.8 * lane_width
                and abs(previous_local_bbox[0] - current_local_bbox[0]) <= 0.75 * pair_height
                and -0.75 * pair_height <= vertical_gap <= 0.75 * pair_height
                and not fonts_conflict
                and not _component_connection_skips_block(
                    blocks,
                    previous_index,
                    current_index,
                    pair_height,
                )
            )
            if connects:
                chain.append(current_index)
            else:
                candidate_chains.append(chain)
                chain = [current_index]
        candidate_chains.append(chain)

    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for chain in candidate_chains:
        if len(chain) < 3 or sum(blocks[index].get(_INLINE_MATH_RECOVERY_MARKER) is True for index in chain) < 2:
            continue
        chain_heights = [
            float(height)
            for index in chain
            for height in blocks[index].get("_line_heights", [])
            if isinstance(height, (int, float)) and height > 0
        ]
        median_height = statistics.median(chain_heights)
        local_union = _bbox_union_many([bbox for index in chain for bbox in blocks[index]["_local_line_bboxes"]])
        if local_union[3] - local_union[1] > 24.0 * median_height:
            continue
        replacement_index = min(chain)
        merged = _merge_internal_text_block_group(blocks, chain)
        merged[_INLINE_MATH_RECOVERY_MARKER] = True
        replacements[replacement_index] = merged
        consumed.update(chain)
    return [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]


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
        merged["content"] = _merge_text_line_content([str(blocks[index].get("content", "")) for index in ordered_indices])
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
            bbox for index in ordered_indices for bbox in blocks[index].get("_local_line_bboxes", [])
        ]
        merged["_local_output_line_bboxes"] = [
            bbox for index in ordered_indices for bbox in blocks[index].get("_local_output_line_bboxes", [])
        ]
        merged["_output_bbox_repaired"] = any(blocks[index].get("_output_bbox_repaired") is True for index in ordered_indices)
        merged["_line_heights"] = [height for index in ordered_indices for height in blocks[index].get("_line_heights", [])]
        merged["_font_signatures"] = set().union(
            *[
                signatures
                for index in ordered_indices
                if isinstance(
                    (signatures := blocks[index].get("_font_signatures")),
                    set,
                )
            ]
        )
        merged["_inline_math_regions"] = [
            region for index in ordered_indices for region in blocks[index].get("_inline_math_regions", [])
        ]
        output.append(merged)
    return output


def _merge_list_intro_text_components(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """在编号列表硬边界前合并被误拆的连续引导段和冒号短尾。"""

    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    text_indices = [
        index
        for index, block in enumerate(blocks)
        if block.get("type") == "text" and isinstance(block.get("_local_line_bboxes"), list) and block["_local_line_bboxes"]
    ]
    for boundary_index in text_indices:
        boundary = blocks[boundary_index]
        if (
            boundary.get("_hard_break_before") is not True
            or _LIST_ITEM_RE.match(
                str(boundary.get("content") or ""),
            )
            is None
        ):
            continue
        preceding = [
            index
            for index in text_indices
            if index not in consumed and _text_component_sort_key(blocks[index]) < _text_component_sort_key(boundary)
        ]
        if not preceding:
            continue
        immediate_index = max(
            preceding,
            key=lambda index: _text_component_sort_key(blocks[index]),
        )
        immediate = blocks[immediate_index]
        immediate_rows = immediate["_local_line_bboxes"]
        immediate_heights = [
            float(height) for height in immediate.get("_line_heights", []) if isinstance(height, (int, float)) and height > 0
        ]
        pair_height = statistics.median(
            immediate_heights
            or [
                immediate_rows[-1][3] - immediate_rows[-1][1],
            ],
        )
        interval = _component_lane_interval(immediate)
        if (
            interval is None
            or immediate_rows[-1][2] - immediate_rows[-1][0] > 0.35 * (interval[1] - interval[0])
            or not str(immediate.get("content") or "").rstrip().endswith((":", "："))
            or not _components_share_lane_role(
                immediate,
                boundary,
                pair_height,
            )
        ):
            continue

        group = [immediate_index]
        cursor_index = immediate_index
        while len(group) < 3:
            earlier = [
                index
                for index in preceding
                if index not in group
                and _text_component_sort_key(blocks[index]) < _text_component_sort_key(blocks[cursor_index])
                and _components_share_lane_role(
                    blocks[index],
                    blocks[cursor_index],
                    pair_height,
                )
            ]
            if not earlier:
                break
            previous_index = max(
                earlier,
                key=lambda index: _text_component_sort_key(blocks[index]),
            )
            previous = blocks[previous_index]
            current = blocks[cursor_index]
            previous_rows = previous["_local_line_bboxes"]
            current_rows = current["_local_line_bboxes"]
            heights = [
                float(height)
                for block in (previous, current)
                for height in block.get("_line_heights", [])
                if isinstance(height, (int, float)) and height > 0
            ]
            local_height = statistics.median(heights or [pair_height])
            vertical_gap = current_rows[0][1] - previous_rows[-1][3]
            if (
                previous.get("_hard_break_before") is True
                or not _components_share_lane_role(
                    previous,
                    current,
                    local_height,
                )
                or not -local_height <= vertical_gap <= local_height
            ):
                break
            group.append(previous_index)
            cursor_index = previous_index
        if len(group) < 2:
            continue
        ordered_group = sorted(
            group,
            key=lambda index: _text_component_sort_key(blocks[index]),
        )
        replacement_index = min(ordered_group)
        replacements[replacement_index] = _merge_internal_text_block_group(
            blocks,
            ordered_group,
        )
        consumed.update(ordered_group)

    return [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]


def _merge_unterminated_text_components(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并普通栏未终止正文，以及满足严格结构约束的满宽 span 正文。"""

    output = list(blocks)
    terminal_re = re.compile(
        r"[.!?。！？:：;；][\]\)}）】》”’'\"]*$",
    )
    while True:
        text_indices = sorted(
            (
                index
                for index, block in enumerate(output)
                if block.get("type") == "text"
                and isinstance(
                    block.get("_local_line_bboxes"),
                    list,
                )
                and block["_local_line_bboxes"]
            ),
            key=lambda index: _text_component_sort_key(output[index]),
        )
        merged_pair: tuple[int, int] | None = None
        for first_index, second_index in zip(
            text_indices,
            text_indices[1:],
        ):
            first = output[first_index]
            second = output[second_index]
            second_rows = second["_local_line_bboxes"]
            first_rows = first["_local_line_bboxes"]
            heights = [
                float(height)
                for block in (first, second)
                for height in block.get("_line_heights", [])
                if isinstance(height, (int, float)) and height > 0
            ]
            pair_height = statistics.median(
                heights
                or [
                    first_rows[-1][3] - first_rows[-1][1],
                    second_rows[0][3] - second_rows[0][1],
                ],
            )
            second_interval = _component_lane_interval(second)
            first_declared_interval = _component_declared_lane_interval(
                first,
            )
            second_declared_interval = _component_declared_lane_interval(
                second,
            )
            row_pair_height = statistics.median(
                [
                    max(
                        0.1,
                        first_rows[-1][3] - first_rows[-1][1],
                    ),
                    max(
                        0.1,
                        second_rows[0][3] - second_rows[0][1],
                    ),
                ],
            )
            span_connection_height = max(
                pair_height,
                row_pair_height,
            )
            span_tolerance = 0.75 * span_connection_height
            span_pair = (
                first.get("_lane_is_span") is True
                and second.get("_lane_is_span") is True
                and int(first.get("angle", 0) or 0) % 360 == int(second.get("angle", 0) or 0) % 360
                and first_declared_interval is not None
                and second_declared_interval is not None
                and abs(
                    first_declared_interval[0] - second_declared_interval[0],
                )
                <= span_tolerance
                and abs(
                    first_declared_interval[1] - second_declared_interval[1],
                )
                <= span_tolerance
            )
            first_content = str(first.get("content") or "")
            second_content = str(second.get("content") or "")
            single_numbered_tail = (
                len(second_rows) == 1
                and second.get("_hard_break_before") is not True
                and _LIST_ITEM_RE.match(second_content) is not None
                and first_content.rstrip().endswith((":", "："))
            )
            url_continuation = _URL_LINE_RE.match(second_content) is not None
            aligned_short_tail = (
                len(first_rows) >= 2
                and len(second_rows) == 1
                and abs(second_rows[0][0] - first_rows[-1][0]) <= 0.75 * pair_height
            )
            narrow_continuation = single_numbered_tail or url_continuation or aligned_short_tail
            starts_wide_label = (
                not url_continuation
                and _LABELLED_METADATA_RE.match(
                    second_content,
                )
                is not None
                and (reference_interval := (second_declared_interval if span_pair else second_interval)) is not None
                and second_rows[0][2] - second_rows[0][0] >= 0.5 * (reference_interval[1] - reference_interval[0])
            )
            if (
                second.get("_protected_hard_break_before") is True
                or (second.get("_hard_break_before") is True and (span_pair or not narrow_continuation))
                or second.get("_leading_emphasis_start") is True
                or (starts_wide_label and (span_pair or not aligned_short_tail))
                or first.get("_hanging_indent_group") is not None
                or second.get("_hanging_indent_group") is not None
                or _FIGURE_CAPTION_MARKER_RE.match(
                    first_content,
                )
                is not None
                or (
                    not span_pair
                    and not single_numbered_tail
                    and not url_continuation
                    and terminal_re.search(
                        first_content.rstrip(),
                    )
                    is not None
                )
            ):
                continue
            interval = _component_lane_interval(first)
            if span_pair:
                interval = first_declared_interval
            elif interval is None or not _components_share_lane_role(
                first,
                second,
                pair_height,
            ):
                continue
            if interval is None:
                continue
            lane_width = interval[1] - interval[0]
            vertical_gap = second_rows[0][1] - first_rows[-1][3]
            minimum_first_fill = 0.8 if span_pair else 0.5 if single_numbered_tail else 0.65
            minimum_second_fill = 0.8 if span_pair else 0.65
            connection_height = span_connection_height if span_pair else pair_height
            first_reference_width = (
                max(row[2] - row[0] for row in first_rows) if narrow_continuation else first_rows[-1][2] - first_rows[-1][0]
            )
            first_fonts = first.get("_font_signatures")
            second_fonts = second.get("_font_signatures")
            fonts_conflict = (
                isinstance(first_fonts, set)
                and isinstance(second_fonts, set)
                and first_fonts
                and second_fonts
                and first_fonts.isdisjoint(second_fonts)
            )
            if (
                first_reference_width < minimum_first_fill * lane_width
                or (
                    (span_pair or not narrow_continuation)
                    and second_rows[0][2] - second_rows[0][0] < minimum_second_fill * lane_width
                )
                or not -connection_height <= vertical_gap <= 1.5 * connection_height
                or fonts_conflict
                or _component_connection_skips_block(
                    output,
                    first_index,
                    second_index,
                    connection_height,
                )
            ):
                continue
            merged_pair = (first_index, second_index)
            break
        if merged_pair is None:
            return output
        first_index, second_index = merged_pair
        output[first_index] = _merge_internal_text_block_group(
            output,
            [first_index, second_index],
        )
        output.pop(second_index)


__all__ = [
    "_merge_short_same_baseline_prefix_blocks",
    "_blocks_share_boundary_visual_row",
    "_merge_overlapping_same_line_text_blocks",
    "_merge_inline_math_fragment_text_blocks",
    "_component_local_union_bbox",
    "_merge_paragraph_formula_context_blocks",
    "_merge_residual_narrow_math_text_blocks",
    "_merge_hostless_inline_math_fragment_blocks",
    "_merge_inline_math_recovery_group",
    "_merge_inline_math_paragraph_continuations",
    "_merge_spatial_text_components",
    "_merge_list_intro_text_components",
    "_merge_unterminated_text_components",
]
