# Copyright (c) Opendatalab. All rights reserved.
"""组装跨行标题、图片注释、页眉及首页信息块。"""

from __future__ import annotations

import statistics
from typing import Any

from .....types import BBox
from ..geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_union_many,
)
from ..line_layout import (
    _is_structural_typography_gap,
)
from .common import _FIGURE_CAPTION_MARKER_RE, _components_share_lane_role, _merge_internal_text_block_group


def _merge_image_caption_text_blocks(
    blocks: list[dict[str, Any]],
    image_bboxes: list[BBox],
) -> list[dict[str, Any]]:
    """在图像邻接已成立后，用通用图注标记确认锚点并吸收同字体续行。"""

    if not image_bboxes:
        return blocks
    text_indices = [
        index
        for index, block in enumerate(blocks)
        if block.get("type") == "text"
        and isinstance(block.get("content"), str)
        and isinstance(block.get("bbox"), (list, tuple))
    ]
    all_heights = [
        float(height)
        for index in text_indices
        for height in blocks[index].get("_line_heights", [])
        if isinstance(height, (int, float)) and height > 0
    ]
    median_height = statistics.median(all_heights) if all_heights else 1.0
    caption_image_bboxes = _caption_image_group_bboxes(
        image_bboxes,
        median_height,
    )
    seed_indices = {
        index
        for index in text_indices
        if _FIGURE_CAPTION_MARKER_RE.match(str(blocks[index]["content"]).strip())
        and any(
            _caption_seed_matches_image(
                blocks[index],
                image_bbox,
                median_height,
            )
            for image_bbox in caption_image_bboxes
        )
    }
    if not seed_indices:
        return blocks

    assignments: dict[int, list[int]] = {index: [] for index in seed_indices}
    for candidate_index in text_indices:
        if candidate_index in seed_indices:
            continue
        candidate = blocks[candidate_index]
        matches: list[tuple[float, float, int]] = []
        for seed_index in seed_indices:
            seed = blocks[seed_index]
            if not _caption_tail_matches_seed(
                seed,
                candidate,
                median_height,
            ):
                continue
            seed_bbox = seed["bbox"]
            candidate_bbox = candidate["bbox"]
            matches.append(
                (
                    _bbox_center_y(candidate_bbox) - _bbox_center_y(seed_bbox),
                    abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(seed_bbox)),
                    seed_index,
                )
            )
        if matches:
            assignments[min(matches)[2]].append(candidate_index)

    merged_indices: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for seed_index, tail_indices in assignments.items():
        if not tail_indices:
            continue
        group_indices = [seed_index, *tail_indices]
        replacements[seed_index] = _merge_internal_text_block_group(
            blocks,
            group_indices,
        )
        merged_indices.update(tail_indices)
    return [replacements.get(index, block) for index, block in enumerate(blocks) if index not in merged_indices]


def _caption_image_group_bboxes(
    image_bboxes: list[BBox],
    median_height: float,
) -> list[BBox]:
    """合并同一视觉行的并排图片 bbox，使跨多图的统一图注也能建立邻接。"""

    remaining = list(image_bboxes)
    grouped_bboxes = list(image_bboxes)
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                aligned = False
                for member in group:
                    overlap = max(
                        0.0,
                        min(candidate[3], member[3]) - max(candidate[1], member[1]),
                    )
                    minimum_height = max(
                        0.1,
                        min(
                            candidate[3] - candidate[1],
                            member[3] - member[1],
                        ),
                    )
                    horizontal_gap = max(
                        0.0,
                        max(candidate[0], member[0]) - min(candidate[2], member[2]),
                    )
                    if overlap / minimum_height >= 0.7 and horizontal_gap <= 2.0 * median_height:
                        aligned = True
                        break
                if aligned:
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        if len(group) >= 2:
            grouped_bboxes.append(_bbox_union_many(group))
    return grouped_bboxes


def _caption_seed_matches_image(
    block: dict[str, Any],
    image_bbox: BBox,
    median_height: float,
) -> bool:
    """用上下位置、水平投影和居中关系确认图像下方的图注空间候选。"""

    bbox = block["bbox"]
    image_width = max(0.1, image_bbox[2] - image_bbox[0])
    block_width = max(0.1, bbox[2] - bbox[0])
    vertical_gap = max(0.0, bbox[1] - image_bbox[3])
    return (
        _bbox_center_y(bbox) >= image_bbox[3] - 0.25 * median_height
        and vertical_gap <= 2.5 * median_height
        and _bbox_axis_overlap_ratio(bbox, image_bbox, axis="x") >= 0.35
        and abs(_bbox_center_x(bbox) - _bbox_center_x(image_bbox)) <= 0.35 * max(image_width, block_width)
        and block_width <= 1.75 * image_width
    )


def _caption_body_has_structural_gap(
    seed: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    """用图注末行、候选首行和图注内部行距阻止跨排版层级回并。"""

    seed_bboxes = seed.get("_local_line_bboxes")
    seed_heights = seed.get("_line_heights")
    candidate_bboxes = candidate.get("_local_line_bboxes")
    candidate_heights = candidate.get("_line_heights")
    if not (
        isinstance(seed_bboxes, list)
        and isinstance(seed_heights, list)
        and len(seed_bboxes) == len(seed_heights)
        and seed_bboxes
        and isinstance(candidate_bboxes, list)
        and isinstance(candidate_heights, list)
        and len(candidate_bboxes) == len(candidate_heights)
        and candidate_bboxes
    ):
        return False

    seed_rows = sorted(
        zip(seed_bboxes, seed_heights, strict=True),
        key=lambda item: (item[0][1], item[0][0]),
    )
    candidate_rows = sorted(
        zip(candidate_bboxes, candidate_heights, strict=True),
        key=lambda item: (item[0][1], item[0][0]),
    )
    previous_bbox, previous_height = seed_rows[-1]
    current_bbox, current_height = candidate_rows[0]
    internal_gaps = [
        max(0.0, current[0][1] - (previous[0][1] + float(previous[1]))) for previous, current in zip(seed_rows, seed_rows[1:])
    ]
    regular_gap = statistics.median(internal_gaps) if internal_gaps else 0.0
    gap_mad = statistics.median(abs(gap - regular_gap) for gap in internal_gaps) if internal_gaps else 0.0
    seed_fonts = seed.get("_font_signatures")
    candidate_fonts = candidate.get("_font_signatures")
    reliable_style_change = (
        isinstance(seed_fonts, set)
        and bool(seed_fonts)
        and isinstance(candidate_fonts, set)
        and bool(candidate_fonts)
        and seed_fonts.isdisjoint(candidate_fonts)
    )
    return _is_structural_typography_gap(
        float(previous_height),
        float(current_height),
        current_bbox[1] - (previous_bbox[1] + float(previous_height)),
        regular_gap,
        gap_mad,
        reliable_style_change=reliable_style_change,
    )


def _caption_tail_matches_seed(
    seed: dict[str, Any],
    candidate: dict[str, Any],
    median_height: float,
) -> bool:
    """只用同栏角色、字体、邻接和投影把无标记的图注续行接回锚点。"""

    seed_bbox = seed["bbox"]
    candidate_bbox = candidate["bbox"]
    if not _components_share_lane_role(seed, candidate, median_height) and (
        _bbox_axis_overlap_ratio(seed_bbox, candidate_bbox, axis="x") < 0.75
        or abs(seed_bbox[0] - candidate_bbox[0]) > median_height
    ):
        return False
    if _bbox_center_y(candidate_bbox) <= _bbox_center_y(seed_bbox):
        return False
    if _caption_body_has_structural_gap(seed, candidate):
        return False
    vertical_gap = max(0.0, candidate_bbox[1] - seed_bbox[3])
    if vertical_gap > 0.5 * median_height or _bbox_axis_overlap_ratio(seed_bbox, candidate_bbox, axis="x") < 0.35:
        return False
    seed_fonts = seed.get("_font_signatures")
    candidate_fonts = candidate.get("_font_signatures")
    return not (
        isinstance(seed_fonts, set)
        and seed_fonts
        and isinstance(candidate_fonts, set)
        and candidate_fonts
        and seed_fonts.isdisjoint(candidate_fonts)
    )


def _merge_multiline_title_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """跨错误栏带合并紧贴且字体兼容的多行文档标题和段落标题。"""

    replacements: dict[int, dict[str, Any]] = {}
    consumed: set[int] = set()
    for block_type in ("doc_title", "paragraph_title"):
        indices = [
            index
            for index, block in enumerate(blocks)
            if block.get("type") == block_type and isinstance(block.get("bbox"), (list, tuple))
        ]
        indices.sort(
            key=lambda index: (
                blocks[index]["bbox"][1],
                blocks[index]["bbox"][0],
            )
        )
        groups: list[list[int]] = []
        for index in indices:
            if not groups:
                groups.append([index])
                continue
            previous_index = groups[-1][-1]
            previous = blocks[previous_index]
            current = blocks[index]
            previous_bbox = previous["bbox"]
            current_bbox = current["bbox"]
            previous_heights = previous.get("_line_heights", [])
            current_heights = current.get("_line_heights", [])
            previous_height = (
                statistics.median(previous_heights)
                if isinstance(previous_heights, list) and previous_heights
                else previous_bbox[3] - previous_bbox[1]
            )
            current_height = (
                statistics.median(current_heights)
                if isinstance(current_heights, list) and current_heights
                else current_bbox[3] - current_bbox[1]
            )
            vertical_gap = current_bbox[1] - previous_bbox[3]
            previous_fonts = previous.get("_font_signatures")
            current_fonts = current.get("_font_signatures")
            fonts_conflict = (
                isinstance(previous_fonts, set)
                and previous_fonts
                and isinstance(current_fonts, set)
                and current_fonts
                and previous_fonts.isdisjoint(current_fonts)
            )
            if (
                -0.2 * max(previous_height, current_height) <= vertical_gap <= 0.4 * max(previous_height, current_height)
                and _bbox_axis_overlap_ratio(
                    previous_bbox,
                    current_bbox,
                    axis="x",
                )
                >= 0.2
                and not fonts_conflict
            ):
                groups[-1].append(index)
            else:
                groups.append([index])
        for group in groups:
            if len(group) < 2:
                continue
            replacements[group[0]] = _merge_internal_text_block_group(
                blocks,
                group,
            )
            consumed.update(group[1:])
    return [replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed]


def _merge_fragmented_header_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """聚合同一视觉行中等距分散的页眉页脚片段。"""

    grouped: dict[tuple[int, int], list[int]] = {}
    for index, block in enumerate(blocks):
        row_id = block.get("_single_run_row_id")
        angle = int(block.get("angle", 0) or 0) % 360
        if block.get("type") in {"header", "footer"} and isinstance(row_id, int):
            grouped.setdefault((angle, row_id), []).append(index)

    replacements: dict[int, dict[str, Any]] = {}
    consumed: set[int] = set()
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: blocks[index]["bbox"][0])
        components: list[list[int]] = []
        for index in ordered:
            bbox = blocks[index]["bbox"]
            heights = blocks[index].get("_line_heights", [])
            effective_height = (
                statistics.median(heights) if isinstance(heights, list) and heights else max(0.1, bbox[3] - bbox[1])
            )
            if blocks[index].get("type") == "header" and bbox[2] - bbox[0] > 1.25 * effective_height:
                continue
            if not components:
                components.append([index])
                continue
            previous_index = components[-1][-1]
            previous_bbox = blocks[previous_index]["bbox"]
            previous_heights = blocks[previous_index].get("_line_heights", [])
            previous_height = (
                statistics.median(previous_heights)
                if isinstance(previous_heights, list) and previous_heights
                else max(0.1, previous_bbox[3] - previous_bbox[1])
            )
            if (
                bbox[0] - previous_bbox[2] <= 4.0 * max(effective_height, previous_height)
                and _bbox_axis_overlap_ratio(previous_bbox, bbox, axis="y") >= 0.5
            ):
                components[-1].append(index)
            else:
                components.append([index])
        for component in components:
            if len(component) < 2:
                continue
            replacement = _merge_internal_text_block_group(
                blocks,
                component,
                preserve_visual_spaces=True,
            )
            replacement["_single_run_row_id"] = None
            replacements[component[0]] = replacement
            consumed.update(component[1:])
    return [replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed]


def _merge_front_matter_column_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    page_index: int,
) -> list[dict[str, Any]]:
    """把首页标题下方规则排列的多列作者信息按列聚合。"""

    if page_index != 0:
        return blocks
    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return blocks
    title_blocks = [
        block for block in blocks if block.get("type") == "doc_title" and isinstance(block.get("bbox"), (list, tuple))
    ]
    if not title_blocks:
        return blocks
    title_bottom = max(block["bbox"][3] for block in title_blocks)
    candidates = [
        index
        for index, block in enumerate(blocks)
        if block.get("type") == "text"
        and isinstance(block.get("bbox"), (list, tuple))
        and title_bottom < block["bbox"][1]
        and block["bbox"][3]
        <= min(
            0.4 * page_height,
            title_bottom + 0.22 * page_height,
        )
        and block["bbox"][2] - block["bbox"][0] <= 0.32 * page_width
        and block["bbox"][3] - block["bbox"][1] <= 0.035 * page_height
    ]
    if len(candidates) < 9:
        return blocks
    median_height = statistics.median(blocks[index]["bbox"][3] - blocks[index]["bbox"][1] for index in candidates)
    row_groups: list[list[int]] = []
    for index in sorted(
        candidates,
        key=lambda item: (
            _bbox_center_y(blocks[item]["bbox"]),
            blocks[item]["bbox"][0],
        ),
    ):
        center_y = _bbox_center_y(blocks[index]["bbox"])
        target = next(
            (
                row
                for row in row_groups
                if abs(center_y - statistics.median(_bbox_center_y(blocks[member]["bbox"]) for member in row))
                <= 0.6 * median_height
            ),
            None,
        )
        if target is None:
            row_groups.append([index])
        else:
            target.append(index)
    dense_rows = [
        row
        for row in row_groups
        if 3 <= len(row) <= 6
        and (
            max(blocks[index]["bbox"][2] for index in row) - min(blocks[index]["bbox"][0] for index in row) >= 0.55 * page_width
        )
    ]
    if len(dense_rows) < 2:
        return blocks
    anchor_row = min(
        dense_rows,
        key=lambda row: (
            -len(row),
            statistics.median(_bbox_center_y(blocks[index]["bbox"]) for index in row),
        ),
    )
    anchor_centers = sorted(_bbox_center_x(blocks[index]["bbox"]) for index in anchor_row)
    if len(anchor_centers) != 4:
        return blocks
    boundaries = [0.5 * (left + right) for left, right in zip(anchor_centers, anchor_centers[1:])]
    band_top = min(min(blocks[index]["bbox"][1] for index in row) for row in dense_rows) - median_height
    band_bottom = max(max(blocks[index]["bbox"][3] for index in row) for row in dense_rows) + median_height
    column_groups: list[list[int]] = [[] for _center in anchor_centers]
    for index in candidates:
        bbox = blocks[index]["bbox"]
        if not band_top <= _bbox_center_y(bbox) <= band_bottom:
            continue
        center_x = _bbox_center_x(bbox)
        column_index = sum(center_x > boundary for boundary in boundaries)
        if column_index >= len(column_groups):
            continue
        column_groups[column_index].append(index)
    if any(len(group) < 3 for group in column_groups):
        return blocks

    replacements: dict[int, dict[str, Any]] = {}
    consumed: set[int] = set()
    for group in column_groups:
        ordered = sorted(
            group,
            key=lambda index: (
                blocks[index]["bbox"][1],
                blocks[index]["bbox"][0],
            ),
        )
        replacements[ordered[0]] = _merge_internal_text_block_group(
            blocks,
            ordered,
        )
        consumed.update(ordered[1:])
    return [replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed]


def _merge_repeated_compact_title_continuations(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """把重复出现的两行弱标题与紧邻异字体续行恢复为普通文本块。"""

    candidate_pairs: list[tuple[int, int, float]] = []
    for title_index, title in enumerate(blocks):
        title_lines = title.get("_local_line_bboxes")
        title_fonts = title.get("_font_signatures")
        title_bbox = title.get("bbox")
        if (
            title.get("type") != "paragraph_title"
            or not isinstance(title_bbox, (list, tuple))
            or not isinstance(title_lines, list)
            or len(title_lines) < 2
            or not isinstance(title_fonts, set)
            or not title_fonts
        ):
            continue
        angle = int(title.get("angle", 0) or 0) % 360
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        line_heights = [
            float(height) for height in title.get("_line_heights", []) if isinstance(height, (int, float)) and height > 0
        ]
        title_height = statistics.median(line_heights) if line_heights else 0.0
        if title_height <= 0 or title_bbox[2] - title_bbox[0] > 0.55 * local_page_width:
            continue

        continuations: list[tuple[float, int]] = []
        for text_index, text_block in enumerate(blocks):
            text_bbox = text_block.get("bbox")
            text_fonts = text_block.get("_font_signatures")
            if (
                text_block.get("type") != "text"
                or int(text_block.get("angle", 0) or 0) % 360 != angle
                or not isinstance(text_bbox, (list, tuple))
                or not isinstance(text_fonts, set)
                or not text_fonts
                or not title_fonts.isdisjoint(text_fonts)
                or text_bbox[2] - text_bbox[0] > 0.6 * local_page_width
            ):
                continue
            gap = text_bbox[1] - title_bbox[3]
            if -0.25 * title_height <= gap <= 0.6 * title_height and abs(text_bbox[0] - title_bbox[0]) <= 0.75 * title_height:
                continuations.append((max(0.0, gap), text_index))
        if continuations:
            _gap, text_index = min(continuations)
            candidate_pairs.append((title_index, text_index, title_height))

    supported_pairs: list[tuple[int, int]] = []
    for title_index, text_index, title_height in candidate_pairs:
        title_bbox = blocks[title_index]["bbox"]
        support_count = sum(
            abs(blocks[other_title]["bbox"][0] - title_bbox[0]) <= max(title_height, other_height)
            and 0.75 <= other_height / title_height <= 1.25
            for other_title, _other_text, other_height in candidate_pairs
        )
        if support_count >= 2:
            supported_pairs.append((title_index, text_index))
    if not supported_pairs:
        return blocks

    replacements: dict[int, dict[str, Any]] = {}
    consumed: set[int] = set()
    for title_index, text_index in supported_pairs:
        if title_index in consumed or text_index in consumed:
            continue
        merged = _merge_internal_text_block_group(
            blocks,
            [title_index, text_index],
        )
        merged["type"] = "text"
        replacements[min(title_index, text_index)] = merged
        consumed.update({title_index, text_index})
    return [
        replacements.get(index, block) for index, block in enumerate(blocks) if index not in consumed or index in replacements
    ]


__all__ = [
    "_merge_image_caption_text_blocks",
    "_caption_image_group_bboxes",
    "_caption_seed_matches_image",
    "_caption_body_has_structural_gap",
    "_caption_tail_matches_seed",
    "_merge_multiline_title_blocks",
    "_merge_fragmented_header_blocks",
    "_merge_front_matter_column_blocks",
    "_merge_repeated_compact_title_continuations",
]
