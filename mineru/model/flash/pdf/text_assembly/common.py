# Copyright (c) Opendatalab. All rights reserved.
"""共享正文块连接的几何、内容拼接和来源规则。"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .....utils.language import detect_lang
from .....utils.text import merge_text_line_contents
from ..geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_y,
    _bbox_union_many,
)
from ..native_text import _normalize_native_run_text

_REFERENCE_ENTRY_RE = re.compile(r"^[［\[]\s*\d+\s*[］\]]")


_FIGURE_CAPTION_MARKER_RE = re.compile(
    r"^(?:图\s*[0-9０-９一二三四五六七八九十]|fig(?:ure)?\.?\s*[0-9])",
    re.IGNORECASE,
)


_INLINE_MATH_RECOVERY_MARKER = "_recovered_inline_math_fragments"


_PARAGRAPH_FORMULA_CONTEXT_MARKER = "_paragraph_formula_context"


_FRONT_MATTER_FIELD_RE = re.compile(
    r"^\s*(?:keywords?|key\s+words?|关键词|中图分类号|文献标识码|文章编号)\s*[:：]",
    re.IGNORECASE,
)


_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[（(]\s*(?:\d+|[ivxlcdm]+)\s*[）)]|[①-⑳]|[•●▪])",
    re.IGNORECASE,
)


_BULLET_ITEM_RE = re.compile(
    r"^\s*[•●▪]",
)


_EMAIL_METADATA_RE = re.compile(
    r"^\s*e[\s-]*mail\s*[:：]",
    re.IGNORECASE,
)


_ABSTRACT_METADATA_RE = re.compile(
    r"^\s*(?:abstract|摘\s*要)\s*[:：]",
    re.IGNORECASE,
)


_LABELLED_METADATA_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z\u3400-\u9fff]{1,12})\s*[:：]",
)


_URL_LINE_RE = re.compile(
    r"^\s*(?:https?://|www\.)",
    re.IGNORECASE,
)


_SHORT_SAME_BASELINE_PREFIX_RE = re.compile(
    r"^(?:[（(]\s*\d{1,3}|\d{1,2}\s*[:：]\s*\d{2})$",
)


def _merge_internal_text_block_group(
    blocks: list[dict[str, Any]],
    indices: list[int],
    *,
    preserve_visual_spaces: bool = False,
) -> dict[str, Any]:
    """合并内部文本块及其版面元数据，最终输出阶段仍会统一移除这些字段。"""

    ordered_indices = sorted(
        indices,
        key=(
            (lambda index: blocks[index]["bbox"][0])
            if preserve_visual_spaces
            else (lambda index: _text_component_sort_key(blocks[index]))
        ),
    )
    merged = dict(blocks[ordered_indices[0]])
    merged["bbox"] = _bbox_union_many([blocks[index]["bbox"] for index in ordered_indices])
    contents = [str(blocks[index].get("content", "")) for index in ordered_indices]
    merged["content"] = (
        " ".join(content.strip() for content in contents if content.strip())
        if preserve_visual_spaces
        else _merge_text_line_content(contents)
    )
    merged["_visual_row_ids"] = set().union(
        *[row_ids for index in ordered_indices if isinstance((row_ids := blocks[index].get("_visual_row_ids")), set)]
    )
    merged["_single_run_row_id"] = None
    merged["_local_line_bboxes"] = [bbox for index in ordered_indices for bbox in blocks[index].get("_local_line_bboxes", [])]
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
    merged[_PARAGRAPH_FORMULA_CONTEXT_MARKER] = any(
        blocks[index].get(_PARAGRAPH_FORMULA_CONTEXT_MARKER) is True for index in ordered_indices
    )
    return merged


def _component_declared_lane_interval(
    block: dict[str, Any],
) -> tuple[float, float] | None:
    """读取组件声明的有效栏带区间，不区分普通栏或跨栏。"""

    interval = block.get("_lane_interval")
    if (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or not all(isinstance(value, (int, float)) for value in interval)
    ):
        return None
    left, right = float(interval[0]), float(interval[1])
    return (left, right) if right > left else None


def _component_lane_interval(
    block: dict[str, Any],
) -> tuple[float, float] | None:
    """读取普通文本组件所属的有效非跨栏栏带区间。"""

    if block.get("_lane_is_span") is not False:
        return None
    return _component_declared_lane_interval(block)


def _component_reference_width(
    block: dict[str, Any],
    local_page_width: float,
) -> float:
    """优先返回组件的局部栏宽，缺少可靠栏带时回退页面宽度。"""

    interval = _component_lane_interval(block)
    return interval[1] - interval[0] if interval is not None else local_page_width


def _compatible_component_lane_width(
    first_block: dict[str, Any],
    second_block: dict[str, Any],
    local_page_width: float,
    median_height: float,
) -> float:
    """同一局部栏带的两个组件使用栏宽，否则继续使用页面宽度。"""

    first_interval = _component_lane_interval(first_block)
    second_interval = _component_lane_interval(second_block)
    if first_interval is None or second_interval is None:
        return local_page_width
    tolerance = 0.75 * median_height
    if abs(first_interval[0] - second_interval[0]) > tolerance or abs(first_interval[1] - second_interval[1]) > tolerance:
        return local_page_width
    return min(
        first_interval[1] - first_interval[0],
        second_interval[1] - second_interval[0],
    )


def _components_share_lane_role(
    first_block: dict[str, Any],
    second_block: dict[str, Any],
    median_height: float,
) -> bool:
    """要求二次合并组件同为跨栏或属于同一普通栏带。"""

    first_role = first_block.get("_lane_is_span")
    second_role = second_block.get("_lane_is_span")
    if isinstance(first_role, bool) and isinstance(second_role, bool):
        if first_role != second_role:
            return False
        if first_role:
            return True
        first_interval = _component_lane_interval(first_block)
        second_interval = _component_lane_interval(second_block)
        if first_interval is None or second_interval is None:
            return False
        tolerance = 0.75 * median_height
        return (
            abs(first_interval[0] - second_interval[0]) <= tolerance
            and abs(first_interval[1] - second_interval[1]) <= tolerance
        )
    # 兼容缺少内部栏元数据的旧调用；生产路径始终会携带该字段。
    return not isinstance(first_role, bool) and not isinstance(second_role, bool)


def _block_starts_with_short_wide_rows(
    block: dict[str, Any],
    local_page_width: float,
) -> bool:
    """判断组件内部是否以短首行和紧邻宽正文形成明确分组起点。"""

    line_bboxes = block.get("_local_line_bboxes")
    if not isinstance(line_bboxes, list) or len(line_bboxes) < 2:
        return False
    first_bbox, second_bbox = line_bboxes[:2]
    first_width = first_bbox[2] - first_bbox[0]
    second_width = second_bbox[2] - second_bbox[0]
    reference_width = _component_reference_width(block, local_page_width)
    return first_width <= 0.35 * reference_width and second_width >= max(0.25 * reference_width, 1.8 * first_width)


def _find_short_opener_pairs(
    blocks: list[dict[str, Any]],
    candidate_indices: list[int],
    local_page_width: float,
    median_height: float,
) -> list[tuple[int, int]]:
    """查找跨栏拆开的短首行与紧邻满宽正文组件。"""

    output: list[tuple[int, int]] = []
    for opener_index in candidate_indices:
        opener_bboxes = blocks[opener_index]["_local_line_bboxes"]
        if len(opener_bboxes) != 1:
            continue
        opener_bbox = opener_bboxes[0]
        opener_width = opener_bbox[2] - opener_bbox[0]
        matches: list[tuple[float, int]] = []
        for body_index in candidate_indices:
            if body_index == opener_index:
                continue
            if blocks[body_index].get("_hard_break_before") is True:
                continue
            if not _components_share_lane_role(
                blocks[opener_index],
                blocks[body_index],
                median_height,
            ):
                continue
            body_bbox = blocks[body_index]["_local_line_bboxes"][0]
            gap = body_bbox[1] - opener_bbox[3]
            body_width = body_bbox[2] - body_bbox[0]
            reference_width = _compatible_component_lane_width(
                blocks[opener_index],
                blocks[body_index],
                local_page_width,
                median_height,
            )
            if (
                not 0.0 <= gap <= 0.75 * median_height
                or abs(body_bbox[0] - opener_bbox[0]) > 0.75 * median_height
                or opener_width > 0.35 * reference_width
                or body_width < 0.6 * reference_width
                or _component_connection_skips_block(
                    blocks,
                    opener_index,
                    body_index,
                    median_height,
                )
            ):
                continue
            matches.append((gap, body_index))
        if matches:
            output.append((opener_index, min(matches)[1]))
    return output


def _nearest_following_text_component(
    blocks: list[dict[str, Any]],
    current_index: int,
    candidate_indices: list[int],
    *,
    maximum_gap: float,
    section_starts: set[int],
) -> int | None:
    """返回同一水平流中紧邻且不是新分组起点的下一正文组件。"""

    current_bbox = blocks[current_index]["_local_line_bboxes"][-1]
    matches: list[tuple[float, float, int]] = []
    for candidate_index in candidate_indices:
        if candidate_index == current_index or candidate_index in section_starts:
            continue
        if blocks[candidate_index].get("_hard_break_before") is True:
            continue
        if not _components_share_lane_role(
            blocks[current_index],
            blocks[candidate_index],
            maximum_gap / 0.75,
        ):
            continue
        candidate_bbox = blocks[candidate_index]["_local_line_bboxes"][0]
        gap = candidate_bbox[1] - current_bbox[3]
        if not -0.25 * maximum_gap <= gap <= maximum_gap:
            continue
        overlap = _bbox_axis_overlap_ratio(current_bbox, candidate_bbox, axis="x")
        left_gap = abs(candidate_bbox[0] - current_bbox[0])
        if overlap < 0.5 and left_gap > 3.0 * maximum_gap:
            continue
        if _component_connection_skips_block(
            blocks,
            current_index,
            candidate_index,
            maximum_gap / 0.75,
        ):
            continue
        matches.append((max(0.0, gap), left_gap, candidate_index))
    return min(matches)[2] if matches else None


def _has_parallel_text_component(
    blocks: list[dict[str, Any]],
    current_index: int,
    candidate_indices: list[int],
) -> bool:
    """检查当前组件同一纵向带内是否存在水平分离的并栏组件。"""

    current_bbox = blocks[current_index]["bbox"]
    for candidate_index in candidate_indices:
        if candidate_index == current_index:
            continue
        candidate_bbox = blocks[candidate_index]["bbox"]
        vertical_overlap = max(
            0.0,
            min(current_bbox[3], candidate_bbox[3]) - max(current_bbox[1], candidate_bbox[1]),
        )
        horizontally_separate = candidate_bbox[0] >= current_bbox[2] or current_bbox[0] >= candidate_bbox[2]
        if vertical_overlap > 0 and horizontally_separate:
            return True
    return False


def _nearest_tapered_tail_component(
    blocks: list[dict[str, Any]],
    current_index: int,
    candidate_indices: list[int],
    median_height: float,
    section_starts: set[int],
) -> int | None:
    """查找左对齐、行宽递减且行距略大的信息组尾行组件。"""

    previous_bbox = blocks[current_index]["_local_line_bboxes"][-1]
    previous_width = previous_bbox[2] - previous_bbox[0]
    matches: list[tuple[float, int]] = []
    for candidate_index in candidate_indices:
        if candidate_index == current_index or candidate_index in section_starts:
            continue
        if blocks[candidate_index].get("_hard_break_before") is True:
            continue
        if not _components_share_lane_role(
            blocks[current_index],
            blocks[candidate_index],
            median_height,
        ):
            continue
        candidate_bboxes = blocks[candidate_index]["_local_line_bboxes"]
        first_bbox = candidate_bboxes[0]
        last_bbox = candidate_bboxes[-1]
        gap = first_bbox[1] - previous_bbox[3]
        first_width = first_bbox[2] - first_bbox[0]
        last_width = last_bbox[2] - last_bbox[0]
        if (
            not 0.75 * median_height < gap <= 1.5 * median_height
            or len(candidate_bboxes) > 2
            or abs(first_bbox[0] - previous_bbox[0]) > 0.75 * median_height
            or first_width > 0.85 * previous_width
            or last_width > first_width + 0.5 * median_height
            or _component_connection_skips_block(
                blocks,
                current_index,
                candidate_index,
                median_height,
            )
        ):
            continue
        matches.append((gap, candidate_index))
    return min(matches)[1] if matches else None


def _component_connection_skips_block(
    blocks: list[dict[str, Any]],
    first_index: int,
    second_index: int,
    median_height: float,
) -> bool:
    """检查两个正文组件之间是否已有同水平流的中间块，禁止二次合并跨越它。"""

    first_bbox = blocks[first_index]["_local_line_bboxes"][-1]
    second_bbox = blocks[second_index]["_local_line_bboxes"][0]
    first_center = _bbox_center_y(first_bbox)
    second_center = _bbox_center_y(second_bbox)
    corridor_top, corridor_bottom = sorted((first_center, second_center))
    corridor_bbox = (
        min(first_bbox[0], second_bbox[0]),
        corridor_top,
        max(first_bbox[2], second_bbox[2]),
        corridor_bottom,
    )
    for index, block in enumerate(blocks):
        if index in {first_index, second_index}:
            continue
        local_bboxes = block.get("_local_line_bboxes")
        if not isinstance(local_bboxes, list) or not local_bboxes:
            continue
        other_bbox = local_bboxes[0]
        other_center = _bbox_center_y(other_bbox)
        if not (corridor_top + 0.1 * median_height < other_center < corridor_bottom - 0.1 * median_height):
            continue
        if _bbox_axis_overlap_ratio(corridor_bbox, other_bbox, axis="x") >= 0.2:
            return True
    return False


def _text_component_sort_key(block: dict[str, Any]) -> tuple[float, float]:
    """返回正文组件按局部首行位置排序的稳定键。"""

    first_bbox = block["_local_line_bboxes"][0]
    return first_bbox[1], first_bbox[0]


def _merge_text_line_content(line_texts: Sequence[str]) -> str:
    """按 Hybrid 语言与行末连字规则折叠普通文本行。"""

    normalized_lines = [_normalize_native_run_text(str(text or "")) for text in line_texts]
    normalized_lines = [text for text in normalized_lines if text]
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


__all__ = [
    "_REFERENCE_ENTRY_RE",
    "_FIGURE_CAPTION_MARKER_RE",
    "_INLINE_MATH_RECOVERY_MARKER",
    "_PARAGRAPH_FORMULA_CONTEXT_MARKER",
    "_FRONT_MATTER_FIELD_RE",
    "_LIST_ITEM_RE",
    "_BULLET_ITEM_RE",
    "_EMAIL_METADATA_RE",
    "_ABSTRACT_METADATA_RE",
    "_LABELLED_METADATA_RE",
    "_URL_LINE_RE",
    "_SHORT_SAME_BASELINE_PREFIX_RE",
    "_merge_internal_text_block_group",
    "_component_declared_lane_interval",
    "_component_lane_interval",
    "_component_reference_width",
    "_compatible_component_lane_width",
    "_components_share_lane_role",
    "_block_starts_with_short_wide_rows",
    "_find_short_opener_pairs",
    "_nearest_following_text_component",
    "_has_parallel_text_component",
    "_nearest_tapered_tail_component",
    "_component_connection_skips_block",
    "_text_component_sort_key",
    "_merge_text_line_content",
]
