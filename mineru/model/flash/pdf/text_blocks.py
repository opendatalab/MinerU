# Copyright (c) Opendatalab. All rights reserved.

"""将剩余文本行构造成正文和页脚注块。"""

from __future__ import annotations

import re
import statistics
from typing import Any, Sequence


from ....utils.text import (
    is_hyphen_at_line_end,
    merge_text_line_contents,
)
from ....types import BBox
from ....utils.language import detect_lang

from .models import (
    _AxisLine,
    _LineItem,
    _LocalAxisLine,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_union_many,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from .native_text import _normalize_native_run_text
from .line_layout import (
    _connection_crosses_table,
    _effective_body_text_row_gap,
    _effective_text_row_gap,
    _estimate_lane_gap,
    _horizontal_rule_separates_rows,
    _infer_text_lanes,
    _is_structural_typography_gap,
    _line_effective_height,
    _line_tight_output_bbox,
    _lines_tight_output_bbox,
    _should_connect_semantic_rows,
    _should_connect_text_rows,
    _title_fonts_compatible,
)


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


def _local_tight_output_line_bboxes(
    lines: Sequence[_LineItem],
    page_size: tuple[float, float],
    angle: int,
) -> tuple[list[BBox], bool]:
    """返回与原行顺序一致的 tight+1pt 局部框及是否存在可靠候选。"""

    output = []
    changed = False
    for line in lines:
        candidate = _line_tight_output_bbox(line, page_size)
        output.append(
            _rotate_bbox_to_upright(
                candidate or line.bbox,
                page_size,
                angle,
            )
        )
        changed = changed or candidate is not None
    return output, changed


def _starts_structural_reference_entry(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
) -> bool:
    """仅在编号行相对续行明显左突时确认新的参考文献条目。"""

    if _REFERENCE_ENTRY_RE.match(current[0].text.strip()) is None:
        return False
    previous_height = _line_effective_height(*previous)
    current_height = _line_effective_height(*current)
    pair_height = max(previous_height, current_height)
    return (
        current[1][0] <= previous[1][0] - max(5.0, 0.6 * min(previous_height, current_height))
        and -0.75 * pair_height <= _effective_text_row_gap(previous, current) <= 1.5 * pair_height
    )


def _build_hanging_indent_group_map(
    lane: _TextLane,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> dict[int, int]:
    """仅按重复的左突首行和稳定续行缩进识别悬挂缩进条目。"""

    if len(lane.lines) < 4:
        return {}
    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    if len(rows) < 4:
        return {}
    median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in rows)
    start_tolerance = max(5.0, 0.65 * median_height)
    minimum_indent = max(7.0, 0.8 * median_height)
    continuation_tolerance = max(4.0, 0.55 * median_height)

    def rows_are_adjacent(
        previous: tuple[_LineItem, BBox],
        current: tuple[_LineItem, BBox],
    ) -> bool:
        """检查相邻行的净空和几何障碍是否允许组成同一缩进序列。"""

        effective_gap = _effective_text_row_gap(previous, current)
        top_pitch = current[1][1] - previous[1][1]
        robust_pitch_fallback = 0.5 * median_height <= top_pitch <= 1.8 * median_height
        if not -0.6 * median_height <= effective_gap <= 1.3 * median_height and not robust_pitch_fallback:
            return False
        if _connection_crosses_table(
            previous[0].bbox,
            current[0].bbox,
            table_bboxes,
        ):
            return False
        return not _horizontal_rule_separates_rows(
            previous[1],
            current[1],
            lane,
            axis_lines,
        )

    def consume_entry(
        start_index: int,
        start_left: float,
        expected_continuation_left: float | None,
        *,
        require_next_start: bool,
    ) -> tuple[int, float] | None:
        """消费一个左突首行及其续行，并返回下一条首行位置。"""

        lane_width = max(0.1, lane.right - lane.left)
        full_width_midparagraph_entry = (
            rows[start_index][1][2] - rows[start_index][1][0] >= 0.8 * lane_width
            and start_index + 1 < len(rows)
            and rows[start_index + 1][1][2] - rows[start_index + 1][1][0] <= 0.75 * lane_width
        )
        if (
            start_index > 0
            and rows_are_adjacent(
                rows[start_index - 1],
                rows[start_index],
            )
            and abs(rows[start_index - 1][1][0] - start_left) <= start_tolerance
            and not full_width_midparagraph_entry
        ):
            # 同左缘正文仍在连续时不能从段落中部启动悬挂条目序列。
            return None
        if (
            start_index > 0
            and is_hyphen_at_line_end(rows[start_index - 1][0].text)
            and rows_are_adjacent(rows[start_index - 1], rows[start_index])
        ):
            # 排版断词后的下一物理行属于前文，不能被缩进几何误当成新条目首行。
            return None
        continuation_index = start_index + 1
        if continuation_index >= len(rows):
            return None
        first_continuation = rows[continuation_index]
        if not rows_are_adjacent(rows[start_index], first_continuation):
            return None
        continuation_left = first_continuation[1][0]
        if continuation_left < start_left + minimum_indent:
            return None
        if (
            expected_continuation_left is not None
            and abs(continuation_left - expected_continuation_left) > continuation_tolerance
        ):
            return None

        continuation_index += 1
        while continuation_index < len(rows):
            previous = rows[continuation_index - 1]
            current = rows[continuation_index]
            current_left = current[1][0]
            if not rows_are_adjacent(previous, current):
                break
            if current_left < start_left + minimum_indent:
                break
            if abs(current_left - continuation_left) > continuation_tolerance:
                break
            continuation_index += 1

        if not require_next_start:
            return continuation_index, continuation_left
        if continuation_index >= len(rows):
            return None
        if not rows_are_adjacent(rows[continuation_index - 1], rows[continuation_index]):
            return None
        if abs(rows[continuation_index][1][0] - start_left) > start_tolerance:
            return None
        return continuation_index, continuation_left

    group_map: dict[int, int] = {}
    group_index = 0
    row_index = 0
    while row_index < len(rows) - 3:
        start_left = rows[row_index][1][0]
        first_entry = consume_entry(
            row_index,
            start_left,
            None,
            require_next_start=True,
        )
        if first_entry is None:
            row_index += 1
            continue

        _next_start_index, continuation_left = first_entry
        start_indices = [row_index]
        current_start_index = row_index
        end_index: int | None = None
        while True:
            next_entry = consume_entry(
                current_start_index,
                start_left,
                continuation_left,
                require_next_start=True,
            )
            if next_entry is None:
                final_entry = consume_entry(
                    current_start_index,
                    start_left,
                    continuation_left,
                    require_next_start=False,
                )
                if final_entry is not None:
                    end_index = final_entry[0]
                break
            next_start_index, _continuation_left = next_entry
            prospective_entry = consume_entry(
                next_start_index,
                start_left,
                continuation_left,
                require_next_start=False,
            )
            if prospective_entry is None:
                # 当前条目已经完整确认；后面的普通左对齐段落只作为终止边界，
                # 不能让它反向使此前所有悬挂缩进条目失效。
                end_index = next_start_index
                break
            start_indices.append(next_start_index)
            current_start_index = next_start_index
        if len(start_indices) < 2 or end_index is None:
            row_index += 1
            continue

        entry_ranges = [
            (start, end)
            for start, end in zip(
                start_indices,
                [*start_indices[1:], end_index],
                strict=True,
            )
        ]
        for start, end in entry_ranges:
            for line, _bbox in rows[start:end]:
                group_map[line.source_index] = group_index
            group_index += 1
        row_index = end_index

    return group_map


def _build_grouped_page_footnote_blocks(
    lines: list[_LineItem],
    source_groups: Sequence[set[int]],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], set[int]]:
    """按分隔线组切分几何脚注条目，并返回已消费的来源编号。"""

    footnote_by_source = {line.source_index: line for line in lines if line.semantic_type == "page_footnote"}
    blocks: list[dict[str, Any]] = []
    consumed_source_indices: set[int] = set()
    for source_group in source_groups:
        group_lines = [footnote_by_source[source_index] for source_index in source_group if source_index in footnote_by_source]
        for angle in sorted({line.angle for line in group_lines}):
            directional_lines = [line for line in group_lines if line.angle == angle]
            tight_bboxes = _tight_page_footnote_bboxes(
                directional_lines,
                page_size,
            )
            for entry_lines in _split_page_footnote_entries(
                directional_lines,
                page_size,
            ):
                ordered_lines = sorted(
                    entry_lines,
                    key=lambda line: (
                        _rotate_bbox_to_upright(line.bbox, page_size, angle)[1],
                        _rotate_bbox_to_upright(line.bbox, page_size, angle)[0],
                        line.source_index,
                    ),
                )
                content = _merge_text_line_content([line.text for line in ordered_lines])
                if not content:
                    continue
                visual_row_ids = {line.visual_row_id for line in ordered_lines if line.visual_row_id is not None}
                single_run_row_id = (
                    ordered_lines[0].visual_row_id
                    if len(ordered_lines) == 1
                    and ordered_lines[0].split_from_row
                    and ordered_lines[0].visual_row_id is not None
                    else None
                )
                block = {
                    "type": "page_footnote",
                    "bbox": _bbox_union_many([tight_bboxes[line.source_index] for line in ordered_lines]),
                    "angle": angle,
                    "content": content,
                    "_visual_row_ids": visual_row_ids,
                    "_single_run_row_id": single_run_row_id,
                    "_inline_math_regions": [region for line in ordered_lines for region in line.inline_math_regions],
                }
                tight_output_bbox = _lines_tight_output_bbox(
                    ordered_lines,
                    page_size,
                )
                if tight_output_bbox is not None:
                    block["_tight_output_bbox"] = tight_output_bbox
                blocks.append(block)
                consumed_source_indices.update(line.source_index for line in ordered_lines)
    return blocks, consumed_source_indices


def _split_page_footnote_entries(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[list[_LineItem]]:
    """按首行或续行缩进模式，把同一分隔线下的脚注行切成条目。"""

    if not lines:
        return []
    angle = lines[0].angle
    line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines]
    line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    if len(line_geometry) == 1:
        return [[line_geometry[0][0]]]

    effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
    median_height = statistics.median(effective_heights)
    glyph_widths = [line.median_glyph_width for line, _bbox in line_geometry if line.median_glyph_width is not None]
    median_glyph_width = statistics.median(glyph_widths) if glyph_widths else 0.5 * median_height
    indent_threshold = max(1.5 * median_height, 2.0 * median_glyph_width)
    base_left = min(bbox[0] for _line, bbox in line_geometry)
    maximum_row_width = max(bbox[2] - bbox[0] for _line, bbox in line_geometry)
    marker_rows = _find_page_footnote_marker_rows(
        line_geometry,
        median_height,
        median_glyph_width,
        base_left,
    )
    if marker_rows:
        return _split_marked_page_footnote_entries(
            line_geometry,
            marker_rows,
            median_height,
            median_glyph_width,
            base_left,
            maximum_row_width,
            indent_threshold,
        )
    return _split_unmarked_page_footnote_entries(
        line_geometry,
        base_left,
        maximum_row_width,
        indent_threshold,
    )


def _find_page_footnote_marker_rows(
    line_geometry: list[tuple[_LineItem, BBox]],
    median_height: float,
    median_glyph_width: float,
    base_left: float,
) -> list[tuple[tuple[_LineItem, BBox], tuple[_LineItem, BBox]]]:
    """用同视觉行的窄左片段和右侧正文识别脚注编号锚点。"""

    by_visual_row: dict[int, list[tuple[_LineItem, BBox]]] = {}
    for item in line_geometry:
        line = item[0]
        if line.visual_row_id is None or not line.split_from_row:
            continue
        by_visual_row.setdefault(line.visual_row_id, []).append(item)

    marker_rows: list[tuple[tuple[_LineItem, BBox], tuple[_LineItem, BBox]]] = []
    marker_width_limit = max(1.5 * median_glyph_width, 0.75 * median_height)
    for members in by_visual_row.values():
        ordered = sorted(members, key=lambda item: item[1][0])
        if len(ordered) < 2:
            continue
        marker, body = ordered[0], ordered[1]
        marker_width = marker[1][2] - marker[1][0]
        body_width = body[1][2] - body[1][0]
        horizontal_gap = body[1][0] - marker[1][2]
        if (
            marker_width > marker_width_limit
            or marker[1][0] - base_left > 0.5 * median_height
            or not 0.0 <= horizontal_gap <= 1.5 * median_height
            or body[1][0] - base_left < max(1.5 * marker_width, 0.75 * median_height)
            or body_width < max(4.0 * marker_width, 4.0 * median_glyph_width)
            or abs(_bbox_center_y(marker[1]) - _bbox_center_y(body[1])) > 0.4 * median_height
        ):
            continue
        marker_rows.append((marker, body))
    marker_rows.sort(key=lambda pair: (min(pair[0][1][1], pair[1][1][1]), pair[0][1][0]))
    if not marker_rows:
        marker_rows = _find_geometric_page_footnote_marker_rows(
            line_geometry,
            median_height,
            median_glyph_width,
            base_left,
        )
    return marker_rows


def _find_geometric_page_footnote_marker_rows(
    line_geometry: list[tuple[_LineItem, BBox]],
    median_height: float,
    median_glyph_width: float,
    base_left: float,
) -> list[tuple[tuple[_LineItem, BBox], tuple[_LineItem, BBox]]]:
    """在 row id 缺失时用同基线窄标记和右侧正文恢复脚注首行。"""

    marker_width_limit = max(
        1.5 * median_glyph_width,
        0.75 * median_height,
    )
    candidates: list[
        tuple[
            float,
            tuple[_LineItem, BBox],
            tuple[_LineItem, BBox],
        ]
    ] = []
    for marker in line_geometry:
        marker_width = marker[1][2] - marker[1][0]
        if (
            marker_width > marker_width_limit
            or marker[1][0] - base_left > 0.5 * median_height
        ):
            continue
        for body in line_geometry:
            if body is marker:
                continue
            body_width = body[1][2] - body[1][0]
            horizontal_gap = body[1][0] - marker[1][2]
            if (
                not 0.0 <= horizontal_gap <= 1.5 * median_height
                or body[1][0] - base_left
                < max(
                    1.5 * marker_width,
                    0.4 * median_height,
                )
                or body_width
                < max(
                    4.0 * marker_width,
                    4.0 * median_glyph_width,
                )
                or abs(
                    _bbox_center_y(marker[1])
                    - _bbox_center_y(body[1])
                )
                > 0.4 * median_height
                or not _title_fonts_compatible(
                    marker[0],
                    body[0],
                )
            ):
                continue
            candidates.append(
                (
                    horizontal_gap,
                    marker,
                    body,
                )
            )

    output = []
    consumed_sources: set[int] = set()
    for _gap, marker, body in sorted(
        candidates,
        key=lambda item: (
            min(item[1][1][1], item[2][1][1]),
            item[0],
            item[1][0].source_index,
            item[2][0].source_index,
        ),
    ):
        pair_sources = {
            marker[0].source_index,
            body[0].source_index,
        }
        if pair_sources & consumed_sources:
            continue
        output.append((marker, body))
        consumed_sources.update(pair_sources)
    return output


def _split_marked_page_footnote_entries(
    line_geometry: list[tuple[_LineItem, BBox]],
    marker_rows: list[tuple[tuple[_LineItem, BBox], tuple[_LineItem, BBox]]],
    median_height: float,
    median_glyph_width: float,
    base_left: float,
    maximum_row_width: float,
    indent_threshold: float,
) -> list[list[_LineItem]]:
    """把编号、右侧首行和对齐续行聚合，并保留编号区之前的独立脚注。"""

    first_marker_top = min(marker_rows[0][0][1][1], marker_rows[0][1][1][1])
    prefix_geometry = [item for item in line_geometry if item[1][1] < first_marker_top]
    entries = _split_unmarked_page_footnote_entries(
        prefix_geometry,
        base_left,
        maximum_row_width,
        indent_threshold,
    )
    consumed = {line.source_index for entry in entries for line in entry}
    continuation_tolerance = max(0.75 * median_height, 2.0 * median_glyph_width)
    for marker_index, (marker, body) in enumerate(marker_rows):
        next_marker_top = (
            min(marker_rows[marker_index + 1][0][1][1], marker_rows[marker_index + 1][1][1][1])
            if marker_index + 1 < len(marker_rows)
            else float("inf")
        )
        entry_items = [marker, body]
        consumed.update((marker[0].source_index, body[0].source_index))
        previous = body
        for candidate in line_geometry:
            if candidate[0].source_index in consumed:
                continue
            if candidate[1][1] < body[1][1] or candidate[1][1] >= next_marker_top:
                continue
            if abs(candidate[1][0] - body[1][0]) > continuation_tolerance:
                continue
            if _effective_text_row_gap(previous, candidate) > 1.5 * median_height:
                continue
            entry_items.append(candidate)
            consumed.add(candidate[0].source_index)
            previous = candidate
        entry_items.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        entries.append([line for line, _bbox in entry_items])

    leftovers = [item for item in line_geometry if item[0].source_index not in consumed]
    entries.extend(
        _split_unmarked_page_footnote_entries(
            leftovers,
            base_left,
            maximum_row_width,
            indent_threshold,
        )
    )
    entries.sort(
        key=lambda entry: min(
            bbox[1] for line in entry for candidate, bbox in line_geometry if candidate.source_index == line.source_index
        )
    )
    return entries


def _split_unmarked_page_footnote_entries(
    line_geometry: list[tuple[_LineItem, BBox]],
    base_left: float,
    maximum_row_width: float,
    indent_threshold: float,
) -> list[list[_LineItem]]:
    """沿用首行和续行缩进模式切分没有稳定编号锚点的脚注行。"""

    if not line_geometry:
        return []
    first_line_is_indented = line_geometry[0][1][0] - base_left > indent_threshold

    entries: list[list[_LineItem]] = [[line_geometry[0][0]]]
    for previous, current in zip(line_geometry, line_geometry[1:]):
        previous_width = previous[1][2] - previous[1][0]
        previous_is_near_full = previous_width >= 0.8 * maximum_row_width
        current_is_indented = current[1][0] - base_left > indent_threshold
        same_left_compact_continuation = (
            not first_line_is_indented
            and previous_width >= 0.7 * maximum_row_width
            and abs(current[1][0] - previous[1][0]) <= 0.5 * indent_threshold
            and _title_fonts_compatible(previous[0], current[0])
            and _effective_text_row_gap(previous, current) <= indent_threshold
        )
        # 满行后必然续接；其余行按首页观测到的首行/续行缩进模式决定边界。
        continues_previous = (
            previous_is_near_full
            or same_left_compact_continuation
            or (not current_is_indented if first_line_is_indented else current_is_indented)
        )
        if continues_previous:
            entries[-1].append(current[0])
        else:
            entries.append([current[0]])
    return entries


def _tight_page_footnote_bboxes(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> dict[int, BBox]:
    """以有效行高和相邻行距收紧脚注框，避免异常字体框跨入相邻条目。"""

    if not lines:
        return {}
    angle = lines[0].angle
    line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines]
    line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
    row_pitches = [
        current[1][1] - previous[1][1]
        for previous, current in zip(line_geometry, line_geometry[1:])
        if current[1][1] - previous[1][1] > 0.25 * median_height
    ]
    median_pitch = statistics.median(row_pitches) if row_pitches else None

    output: dict[int, BBox] = {}
    for line, bbox in line_geometry:
        tight_height = min(
            bbox[3] - bbox[1],
            _line_effective_height(line, bbox),
        )
        if median_pitch is not None:
            tight_height = min(tight_height, 0.9 * median_pitch)
        tight_height = max(0.1, tight_height)
        center_y = _bbox_center_y(bbox)
        tight_local_bbox = (
            bbox[0],
            center_y - 0.5 * tight_height,
            bbox[2],
            center_y + 0.5 * tight_height,
        )
        output[line.source_index] = _rotate_bbox_from_upright(
            tight_local_bbox,
            page_size,
            angle,
        )
    return output


def _infer_local_text_lane_map(lane: _TextLane) -> dict[int, _TextLane]:
    """从连续同左缘正文推导局部栏宽，修正跨栏上文污染的全宽栏带。"""

    if lane.is_span or len(lane.lines) < 3:
        return {}
    rows = sorted(
        lane.lines,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in rows)
    left_tolerance = max(3.0, 0.75 * median_height)
    height_ratio_limit = 1.25
    runs: list[list[tuple[_LineItem, BBox]]] = []
    current_run: list[tuple[_LineItem, BBox]] = []

    def submit_run() -> None:
        """提交当前连续正文行，语义行和明显左缘变化都会结束该局部区段。"""

        nonlocal current_run
        if current_run:
            runs.append(current_run)
            current_run = []

    for item in rows:
        line, bbox = item
        if line.semantic_type is not None:
            submit_run()
            continue
        if not current_run:
            current_run = [item]
            continue
        run_left = statistics.median(member[1][0] for member in current_run)
        run_heights = [_line_effective_height(member, member_bbox) for member, member_bbox in current_run]
        current_height = _line_effective_height(line, bbox)
        if (
            abs(bbox[0] - run_left) <= left_tolerance
            and max([*run_heights, current_height]) / max(0.1, min([*run_heights, current_height])) <= height_ratio_limit
        ):
            current_run.append(item)
        else:
            submit_run()
            current_run = [item]
    submit_run()

    global_width = max(0.1, lane.right - lane.left)
    local_by_source: dict[int, _TextLane] = {}
    for run in runs:
        if len(run) < 3:
            continue
        local_left = statistics.median(bbox[0] for _line, bbox in run)
        local_right = max(bbox[2] for _line, bbox in run)
        local_width = max(0.1, local_right - local_left)
        wide_support = sum(bbox[2] - bbox[0] >= 0.7 * local_width for _line, bbox in run)
        if global_width < 1.4 * local_width or wide_support < 3:
            continue
        local_lane = _TextLane(
            left=local_left,
            right=local_right,
            lines=run,
            is_span=False,
        )
        for line, _bbox in run:
            local_by_source[line.source_index] = local_lane
    return local_by_source


def _structured_text_break_sources(
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
) -> set[int]:
    """用重复强调首行和前行右侧留白确认结构化正文的新段起点。"""

    rows = sorted(
        lane.lines,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    lane_width = max(0.1, lane.right - lane.left)
    break_sources: set[int] = set()
    regions: list[list[tuple[_LineItem, BBox]]] = []
    for row in rows:
        if row[0].semantic_type is not None:
            if regions and regions[-1]:
                regions.append([])
            continue
        if not regions:
            regions.append([])
        regions[-1].append(row)

    for region in regions:
        candidates: list[int] = []
        for index, (line, bbox) in enumerate(region):
            height = _line_effective_height(line, bbox)
            line_width = bbox[2] - bbox[0]
            if (
                line.leading_emphasis_width is not None
                and line.leading_emphasis_width <= 0.2 * lane_width
                and line_width >= 0.95 * lane_width
                and abs(bbox[0] - lane.left) <= 0.75 * height
            ):
                candidates.append(index)
        if len(candidates) < 3:
            continue
        for index in candidates:
            if index == 0:
                continue
            previous = region[index - 1]
            current = region[index]
            pair_height = max(
                _line_effective_height(*previous),
                _line_effective_height(*current),
            )
            previous_fill = (previous[1][2] - lane.left) / lane_width
            vertical_gap = _effective_text_row_gap(previous, current)
            if previous_fill <= 0.8 and -0.25 * pair_height <= vertical_gap <= regular_gap + max(
                0.75 * pair_height, 3.0 * gap_mad
            ):
                break_sources.add(current[0].source_index)
    return break_sources


def _isolated_indented_paragraph_break_sources(
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
) -> set[int]:
    """识别短终止尾行之后的缩进首行，并要求下一行回到稳定栏左缘。"""

    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    lane_width = max(0.1, lane.right - lane.left)
    output: set[int] = set()
    terminal_re = re.compile(r"[.!?。！？:：;；][\]\)}）】》”’'\"]*$")
    for previous, current, following in zip(
        rows,
        rows[1:],
        rows[2:],
    ):
        previous_height = _line_effective_height(*previous)
        current_height = _line_effective_height(*current)
        following_height = _line_effective_height(*following)
        pair_height = max(
            previous_height,
            current_height,
            following_height,
        )
        current_indent = current[1][0] - lane.left
        if (
            previous[1][2] - previous[1][0] > 0.3 * lane_width
            or terminal_re.search(previous[0].text.rstrip()) is None
            or not max(5.0, 0.65 * pair_height) <= current_indent <= 3.0 * pair_height
            or current[1][2] - current[1][0] < 0.75 * lane_width
            or abs(following[1][0] - lane.left) > 0.75 * pair_height
            or following[1][2] - following[1][0] < 0.65 * lane_width
            or not _title_fonts_compatible(current[0], following[0])
        ):
            continue
        first_gap = _effective_body_text_row_gap(previous, current)
        second_gap = _effective_body_text_row_gap(current, following)
        gap_limit = regular_gap + max(
            0.75 * pair_height,
            3.0 * gap_mad,
        )
        if -0.25 * pair_height <= first_gap <= gap_limit and -0.25 * pair_height <= second_gap <= gap_limit:
            output.add(current[0].source_index)
    return output


def _centered_visual_reset_break_sources(
    lane: _TextLane,
    visual_bboxes: Sequence[BBox],
    local_page_height: float,
) -> set[int]:
    """识别视觉主体下方短居中行到更宽居中行的独立注释重启。"""

    if not visual_bboxes:
        return set()
    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    output: set[int] = set()
    for previous, current in zip(rows, rows[1:]):
        previous_bbox = previous[1]
        current_bbox = current[1]
        previous_width = previous_bbox[2] - previous_bbox[0]
        current_width = current_bbox[2] - current_bbox[0]
        pair_height = max(
            _line_effective_height(*previous),
            _line_effective_height(*current),
        )
        if (
            previous_width > 0.7 * current_width
            or current_bbox[0] > previous_bbox[0] - 0.25 * pair_height
            or current_bbox[2] < previous_bbox[2] + 0.25 * pair_height
            or abs(_bbox_center_x(previous_bbox) - _bbox_center_x(current_bbox)) > 0.1 * current_width
        ):
            continue
        vertical_gap = _effective_text_row_gap(previous, current)
        if not -0.25 * pair_height <= vertical_gap <= 0.75 * pair_height:
            continue
        if any(
            -0.25 * pair_height <= previous_bbox[1] - visual_bbox[3] <= max(2.0 * pair_height, 0.03 * local_page_height)
            and _bbox_axis_overlap_ratio(current_bbox, visual_bbox, axis="x") >= 0.8
            and abs(_bbox_center_x(current_bbox) - _bbox_center_x(visual_bbox))
            <= 0.12 * max(current_width, visual_bbox[2] - visual_bbox[0])
            for visual_bbox in visual_bboxes
        ):
            output.add(current[0].source_index)
    return output


def _leading_typography_reset_break_sources(
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
) -> set[int]:
    """识别短尾之后以独立行首字体 run 开启的宽行结构段。"""

    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    lane_width = max(0.1, lane.right - lane.left)
    output: set[int] = set()
    for previous, current in zip(rows, rows[1:]):
        previous_width = previous[1][2] - previous[1][0]
        current_width = current[1][2] - current[1][0]
        pair_height = max(
            _line_effective_height(*previous),
            _line_effective_height(*current),
        )
        if (
            current[0].leading_typography_width is None
            or current[0].leading_typography_width > 0.2 * lane_width
            or previous_width > 0.45 * lane_width
            or current_width < 0.75 * lane_width
            or abs(previous[1][0] - lane.left) > 0.75 * pair_height
            or abs(current[1][0] - lane.left) > 0.75 * pair_height
            or current[0].formula_candidate_only
            or current[0].compact_formula_cluster
            or current[0].inline_math_regions
        ):
            continue
        vertical_gap = _effective_body_text_row_gap(previous, current)
        if (
            -0.25 * pair_height
            <= vertical_gap
            <= regular_gap
            + max(
                0.75 * pair_height,
                3.0 * gap_mad,
            )
        ):
            output.add(current[0].source_index)
    return output


def _formula_style_text_row_break_sources(
    lane: _TextLane,
) -> set[int]:
    """按相邻显示行几何拆分被公式检测回退为正文的独立文本行。"""

    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    lane_width = max(0.1, lane.right - lane.left)
    matching_edges: set[int] = set()
    for index, (previous, current) in enumerate(zip(rows, rows[1:])):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        if not (
            previous_line.paragraph_formula_context
            and current_line.paragraph_formula_context
        ):
            continue
        previous_height = _line_effective_height(*previous)
        current_height = _line_effective_height(*current)
        minimum_height = min(previous_height, current_height)
        maximum_height = max(previous_height, current_height)
        if minimum_height < 0.75 * maximum_height:
            continue
        previous_width = previous_bbox[2] - previous_bbox[0]
        current_width = current_bbox[2] - current_bbox[0]
        if (
            min(previous_width, current_width) < 0.45 * lane_width
            or max(previous_width, current_width) > 0.95 * lane_width
        ):
            continue
        lane_center = 0.5 * (lane.left + lane.right)
        if (
            abs(_bbox_center_x(previous_bbox) - lane_center) > 0.15 * lane_width
            or abs(_bbox_center_x(current_bbox) - lane_center) > 0.15 * lane_width
        ):
            continue
        vertical_overlap = max(
            0.0,
            min(previous_bbox[3], current_bbox[3])
            - max(previous_bbox[1], current_bbox[1]),
        )
        top_pitch = current_bbox[1] - previous_bbox[1]
        pair_height = statistics.median((previous_height, current_height))
        if (
            vertical_overlap <= 0.2 * minimum_height
            and 0.9 * pair_height <= top_pitch <= 2.0 * pair_height
        ):
            matching_edges.add(index)

    output: set[int] = set()
    for index in matching_edges:
        output.add(rows[index][0].source_index)
        output.add(rows[index + 1][0].source_index)
        if index + 2 < len(rows):
            # 同时保护显示行组后的正文起点，避免上下文恢复阶段重新跨界合并。
            output.add(rows[index + 2][0].source_index)
    return output


def _front_matter_keyword_break_sources(
    lane: _TextLane,
    local_page_height: float,
    page_index: int | None,
) -> set[int]:
    """把首页关键词和文献元数据行固定为独立文本块起点。"""

    if page_index != 0:
        return set()
    return {
        line.source_index
        for line, bbox in lane.lines
        if line.semantic_type is None
        and bbox[1] <= 0.65 * local_page_height
        and _FRONT_MATTER_FIELD_RE.match(line.text) is not None
    }


def _component_starts_with_emphasized_row(
    lines: list[_LineItem],
) -> bool:
    """识别行内强调或首行字重显著高于后续正文的组件起点。"""

    if not lines:
        return False
    if lines[0].leading_emphasis_width is not None:
        return True
    first_weight = lines[0].dominant_font_weight
    following_weights = [line.dominant_font_weight for line in lines[1:] if line.dominant_font_weight is not None]
    if first_weight is None or not following_weights:
        return False
    body_weight = statistics.median(following_weights)
    return first_weight - body_weight >= 100.0 and first_weight >= 1.15 * max(1.0, body_weight)


def _explicit_text_break_sources(
    lane: _TextLane,
) -> set[int]:
    """用通用列表标记和 E-mail 元数据确认正文中的显式硬分段。"""

    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    output = {line.source_index for line, _bbox in rows if _ABSTRACT_METADATA_RE.match(line.text) is not None}
    lane_width = max(0.1, lane.right - lane.left)
    output.update(
        line.source_index
        for line, bbox in rows
        if _BULLET_ITEM_RE.match(line.text) is not None and bbox[2] - bbox[0] >= 0.8 * lane_width
    )
    for index, (line, bbox) in enumerate(rows):
        if _EMAIL_METADATA_RE.match(line.text) is None or index == 0:
            continue
        previous_line, previous_bbox = rows[index - 1]
        pair_height = max(
            _line_effective_height(previous_line, previous_bbox),
            _line_effective_height(line, bbox),
        )
        if abs(bbox[0] - previous_bbox[0]) <= 0.75 * pair_height:
            output.add(line.source_index)
    for row_index, (previous, current) in enumerate(
        zip(rows, rows[1:]),
    ):
        previous_is_label = _LABELLED_METADATA_RE.match(
            previous[0].text,
        )
        current_is_label = _LABELLED_METADATA_RE.match(
            current[0].text,
        )
        label_pair_height = max(
            _line_effective_height(*previous),
            _line_effective_height(*current),
        )
        if (
            previous_is_label is not None
            and current_is_label is not None
            and _URL_LINE_RE.match(current[0].text) is None
            and len(previous_is_label.group("label")) >= 4
            and len(current_is_label.group("label")) >= 4
            and any("\u3400" <= char <= "\u9fff" for char in previous_is_label.group("label"))
            and any("\u3400" <= char <= "\u9fff" for char in current_is_label.group("label"))
            and previous_is_label.group("label").casefold() != current_is_label.group("label").casefold()
            and current[1][1] - previous[1][1] <= 2.0 * label_pair_height
            and previous[1][2] - previous[1][0] <= 0.75 * lane_width
            and current[1][2] - current[1][0] <= 0.75 * lane_width
        ):
            output.add(current[0].source_index)
        pair_height = max(
            _line_effective_height(*previous),
            _line_effective_height(*current),
        )
        next_row = rows[row_index + 2] if row_index + 2 < len(rows) else None
        indented_item_continuation = (
            current[1][0] - lane.left >= max(5.0, 0.65 * pair_height)
            and next_row is not None
            and next_row[1][0] - lane.left <= 0.5 * pair_height
            and 0.5 * pair_height <= next_row[1][1] - current[1][1] <= 2.25 * pair_height
        )
        if (
            _LIST_ITEM_RE.match(current[0].text) is not None
            and previous[0].text.rstrip().endswith((":", "："))
            and previous[1][2] - previous[1][0] <= 0.8 * lane_width
            and indented_item_continuation
        ):
            output.add(current[0].source_index)
    return output


def _build_text_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    drawing_lines: list[_AxisLine] | None = None,
    *,
    page_footnote_groups: Sequence[set[int]] | None = None,
    page_index: int | None = None,
    visual_bboxes: Sequence[BBox] | None = None,
) -> list[dict[str, Any]]:
    """先构建分组脚注，再按类型屏障、栏带和自然段边界聚合其余文本。"""

    blocks, grouped_footnote_indices = _build_grouped_page_footnote_blocks(
        lines,
        page_footnote_groups or [],
        page_size,
    )
    lines = [line for line in lines if line.source_index not in grouped_footnote_indices]
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if not line_geometry:
            continue
        line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        local_visual_bboxes = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in (visual_bboxes or [])]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        local_axis_lines = _transform_axis_lines(drawing_lines or [], page_size, angle)
        split_row_counts: dict[int, int] = {}
        for line, _bbox in line_geometry:
            if line.visual_row_id is not None and line.split_from_row:
                split_row_counts[line.visual_row_id] = split_row_counts.get(line.visual_row_id, 0) + 1

        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            if not lane.lines:
                continue
            regular_gap, gap_mad = _estimate_lane_gap(lane)
            local_lane_by_source = _infer_local_text_lane_map(lane)
            structured_break_sources = _structured_text_break_sources(
                lane,
                regular_gap,
                gap_mad,
            )
            isolated_break_sources = _isolated_indented_paragraph_break_sources(
                lane,
                regular_gap,
                gap_mad,
            )
            structured_break_sources.update(
                isolated_break_sources,
            )
            visual_reset_sources = _centered_visual_reset_break_sources(
                lane,
                local_visual_bboxes,
                local_page_height,
            )
            typography_reset_sources = _leading_typography_reset_break_sources(
                lane,
                regular_gap,
                gap_mad,
            )
            formula_text_break_sources = _formula_style_text_row_break_sources(
                lane,
            )
            structured_break_sources.update(visual_reset_sources)
            structured_break_sources.update(typography_reset_sources)
            structured_break_sources.update(formula_text_break_sources)
            protected_break_sources: set[int] = set()
            protected_break_sources.update(visual_reset_sources)
            protected_break_sources.update(typography_reset_sources)
            protected_break_sources.update(formula_text_break_sources)
            protected_break_sources.update(
                _front_matter_keyword_break_sources(
                    lane,
                    local_page_height,
                    page_index,
                )
            )
            protected_break_sources.update(
                _explicit_text_break_sources(lane),
            )
            structured_break_sources.update(
                protected_break_sources,
            )
            hanging_indent_groups = _build_hanging_indent_group_map(
                lane,
                table_bboxes,
                local_axis_lines,
            )
            component: list[tuple[_LineItem, BBox]] = [lane.lines[0]]
            components: list[list[tuple[_LineItem, BBox]]] = []
            for previous, current in zip(lane.lines, lane.lines[1:]):
                previous_type = previous[0].semantic_type
                current_type = current[0].semantic_type
                if previous_type != current_type:
                    should_connect = False
                elif previous_type is not None:
                    should_connect = _should_connect_semantic_rows(
                        previous,
                        current,
                        lane,
                        regular_gap,
                        table_bboxes,
                        local_axis_lines,
                    )
                else:
                    previous_group = hanging_indent_groups.get(previous[0].source_index)
                    current_group = hanging_indent_groups.get(current[0].source_index)
                    previous_local_lane = local_lane_by_source.get(previous[0].source_index)
                    current_local_lane = local_lane_by_source.get(current[0].source_index)
                    connection_lane = (
                        current_local_lane
                        if current_local_lane is not None and previous_local_lane is current_local_lane
                        else lane
                    )
                    if (
                        current[0].style_scale_repaired
                        and current[0].split_from_row
                        and current[0].visual_row_id is not None
                        and split_row_counts.get(
                            current[0].visual_row_id,
                            0,
                        )
                        >= 2
                    ):
                        should_connect = False
                    elif current[0].source_index in structured_break_sources:
                        should_connect = False
                    elif _starts_structural_reference_entry(previous, current):
                        # 编号只确认已经由悬挂缩进几何形成的新条目，不能单独扩张范围。
                        should_connect = False
                    elif is_hyphen_at_line_end(previous[0].text):
                        # 断词续行优先于悬挂缩进分组，但仍复用正文连接中的距离和障碍限制。
                        should_connect = _should_connect_text_rows(
                            previous,
                            current,
                            connection_lane,
                            regular_gap,
                            gap_mad,
                            table_bboxes,
                            local_axis_lines,
                        )
                    elif previous_group is not None or current_group is not None:
                        should_connect = previous_group is not None and previous_group == current_group
                    else:
                        should_connect = _should_connect_text_rows(
                            previous,
                            current,
                            connection_lane,
                            regular_gap,
                            gap_mad,
                            table_bboxes,
                            local_axis_lines,
                        )
                if should_connect:
                    component.append(current)
                else:
                    components.append(component)
                    component = [current]
            components.append(component)

            for component_geometry in components:
                component_lines = [item[0] for item in component_geometry]
                component_local_lane = local_lane_by_source.get(component_lines[0].source_index)
                if component_local_lane is None or not all(
                    local_lane_by_source.get(line.source_index) is component_local_lane for line in component_lines
                ):
                    component_local_lane = lane
                if component_lines[0].semantic_type == "doc_title":
                    # 文档标题保留自然换行，避免混排标题因语言检测在中文折行处插入空格。
                    content = "\n".join(
                        normalized for line in component_lines if (normalized := _normalize_native_run_text(line.text))
                    )
                else:
                    content = _merge_text_line_content([line.text for line in component_lines])
                if not content:
                    continue
                visual_row_ids = {line.visual_row_id for line in component_lines if line.visual_row_id is not None}
                single_run_row_id = (
                    component_lines[0].visual_row_id
                    if len(component_lines) == 1
                    and component_lines[0].split_from_row
                    and component_lines[0].visual_row_id is not None
                    else None
                )
                local_output_line_bboxes, output_bbox_repaired = _local_tight_output_line_bboxes(
                    component_lines,
                    page_size,
                    angle,
                )
                blocks.append(
                    {
                        "type": component_lines[0].semantic_type or "text",
                        "bbox": _bbox_union_many([line.bbox for line in component_lines]),
                        "angle": angle,
                        "content": content,
                        "_visual_row_ids": visual_row_ids,
                        "_single_run_row_id": single_run_row_id,
                        "_local_line_bboxes": [bbox for _line, bbox in component_geometry],
                        "_local_output_line_bboxes": local_output_line_bboxes,
                        "_output_bbox_repaired": output_bbox_repaired,
                        "_line_heights": [_line_effective_height(line, bbox) for line, bbox in component_geometry],
                        "_font_signatures": {
                            line.font_signature
                            for line in component_lines
                            if line.font_signature is not None and line.font_coverage >= 0.5
                        },
                        "_inline_math_regions": [region for line in component_lines for region in line.inline_math_regions],
                        _PARAGRAPH_FORMULA_CONTEXT_MARKER: any(line.paragraph_formula_context for line in component_lines),
                        "_lane_interval": (
                            component_local_lane.left,
                            component_local_lane.right,
                        ),
                        "_lane_is_span": component_local_lane.is_span,
                        "_hard_break_before": (component_lines[0].source_index in structured_break_sources),
                        "_protected_hard_break_before": (component_lines[0].source_index in protected_break_sources),
                        "_hanging_indent_group": hanging_indent_groups.get(
                            component_lines[0].source_index,
                        ),
                        "_leading_emphasis_start": _component_starts_with_emphasized_row(
                            component_lines,
                        ),
                    }
                )
    blocks = _merge_short_same_baseline_prefix_blocks(
        blocks,
        page_size,
    )
    blocks = _merge_spatial_text_components(blocks, page_size)
    blocks = _merge_list_intro_text_components(blocks)
    blocks = _merge_unterminated_text_components(blocks)
    blocks = _merge_overlapping_same_line_text_blocks(blocks, page_size)
    blocks = _merge_inline_math_fragment_text_blocks(
        blocks,
        page_size,
    )
    return _merge_paragraph_formula_context_blocks(
        blocks,
        page_size,
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
