# Copyright (c) Opendatalab. All rights reserved.
"""依据栏带、缩进和排版重置寻找正文行分组边界。"""

from __future__ import annotations

import re
import statistics
from typing import Sequence

from .....types import BBox
from .....utils.text import is_hyphen_at_line_end
from ..geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _rotate_bbox_to_upright,
)
from ..line_layout import (
    _connection_crosses_table,
    _effective_body_text_row_gap,
    _effective_text_row_gap,
    _horizontal_rule_separates_rows,
    _line_effective_height,
    _line_tight_output_bbox,
    _title_fonts_compatible,
)
from ..models import _LineItem, _LocalAxisLine, _TextLane
from .common import (
    _ABSTRACT_METADATA_RE,
    _BULLET_ITEM_RE,
    _EMAIL_METADATA_RE,
    _FRONT_MATTER_FIELD_RE,
    _LABELLED_METADATA_RE,
    _LIST_ITEM_RE,
    _REFERENCE_ENTRY_RE,
    _URL_LINE_RE,
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
        if not (previous_line.paragraph_formula_context and current_line.paragraph_formula_context):
            continue
        previous_height = _line_effective_height(*previous)
        current_height = _line_effective_height(*current)
        minimum_height = min(previous_height, current_height)
        maximum_height = max(previous_height, current_height)
        if minimum_height < 0.75 * maximum_height:
            continue
        previous_width = previous_bbox[2] - previous_bbox[0]
        current_width = current_bbox[2] - current_bbox[0]
        if min(previous_width, current_width) < 0.45 * lane_width or max(previous_width, current_width) > 0.95 * lane_width:
            continue
        lane_center = 0.5 * (lane.left + lane.right)
        if (
            abs(_bbox_center_x(previous_bbox) - lane_center) > 0.15 * lane_width
            or abs(_bbox_center_x(current_bbox) - lane_center) > 0.15 * lane_width
        ):
            continue
        vertical_overlap = max(
            0.0,
            min(previous_bbox[3], current_bbox[3]) - max(previous_bbox[1], current_bbox[1]),
        )
        top_pitch = current_bbox[1] - previous_bbox[1]
        pair_height = statistics.median((previous_height, current_height))
        if vertical_overlap <= 0.2 * minimum_height and 0.9 * pair_height <= top_pitch <= 2.0 * pair_height:
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


__all__ = [
    "_local_tight_output_line_bboxes",
    "_starts_structural_reference_entry",
    "_build_hanging_indent_group_map",
    "_infer_local_text_lane_map",
    "_structured_text_break_sources",
    "_isolated_indented_paragraph_break_sources",
    "_centered_visual_reset_break_sources",
    "_leading_typography_reset_break_sources",
    "_formula_style_text_row_break_sources",
    "_front_matter_keyword_break_sources",
    "_component_starts_with_emphasized_row",
    "_explicit_text_break_sources",
]
