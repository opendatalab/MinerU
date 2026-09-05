# Copyright (c) Opendatalab. All rights reserved.
"""保持既有顺序编排正文行分组与块级组装。"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from .....types import BBox
from .....utils.text import is_hyphen_at_line_end
from ..geometry import (
    _bbox_union_many,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from ..line_layout import (
    _estimate_lane_gap,
    _infer_text_lanes,
    _line_effective_height,
    _should_connect_semantic_rows,
    _should_connect_text_rows,
)
from ..models import _AxisLine, _LineItem
from ..native_text import _normalize_native_run_text
from .common import _PARAGRAPH_FORMULA_CONTEXT_MARKER, _merge_text_line_content
from .footnotes import _build_grouped_page_footnote_blocks
from .merging import (
    _merge_inline_math_fragment_text_blocks,
    _merge_list_intro_text_components,
    _merge_overlapping_same_line_text_blocks,
    _merge_paragraph_formula_context_blocks,
    _merge_short_same_baseline_prefix_blocks,
    _merge_spatial_text_components,
    _merge_unterminated_text_components,
)
from .rows import (
    _build_hanging_indent_group_map,
    _centered_visual_reset_break_sources,
    _component_starts_with_emphasized_row,
    _explicit_text_break_sources,
    _formula_style_text_row_break_sources,
    _front_matter_keyword_break_sources,
    _infer_local_text_lane_map,
    _isolated_indented_paragraph_break_sources,
    _leading_typography_reset_break_sources,
    _local_tight_output_line_bboxes,
    _starts_structural_reference_entry,
    _structured_text_break_sources,
)


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


__all__ = ["_build_text_blocks"]
