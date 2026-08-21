# Copyright (c) Opendatalab. All rights reserved.

"""检测 Form、矢量图形和栅格图片并认领内部文本。"""

from __future__ import annotations

from dataclasses import replace
import math
import re
import statistics
from typing import Any


from mineru.types import BBox
from mineru.utils.pdf_document import PDFPathInfo

from .models import (
    _AxisLine,
    _GraphicCandidate,
    _LineItem,
    _PageSource,
    _TextLane,
)
from .geometry import (
    _bbox_area,
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_distance,
    _bbox_intersects,
    _bbox_overlap_in_first,
    _bbox_overlap_in_smaller,
    _bbox_union,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _point_in_bbox,
    _rotate_bbox_to_upright,
)
from .native_text import (
    _fill_native_typography,
    _normalize_native_run_text,
    _sanitize_pdf_control_text,
)
from .line_layout import (
    _infer_text_lanes,
    _line_effective_height,
)
from .line_merging import _join_formula_visual_row


_MIN_RASTER_IMAGE_PAGE_AREA_RATIO = 0.0038


_SIGNATURE_IMAGE_BBOX_DEDUP_TOLERANCE = 0.5


_MIN_FORM_IMAGE_PAGE_AREA_RATIO = 0.01


_MAX_FORM_IMAGE_PAGE_AREA_RATIO = 0.8


_IMAGE_CONTAINER_OVERLAP_THRESHOLD = 0.5


_FIGURE_CAPTION_LINE_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?)[ \t]*\d+[A-Za-z]?(?:\s*[.:])?",
    re.IGNORECASE,
)


def _form_supersedes_nested_bbox(form_bbox: BBox, nested_bbox: BBox) -> bool:
    """判断 Form 是否应整体吞并其内部面积明显更小的候选容器。"""

    form_area = _bbox_area(form_bbox)
    nested_area = _bbox_area(nested_bbox)
    return (
        form_area > 0
        and nested_area < 0.5 * form_area
        and _bbox_overlap_in_first(nested_bbox, form_bbox) >= 0.9
    )


def _tighten_form_image_bbox(
    source: _PageSource,
    form_bbox: BBox,
) -> BBox:
    """用充分的 Form 内部矢量与文本证据收紧空白容器，证据不足时保留原框。"""

    internal_paths = [
        path_info.bbox
        for path_info in source.path_infos
        if path_info.form_depth > 0
        and _bbox_overlap_in_first(path_info.bbox, form_bbox) >= 0.9
    ]
    internal_drawing_lines = [
        drawing_line.bbox
        for drawing_line in source.drawing_lines
        if _bbox_overlap_in_first(drawing_line.bbox, form_bbox) >= 0.9
    ]
    # 至少两个嵌套 Path 和四个矢量元素，避免只凭普通边框或少量文本裁剪 Form。
    if len(internal_paths) < 2 or len(internal_paths) + len(internal_drawing_lines) < 4:
        return form_bbox
    internal_text = [
        line.bbox
        for line in source.lines
        if _bbox_overlap_in_first(line.bbox, form_bbox) >= 0.9
    ]
    evidence_bbox = _clip_bbox(
        _bbox_union_many(
            internal_paths + internal_drawing_lines + internal_text
        ),
        source.page_size,
    )
    if evidence_bbox is None:
        return form_bbox
    form_width = max(0.1, form_bbox[2] - form_bbox[0])
    form_height = max(0.1, form_bbox[3] - form_bbox[1])
    evidence_width = evidence_bbox[2] - evidence_bbox[0]
    evidence_height = evidence_bbox[3] - evidence_bbox[1]
    if (
        evidence_width < 0.5 * form_width
        or evidence_height < 0.5 * form_height
        or _bbox_area(evidence_bbox) < 0.25 * _bbox_area(form_bbox)
    ):
        return form_bbox
    return evidence_bbox


def _select_form_image_bboxes(source: _PageSource) -> list[BBox]:
    """按页面占比、行高和内部视觉证据筛选矢量 Form 图片候选。"""

    page_area = max(0.0, source.page_size[0]) * max(0.0, source.page_size[1])
    if page_area <= 0 or not source.form_bboxes:
        return []
    effective_heights = [
        _line_effective_height(line, line.bbox)
        for line in source.lines
        if line.angle == 0
    ]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0
    output: list[BBox] = []
    for raw_bbox in source.form_bboxes:
        bbox = _clip_bbox(_coerce_bbox(raw_bbox), source.page_size)
        if bbox is None:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area_ratio = _bbox_area(bbox) / page_area
        if not (
            _MIN_FORM_IMAGE_PAGE_AREA_RATIO <= area_ratio <= _MAX_FORM_IMAGE_PAGE_AREA_RATIO
            and width >= 4.0 * median_height
            and height >= 4.0 * median_height
        ):
            continue

        member_rows = {
            line.visual_row_id if line.visual_row_id is not None else line.source_index
            for line in source.lines
            if _bbox_overlap_in_first(line.bbox, bbox) >= 0.9
        }
        internal_drawing_count = sum(
            _bbox_overlap_in_first(drawing_line.bbox, bbox) >= 0.9
            for drawing_line in source.drawing_lines
        )
        if len(member_rows) < 2 and internal_drawing_count < 4:
            continue
        output.append(_tighten_form_image_bbox(source, bbox))
    return sorted(output, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _build_form_image_blocks(
    source: _PageSource,
    form_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """把 Form 及其完整内含文本输出为 image，并保持 source_index 唯一认领。"""

    if not form_bboxes:
        return [], set()
    members_by_candidate: list[list[_LineItem]] = [[] for _ in form_bboxes]
    claimed: set[int] = set()
    for line in source.lines:
        if line.source_index in claimed_line_indices:
            continue
        matching_indices = [
            candidate_index
            for candidate_index, bbox in enumerate(form_bboxes)
            if _bbox_overlap_in_first(line.bbox, bbox) >= 0.9
        ]
        if not matching_indices:
            continue
        candidate_index = min(
            matching_indices,
            key=lambda index: (_bbox_area(form_bboxes[index]), index),
        )
        members_by_candidate[candidate_index].append(line)
        claimed.add(line.source_index)

    blocks = [
        {
            "type": "image",
            "bbox": bbox,
            "angle": 0,
            "content": _image_members_to_content(members, source.page_size),
        }
        for bbox, members in zip(form_bboxes, members_by_candidate, strict=True)
    ]
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _build_graphic_like_blocks(
    source: _PageSource,
    table_bboxes: list[BBox],
    claimed_line_indices: set[int],
    strong_core_bboxes: list[BBox] | None = None,
) -> tuple[list[dict[str, Any]], set[int]]:
    """在表格认领后把紧凑绘图组件及其短标签聚成内部图形文本块。"""

    lines = [line for line in source.lines if line.source_index not in claimed_line_indices]
    if strong_core_bboxes is None:
        strong_core_bboxes = _detect_strong_graphic_bboxes(source)
    if len(lines) < 2 or (len(source.drawing_lines) < 4 and not strong_core_bboxes):
        return [], set()

    effective_heights = [
        max(
            0.1,
            line.effective_height
            or min(
                max(0.1, line.bbox[2] - line.bbox[0]),
                max(0.1, line.bbox[3] - line.bbox[1]),
            ),
        )
        for line in lines
    ]
    median_height = statistics.median(effective_heights)
    lanes = _infer_graphic_text_lanes(lines, source.page_size, median_height)
    line_candidates = _detect_graphic_candidates(
        source.drawing_lines,
        source.page_size,
        median_height,
        lanes,
        table_bboxes,
    )
    # 复杂 Path 或成对坐标轴形成的强图形核心优先于普通绘图线组件，
    # 避免同一图表被拆成多个相互重叠的 image。
    candidates = [
        candidate
        for candidate in line_candidates
        if not any(
            _bbox_overlap_in_smaller(candidate.core_bbox, core_bbox) >= 0.5
            for core_bbox in strong_core_bboxes
        )
    ]
    candidates.extend(
        _GraphicCandidate(
            core_bbox=core_bbox,
            lane_index=_strong_graphic_lane_index(
                core_bbox,
                lanes,
                median_height,
            ),
            label_margin_scale=(
                2.5
                if any(
                    _bbox_overlap_in_smaller(candidate.core_bbox, core_bbox) >= 0.5
                    and _bbox_area(candidate.core_bbox) >= 0.8 * _bbox_area(core_bbox)
                    for candidate in line_candidates
                )
                else 1.0
            ),
        )
        for core_bbox in strong_core_bboxes
        if not any(
            _bbox_overlap_in_smaller(core_bbox, table_bbox) >= 0.5
            for table_bbox in table_bboxes
        )
    )
    if not candidates:
        return [], set()

    row_groups: dict[tuple[int, int, int, int], list[_LineItem]] = {}
    for line in lines:
        lane_index = _graphic_lane_index(line.bbox, lanes)
        if line.visual_row_id is None:
            row_kind, row_identity = 1, line.source_index
        else:
            row_kind, row_identity = 0, line.visual_row_id
        row_groups.setdefault(
            (line.angle, row_kind, row_identity, lane_index),
            [],
        ).append(line)

    protected_caption_indices = _graphic_caption_line_indices_to_preserve(
        lines,
        candidates,
        median_height,
    )
    protected_body_tail_indices = _graphic_body_tail_line_indices_to_preserve(
        lines,
        candidates,
        lanes,
        median_height,
    )

    for row_lines in row_groups.values():
        if any(
            line.source_index in protected_caption_indices | protected_body_tail_indices
            for line in row_lines
        ):
            continue
        row_lane_index = _graphic_lane_index(row_lines[0].bbox, lanes)
        matches: list[tuple[int, float, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            if candidate.lane_index >= 0 and candidate.lane_index != row_lane_index:
                continue
            member_flags = [
                _is_graphic_label_member(
                    line,
                    candidate.core_bbox,
                    median_height,
                    margin_scale=candidate.label_margin_scale,
                )
                for line in row_lines
            ]
            # 同一 pdftext 视觉行必须整体归属或整体保留，避免只吞掉 caption 的短碎片。
            if not all(member_flags):
                continue
            inside_count = sum(
                _point_in_bbox(
                    (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                    candidate.core_bbox,
                )
                for line in row_lines
            )
            mean_distance = statistics.fmean(
                _bbox_distance(line.bbox, candidate.core_bbox) for line in row_lines
            )
            matches.append((-inside_count, mean_distance, candidate_index))
        if not matches:
            continue
        candidate_index = min(matches)[2]
        candidates[candidate_index].line_indices.update(line.source_index for line in row_lines)

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    lines_by_index = {line.source_index: line for line in lines}
    for candidate in candidates:
        members = [
            lines_by_index[source_index]
            for source_index in sorted(candidate.line_indices)
            if source_index in lines_by_index
        ]
        if len(members) < 2:
            continue
        block = _graphic_members_to_block(candidate, members, source.page_size)
        if block is None:
            continue
        blocks.append(block)
        claimed.update(line.source_index for line in members)

    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _parallel_graphic_rule_pairs(
    drawing_lines: list[_AxisLine],
    image_bboxes: list[BBox],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    median_height: float,
) -> list[tuple[BBox, BBox]]:
    """筛选分别贴近两个并排图形上沿的同高长横线。"""

    minimum_rule_width = max(8.0 * median_height, 0.18 * page_size[0])
    long_rules = [
        line.bbox
        for line in drawing_lines
        if line.orientation == "horizontal"
        and line.bbox[2] - line.bbox[0] >= minimum_rule_width
        and not any(
            _point_in_bbox(
                (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                table_bbox,
            )
            for table_bbox in table_bboxes
        )
    ]
    ordered_images = sorted(image_bboxes, key=lambda bbox: (bbox[0], bbox[1]))
    rule_pairs: list[tuple[BBox, BBox]] = []
    seen_pairs: set[tuple[BBox, BBox]] = set()
    for left_index, left_image in enumerate(ordered_images):
        left_height = max(0.1, left_image[3] - left_image[1])
        for right_image in ordered_images[left_index + 1 :]:
            if left_image[2] >= right_image[0]:
                continue
            right_height = max(0.1, right_image[3] - right_image[1])
            image_overlap = max(
                0.0,
                min(left_image[3], right_image[3])
                - max(left_image[1], right_image[1]),
            )
            if image_overlap < 0.7 * min(left_height, right_height):
                continue
            if any(
                _bbox_overlap_in_smaller(image_bbox, table_bbox) >= 0.5
                for image_bbox in (left_image, right_image)
                for table_bbox in table_bboxes
            ):
                continue

            left_rules = [
                rule_bbox
                for rule_bbox in long_rules
                if _bbox_axis_overlap_ratio(rule_bbox, left_image, axis="x") >= 0.8
                and -0.25 * median_height
                <= left_image[1] - rule_bbox[3]
                <= 3.0 * median_height
            ]
            right_rules = [
                rule_bbox
                for rule_bbox in long_rules
                if _bbox_axis_overlap_ratio(rule_bbox, right_image, axis="x") >= 0.8
                and -0.25 * median_height
                <= right_image[1] - rule_bbox[3]
                <= 3.0 * median_height
            ]
            for left_rule in left_rules:
                for right_rule in right_rules:
                    if left_rule[2] >= right_rule[0]:
                        continue
                    if (
                        abs(_bbox_center_y(left_rule) - _bbox_center_y(right_rule))
                        > 0.5 * median_height
                    ):
                        continue
                    rule_gap = right_rule[0] - left_rule[2]
                    if not 0.5 * median_height <= rule_gap <= 5.0 * median_height:
                        continue
                    pair = (left_rule, right_rule)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        rule_pairs.append(pair)
    return rule_pairs


def _parallel_graphic_row_split_boundary(
    members: list[_LineItem],
    left_rule: BBox,
    right_rule: BBox,
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    median_height: float,
) -> float | None:
    """用横线栏沟和字符投影确认并排图形上方文本的安全切分点。"""

    if (
        not members
        or any(member.angle != 0 for member in members)
        or len({member.semantic_type for member in members}) != 1
    ):
        return None
    row_bbox = _bbox_union_many([member.bbox for member in members])
    if any(_bbox_intersects(row_bbox, table_bbox) for table_bbox in table_bboxes):
        return None
    rule_top = min(left_rule[1], right_rule[1])
    if not -0.2 * median_height <= rule_top - row_bbox[3] <= 1.5 * median_height:
        return None

    glyph_bboxes = [
        bbox
        for member in members
        for char in member.chars
        if str(char.get("char") or "").isprintable()
        and not str(char.get("char") or "").isspace()
        and (bbox := _clip_bbox(_coerce_bbox(char.get("bbox")), page_size))
        is not None
    ]
    if not glyph_bboxes:
        return None
    boundary = 0.5 * (left_rule[2] + right_rule[0])
    if any(
        _bbox_center_x(bbox) < left_rule[0] - median_height
        or _bbox_center_x(bbox) > right_rule[2] + median_height
        for bbox in glyph_bboxes
    ):
        return None
    left_glyphs = [bbox for bbox in glyph_bboxes if _bbox_center_x(bbox) < boundary]
    right_glyphs = [bbox for bbox in glyph_bboxes if _bbox_center_x(bbox) > boundary]
    if len(left_glyphs) < 3 or len(right_glyphs) < 3:
        return None
    left_width = max(bbox[2] for bbox in left_glyphs) - min(
        bbox[0] for bbox in left_glyphs
    )
    right_width = max(bbox[2] for bbox in right_glyphs) - min(
        bbox[0] for bbox in right_glyphs
    )
    if min(left_width, right_width) < 4.0 * median_height:
        return None
    left_edge = max(bbox[2] for bbox in left_glyphs)
    right_edge = min(bbox[0] for bbox in right_glyphs)
    if right_edge - left_edge < 0.75 * median_height:
        return None
    if not (
        left_edge <= left_rule[2] <= right_edge
        and left_edge <= right_rule[0] <= right_edge
    ):
        return None
    return boundary


def _split_parallel_graphic_rule_rows(
    lines: list[_LineItem],
    drawing_lines: list[_AxisLine],
    image_bboxes: list[BBox],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    *,
    source_index_start: int | None = None,
) -> list[_LineItem]:
    """按成对图形、独立顶边横线和栏沟字符投影拆分并排图形上方文本。"""

    horizontal_lines = [
        line for line in lines if line.angle == 0 and line.effective_height > 0
    ]
    if len(horizontal_lines) < 1 or len(image_bboxes) < 2:
        return list(lines)
    median_height = statistics.median(
        line.effective_height for line in horizontal_lines
    )
    rule_pairs = _parallel_graphic_rule_pairs(
        drawing_lines,
        image_bboxes,
        table_bboxes,
        page_size,
        median_height,
    )
    if not rule_pairs:
        return list(lines)

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in horizontal_lines:
        if line.visual_row_id is not None:
            row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)
    boundaries_by_row: dict[tuple[int, int], list[float]] = {}
    for row_key, members in row_groups.items():
        for left_rule, right_rule in rule_pairs:
            boundary = _parallel_graphic_row_split_boundary(
                members,
                left_rule,
                right_rule,
                table_bboxes,
                page_size,
                median_height,
            )
            if boundary is None:
                continue
            row_boundaries = boundaries_by_row.setdefault(row_key, [])
            if not any(
                abs(boundary - existing) <= 0.5 * median_height
                for existing in row_boundaries
            ):
                row_boundaries.append(boundary)
    if not boundaries_by_row:
        return list(lines)

    next_source_index = max(
        max((line.source_index for line in lines), default=-1) + 1,
        source_index_start or 0,
    )
    consumed_source_indices: set[int] = set()
    split_lines: list[_LineItem] = []
    for row_key, boundaries in boundaries_by_row.items():
        members = sorted(
            row_groups[row_key],
            key=lambda line: (line.bbox[0], line.run_index, line.source_index),
        )
        ordered_chars = [char for member in members for char in member.chars]
        split_indices: list[int] = []
        for boundary in sorted(boundaries):
            split_index = next(
                (
                    index
                    for index, char in enumerate(ordered_chars)
                    if str(char.get("char") or "").isprintable()
                    and not str(char.get("char") or "").isspace()
                    and (
                        bbox := _clip_bbox(
                            _coerce_bbox(char.get("bbox")),
                            page_size,
                        )
                    )
                    is not None
                    and _bbox_center_x(bbox) > boundary
                ),
                None,
            )
            if split_index is not None and split_index not in split_indices:
                split_indices.append(split_index)
        if not split_indices:
            continue
        ranges: list[tuple[int, int]] = []
        start = 0
        for split_index in sorted(split_indices):
            ranges.append((start, split_index))
            start = split_index
        ranges.append((start, len(ordered_chars)))

        source_indices = [member.source_index for member in members]
        rebuilt: list[_LineItem] = []
        for run_index, (start, end) in enumerate(ranges):
            run_chars = ordered_chars[start:end]
            run_bboxes = [
                bbox
                for char in run_chars
                if str(char.get("char") or "").isprintable()
                and not str(char.get("char") or "").isspace()
                and (
                    bbox := _clip_bbox(
                        _coerce_bbox(char.get("bbox")),
                        page_size,
                    )
                )
                is not None
            ]
            run_text = _normalize_native_run_text(
                "".join(str(char.get("char") or "") for char in run_chars)
            )
            if not run_text or not run_bboxes:
                continue
            if run_index < len(source_indices):
                source_index = source_indices[run_index]
            else:
                source_index = next_source_index
                next_source_index += 1
            template = members[min(run_index, len(members) - 1)]
            rebuilt_line = replace(
                template,
                text=run_text,
                bbox=_bbox_union_many(run_bboxes),
                source_index=source_index,
                chars=list(run_chars),
                visual_row_id=row_key[1],
                run_index=run_index,
                split_from_row=True,
                preserve_split_boundary=True,
            )
            _fill_native_typography(rebuilt_line, page_size)
            rebuilt.append(rebuilt_line)
        if len(rebuilt) < 2:
            continue
        consumed_source_indices.update(member.source_index for member in members)
        split_lines.extend(rebuilt)

    output = [
        line for line in lines if line.source_index not in consumed_source_indices
    ]
    output.extend(split_lines)
    output.sort(key=lambda line: (line.angle, line.bbox[1], line.bbox[0], line.source_index))
    return output


def _graphic_caption_line_indices_to_preserve(
    lines: list[_LineItem],
    candidates: list[_GraphicCandidate],
    median_height: float,
) -> set[int]:
    """保护贴近图形下沿的图注及其同字体续行，避免末词被图片容器认领。"""

    protected: set[int] = set()
    ordered_lines = sorted(
        (line for line in lines if line.angle == 0),
        key=lambda line: (line.bbox[1], line.bbox[0], line.source_index),
    )
    for seed_index, seed in enumerate(ordered_lines):
        if not _FIGURE_CAPTION_LINE_RE.match(seed.text):
            continue
        matching_candidates = [
            candidate
            for candidate in candidates
            if _bbox_axis_overlap_ratio(
                seed.bbox,
                candidate.core_bbox,
                axis="x",
            )
            >= 0.35
            and candidate.core_bbox[3] - 2.5 * median_height
            <= _bbox_center_y(seed.bbox)
            <= candidate.core_bbox[3] + 2.5 * median_height
        ]
        if not matching_candidates:
            continue
        protected.add(seed.source_index)
        previous = seed
        for candidate_line in ordered_lines[seed_index + 1 :]:
            if candidate_line.bbox[1] - previous.bbox[3] > 0.75 * median_height:
                break
            if _bbox_center_y(candidate_line.bbox) <= _bbox_center_y(previous.bbox):
                continue
            if (
                abs(candidate_line.bbox[0] - seed.bbox[0]) > median_height
                or (
                    seed.font_signature is not None
                    and candidate_line.font_signature is not None
                    and seed.font_signature != candidate_line.font_signature
                )
            ):
                continue
            protected.add(candidate_line.source_index)
            previous = candidate_line
            if candidate_line.text.rstrip().endswith((".", "!", "?")):
                break
    return protected


def _graphic_body_tail_line_indices_to_preserve(
    lines: list[_LineItem],
    candidates: list[_GraphicCandidate],
    lanes: list[_TextLane],
    median_height: float,
) -> set[int]:
    """保护贴近图形上沿但延续上方满栏正文排版的短尾行。"""

    protected: set[int] = set()
    horizontal_lines = [line for line in lines if line.angle == 0]
    for tail in horizontal_lines:
        lane_index = _graphic_lane_index(tail.bbox, lanes)
        lane = lanes[lane_index]
        lane_width = max(0.1, lane.right - lane.left)
        tail_width = tail.bbox[2] - tail.bbox[0]
        if tail_width > 0.5 * lane_width:
            continue

        matching_candidates = [
            candidate
            for candidate in candidates
            if (candidate.lane_index < 0 or candidate.lane_index == lane_index)
            and tail.bbox[3] <= candidate.core_bbox[1] + 0.25 * median_height
            and _is_graphic_label_member(
                tail,
                candidate.core_bbox,
                median_height,
                margin_scale=candidate.label_margin_scale,
            )
        ]
        if not matching_candidates:
            continue

        tail_height = _line_effective_height(tail, tail.bbox)
        for previous in horizontal_lines:
            if previous.source_index == tail.source_index:
                continue
            if _graphic_lane_index(previous.bbox, lanes) != lane_index:
                continue
            vertical_gap = tail.bbox[1] - previous.bbox[3]
            if not -0.25 * median_height <= vertical_gap <= 0.75 * median_height:
                continue
            if abs(previous.bbox[0] - tail.bbox[0]) > 0.75 * median_height:
                continue

            previous_width = previous.bbox[2] - previous.bbox[0]
            if (
                previous_width < 0.75 * lane_width
                or lane.right - previous.bbox[2] > median_height
                or previous_width < 1.5 * tail_width
            ):
                continue
            previous_height = _line_effective_height(previous, previous.bbox)
            if max(previous_height, tail_height) > 1.25 * min(previous_height, tail_height):
                continue
            if (
                previous.font_signature is not None
                and tail.font_signature is not None
                and previous.font_signature != tail.font_signature
            ):
                continue
            if any(
                _is_graphic_label_member(
                    previous,
                    candidate.core_bbox,
                    median_height,
                    margin_scale=candidate.label_margin_scale,
                )
                for candidate in matching_candidates
            ):
                continue
            protected.add(tail.source_index)
            break
    return protected


def _detect_strong_graphic_bboxes(source: _PageSource) -> list[BBox]:
    """仅按复杂 Path、容器尺度与成对坐标轴识别高置信图形核心。"""

    if not source.path_infos:
        return []
    effective_heights = [
        _line_effective_height(line, line.bbox)
        for line in source.lines
        if line.angle == 0
    ]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0
    candidates = [
        *_detect_complex_path_containers(
            source.path_infos,
            source.page_size,
            median_height,
        ),
        *_detect_axis_path_graphics(
            source.path_infos,
            source.page_size,
            median_height,
        ),
        *_detect_complex_drawing_components(
            source.drawing_lines,
            source.path_infos,
            source.page_size,
            median_height,
        ),
    ]

    output: list[BBox] = []
    for bbox in sorted(candidates, key=_bbox_area, reverse=True):
        if any(_bbox_overlap_in_first(bbox, accepted) >= 0.9 for accepted in output):
            continue
        output.append(bbox)
    return sorted(output, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _detect_complex_path_containers(
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    median_height: float,
) -> list[BBox]:
    """筛选包含多个内部 Path 且至少含一个二维复杂轮廓的大容器。"""

    page_area = max(0.1, page_size[0] * page_size[1])
    output: list[BBox] = []
    for path_info in path_infos:
        bbox = path_info.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area_ratio = _bbox_area(bbox) / page_area
        if (
            path_info.form_depth != 0
            or not path_info.fill_visible
            or not 0.005 <= area_ratio <= 0.5
            or width < 4.0 * median_height
            or height < 4.0 * median_height
        ):
            continue
        inner_paths = [
            item
            for item in path_infos
            if item.source_index != path_info.source_index
            and _bbox_overlap_in_first(item.bbox, bbox) >= 0.9
            and _bbox_area(item.bbox) < 0.95 * _bbox_area(bbox)
        ]
        if len(inner_paths) < 4:
            continue
        if not any(_is_two_dimensional_complex_path(item, median_height) for item in inner_paths):
            continue
        output.append(bbox)
    return output


def _detect_axis_path_graphics(
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    median_height: float,
) -> list[BBox]:
    """用相交的长横纵轴和内部二维复杂路径补充无外框图表。"""

    thin_limit = max(1.0, 0.5 * median_height)
    minimum_axis_length = 6.0 * median_height
    horizontal_axes = [
        item
        for item in path_infos
        if item.form_depth == 0
        and item.stroke_visible
        and item.bbox[3] - item.bbox[1] <= thin_limit
        and item.bbox[2] - item.bbox[0] >= minimum_axis_length
    ]
    vertical_axes = [
        item
        for item in path_infos
        if item.form_depth == 0
        and item.stroke_visible
        and item.bbox[2] - item.bbox[0] <= thin_limit
        and item.bbox[3] - item.bbox[1] >= minimum_axis_length
    ]
    tolerance = max(2.0, median_height)
    output: list[BBox] = []
    for horizontal in horizontal_axes:
        horizontal_y = _bbox_center_y(horizontal.bbox)
        for vertical in vertical_axes:
            vertical_x = _bbox_center_x(vertical.bbox)
            touches_x = min(
                abs(vertical_x - horizontal.bbox[0]),
                abs(vertical_x - horizontal.bbox[2]),
            ) <= tolerance
            touches_y = min(
                abs(horizontal_y - vertical.bbox[1]),
                abs(horizontal_y - vertical.bbox[3]),
            ) <= tolerance
            if not (touches_x and touches_y):
                continue
            plot_bbox = _bbox_union(horizontal.bbox, vertical.bbox)
            width = plot_bbox[2] - plot_bbox[0]
            height = plot_bbox[3] - plot_bbox[1]
            if width > 0.65 * page_size[0] or height > 0.5 * page_size[1]:
                continue
            complex_paths = [
                item
                for item in path_infos
                if _is_two_dimensional_complex_path(item, median_height)
                and _bbox_overlap_in_smaller(item.bbox, plot_bbox) >= 0.2
            ]
            if not complex_paths:
                continue
            output.append(
                _bbox_union(
                    plot_bbox,
                    _bbox_union_many([item.bbox for item in complex_paths]),
                )
            )
    return output


def _is_two_dimensional_complex_path(
    path_info: PDFPathInfo,
    median_height: float,
) -> bool:
    """排除细轴线，只保留横纵均有尺寸且段数较多的图形轮廓。"""

    width = path_info.bbox[2] - path_info.bbox[0]
    height = path_info.bbox[3] - path_info.bbox[1]
    return (
        path_info.form_depth == 0
        and path_info.segment_count >= 6
        and width >= 1.5 * median_height
        and height >= 1.5 * median_height
    )


def _detect_complex_drawing_components(
    drawing_lines: list[_AxisLine],
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    median_height: float,
) -> list[BBox]:
    """以横纵绘图线组件和内部二维复杂 Path 识别坐标图或嵌入式图表。"""

    tolerance = max(2.0, 0.75 * median_height)
    output: list[BBox] = []
    for component in _connected_drawing_line_components(drawing_lines, tolerance):
        horizontal_count = sum(line.orientation == "horizontal" for line in component)
        vertical_count = len(component) - horizontal_count
        core_bbox = _bbox_union_many([line.bbox for line in component])
        width = core_bbox[2] - core_bbox[0]
        height = core_bbox[3] - core_bbox[1]
        if (
            len(component) < 4
            or horizontal_count < 2
            or vertical_count < 2
            or width < 4.0 * median_height
            or height < 3.0 * median_height
            or width > 0.65 * page_size[0]
            or height > 0.5 * page_size[1]
        ):
            continue
        complex_paths = [
            path_info
            for path_info in path_infos
            if _is_two_dimensional_complex_path(path_info, median_height)
            and _bbox_overlap_in_smaller(path_info.bbox, core_bbox) >= 0.2
        ]
        if not complex_paths:
            continue
        output.append(
            _bbox_union(
                core_bbox,
                _bbox_union_many([path_info.bbox for path_info in complex_paths]),
            )
        )
    return output


def _infer_graphic_text_lanes(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    median_height: float,
) -> list[_TextLane]:
    """用横排正文推断页内栏带，供不同角度的图形标签共享栏归属。"""

    line_geometry = [(line, line.bbox) for line in lines if line.angle == 0]
    if not line_geometry:
        return [_TextLane(left=0.0, right=page_size[0])]
    angle_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
    angle_median_height = statistics.median(angle_heights) if angle_heights else median_height
    lanes = [
        lane
        for lane in _infer_text_lanes(
            line_geometry,
            page_size[0],
            angle_median_height,
        )
        if not lane.is_span
    ]
    return lanes or [_TextLane(left=0.0, right=page_size[0])]


def _graphic_lane_index(bbox: BBox, lanes: list[_TextLane]) -> int:
    """按中心点、水平覆盖和距离为 bbox 选择唯一栏带。"""

    center_x = _bbox_center_x(bbox)
    best_index = 0
    best_score = (-1, -1.0, -math.inf)
    for lane_index, lane in enumerate(lanes):
        inside = int(lane.left <= center_x <= lane.right)
        overlap = max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left))
        if inside:
            distance = 0.0
        else:
            distance = min(abs(center_x - lane.left), abs(center_x - lane.right))
        score = (inside, overlap, -distance)
        if score > best_score:
            best_score = score
            best_index = lane_index
    return best_index


def _strong_graphic_lane_index(
    core_bbox: BBox,
    lanes: list[_TextLane],
    median_height: float,
) -> int:
    """仅把几乎完整落入唯一栏带的强图形核心绑定到该栏。"""

    core_width = max(0.1, core_bbox[2] - core_bbox[0])
    tolerance = max(1.0, median_height)
    matching_indices = []
    for lane_index, lane in enumerate(lanes):
        overlap = max(
            0.0,
            min(core_bbox[2], lane.right) - max(core_bbox[0], lane.left),
        )
        if (
            overlap / core_width >= 0.9
            and core_bbox[0] >= lane.left - tolerance
            and core_bbox[2] <= lane.right + tolerance
        ):
            matching_indices.append(lane_index)
    return matching_indices[0] if len(matching_indices) == 1 else -1


def _detect_graphic_candidates(
    drawing_lines: list[_AxisLine],
    page_size: tuple[float, float],
    median_height: float,
    lanes: list[_TextLane],
    table_bboxes: list[BBox],
) -> list[_GraphicCandidate]:
    """从非表格绘图线连通分量中筛选尺寸受限的图形容器。"""

    tolerance = max(2.0, 0.75 * median_height)
    candidates: list[_GraphicCandidate] = []
    for component in _connected_drawing_line_components(drawing_lines, tolerance):
        horizontal_count = sum(line.orientation == "horizontal" for line in component)
        vertical_count = len(component) - horizontal_count
        core_bbox = _bbox_union_many([line.bbox for line in component])
        width = core_bbox[2] - core_bbox[0]
        height = core_bbox[3] - core_bbox[1]
        if (
            len(component) < 4
            or horizontal_count < 2
            or vertical_count < 2
            or width < 4.0 * median_height
            or height < 3.0 * median_height
            or width > 0.5 * page_size[0]
            or height > 0.5 * page_size[1]
        ):
            continue
        if any(_bbox_overlap_in_smaller(core_bbox, table_bbox) >= 0.5 for table_bbox in table_bboxes):
            continue
        candidates.append(
            _GraphicCandidate(
                core_bbox=core_bbox,
                lane_index=_graphic_lane_index(core_bbox, lanes),
            )
        )
    return candidates


def _connected_drawing_line_components(
    drawing_lines: list[_AxisLine],
    tolerance: float,
) -> list[list[_AxisLine]]:
    """按 bbox 间距连接相邻绘图线，并返回互不重叠的连通分量。"""

    parents = list(range(len(drawing_lines)))

    def find(index: int) -> int:
        """查找绘图线连通分量的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并两个距离满足条件的绘图线分量。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(drawing_lines):
        for second_index in range(first_index + 1, len(drawing_lines)):
            if _bbox_distance(first.bbox, drawing_lines[second_index].bbox) <= tolerance:
                union(first_index, second_index)

    components: dict[int, list[_AxisLine]] = {}
    for line_index, line in enumerate(drawing_lines):
        components.setdefault(find(line_index), []).append(line)
    return list(components.values())


def _is_graphic_label_member(
    line: _LineItem,
    core_bbox: BBox,
    median_height: float,
    *,
    margin_scale: float = 2.5,
) -> bool:
    """判断短文本是否位于图形核心内部或对应轴向的邻近标签区。"""

    center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
    if _point_in_bbox(center, core_bbox):
        return True

    line_height = max(
        0.1,
        line.effective_height
        or min(
            max(0.1, line.bbox[2] - line.bbox[0]),
            max(0.1, line.bbox[3] - line.bbox[1]),
        ),
    )
    if line.angle in {90, 270}:
        primary_length = line.bbox[3] - line.bbox[1]
        core_primary_length = core_bbox[3] - core_bbox[1]
    else:
        primary_length = line.bbox[2] - line.bbox[0]
        core_primary_length = core_bbox[2] - core_bbox[0]

    horizontal_gap = max(core_bbox[0] - line.bbox[2], line.bbox[0] - core_bbox[2], 0.0)
    vertical_gap = max(core_bbox[1] - line.bbox[3], line.bbox[1] - core_bbox[3], 0.0)
    # 横排坐标轴标题允许比刻度标签略长，但必须与图宽、行高和上下间距同时相容。
    is_horizontal_axis_title = (
        line.angle in {0, 180}
        and primary_length <= 8.0 * line_height
        and primary_length <= 0.45 * core_primary_length
        and _bbox_axis_overlap_ratio(line.bbox, core_bbox, axis="x") >= 0.15
        and vertical_gap <= 2.5 * median_height
    )
    if is_horizontal_axis_title:
        return True
    if primary_length > min(5.0 * line_height, 0.5 * core_primary_length):
        return False

    if _bbox_axis_overlap_ratio(line.bbox, core_bbox, axis="x") >= 0.15:
        return vertical_gap <= margin_scale * median_height
    if _bbox_axis_overlap_ratio(line.bbox, core_bbox, axis="y") >= 0.15:
        horizontal_limit = max(
            margin_scale * median_height,
            0.2 * (core_bbox[2] - core_bbox[0]),
        )
        return horizontal_gap <= horizontal_limit
    corner_limit = min(margin_scale, 1.5) * median_height
    return (
        horizontal_gap <= corner_limit
        and vertical_gap <= corner_limit
        and math.hypot(horizontal_gap, vertical_gap) <= corner_limit
    )


def _image_members_to_content(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> str:
    """按视觉行和页内位置生成图片内部文本，保留不同视觉行之间的换行。"""

    row_groups: dict[tuple[int, int, int], list[_LineItem]] = {}
    for line in members:
        if line.visual_row_id is None:
            row_kind, row_identity = 1, line.source_index
        else:
            row_kind, row_identity = 0, line.visual_row_id
        row_groups.setdefault((line.angle, row_kind, row_identity), []).append(line)

    rows: list[tuple[BBox, str]] = []
    for row_lines in row_groups.values():
        row_bbox = _bbox_union_many([line.bbox for line in row_lines])
        angle = row_lines[0].angle
        local_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in row_lines
        ]
        content = _join_formula_visual_row(local_geometry, page_size)
        if content:
            rows.append((row_bbox, content))
    rows.sort(key=lambda item: (item[0][1], item[0][0]))
    return _sanitize_pdf_control_text(
        "\n".join(row_content for _row_bbox, row_content in rows),
        preserve_newlines=True,
    ).strip()


def _graphic_members_to_block(
    candidate: _GraphicCandidate,
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> dict[str, Any] | None:
    """生成含内部文本的矢量图 image block，并合并绘图核心与标签 bbox。"""

    content = _image_members_to_content(members, page_size)
    if not content:
        return None
    return {
        "type": "image",
        "bbox": _bbox_union(candidate.core_bbox, _bbox_union_many([line.bbox for line in members])),
        "angle": 0,
        "content": content,
    }


def _inline_raster_gap_member(
    source: _PageSource,
    left_bbox: BBox,
    right_bbox: BBox,
    claimed_line_indices: set[int],
    median_height: float,
) -> _LineItem | None:
    """查找恰好填充两张同行图片间隙的唯一拆分文本 run。"""

    left_height = max(0.1, left_bbox[3] - left_bbox[1])
    right_height = max(0.1, right_bbox[3] - right_bbox[1])
    vertical_overlap = max(
        0.0,
        min(left_bbox[3], right_bbox[3])
        - max(left_bbox[1], right_bbox[1]),
    )
    horizontal_gap = right_bbox[0] - left_bbox[2]
    if (
        max(left_height, right_height) / min(left_height, right_height) > 1.25
        or vertical_overlap / min(left_height, right_height) < 0.8
        or not 0.0 <= horizontal_gap <= 1.5 * median_height
    ):
        return None

    edge_tolerance = max(0.5, 0.25 * median_height)
    band_top = max(left_bbox[1], right_bbox[1])
    band_bottom = min(left_bbox[3], right_bbox[3])
    gap_members = [
        line
        for line in source.lines
        if line.source_index not in claimed_line_indices
        and line.angle == 0
        and line.split_from_row
        and line.visual_row_id is not None
        and left_bbox[2] - edge_tolerance
        <= _bbox_center_x(line.bbox)
        <= right_bbox[0] + edge_tolerance
        and band_top - edge_tolerance
        <= _bbox_center_y(line.bbox)
        <= band_bottom + edge_tolerance
    ]
    if len(gap_members) != 1:
        return None
    member = gap_members[0]
    if (
        abs(member.bbox[0] - left_bbox[2]) > edge_tolerance
        or abs(member.bbox[2] - right_bbox[0]) > edge_tolerance
    ):
        return None
    return member


def _inline_raster_group_has_only_expected_text(
    source: _PageSource,
    image_bboxes: list[BBox],
    gap_members: list[_LineItem],
    group_bbox: BBox,
    claimed_line_indices: set[int],
) -> bool:
    """确认复合图片框内没有图片内部文本和间隔符之外的正文。"""

    gap_member_indices = {line.source_index for line in gap_members}
    for line in source.lines:
        if line.source_index in claimed_line_indices:
            continue
        center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
        if not _point_in_bbox(center, group_bbox):
            continue
        if line.source_index in gap_member_indices:
            continue
        if any(_point_in_bbox(center, image_bbox) for image_bbox in image_bboxes):
            continue
        return False
    return True


def _merge_inline_raster_image_candidates(
    source: _PageSource,
    candidate_bboxes: list[BBox],
    container_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> list[tuple[BBox, int | None]]:
    """把由同一视觉行间隔符连接的已准入图片合成为单一候选。"""

    if len(candidate_bboxes) < 3:
        return [(bbox, None) for bbox in candidate_bboxes]
    effective_heights = [
        _line_effective_height(line, line.bbox)
        for line in source.lines
        if line.source_index not in claimed_line_indices and line.angle == 0
    ]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0

    adjacency: dict[int, set[int]] = {
        index: set() for index in range(len(candidate_bboxes))
    }
    gap_members: dict[tuple[int, int], _LineItem] = {}
    for first_index, first_bbox in enumerate(candidate_bboxes):
        for second_index, second_bbox in enumerate(candidate_bboxes):
            if first_index == second_index or first_bbox[0] >= second_bbox[0]:
                continue
            member = _inline_raster_gap_member(
                source,
                first_bbox,
                second_bbox,
                claimed_line_indices,
                median_height,
            )
            if member is None:
                continue
            adjacency[first_index].add(second_index)
            adjacency[second_index].add(first_index)
            gap_members[(first_index, second_index)] = member

    components: list[list[int]] = []
    visited: set[int] = set()
    for start_index in range(len(candidate_bboxes)):
        if start_index in visited:
            continue
        component: list[int] = []
        pending = [start_index]
        while pending:
            current_index = pending.pop()
            if current_index in visited:
                continue
            visited.add(current_index)
            component.append(current_index)
            pending.extend(adjacency[current_index] - visited)
        components.append(component)

    merged_specs: list[tuple[BBox, int | None]] = []
    consumed_indices: set[int] = set()
    for component in components:
        if len(component) < 3:
            continue
        ordered_indices = sorted(
            component,
            key=lambda index: candidate_bboxes[index][0],
        )
        ordered_pairs = list(zip(ordered_indices, ordered_indices[1:]))
        if not all(pair in gap_members for pair in ordered_pairs):
            continue
        members = [gap_members[pair] for pair in ordered_pairs]
        visual_row_ids = {member.visual_row_id for member in members}
        if len(visual_row_ids) != 1 or None in visual_row_ids:
            continue
        image_bboxes = [candidate_bboxes[index] for index in ordered_indices]
        group_bbox = _bbox_union_many(
            [*image_bboxes, *[member.bbox for member in members]],
        )
        if any(
            _bbox_overlap_in_smaller(group_bbox, container_bbox)
            >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for container_bbox in container_bboxes
        ):
            continue
        if not _inline_raster_group_has_only_expected_text(
            source,
            image_bboxes,
            members,
            group_bbox,
            claimed_line_indices,
        ):
            continue
        merged_specs.append((group_bbox, next(iter(visual_row_ids))))
        consumed_indices.update(ordered_indices)

    merged_specs.extend(
        (bbox, None)
        for index, bbox in enumerate(candidate_bboxes)
        if index not in consumed_indices
    )
    merged_specs.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
    return merged_specs


def _image_bboxes_are_near_equal(first: BBox, second: BBox) -> bool:
    """用亚 point 边界容差识别同一图片框，避免签名与点阵来源重复输出。"""

    return all(
        abs(first_value - second_value) <= _SIGNATURE_IMAGE_BBOX_DEDUP_TOLERANCE
        for first_value, second_value in zip(first, second, strict=True)
    )


def _build_raster_image_blocks(
    source: _PageSource,
    container_blocks: list[dict[str, Any]],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """过滤点阵图并接纳签名框，避让高优先级容器后唯一认领内部文本。"""

    page_area = max(0.0, source.page_size[0]) * max(0.0, source.page_size[1])
    if page_area <= 0:
        return [], set()

    container_bboxes = [
        bbox
        for block in container_blocks
        if (bbox := _coerce_bbox(block.get("bbox"))) is not None
    ]
    signature_bboxes: list[BBox] = []
    for raw_bbox in source.signature_bboxes:
        bbox = _clip_bbox(_coerce_bbox(raw_bbox), source.page_size)
        if bbox is None:
            continue
        if any(
            _bbox_overlap_in_smaller(bbox, container_bbox)
            >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for container_bbox in container_bboxes
        ):
            continue
        if not any(
            _image_bboxes_are_near_equal(bbox, existing_bbox)
            for existing_bbox in signature_bboxes
        ):
            # 已由注释可见性和 /AP 严格确认的签名不再套用普通点阵图面积门槛。
            signature_bboxes.append(bbox)

    raster_bboxes: list[BBox] = []
    for raw_bbox in source.image_bboxes:
        bbox = _clip_bbox(_coerce_bbox(raw_bbox), source.page_size)
        if bbox is None or _bbox_area(bbox) / page_area < _MIN_RASTER_IMAGE_PAGE_AREA_RATIO:
            continue
        if any(
            _bbox_overlap_in_smaller(bbox, container_bbox) >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for container_bbox in container_bboxes
        ):
            continue
        if any(
            _image_bboxes_are_near_equal(bbox, signature_bbox)
            for signature_bbox in signature_bboxes
        ):
            continue
        raster_bboxes.append(bbox)
    if not raster_bboxes and not signature_bboxes:
        return [], set()

    candidate_specs = (
        _merge_inline_raster_image_candidates(
            source,
            raster_bboxes,
            container_bboxes,
            claimed_line_indices,
        )
        if raster_bboxes
        else []
    )
    candidate_specs.extend((bbox, None) for bbox in signature_bboxes)
    candidate_specs.sort(
        key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2])
    )
    candidate_bboxes = [bbox for bbox, _row_id in candidate_specs]

    members_by_candidate: list[list[_LineItem]] = [[] for _ in candidate_bboxes]
    claimed: set[int] = set()
    for line in source.lines:
        if line.source_index in claimed_line_indices:
            continue
        center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
        matching_indices = [
            candidate_index
            for candidate_index, bbox in enumerate(candidate_bboxes)
            if _point_in_bbox(center, bbox)
        ]
        if not matching_indices:
            continue
        # 重叠点阵图共享内部文本时归属最小容器，避免 content 重复。
        candidate_index = min(
            matching_indices,
            key=lambda index: (_bbox_area(candidate_bboxes[index]), index),
        )
        members_by_candidate[candidate_index].append(line)
        claimed.add(line.source_index)

    blocks: list[dict[str, Any]] = []
    for (bbox, visual_row_id), members in zip(
        candidate_specs,
        members_by_candidate,
        strict=True,
    ):
        block = {
            "type": "image",
            "bbox": bbox,
            "angle": 0,
            "content": _image_members_to_content(members, source.page_size),
        }
        if visual_row_id is not None:
            block["_inline_visual_row_id"] = visual_row_id
        blocks.append(block)
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed
