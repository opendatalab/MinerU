# Copyright (c) Opendatalab. All rights reserved.
"""识别编号、排版重置及跨页一致的结构标题。"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import replace

from .....types import BBox
from ..geometry import _bbox_axis_overlap_ratio, _bbox_center_x, _bbox_center_y, _bbox_union_many, _rotate_bbox_to_upright
from ..line_layout import (
    _effective_text_row_gap,
    _estimate_lane_gap,
    _font_signatures_share_family,
    _infer_text_lanes,
    _line_canonical_style_scale,
    _line_effective_height,
    _normalized_font_family,
    _title_fonts_compatible,
)
from ..models import (
    _DocumentBodyProfile,
    _DocumentTitleProfile,
    _LineItem,
    _PreparedPage,
    _TextLane,
)
from .body_profile import _line_uses_document_regular_font
from .common import (
    _NUMBERED_SECTION_TITLE_RE,
    _SECTION_NUMBER_ONLY_RE,
    _SECTION_TITLE_TERMINAL_RE,
    _UNNUMBERED_SECTION_HEADING_RE,
    _build_physical_title_gap_map,
    _line_inside_visual_container,
)
from .page_titles import _classify_page_titles


def _normalized_section_title_text(text: str) -> str:
    """规范全半角编号和空白，仅用于结构标题规则判断。"""

    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", text),
    ).strip()


def _is_plausible_section_number(
    number: str,
    label: str = "",
) -> bool:
    """排除年代、小数值和整句正文冒充的章节编号。"""

    parts = [int(part) for part in re.findall(r"\d+", number)]
    if not parts or any(part > 99 for part in parts):
        return False
    if len(parts) > 1 and parts[0] == 0:
        return False
    if len(parts) == 1 and parts[0] > 12:
        return False
    stripped_label = label.lstrip()
    if stripped_label and not stripped_label[0].isalpha():
        return False
    if stripped_label and stripped_label[0].isascii() and stripped_label[0].isalpha() and not stripped_label[0].isupper():
        return False
    if len(re.findall(r"\d+(?:\.\d+)?", label)) >= 2:
        return False
    return not any(char in label for char in ",，;；:：。!?！？")


def _section_title_has_body_followers(
    title_bbox: BBox,
    geometry: list[tuple[_LineItem, BBox]],
    body_height: float,
    local_page_width: float,
    *,
    minimum_count: int,
) -> bool:
    """检查紧随标题的同栏常规正文行，避免把页码和数值标成标题。"""

    followers = 0
    for line, bbox in sorted(
        geometry,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    ):
        if bbox[1] <= title_bbox[1] + 0.4 * body_height:
            continue
        if bbox[1] - title_bbox[3] > 8.0 * body_height:
            break
        line_height = _line_effective_height(line, bbox)
        horizontally_related = (
            _bbox_axis_overlap_ratio(title_bbox, bbox, axis="x") >= 0.15 or abs(bbox[0] - title_bbox[0]) <= 2.5 * body_height
        )
        if (
            line.semantic_type is None
            and horizontally_related
            and bbox[2] - bbox[0] >= 0.25 * local_page_width
            and 0.7 <= line_height / body_height <= 1.4
        ):
            followers += 1
            if followers >= minimum_count:
                return True
    return False


def _classify_explicit_section_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
) -> None:
    """以通用编号行和紧凑结构转折补齐正文同字号章节标题。"""

    if document_body_profile is None or document_body_profile.body_height <= 0:
        return
    body_height = document_body_profile.body_height
    for angle in sorted(
        {
            line.angle
            for line in lines
            if (line.semantic_type is None or line.explicit_section_title) and not line.title_suppressed
        }
    ):
        geometry = sorted(
            [
                (
                    line,
                    _rotate_bbox_to_upright(
                        line.bbox,
                        page_size,
                        angle,
                    ),
                )
                for line in lines
                if line.angle == angle and line.semantic_type is None and not line.title_suppressed
            ],
            key=lambda item: (
                item[1][1],
                item[1][0],
                item[0].source_index,
            ),
        )
        if not geometry:
            continue
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        local_containers = [
            _rotate_bbox_to_upright(
                bbox,
                page_size,
                angle,
            )
            for bbox in container_bboxes
        ]
        numbered_groups: list[list[tuple[_LineItem, BBox]]] = []
        grouped_sources: set[int] = set()
        for line, bbox in geometry:
            normalized = _normalized_section_title_text(line.text)
            numbered_match = _NUMBERED_SECTION_TITLE_RE.match(
                normalized,
            )
            if numbered_match is not None and _is_plausible_section_number(
                numbered_match.group("number"),
                numbered_match.group("label"),
            ):
                numbered_groups.append([(line, bbox)])
                grouped_sources.add(line.source_index)
                continue
            if _SECTION_NUMBER_ONLY_RE.match(normalized) is None or not _is_plausible_section_number(normalized):
                continue
            marker_height = _line_effective_height(line, bbox)
            companions = [
                (candidate, candidate_bbox)
                for candidate, candidate_bbox in geometry
                if candidate is not line
                and candidate.source_index not in grouped_sources
                and candidate_bbox[0] >= bbox[2]
                and candidate_bbox[0] - bbox[2] <= 4.0 * max(body_height, marker_height)
                and _bbox_axis_overlap_ratio(
                    bbox,
                    candidate_bbox,
                    axis="y",
                )
                >= 0.5
                and candidate_bbox[2] - candidate_bbox[0] <= 0.55 * local_page_width
                and _SECTION_TITLE_TERMINAL_RE.search(
                    _normalized_section_title_text(candidate.text),
                )
                is None
            ]
            if not companions:
                continue
            companion = min(
                companions,
                key=lambda item: (
                    item[1][0] - bbox[2],
                    item[1][1],
                ),
            )
            numbered_groups.append([(line, bbox), companion])
            grouped_sources.update({line.source_index, companion[0].source_index})

        for group in numbered_groups:
            title_bbox = _bbox_union_many(
                [bbox for _line, bbox in group],
            )
            group_line_ids = {id(line) for line, _bbox in group}
            preceding = [
                previous_bbox
                for previous_line, previous_bbox in geometry
                if id(previous_line) not in group_line_ids
                and previous_bbox[3] <= title_bbox[1]
                and (
                    _bbox_axis_overlap_ratio(
                        previous_bbox,
                        title_bbox,
                        axis="x",
                    )
                    >= 0.15
                    or abs(previous_bbox[0] - title_bbox[0]) <= 2.5 * body_height
                )
            ]
            gap_above = title_bbox[1] - max(previous_bbox[3] for previous_bbox in preceding) if preceding else body_height
            if (
                title_bbox[2] - title_bbox[0] > 0.7 * local_page_width
                or not 0.1 * local_page_height <= _bbox_center_y(title_bbox) <= 0.93 * local_page_height
                or any(
                    _bbox_axis_overlap_ratio(
                        title_bbox,
                        container_bbox,
                        axis="x",
                    )
                    >= 0.8
                    and _bbox_axis_overlap_ratio(
                        title_bbox,
                        container_bbox,
                        axis="y",
                    )
                    >= 0.8
                    for container_bbox in local_containers
                )
                or not _section_title_has_body_followers(
                    title_bbox,
                    geometry,
                    body_height,
                    local_page_width,
                    minimum_count=1,
                )
                or gap_above < 0.4 * body_height
            ):
                continue
            for line, _bbox in group:
                line.semantic_type = "paragraph_title"
                line.structural_title = True
                line.explicit_section_title = True

        for line, bbox in geometry:
            if line.semantic_type is not None:
                continue
            normalized = _normalized_section_title_text(line.text)
            canonical_heading = normalized.strip(
                "[]［］【】()（）",
            ).replace(" ", "")
            if (
                _UNNUMBERED_SECTION_HEADING_RE.fullmatch(
                    canonical_heading,
                )
                is None
                or not 2 <= len(normalized) <= 24
                or _SECTION_TITLE_TERMINAL_RE.search(normalized) is not None
                or any(char in normalized for char in "[]［］")
                or bbox[2] - bbox[0] > 0.3 * local_page_width
                or _bbox_center_y(bbox) < 0.35 * local_page_height
                or not 0.75 <= _line_effective_height(line, bbox) / body_height <= 1.4
                or any(
                    _bbox_axis_overlap_ratio(
                        bbox,
                        container_bbox,
                        axis="x",
                    )
                    >= 0.8
                    and _bbox_axis_overlap_ratio(
                        bbox,
                        container_bbox,
                        axis="y",
                    )
                    >= 0.8
                    for container_bbox in local_containers
                )
                or not _section_title_has_body_followers(
                    bbox,
                    geometry,
                    body_height,
                    local_page_width,
                    minimum_count=2,
                )
            ):
                continue
            preceding = [
                previous_bbox
                for previous_line, previous_bbox in geometry
                if previous_line is not line
                and previous_bbox[3] <= bbox[1]
                and (
                    _bbox_axis_overlap_ratio(
                        previous_bbox,
                        bbox,
                        axis="x",
                    )
                    >= 0.15
                    or abs(previous_bbox[0] - bbox[0]) <= 2.5 * body_height
                )
            ]
            gap_above = bbox[1] - max(previous_bbox[3] for previous_bbox in preceding) if preceding else body_height
            if gap_above >= 0.5 * body_height:
                line.semantic_type = "paragraph_title"
                line.structural_title = True
                line.explicit_section_title = True


def _classify_document_structural_titles(
    prepared_pages: list[_PreparedPage],
    document_body_profile: _DocumentBodyProfile | None,
    *,
    legacy_body_profile: _DocumentBodyProfile | None,
    document_title_profile: _DocumentTitleProfile | None,
) -> None:
    """用跨页稳定栏带和段前后转折补齐正文同字号标题。"""

    if document_body_profile is None or not document_body_profile.has_style_scale_repairs:
        return
    probe_pages = [
        replace(
            prepared,
            remaining_lines=[
                replace(
                    line,
                    style_scale_repaired=True,
                    structural_title=False,
                )
                for line in prepared.remaining_lines
            ],
        )
        for prepared in prepared_pages
    ]
    _classify_document_structural_title_candidates(
        probe_pages,
        document_body_profile,
    )
    canonical_candidate_sources = {
        (page_index, line.source_index)
        for page_index, prepared in enumerate(probe_pages)
        for line in prepared.remaining_lines
        if line.structural_title
    }
    legacy_title_sources = _collect_legacy_paragraph_title_sources(
        prepared_pages,
        legacy_body_profile,
        document_title_profile,
    )
    body_height = max(0.1, document_body_profile.body_height)
    canonical_style_candidate_pages: dict[
        tuple[str, int, float, int],
        list[int],
    ] = {}
    for page_index, prepared in enumerate(prepared_pages):
        for line in prepared.remaining_lines:
            line_key = (page_index, line.source_index)
            if line_key not in canonical_candidate_sources:
                continue
            local_bbox = _rotate_bbox_to_upright(
                line.source_bbox or line.bbox,
                prepared.page_size,
                line.angle,
            )
            layout_ratio = (local_bbox[3] - local_bbox[1]) / body_height
            if (
                line_key in legacy_title_sources
                or layout_ratio >= 1.8
                or not _line_uses_document_regular_font(
                    line,
                    document_body_profile,
                )
            ):
                line.semantic_type = "paragraph_title"
                line.structural_title = True
                if (
                    line_key not in legacy_title_sources
                    and layout_ratio >= 1.8
                    and (
                        style_key := _canonical_title_style_key(
                            line,
                        )
                    )
                    is not None
                ):
                    canonical_style_candidate_pages.setdefault(
                        style_key,
                        [],
                    ).append(page_index)
    canonical_style_prototypes = {
        style_key
        for style_key, page_indices in canonical_style_candidate_pages.items()
        if len(set(page_indices)) >= 2 or max(Counter(page_indices).values(), default=0) >= 3
    }
    if canonical_style_prototypes:
        for prepared in prepared_pages:
            prepared.canonical_formula_geometry = True
            for line in prepared.remaining_lines:
                style_key = _canonical_title_style_key(line)
                if style_key in canonical_style_prototypes:
                    line.style_scale_repaired = True


def _promote_noninitial_document_title_band(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    page_index: int,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
    title_candidate_source_indices: set[int],
) -> None:
    """把非首页中已确认且显著大于正文的最强段落标题带升为文档标题。"""

    if page_index == 0 or document_body_profile is None:
        return
    body_height = max(0.1, document_body_profile.body_height)
    page_candidates: list[
        tuple[
            tuple[float, float, int, float],
            list[tuple[_LineItem, BBox]],
        ]
    ] = []
    for angle in sorted(
        {
            line.angle
            for line in lines
            if line.source_index in title_candidate_source_indices
            and line.semantic_type in {None, "paragraph_title"}
            and not line.title_suppressed
        }
    ):
        geometry = sorted(
            [
                (
                    line,
                    _rotate_bbox_to_upright(
                        line.bbox,
                        page_size,
                        angle,
                    ),
                )
                for line in lines
                if line.angle == angle
                and line.source_index in title_candidate_source_indices
                and line.semantic_type in {None, "paragraph_title"}
                and not line.title_suppressed
            ],
            key=lambda item: (
                item[1][1],
                item[1][0],
                item[0].source_index,
            ),
        )
        if not geometry:
            continue
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        local_containers = [
            _rotate_bbox_to_upright(
                bbox,
                page_size,
                angle,
            )
            for bbox in container_bboxes
        ]
        for index, (line, bbox) in enumerate(geometry):
            line_scale = _line_effective_height(line, bbox)
            width_ratio = (bbox[2] - bbox[0]) / max(
                0.1,
                local_page_width,
            )
            centered = abs(_bbox_center_x(bbox) - 0.5 * local_page_width) <= 0.08 * local_page_width
            if (
                line_scale < 1.25 * body_height
                or not 0.45 <= width_ratio <= 0.85
                or not centered
                or not 0.12 * local_page_height <= _bbox_center_y(bbox) <= 0.75 * local_page_height
                or _line_inside_visual_container(
                    bbox,
                    local_containers,
                )
            ):
                continue

            title_members = [(line, bbox)]
            cursor = index + 1
            while cursor < len(geometry):
                candidate_line, candidate_bbox = geometry[cursor]
                candidate_scale = _line_effective_height(
                    candidate_line,
                    candidate_bbox,
                )
                vertical_gap = max(
                    0.0,
                    candidate_bbox[1] - title_members[-1][1][3],
                )
                if (
                    candidate_scale < 1.2 * body_height
                    or vertical_gap
                    > 1.5
                    * max(
                        line_scale,
                        candidate_scale,
                    )
                    or abs(_bbox_center_x(candidate_bbox) - 0.5 * local_page_width) > 0.1 * local_page_width
                    or candidate_bbox[2] - candidate_bbox[0] > 0.85 * local_page_width
                    or _line_inside_visual_container(
                        candidate_bbox,
                        local_containers,
                    )
                    or not _title_fonts_compatible(
                        title_members[-1][0],
                        candidate_line,
                    )
                ):
                    break
                title_members.append(
                    (candidate_line, candidate_bbox),
                )
                cursor += 1

            title_scales = [_line_effective_height(title_line, title_bbox) for title_line, title_bbox in title_members]
            title_bbox = _bbox_union_many(
                [member_bbox for _member_line, member_bbox in title_members],
            )
            page_candidates.append(
                (
                    (
                        statistics.median(title_scales) / body_height,
                        sum(member_bbox[2] - member_bbox[0] for _member_line, member_bbox in title_members) / local_page_width,
                        len(title_members),
                        -_bbox_center_y(title_bbox) / local_page_height,
                    ),
                    title_members,
                )
            )

    if not page_candidates:
        return
    _score, title_members = max(
        page_candidates,
        key=lambda item: item[0],
    )
    for title_line, _title_bbox in title_members:
        title_line.semantic_type = "doc_title"


def _canonical_title_style_key(
    line: _LineItem,
) -> tuple[str, int, float, int] | None:
    """返回 canonical-only 标题向同样式正文传播时使用的稳定键。"""

    if line.font_signature is None or line.em_height <= 0:
        return None
    font_family = _normalized_font_family(line.font_signature)
    if font_family is None:
        return None
    return (
        font_family,
        line.font_signature[1],
        round(line.em_height * 4.0) / 4.0,
        line.angle,
    )


def _classify_document_structural_title_candidates(
    prepared_pages: list[_PreparedPage],
    document_body_profile: _DocumentBodyProfile,
) -> None:
    """在 canonical 行副本上收集所有满足结构转折的标题候选。"""

    body_height = max(0.1, document_body_profile.body_height)
    strong_candidates: list[tuple[int, _LineItem, tuple[str, int] | None, float]] = []
    start_candidates: list[tuple[int, _LineItem, tuple[str, int] | None, float]] = []
    accepted_anchor_positions: list[tuple[int, int, float, float]] = []
    for page_index, prepared in enumerate(prepared_pages):
        container_bboxes = [
            block["bbox"] for block in prepared.fixed_blocks if not isinstance(block.get("_inline_visual_row_id"), int)
        ]
        for angle in sorted({line.angle for line in prepared.remaining_lines if line.semantic_type is None}):
            geometry = sorted(
                [
                    (
                        line,
                        _rotate_bbox_to_upright(
                            line.source_bbox or line.bbox,
                            prepared.page_size,
                            angle,
                        ),
                    )
                    for line in prepared.remaining_lines
                    if line.angle == angle and line.semantic_type is None
                ],
                key=lambda item: (
                    item[1][1],
                    item[1][0],
                    item[0].source_index,
                ),
            )
            if len(geometry) < 4:
                continue
            local_page_width = prepared.page_size[1] if angle in {90, 270} else prepared.page_size[0]
            local_page_height = prepared.page_size[0] if angle in {90, 270} else prepared.page_size[1]
            median_height = statistics.median(_line_canonical_style_scale(line, bbox) for line, bbox in geometry)
            lanes = _infer_text_lanes(
                geometry,
                local_page_width,
                median_height,
            )
            physical_gaps = _build_physical_title_gap_map(geometry)
            local_containers = [
                _rotate_bbox_to_upright(
                    bbox,
                    prepared.page_size,
                    angle,
                )
                for bbox in container_bboxes
            ]
            for line, bbox in geometry:
                if page_index == 0 and _bbox_center_y(bbox) < 0.64 * local_page_height:
                    continue
                related_lanes = [
                    lane
                    for lane in lanes
                    if not lane.is_span
                    and len(lane.lines) >= 3
                    and lane.left - body_height <= _bbox_center_x(bbox) <= lane.right + body_height
                ]
                if not related_lanes:
                    continue
                lane = max(
                    related_lanes,
                    key=lambda item: (
                        len(item.lines),
                        item.right - item.left,
                    ),
                )
                lane_width = max(0.1, lane.right - lane.left)
                width_ratio = (bbox[2] - bbox[0]) / lane_width
                style_ratio = _line_canonical_style_scale(line, bbox) / body_height
                left_offset = (bbox[0] - lane.left) / body_height
                regular_font = _line_uses_document_regular_font(
                    line,
                    document_body_profile,
                )
                if (
                    not 0.75 <= style_ratio <= 1.35
                    or width_ratio > 0.8
                    or left_offset > 0.75
                    or left_offset < (-0.75 if regular_font else -3.0)
                    or _line_inside_visual_container(
                        bbox,
                        local_containers,
                    )
                ):
                    continue
                followers = [
                    (other_line, other_bbox)
                    for other_line, other_bbox in geometry
                    if other_line is not line
                    and bbox[1] < other_bbox[1]
                    and other_bbox[1] - bbox[3] <= 3.0 * body_height
                    and lane.left - body_height <= other_bbox[0] <= lane.left + 3.5 * body_height
                    and other_bbox[2] - other_bbox[0] >= 0.4 * lane_width
                ]
                if not followers:
                    continue
                gap_above, gap_below = physical_gaps.get(
                    line.source_index,
                    (None, None),
                )
                layout_ratio = (bbox[3] - bbox[1]) / body_height
                if regular_font and layout_ratio < 1.3 and line.font_coverage < 0.75:
                    continue
                standard_transition = (
                    gap_above is not None
                    and gap_below is not None
                    and gap_above >= 0.65 * body_height
                    and gap_below >= 0.35 * body_height
                    and (not regular_font or gap_below <= 1.5 * body_height)
                )
                low_coverage_wide_transition = (
                    gap_above is not None
                    and gap_below is not None
                    and gap_above >= 0.5 * body_height
                    and gap_below >= 0.45 * body_height
                    and gap_below <= 0.7 * body_height
                    and width_ratio >= 0.75
                    and line.font_coverage <= 0.7
                    and layout_ratio >= 1.8
                )
                if low_coverage_wide_transition and any(
                    anchor_page_index == page_index
                    and anchor_angle == angle
                    and abs(anchor_left - lane.left) <= body_height
                    and 0 < _bbox_center_y(bbox) - anchor_center_y <= 12.0 * body_height
                    for (
                        anchor_page_index,
                        anchor_angle,
                        anchor_left,
                        anchor_center_y,
                    ) in accepted_anchor_positions
                ):
                    low_coverage_wide_transition = False
                compact_regular_transition = (
                    gap_above is not None
                    and gap_below is not None
                    and gap_above >= 0.65 * body_height
                    and gap_below >= 0.15 * body_height
                    and width_ratio <= 0.45
                    and line.font_coverage >= 0.75
                    and layout_ratio <= 1.3
                )
                family_key = (
                    (
                        _normalized_font_family(line.font_signature),
                        line.font_signature[1],
                    )
                    if line.font_signature is not None
                    else None
                )
                if (
                    standard_transition
                    or low_coverage_wide_transition
                    or compact_regular_transition
                    or (gap_above is None and gap_below is not None and gap_below >= 0.6 * body_height and not regular_font)
                ):
                    strong_candidates.append(
                        (
                            page_index,
                            line,
                            family_key,
                            layout_ratio,
                        ),
                    )
                    accepted_anchor_positions.append(
                        (
                            page_index,
                            angle,
                            lane.left,
                            _bbox_center_y(bbox),
                        )
                    )
                    continue
                if (
                    gap_above is None
                    and bbox[1] <= 0.18 * local_page_height
                    and width_ratio <= 0.6
                    and (line.font_coverage >= 0.75 or (not regular_font and line.font_coverage >= 0.5))
                ):
                    start_candidates.append(
                        (
                            page_index,
                            line,
                            family_key,
                            layout_ratio,
                        ),
                    )

    for _page_index, line, _family_key, _layout_ratio in strong_candidates:
        line.semantic_type = "paragraph_title"
        line.structural_title = True
    strong_families_by_page = {
        (page_index, family_key) for page_index, _line, family_key, _layout_ratio in strong_candidates if family_key is not None
    }
    for page_index, line, family_key, _layout_ratio in start_candidates:
        if not _line_uses_document_regular_font(
            line,
            document_body_profile,
        ) or (family_key is not None and (page_index, family_key) in strong_families_by_page):
            line.semantic_type = "paragraph_title"
            line.structural_title = True


def _collect_legacy_paragraph_title_sources(
    prepared_pages: list[_PreparedPage],
    document_body_profile: _DocumentBodyProfile | None,
    document_title_profile: _DocumentTitleProfile | None,
) -> set[tuple[int, int]]:
    """在行副本上使用 legacy 尺度收集原本成立的段落标题身份。"""

    if document_body_profile is None:
        return set()
    legacy_profile = replace(
        document_body_profile,
        has_style_scale_repairs=False,
    )
    output: set[tuple[int, int]] = set()
    for page_index, prepared in enumerate(prepared_pages):
        probe_lines = [
            replace(
                line,
                style_scale_repaired=False,
                structural_title=False,
            )
            for line in prepared.remaining_lines
        ]
        container_bboxes = [
            block["bbox"] for block in prepared.fixed_blocks if not isinstance(block.get("_inline_visual_row_id"), int)
        ]
        caption_container_bboxes = [block["bbox"] for block in prepared.fixed_blocks if block.get("type") in {"image", "code"}]
        _classify_page_titles(
            probe_lines,
            prepared.page_size,
            page_index=page_index,
            container_bboxes=container_bboxes,
            caption_container_bboxes=caption_container_bboxes,
            document_body_profile=legacy_profile,
            document_title_profile=document_title_profile,
        )
        output.update((page_index, line.source_index) for line in probe_lines if line.semantic_type == "paragraph_title")
    return output


def _classify_inline_typography_reset_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
) -> None:
    """用短段尾、字体切换和缩进正文识别行内结构标题。"""

    if document_body_profile is None:
        return
    for angle in sorted({line.angle for line in lines if line.semantic_type is None and not line.title_suppressed}):
        line_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.semantic_type is None and not line.title_suppressed
        ]
        if len(line_geometry) < 3:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(
            line_geometry,
            local_page_width,
            median_height,
        )
        local_container_bboxes = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in container_bboxes]
        for lane in lanes:
            rows = sorted(
                lane.lines,
                key=lambda item: (
                    item[1][1],
                    item[1][0],
                    item[0].source_index,
                ),
            )
            lane_width = max(0.1, lane.right - lane.left)
            for previous, current, following in zip(
                rows,
                rows[1:],
                rows[2:],
            ):
                previous_line, previous_bbox = previous
                current_line, current_bbox = current
                following_line, following_bbox = following
                if any(
                    line.semantic_type is not None
                    for line in (
                        previous_line,
                        current_line,
                        following_line,
                    )
                ):
                    continue
                if (
                    previous_line.font_signature is None
                    or current_line.font_signature is None
                    or following_line.font_signature is None
                    or previous_line.font_coverage < 0.65
                    or current_line.font_coverage < 0.65
                    or following_line.font_coverage < 0.65
                ):
                    continue
                if not _font_signatures_share_family(
                    previous_line.font_signature,
                    following_line.font_signature,
                ) or _font_signatures_share_family(
                    current_line.font_signature,
                    previous_line.font_signature,
                ):
                    continue
                previous_height = _line_effective_height(*previous)
                current_height = _line_effective_height(*current)
                following_height = _line_effective_height(*following)
                neighbor_height = statistics.median(
                    (previous_height, following_height),
                )
                pair_height = max(
                    previous_height,
                    current_height,
                    following_height,
                )
                previous_width = previous_bbox[2] - previous_bbox[0]
                current_width = current_bbox[2] - current_bbox[0]
                following_width = following_bbox[2] - following_bbox[0]
                following_indent = following_bbox[0] - current_bbox[0]
                if not (
                    previous_width <= 0.45 * lane_width
                    and 0.2 * lane_width <= current_width <= 0.65 * lane_width
                    and following_width >= 0.75 * lane_width
                    and abs(current_bbox[0] - lane.left) <= 0.75 * pair_height
                    and 0.75 * pair_height <= following_indent <= 3.0 * pair_height
                    and 0.85 <= current_height / max(0.1, neighbor_height) <= 1.15
                    and -0.25 * pair_height <= _effective_text_row_gap(previous, current) <= 0.5 * pair_height
                    and -0.25 * pair_height <= _effective_text_row_gap(current, following) <= 0.5 * pair_height
                    and not _line_inside_visual_container(
                        current_bbox,
                        local_container_bboxes,
                    )
                ):
                    continue
                current_line.semantic_type = "paragraph_title"
                current_line.structural_title = True


def _classify_body_height_section_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
    page_index: int = 1,
) -> None:
    """用重复的短行加正文组结构识别与正文同字号的独立章节标题。"""

    if document_body_profile is None:
        return
    body_height = document_body_profile.body_height
    if body_height <= 0:
        return

    for angle in sorted({line.angle for line in lines if line.semantic_type is None and not line.title_suppressed}):
        line_geometry = sorted(
            [
                (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
                for line in lines
                if line.angle == angle and line.semantic_type is None and not line.title_suppressed
            ],
            key=lambda item: (item[1][1], item[1][0], item[0].source_index),
        )
        if len(line_geometry) < 8:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        lanes = _infer_text_lanes(
            line_geometry,
            local_page_width,
            median_height,
        )
        lane_by_source: dict[int, _TextLane] = {}
        regular_gaps: list[float] = []
        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            regular_gap, _gap_mad = _estimate_lane_gap(lane)
            regular_gaps.append(regular_gap)
            for line, _bbox in lane.lines:
                lane_by_source[line.source_index] = lane
        if not lane_by_source:
            continue

        page_regular_gap = statistics.median(regular_gaps) if regular_gaps else 0.2 * body_height
        physical_gaps = _build_physical_title_gap_map(line_geometry)
        local_container_bboxes = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in container_bboxes]
        candidates: list[tuple[_LineItem, BBox, _TextLane]] = []
        for line, bbox in line_geometry:
            lane = lane_by_source.get(line.source_index)
            if lane is None:
                continue
            line_height = _line_effective_height(line, bbox)
            lane_width = max(0.1, lane.right - lane.left)
            if not 0.9 <= line_height / body_height <= 1.1:
                continue
            if bbox[2] - bbox[0] > 0.22 * lane_width:
                continue
            if abs(bbox[0] - lane.left) > 0.75 * body_height:
                continue
            if _line_inside_visual_container(bbox, local_container_bboxes):
                continue

            followers = _body_height_section_followers(
                line,
                bbox,
                line_geometry,
                lane_by_source,
                body_height,
            )
            if len(followers) < 3:
                continue
            gap_above = physical_gaps.get(line.source_index, (None, None))[0]
            if gap_above is not None and gap_above <= 0.25 * body_height:
                continue
            starts_body = gap_above is None and bbox[1] <= 0.2 * local_page_height
            has_extra_gap = gap_above is not None and gap_above - page_regular_gap >= 0.75 * body_height
            has_full_width_follower = any(
                follower_bbox[2] - follower_bbox[0]
                >= 0.75
                * max(
                    0.1,
                    lane_by_source[follower_line.source_index].right - lane_by_source[follower_line.source_index].left,
                )
                for follower_line, follower_bbox in followers
                if follower_line.source_index in lane_by_source
            )
            if starts_body or has_extra_gap or has_full_width_follower:
                candidates.append((line, bbox, lane))

        for line, bbox, _lane in candidates:
            compatible_count = sum(
                1
                for peer_line, peer_bbox, _peer_lane in candidates
                if abs(peer_bbox[0] - bbox[0]) <= body_height
                and 0.9 <= _line_effective_height(peer_line, peer_bbox) / _line_effective_height(line, bbox) <= 1.1
                and _title_fonts_compatible(line, peer_line)
            )
            if compatible_count >= 2:
                # 只标记结构锚点本身，避免普通正文被标题邻行扩展再次吞入。
                line.semantic_type = "paragraph_title"


def _body_height_section_followers(
    candidate_line: _LineItem,
    candidate_bbox: BBox,
    line_geometry: list[tuple[_LineItem, BBox]],
    lane_by_source: dict[int, _TextLane],
    body_height: float,
) -> list[tuple[_LineItem, BBox]]:
    """返回短标题后方同锚点、同正文尺度且行距稳定的前三行。"""

    followers: list[tuple[_LineItem, BBox]] = []
    previous_top = candidate_bbox[1]
    for line, bbox in line_geometry:
        if line is candidate_line or bbox[1] <= candidate_bbox[1] + 0.4 * body_height:
            continue
        if not (candidate_bbox[0] - 0.75 * body_height <= bbox[0] <= candidate_bbox[0] + 1.5 * body_height):
            continue
        top_pitch = bbox[1] - previous_top
        if top_pitch < 0.5 * body_height:
            continue
        if top_pitch > 1.8 * body_height:
            break
        if not 0.9 <= _line_effective_height(line, bbox) / body_height <= 1.1:
            break
        if line.source_index not in lane_by_source:
            break
        followers.append((line, bbox))
        previous_top = bbox[1]
        if len(followers) == 3:
            break
    return followers


__all__ = [
    "_normalized_section_title_text",
    "_is_plausible_section_number",
    "_section_title_has_body_followers",
    "_classify_explicit_section_titles",
    "_classify_document_structural_titles",
    "_promote_noninitial_document_title_band",
    "_canonical_title_style_key",
    "_classify_document_structural_title_candidates",
    "_collect_legacy_paragraph_title_sources",
    "_classify_inline_typography_reset_titles",
    "_classify_body_height_section_titles",
    "_body_height_section_followers",
]
