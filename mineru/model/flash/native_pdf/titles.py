# Copyright (c) Opendatalab. All rights reserved.

"""基于排版差异分类文档标题和段落标题。"""

from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Literal


from mineru.types import BBox
from mineru.utils.pdf_text_styles import (
    PDF_FONT_FORCE_BOLD_FLAG,
    PDF_FONT_ITALIC_FLAG,
)

from .models import (
    _DocumentBodyProfile,
    _DocumentTitleProfile,
    _LaneBodyProfile,
    _LineItem,
    _PreparedPage,
    _TextLane,
    _TitleStylePrototype,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_union_many,
    _rotate_bbox_to_upright,
)
from .line_layout import (
    _effective_text_row_gap,
    _estimate_lane_gap,
    _font_signatures_share_family,
    _font_weights_conflict,
    _infer_text_lanes,
    _line_effective_height,
    _normalized_font_family,
    _should_connect_semantic_rows,
    _title_fonts_compatible,
)

def _classify_page_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    page_index: int,
    container_bboxes: list[BBox],
    caption_container_bboxes: list[BBox] | None = None,
    document_body_profile: _DocumentBodyProfile | None = None,
    document_title_profile: _DocumentTitleProfile | None = None,
) -> None:
    """只用页面几何与字体排版标注首页文档标题和各页段落标题。"""

    for angle in sorted({line.angle for line in lines if line.semantic_type is None}):
        line_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.semantic_type is None
        ]
        if not line_geometry:
            continue
        median_height = statistics.median(
            _line_effective_height(line, bbox) for line, bbox in line_geometry
        )
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        physical_gaps = _build_physical_title_gap_map(line_geometry)
        grid_title_suppressions = _find_repeated_grid_title_suppressions(
            lanes,
            median_height,
        )
        local_container_bboxes = [
            _rotate_bbox_to_upright(bbox, page_size, angle)
            for bbox in container_bboxes
        ]
        grid_title_suppressions.update(
            _find_container_visual_row_title_suppressions(
                line_geometry,
                local_container_bboxes,
                median_height,
            )
        )

        document_title_bottom: float | None = None
        if page_index == 0:
            document_title_bottom = _classify_document_title(
                lanes,
                local_page_height,
                local_page_width,
                document_body_profile=document_body_profile,
            )
            document_title_bottom = _expand_document_title_across_lanes(
                line_geometry,
                local_page_width,
                local_page_height,
                document_title_bottom,
            )
        preserve_front_matter_boundaries = _document_title_uses_page_fallback(
            lanes,
            document_body_profile=document_body_profile,
        )

        _classify_cross_lane_centered_section_titles(
            line_geometry,
            lanes,
            local_page_width,
            local_page_height,
            local_container_bboxes,
            document_title_bottom=document_title_bottom,
        )
        _classify_cross_lane_emphasized_section_titles(
            line_geometry,
            lanes,
            local_page_width,
            local_page_height,
            local_container_bboxes,
            document_title_bottom=document_title_bottom,
            document_body_profile=document_body_profile,
        )

        for lane in lanes:
            profile = _infer_lane_body_profile(lane)
            _classify_paragraph_titles_in_lane(
                lane,
                profile,
                local_page_width,
                local_page_height,
                local_container_bboxes,
                document_title_bottom=document_title_bottom,
                preserve_front_matter_boundaries=preserve_front_matter_boundaries,
                physical_gaps=physical_gaps,
                grid_title_suppressions=grid_title_suppressions,
                document_body_profile=document_body_profile,
                document_title_profile=document_title_profile,
                page_index=page_index,
            )
        _expand_cross_lane_paragraph_title_neighbors(
            line_geometry,
        )
        _demote_cross_lane_body_continuation_titles(
            line_geometry,
            lanes,
        )
        _demote_visual_container_caption_titles(
            line_geometry,
            [
                _rotate_bbox_to_upright(bbox, page_size, angle)
                for bbox in (caption_container_bboxes or [])
            ],
        )


def _infer_document_body_profile(
    prepared_pages: list[_PreparedPage],
) -> _DocumentBodyProfile | None:
    """按跨页覆盖和累计行宽推断全文正文行高及常规字体集合。"""

    samples: list[
        tuple[
            float,
            int,
            float,
            tuple[str, int] | None,
            float | None,
        ]
    ] = []
    for page_index, prepared in enumerate(prepared_pages):
        for line in prepared.remaining_lines:
            if line.semantic_type is not None:
                continue
            local_bbox = _rotate_bbox_to_upright(
                line.bbox,
                prepared.page_size,
                line.angle,
            )
            local_page_width = (
                prepared.page_size[1]
                if line.angle in {90, 270}
                else prepared.page_size[0]
            )
            height = _line_effective_height(line, local_bbox)
            normalized_width = max(0.0, local_bbox[2] - local_bbox[0]) / max(
                0.1,
                local_page_width,
            )
            if height <= 0 or normalized_width <= 0:
                continue
            samples.append(
                (
                    height,
                    page_index,
                    normalized_width,
                    line.font_signature if line.font_coverage >= 0.75 else None,
                    line.dominant_font_weight,
                )
            )
    if not samples:
        return None

    height_clusters: list[list[tuple[float, int, float]]] = []
    for height, page_index, normalized_width, _font, _weight in sorted(
        samples,
        key=lambda item: item[0],
    ):
        target = next(
            (
                cluster
                for cluster in height_clusters
                if abs(height - statistics.median(item[0] for item in cluster))
                <= 0.1 * statistics.median(item[0] for item in cluster)
            ),
            None,
        )
        if target is None:
            height_clusters.append([(height, page_index, normalized_width)])
        else:
            target.append((height, page_index, normalized_width))
    cross_page_clusters = [
        cluster
        for cluster in height_clusters
        if len({item[1] for item in cluster}) >= 2
    ]
    eligible_clusters = cross_page_clusters or height_clusters
    body_cluster = max(
        eligible_clusters,
        key=lambda cluster: (
            sum(item[2] for item in cluster),
            len({item[1] for item in cluster}),
            len(cluster),
        ),
    )
    body_height = statistics.median(item[0] for item in body_cluster)

    body_weights = [
        weight
        for height, _page_index, _width, _font, weight in samples
        if weight is not None and 0.9 <= height / body_height <= 1.1
    ]
    body_weight = statistics.median(body_weights) if body_weights else None

    font_pages: dict[tuple[str, int], set[int]] = {}
    font_widths: dict[tuple[str, int], float] = {}
    font_weights: dict[tuple[str, int], list[float]] = {}
    for height, page_index, width, font, weight in samples:
        if font is None or not 0.9 <= height / body_height <= 1.1:
            continue
        # 常规字体支持必须来自正文高度带；跨页重复的大标题不能反向污染正文画像。
        font_pages.setdefault(font, set()).add(page_index)
        font_widths[font] = font_widths.get(font, 0.0) + width
        if weight is not None:
            font_weights.setdefault(font, []).append(weight)

    regular_fonts = frozenset(
        font
        for font, pages in font_pages.items()
        if len(pages) >= 3
        and font_widths.get(font, 0.0) >= 2.0
        and font_widths.get(font, 0.0) >= 0.75 * len(pages)
        and _document_font_is_regular(
            font,
            font_weights.get(font, []),
            body_weight,
        )
    )
    return _DocumentBodyProfile(
        body_height=max(0.1, body_height),
        body_weight=body_weight,
        regular_fonts=regular_fonts,
    )


def _classify_body_height_section_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
) -> None:
    """用重复的短行加正文组结构识别与正文同字号的独立章节标题。"""

    if document_body_profile is None:
        return
    body_height = document_body_profile.body_height
    if body_height <= 0:
        return

    for angle in sorted({line.angle for line in lines if line.semantic_type is None}):
        line_geometry = sorted(
            [
                (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
                for line in lines
                if line.angle == angle and line.semantic_type is None
            ],
            key=lambda item: (item[1][1], item[1][0], item[0].source_index),
        )
        if len(line_geometry) < 8:
            continue
        median_height = statistics.median(
            _line_effective_height(line, bbox)
            for line, bbox in line_geometry
        )
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
            lane.lines.sort(
                key=lambda item: (item[1][1], item[1][0], item[0].source_index)
            )
            regular_gap, _gap_mad = _estimate_lane_gap(lane)
            regular_gaps.append(regular_gap)
            for line, _bbox in lane.lines:
                lane_by_source[line.source_index] = lane
        if not lane_by_source:
            continue

        page_regular_gap = (
            statistics.median(regular_gaps)
            if regular_gaps
            else 0.2 * body_height
        )
        physical_gaps = _build_physical_title_gap_map(line_geometry)
        local_container_bboxes = [
            _rotate_bbox_to_upright(bbox, page_size, angle)
            for bbox in container_bboxes
        ]
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
            starts_body = (
                gap_above is None
                and bbox[1] <= 0.2 * local_page_height
            )
            has_extra_gap = (
                gap_above is not None
                and gap_above - page_regular_gap >= 0.75 * body_height
            )
            has_full_width_follower = any(
                follower_bbox[2] - follower_bbox[0]
                >= 0.75
                * max(
                    0.1,
                    lane_by_source[follower_line.source_index].right
                    - lane_by_source[follower_line.source_index].left,
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
                and 0.9
                <= _line_effective_height(peer_line, peer_bbox)
                / _line_effective_height(line, bbox)
                <= 1.1
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
        if not (
            candidate_bbox[0] - 0.75 * body_height
            <= bbox[0]
            <= candidate_bbox[0] + 1.5 * body_height
        ):
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


def _infer_document_title_profile(
    prepared_pages: list[_PreparedPage],
    document_body_profile: _DocumentBodyProfile | None,
) -> _DocumentTitleProfile | None:
    """在副本上复用现有标题判定，并把跨页稳定样式聚成标题原型。"""

    if document_body_profile is None:
        return None
    seeds: list[
        tuple[
            str,
            int,
            float,
            float | None,
            Literal["left", "center"],
            float,
            int,
        ]
    ] = []
    for page_index, prepared in enumerate(prepared_pages):
        probe_lines = [replace(line) for line in prepared.remaining_lines]
        container_bboxes = [
            block["bbox"]
            for block in prepared.fixed_blocks
            if not isinstance(block.get("_inline_visual_row_id"), int)
        ]
        _classify_page_titles(
            probe_lines,
            prepared.page_size,
            page_index=page_index,
            container_bboxes=container_bboxes,
            document_body_profile=document_body_profile,
        )
        seen_rows: set[tuple[int, int]] = set()
        for angle in sorted({line.angle for line in probe_lines}):
            geometry = [
                (
                    line,
                    _rotate_bbox_to_upright(
                        line.bbox,
                        prepared.page_size,
                        angle,
                    ),
                )
                for line in probe_lines
                if line.angle == angle
            ]
            if not geometry:
                continue
            median_height = statistics.median(
                _line_effective_height(line, bbox)
                for line, bbox in geometry
            )
            local_page_width = (
                prepared.page_size[1]
                if angle in {90, 270}
                else prepared.page_size[0]
            )
            lanes = _infer_text_lanes(
                geometry,
                local_page_width,
                median_height,
            )
            local_container_bboxes = [
                _rotate_bbox_to_upright(
                    bbox,
                    prepared.page_size,
                    angle,
                )
                for bbox in container_bboxes
            ]
            for lane in lanes:
                for line, bbox in lane.lines:
                    line_height_ratio = (
                        _line_effective_height(line, bbox)
                        / document_body_profile.body_height
                    )
                    large_unresolved_seed = (
                        page_index > 0
                        and
                        line.semantic_type is None
                        and line_height_ratio >= 1.3
                        and bbox[2] - bbox[0] <= 0.8 * local_page_width
                        and not _line_inside_visual_container(
                            bbox,
                            local_container_bboxes,
                        )
                    )
                    if (
                        (
                            line.semantic_type != "paragraph_title"
                            and not large_unresolved_seed
                        )
                        or line.font_signature is None
                        or line.font_coverage < 0.75
                    ):
                        continue
                    row_identity = (
                        line.visual_row_id
                        if line.visual_row_id is not None
                        else line.source_index
                    )
                    row_key = (angle, row_identity)
                    if row_key in seen_rows:
                        continue
                    alignment = _title_profile_alignment(
                        bbox,
                        lane,
                        document_body_profile.body_height,
                    )
                    if alignment is None:
                        continue
                    font_family = _normalized_font_family(line.font_signature)
                    if font_family is None:
                        continue
                    mode, anchor_offset = alignment
                    seeds.append(
                        (
                            font_family,
                            line.font_signature[1],
                            line_height_ratio,
                            line.dominant_font_weight,
                            mode,
                            anchor_offset,
                            page_index,
                        )
                    )
                    seen_rows.add(row_key)

    clusters: list[list[tuple[str, int, float, float | None, Literal["left", "center"], float, int]]] = []
    for seed in sorted(seeds, key=lambda item: (item[0], item[1], item[4], item[2])):
        target = next(
            (
                cluster
                for cluster in clusters
                if _title_profile_seed_matches_cluster(seed, cluster)
            ),
            None,
        )
        if target is None:
            clusters.append([seed])
        else:
            target.append(seed)

    prototypes: list[_TitleStylePrototype] = []
    for cluster in clusters:
        support_pages = len({seed[6] for seed in cluster})
        if len(cluster) < 3 and support_pages < 2:
            continue
        weights = [seed[3] for seed in cluster if seed[3] is not None]
        prototypes.append(
            _TitleStylePrototype(
                font_family=cluster[0][0],
                font_flags=cluster[0][1],
                height_ratio=statistics.median(seed[2] for seed in cluster),
                weight=statistics.median(weights) if weights else None,
                alignment=cluster[0][4],
                anchor_offset=statistics.median(seed[5] for seed in cluster),
                support_count=len(cluster),
                support_pages=support_pages,
            )
        )
    if not prototypes:
        return None
    prototypes.sort(
        key=lambda item: (item.support_pages, item.support_count),
        reverse=True,
    )
    return _DocumentTitleProfile(tuple(prototypes))


def _title_profile_seed_matches_cluster(
    seed: tuple[
        str,
        int,
        float,
        float | None,
        Literal["left", "center"],
        float,
        int,
    ],
    cluster: list[
        tuple[
            str,
            int,
            float,
            float | None,
            Literal["left", "center"],
            float,
            int,
        ]
    ],
) -> bool:
    """判断标题种子是否与已有字体、尺度和对齐簇兼容。"""

    reference = cluster[0]
    median_ratio = statistics.median(item[2] for item in cluster)
    median_anchor = statistics.median(item[5] for item in cluster)
    weights = [item[3] for item in cluster if item[3] is not None]
    weight_compatible = (
        seed[3] is None
        or not weights
        or abs(seed[3] - statistics.median(weights)) < 100.0
        or max(seed[3], statistics.median(weights))
        < 1.15 * min(seed[3], statistics.median(weights))
    )
    return (
        seed[0] == reference[0]
        and seed[1] == reference[1]
        and seed[4] == reference[4]
        and abs(seed[2] - median_ratio) <= 0.1 * median_ratio
        and abs(seed[5] - median_anchor) <= 0.12
        and weight_compatible
    )


def _title_profile_alignment(
    bbox: BBox,
    lane: _TextLane,
    body_height: float,
) -> tuple[Literal["left", "center"], float] | None:
    """返回标题行的栏内对齐模式及归一化锚点偏移。"""

    lane_width = max(0.1, lane.right - lane.left)
    center_offset = (
        _bbox_center_x(bbox) - (lane.left + lane.right) / 2.0
    ) / lane_width
    left_offset = (bbox[0] - lane.left) / lane_width
    if abs(center_offset) <= 0.12 and bbox[2] - bbox[0] <= 0.85 * lane_width:
        return "center", center_offset
    if abs(bbox[0] - lane.left) <= 3.0 * body_height:
        return "left", left_offset
    return None


def _document_font_is_regular(
    font: tuple[str, int],
    weights: list[float],
    body_weight: float | None,
) -> bool:
    """用字体样式位和全文正文基准过滤斜体、粗体等强调字体。"""

    if font[1] & (PDF_FONT_ITALIC_FLAG | PDF_FONT_FORCE_BOLD_FLAG):
        return False
    if not weights or body_weight is None:
        return True
    median_weight = statistics.median(weights)
    return median_weight < max(body_weight + 100.0, 1.15 * body_weight)


def _build_physical_title_gap_map(
    line_geometry: list[tuple[_LineItem, BBox]],
) -> dict[int, tuple[float | None, float | None]]:
    """为每行记录同方向且水平投影相交的最近上、下物理行净空。"""

    output: dict[int, tuple[float | None, float | None]] = {}
    for line, bbox in line_geometry:
        line_center = _bbox_center_y(bbox)
        above_gaps: list[float] = []
        below_gaps: list[float] = []
        for other_line, other_bbox in line_geometry:
            if other_line is line:
                continue
            if _bbox_axis_overlap_ratio(bbox, other_bbox, axis="x") < 0.1:
                # 双栏同高度正文并非当前行的物理上下文，不能抹掉真实标题留白。
                continue
            if (
                line.visual_row_id is not None
                and other_line.visual_row_id == line.visual_row_id
            ):
                continue
            other_center = _bbox_center_y(other_bbox)
            if other_center < line_center:
                above_gaps.append(
                    max(
                        0.0,
                        _effective_text_row_gap(
                            (other_line, other_bbox),
                            (line, bbox),
                        ),
                    )
                )
            elif other_center > line_center:
                below_gaps.append(
                    max(
                        0.0,
                        _effective_text_row_gap(
                            (line, bbox),
                            (other_line, other_bbox),
                        ),
                    )
                )
        output[line.source_index] = (
            min(above_gaps) if above_gaps else None,
            min(below_gaps) if below_gaps else None,
        )
    return output


def _find_repeated_grid_title_suppressions(
    lanes: list[_TextLane],
    median_height: float,
) -> set[int]:
    """识别重复双栏信息网格中的短首行，避免把城市等记录头标成标题。"""

    candidates: list[tuple[int, int, float]] = []
    for lane_index, lane in enumerate(lanes):
        if lane.is_span:
            continue
        lane_width = max(0.1, lane.right - lane.left)
        for row_index, (line, bbox) in enumerate(lane.lines[:-1]):
            next_line, next_bbox = lane.lines[row_index + 1]
            width = bbox[2] - bbox[0]
            next_width = next_bbox[2] - next_bbox[0]
            gap = _effective_text_row_gap(
                (line, bbox),
                (next_line, next_bbox),
            )
            if (
                line.semantic_type is not None
                or next_line.semantic_type is not None
                or width > 0.35 * lane_width
                or next_width < max(0.6 * lane_width, 1.8 * width)
                or abs(next_bbox[0] - bbox[0]) > 0.75 * median_height
                or not -0.25 * median_height <= gap <= 0.75 * median_height
            ):
                continue
            candidates.append((lane_index, line.source_index, _bbox_center_y(bbox)))

    bands: list[list[tuple[int, int, float]]] = []
    for candidate in sorted(candidates, key=lambda item: item[2]):
        target = next(
            (
                band
                for band in bands
                if abs(candidate[2] - statistics.median(item[2] for item in band))
                <= 0.5 * median_height
            ),
            None,
        )
        if target is None:
            bands.append([candidate])
        else:
            target.append(candidate)
    paired_bands = [
        {source_index for _lane_index, source_index, _center_y in band}
        for band in bands
        if len({lane_index for lane_index, _source_index, _center_y in band}) >= 2
    ]
    if len(paired_bands) < 2:
        return set()
    return set().union(*paired_bands)


def _find_container_visual_row_title_suppressions(
    line_geometry: list[tuple[_LineItem, BBox]],
    container_bboxes: list[BBox],
    median_height: float,
) -> set[int]:
    """用完整视觉行与图表容器的邻接关系抑制拆分 caption 标题误报。"""

    visual_rows: dict[int, list[tuple[_LineItem, BBox]]] = {}
    for item in line_geometry:
        row_id = item[0].visual_row_id
        if row_id is not None:
            visual_rows.setdefault(row_id, []).append(item)
    suppressed: set[int] = set()
    for members in visual_rows.values():
        if len(members) < 2:
            continue
        row_bbox = _bbox_union_many([bbox for _line, bbox in members])
        if any(
            _bbox_axis_overlap_ratio(row_bbox, container_bbox, axis="x") >= 0.35
            and max(
                row_bbox[1] - container_bbox[3],
                container_bbox[1] - row_bbox[3],
                0.0,
            )
            <= 1.5 * median_height
            for container_bbox in container_bboxes
        ):
            suppressed.update(line.source_index for line, _bbox in members)
    return suppressed


def _infer_lane_body_profile(lane: _TextLane) -> _LaneBodyProfile:
    """从栏带的长行主体估计正文行高、主字体、字重、常规行距和样式占比。"""

    available = [item for item in lane.lines if item[0].semantic_type is None]
    if not available:
        return _LaneBodyProfile(1.0, None, None, 0.35, {})
    lane_width = max(0.1, lane.right - lane.left)
    long_lines = [item for item in available if item[1][2] - item[1][0] >= 0.45 * lane_width]
    body_rows = long_lines or available
    body_line_ids = {id(line) for line, _bbox in body_rows}
    body_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in body_rows)

    font_support: dict[tuple[str, int], float] = {}
    style_support: dict[tuple[str, int], float] = {}
    total_style_width = 0.0
    for line, bbox in available:
        if line.font_signature is None or line.font_coverage < 0.75:
            continue
        line_width = max(0.1, bbox[2] - bbox[0])
        style_support[line.font_signature] = style_support.get(line.font_signature, 0.0) + line_width
        total_style_width += line_width
        if id(line) in body_line_ids and 0.75 <= _line_effective_height(line, bbox) / body_height <= 1.35:
            font_support[line.font_signature] = font_support.get(line.font_signature, 0.0) + line_width
    body_font = max(font_support, key=font_support.get) if font_support else None
    if total_style_width > 0:
        style_support = {signature: width / total_style_width for signature, width in style_support.items()}

    body_weights = [
        line.dominant_font_weight
        for line, bbox in body_rows
        if line.dominant_font_weight is not None
        and (body_font is None or line.font_signature == body_font)
        and 0.75 <= _line_effective_height(line, bbox) / body_height <= 1.35
    ]
    regular_gap, _gap_mad = _estimate_lane_gap(lane)
    return _LaneBodyProfile(
        body_height=max(0.1, body_height),
        body_font=body_font,
        body_weight=statistics.median(body_weights) if body_weights else None,
        regular_gap=regular_gap,
        style_support=style_support,
        body_row_count=len(body_rows),
    )


def _classify_document_title(
    lanes: list[_TextLane],
    local_page_height: float,
    local_page_width: float,
    *,
    document_body_profile: _DocumentBodyProfile | None = None,
) -> float | None:
    """从首页上部选取显著大字号锚点，并用同版式邻行扩展多行文档标题。"""

    candidates: list[tuple[float, _TextLane, int, tuple[_LineItem, BBox]]] = []
    lane_profiles = [(lane, _infer_lane_body_profile(lane)) for lane in lanes]
    column_body_heights = [profile.body_height for lane, profile in lane_profiles if not lane.is_span]
    if document_body_profile is not None:
        document_body_height = document_body_profile.body_height
    else:
        document_body_height = (
            statistics.median(column_body_heights)
            if column_body_heights
            else min(
                (profile.body_height for _lane, profile in lane_profiles),
                default=1.0,
            )
        )
    for lane, profile in lane_profiles:
        lane_width = max(0.1, lane.right - lane.left)
        available = [item for item in lane.lines if item[0].semantic_type is None]
        for row_index, item in enumerate(available):
            line, bbox = item
            reference_height = min(profile.body_height, 1.25 * document_body_height)
            height_ratio = _line_effective_height(line, bbox) / max(0.1, reference_height)
            document_height_ratio = _line_effective_height(line, bbox) / max(
                0.1,
                document_body_height,
            )
            width_ratio = (bbox[2] - bbox[0]) / lane_width
            centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.15 * lane_width
            page_centered = abs(_bbox_center_x(bbox) - 0.5 * local_page_width) <= 0.15 * local_page_width
            page_width_ratio = (bbox[2] - bbox[0]) / max(0.1, local_page_width)
            spans_columns_fallback = (
                lane.is_span
                and centered
                and width_ratio >= 0.65
                and document_height_ratio >= 1.3
            )
            if (
                _bbox_center_y(bbox) > 0.45 * local_page_height
                or (height_ratio < 1.4 and not spans_columns_fallback)
                or width_ratio < 0.2
                or (not centered and height_ratio < 1.7)
                or (
                    not page_centered
                    and page_width_ratio < 0.45
                    and height_ratio < 1.8
                )
            ):
                continue
            top_preference = max(0.0, 0.45 - _bbox_center_y(bbox) / local_page_height)
            score = (
                height_ratio
                + (0.75 if centered else 0.0)
                + (1.25 if page_centered else 0.0)
                + (0.75 if page_width_ratio >= 0.55 else 0.0)
                + top_preference
                - 0.02 * row_index
            )
            candidates.append((score, lane, row_index, item))
    if not candidates:
        return None

    _score, lane, _row_index, anchor = max(candidates, key=lambda item: item[0])
    anchor_line, anchor_bbox = anchor
    anchor_height = _line_effective_height(anchor_line, anchor_bbox)
    anchor_line.semantic_type = "doc_title"
    selected = [anchor]
    ordered = lane.lines
    anchor_index = ordered.index(anchor)
    for direction in (-1, 1):
        index = anchor_index + direction
        previous_bbox = anchor_bbox
        while 0 <= index < len(ordered):
            candidate_line, candidate_bbox = ordered[index]
            if candidate_line.semantic_type is not None:
                break
            candidate_height = _line_effective_height(candidate_line, candidate_bbox)
            if not 0.8 <= candidate_height / anchor_height <= 1.25:
                break
            if not _document_title_fonts_compatible(anchor_line, candidate_line):
                break
            vertical_gap = max(candidate_bbox[1] - previous_bbox[3], previous_bbox[1] - candidate_bbox[3], 0.0)
            lane_width = max(0.1, lane.right - lane.left)
            centered = abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(anchor_bbox)) <= 0.18 * lane_width
            aligned = abs(candidate_bbox[0] - anchor_bbox[0]) <= 0.75 * anchor_height
            if vertical_gap > 1.1 * anchor_height or not (centered or aligned):
                break
            candidate_line.semantic_type = "doc_title"
            selected.append((candidate_line, candidate_bbox))
            previous_bbox = candidate_bbox
            index += direction
    return max(bbox[3] for _line, bbox in selected)


def _expand_document_title_across_lanes(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    local_page_height: float,
    document_title_bottom: float | None,
) -> float | None:
    """跨错误推断栏扩展紧邻、同字号且对齐的多行文档标题。"""

    title_items = [
        item
        for item in line_geometry
        if item[0].semantic_type == "doc_title"
    ]
    if not title_items:
        return document_title_bottom
    selected_ids = {id(line) for line, _bbox in title_items}
    changed = True
    while changed:
        changed = False
        title_bbox = _bbox_union_many([bbox for _line, bbox in title_items])
        title_height = statistics.median(
            _line_effective_height(line, bbox)
            for line, bbox in title_items
        )
        anchors = [line for line, _bbox in title_items]
        candidates = [
            item
            for item in line_geometry
            if id(item[0]) not in selected_ids
            and item[0].semantic_type is None
            and _bbox_center_y(item[1]) <= 0.45 * local_page_height
        ]
        candidates.sort(
            key=lambda item: min(
                abs(item[1][1] - title_bbox[3]),
                abs(title_bbox[1] - item[1][3]),
            )
        )
        for candidate_line, candidate_bbox in candidates:
            candidate_height = _line_effective_height(
                candidate_line,
                candidate_bbox,
            )
            if not 0.8 <= candidate_height / max(0.1, title_height) <= 1.25:
                continue
            if not any(
                _document_title_fonts_compatible(anchor, candidate_line)
                for anchor in anchors
            ):
                continue
            vertical_gap = max(
                candidate_bbox[1] - title_bbox[3],
                title_bbox[1] - candidate_bbox[3],
                0.0,
            )
            centered = (
                abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(title_bbox))
                <= 0.18 * local_page_width
            )
            aligned = (
                abs(candidate_bbox[0] - title_bbox[0])
                <= 0.75 * title_height
            )
            title_width = max(0.1, title_bbox[2] - title_bbox[0])
            candidate_width = candidate_bbox[2] - candidate_bbox[0]
            aligned_continuation = (
                aligned and candidate_width >= 0.25 * title_width
            )
            centered_continuation = (
                centered and candidate_width >= 0.45 * title_width
            )
            if (
                vertical_gap > 1.1 * title_height
                or not (aligned_continuation or centered_continuation)
            ):
                continue
            candidate_line.semantic_type = "doc_title"
            title_items.append((candidate_line, candidate_bbox))
            selected_ids.add(id(candidate_line))
            changed = True
            break
    return max(bbox[3] for _line, bbox in title_items)


def _classify_cross_lane_centered_section_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
    lanes: list[_TextLane],
    local_page_width: float,
    local_page_height: float,
    container_bboxes: list[BBox],
    *,
    document_title_bottom: float | None,
) -> None:
    """用正文栏中心、上下留白和正文邻行补标被单独推成窄栏的标题。"""

    stable_lanes = [
        lane
        for lane in lanes
        if not lane.is_span
        and len(lane.lines) >= 5
        and lane.right - lane.left >= 0.2 * local_page_width
    ]
    for line, bbox in line_geometry:
        if line.semantic_type is not None:
            continue
        if not 0.07 * local_page_height <= _bbox_center_y(bbox) <= 0.93 * local_page_height:
            continue
        candidate_lanes = [
            lane
            for lane in stable_lanes
            if lane.left <= _bbox_center_x(bbox) <= lane.right
        ]
        if not candidate_lanes:
            continue
        lane = min(
            candidate_lanes,
            key=lambda candidate: abs(
                _bbox_center_x(bbox)
                - 0.5 * (candidate.left + candidate.right)
            ),
        )
        profile = _infer_lane_body_profile(lane)
        lane_width = max(0.1, lane.right - lane.left)
        line_width = bbox[2] - bbox[0]
        line_height = _line_effective_height(line, bbox)
        if not 0.08 * lane_width <= line_width <= 0.7 * lane_width:
            continue
        if (
            abs(_bbox_center_x(bbox) - 0.5 * (lane.left + lane.right))
            > 0.05 * lane_width
        ):
            continue
        if not 0.75 <= line_height / max(0.1, profile.body_height) <= 1.3:
            continue
        if (
            document_title_bottom is not None
            and bbox[1] <= document_title_bottom + profile.body_height
        ):
            continue
        if _line_inside_visual_container(bbox, container_bboxes) or _line_near_visual_container(
            bbox,
            container_bboxes,
            profile.body_height,
        ):
            continue
        body_rows = [
            item
            for item in lane.lines
            if item[0] is not line
            and item[0].semantic_type is None
            and item[1][2] - item[1][0] >= 0.5 * lane_width
            and 0.75
            <= _line_effective_height(*item) / max(0.1, profile.body_height)
            <= 1.3
        ]
        rows_above = [
            item
            for item in body_rows
            if _bbox_center_y(item[1]) < _bbox_center_y(bbox)
        ]
        rows_below = [
            item
            for item in body_rows
            if _bbox_center_y(item[1]) > _bbox_center_y(bbox)
        ]
        if not rows_above or not rows_below:
            continue
        previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
        following_rows = sorted(
            rows_below,
            key=lambda item: _bbox_center_y(item[1]),
        )
        if len(following_rows) < 3 or any(
            _effective_text_row_gap(previous_row, current_row)
            > profile.regular_gap + 0.75 * profile.body_height
            for previous_row, current_row in zip(
                following_rows[:3],
                following_rows[1:3],
            )
        ):
            continue
        following = following_rows[0]
        gap_above = bbox[1] - previous[1][3]
        gap_below = following[1][1] - bbox[3]
        if (
            gap_above < 0.3 * profile.body_height
            or gap_below < -0.1 * profile.body_height
            or max(gap_above, gap_below)
            < profile.regular_gap + 0.2 * profile.body_height
        ):
            continue
        line.semantic_type = "paragraph_title"


def _demote_cross_lane_body_continuation_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
    lanes: list[_TextLane],
) -> None:
    """把紧接上一正文行、同字号同字体的短续行从标题降回正文。"""

    stable_lanes = [
        lane
        for lane in lanes
        if not lane.is_span and len(lane.lines) >= 4
    ]
    for line, bbox in line_geometry:
        if line.semantic_type != "paragraph_title":
            continue
        line_height = _line_effective_height(line, bbox)
        global_preceding = [
            item
            for item in line_geometry
            if item[0] is not line
            and item[0].semantic_type is None
            and item[0].angle == line.angle
            and _bbox_center_y(item[1]) < _bbox_center_y(bbox)
            and abs(item[1][0] - bbox[0]) <= line_height
        ]
        if global_preceding:
            previous_line, previous_bbox = max(
                global_preceding,
                key=lambda item: _bbox_center_y(item[1]),
            )
            previous_height = _line_effective_height(
                previous_line,
                previous_bbox,
            )
            same_family = (
                previous_line.font_signature is None
                or line.font_signature is None
                or _font_signatures_share_family(
                    previous_line.font_signature,
                    line.font_signature,
                )
            )
            if (
                same_family
                and 0.9
                <= line_height / max(0.1, previous_height)
                <= 1.1
                and bbox[1] - previous_bbox[3]
                <= 0.2 * max(line_height, previous_height)
                and abs(bbox[0] - previous_bbox[0])
                <= 0.75 * max(line_height, previous_height)
                and previous_bbox[2] - previous_bbox[0]
                >= 2.0 * (bbox[2] - bbox[0])
            ):
                line.semantic_type = None
                continue
        candidate_lanes = [
            lane
            for lane in stable_lanes
            if lane.left <= _bbox_center_x(bbox) <= lane.right
        ]
        if not candidate_lanes:
            continue
        lane = min(
            candidate_lanes,
            key=lambda candidate: abs(
                _bbox_center_x(bbox)
                - 0.5 * (candidate.left + candidate.right)
            ),
        )
        lane_width = max(0.1, lane.right - lane.left)
        if bbox[2] - bbox[0] > 0.55 * lane_width:
            continue
        preceding = [
            item
            for item in line_geometry
            if item[0] is not line
            and item[0].semantic_type is None
            and _bbox_center_y(item[1]) < _bbox_center_y(bbox)
            and lane.left <= _bbox_center_x(item[1]) <= lane.right
        ]
        if not preceding:
            continue
        previous_line, previous_bbox = max(
            preceding,
            key=lambda item: _bbox_center_y(item[1]),
        )
        previous_height = _line_effective_height(
            previous_line,
            previous_bbox,
        )
        line_height = _line_effective_height(line, bbox)
        if not 0.9 <= line_height / max(0.1, previous_height) <= 1.1:
            continue
        if (
            previous_line.font_signature is not None
            and line.font_signature is not None
            and not _font_signatures_share_family(
                previous_line.font_signature,
                line.font_signature,
            )
        ):
            continue
        if previous_bbox[2] - previous_bbox[0] < 0.5 * lane_width:
            continue
        if bbox[1] - previous_bbox[3] > 0.15 * max(line_height, previous_height):
            continue
        if abs(bbox[0] - lane.left) > max(1.0, 0.75 * line_height):
            continue
        line.semantic_type = None


def _classify_cross_lane_emphasized_section_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
    lanes: list[_TextLane],
    local_page_width: float,
    local_page_height: float,
    container_bboxes: list[BBox],
    *,
    document_title_bottom: float | None,
    document_body_profile: _DocumentBodyProfile | None,
) -> None:
    """用正文栏左缘、强调字体和段间留白补标跨栏推断失败的小节标题。"""

    stable_lanes = [
        lane
        for lane in lanes
        if not lane.is_span
        and len(lane.lines) >= 5
        and lane.right - lane.left >= 0.2 * local_page_width
    ]
    for line, bbox in line_geometry:
        if line.semantic_type is not None or line.font_signature is None:
            continue
        if not line.font_signature[1] & PDF_FONT_ITALIC_FLAG:
            continue
        if _line_uses_document_regular_font(line, document_body_profile):
            continue
        if not 0.07 * local_page_height <= _bbox_center_y(bbox) <= 0.93 * local_page_height:
            continue
        candidate_lanes = [
            lane
            for lane in stable_lanes
            if lane.left <= _bbox_center_x(bbox) <= lane.right
        ]
        if not candidate_lanes:
            continue
        lane = min(
            candidate_lanes,
            key=lambda candidate: abs(bbox[0] - candidate.left),
        )
        profile = _infer_lane_body_profile(lane)
        if (
            profile.body_font is None
            or line.font_signature == profile.body_font
            or line.font_coverage < 0.75
        ):
            continue
        lane_width = max(0.1, lane.right - lane.left)
        line_height = _line_effective_height(line, bbox)
        if bbox[2] - bbox[0] > 0.9 * lane_width:
            continue
        if abs(bbox[0] - lane.left) > 2.0 * profile.body_height:
            continue
        if not 0.8 <= line_height / max(0.1, profile.body_height) <= 1.35:
            continue
        if (
            document_title_bottom is not None
            and bbox[1] <= document_title_bottom + 2.0 * profile.body_height
        ):
            continue
        if _line_inside_visual_container(bbox, container_bboxes) or _line_near_visual_container(
            bbox,
            container_bboxes,
            profile.body_height,
        ):
            continue
        body_rows = [
            item
            for item in lane.lines
            if item[0] is not line
            and item[0].semantic_type is None
            and item[0].font_signature == profile.body_font
            and item[1][2] - item[1][0] >= 0.45 * lane_width
        ]
        rows_above = [
            item
            for item in body_rows
            if _bbox_center_y(item[1]) < _bbox_center_y(bbox)
        ]
        rows_below = [
            item
            for item in body_rows
            if _bbox_center_y(item[1]) > _bbox_center_y(bbox)
        ]
        if not rows_above or not rows_below:
            continue
        previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
        following = min(rows_below, key=lambda item: _bbox_center_y(item[1]))
        gap_above = bbox[1] - previous[1][3]
        gap_below = following[1][1] - bbox[3]
        if (
            gap_above < 0.35 * profile.body_height
            or gap_below > 3.0 * profile.body_height
            or gap_below < -0.1 * profile.body_height
        ):
            continue
        line.semantic_type = "paragraph_title"


def _is_wide_leading_title_continuation(
    title_bbox: BBox,
    candidate_bbox: BBox,
    title_height: float,
    candidate_height: float,
) -> bool:
    """识别紧贴在窄标题锚点上方、同中心的较宽首行。"""

    pair_height = max(0.1, title_height, candidate_height)
    title_width = max(0.1, title_bbox[2] - title_bbox[0])
    width_ratio = (candidate_bbox[2] - candidate_bbox[0]) / title_width
    vertical_gap = title_bbox[1] - candidate_bbox[3]
    return (
        2.0 < width_ratio <= 2.5
        and _bbox_center_y(candidate_bbox) < _bbox_center_y(title_bbox)
        and -0.1 * pair_height <= vertical_gap <= 0.15 * pair_height
        and _bbox_axis_overlap_ratio(title_bbox, candidate_bbox, axis="x") >= 0.8
        and abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(title_bbox))
        <= 0.5 * pair_height
    )


def _expand_cross_lane_paragraph_title_neighbors(
    line_geometry: list[tuple[_LineItem, BBox]],
) -> None:
    """把紧贴标题锚点的同字体相邻行跨栏补标为同一标题。"""

    changed = True
    while changed:
        changed = False
        title_items = [
            item
            for item in line_geometry
            if item[0].semantic_type == "paragraph_title"
        ]
        for title_line, title_bbox in title_items:
            title_height = _line_effective_height(title_line, title_bbox)
            for candidate_line, candidate_bbox in line_geometry:
                if candidate_line.semantic_type is not None:
                    continue
                candidate_height = _line_effective_height(
                    candidate_line,
                    candidate_bbox,
                )
                if not 0.85 <= candidate_height / max(0.1, title_height) <= 1.2:
                    continue
                title_width = max(0.1, title_bbox[2] - title_bbox[0])
                candidate_width = candidate_bbox[2] - candidate_bbox[0]
                if not (
                    0.25 <= candidate_width / title_width <= 2.0
                    or _is_wide_leading_title_continuation(
                        title_bbox,
                        candidate_bbox,
                        title_height,
                        candidate_height,
                    )
                ):
                    continue
                if (
                    title_line.font_signature is not None
                    and candidate_line.font_signature is not None
                    and title_line.font_signature != candidate_line.font_signature
                ):
                    continue
                vertical_gap = max(
                    candidate_bbox[1] - title_bbox[3],
                    title_bbox[1] - candidate_bbox[3],
                    0.0,
                )
                if vertical_gap > 0.35 * max(title_height, candidate_height):
                    continue
                if (
                    _bbox_axis_overlap_ratio(title_bbox, candidate_bbox, axis="x")
                    < 0.2
                    and abs(candidate_bbox[0] - title_bbox[0])
                    > 2.0 * title_height
                ):
                    continue
                candidate_line.semantic_type = "paragraph_title"
                changed = True
                break
            if changed:
                break


def _demote_visual_container_caption_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
    container_bboxes: list[BBox],
) -> None:
    """把紧贴视觉容器下缘且水平居中的标题候选降回普通图注文本。"""

    for line, bbox in line_geometry:
        if line.semantic_type != "paragraph_title":
            continue
        line_height = _line_effective_height(line, bbox)
        for container_bbox in container_bboxes:
            vertical_gap = bbox[1] - container_bbox[3]
            if not -0.25 * line_height <= vertical_gap <= 1.5 * line_height:
                continue
            if _bbox_axis_overlap_ratio(bbox, container_bbox, axis="x") < 0.35:
                continue
            if (
                abs(_bbox_center_x(bbox) - _bbox_center_x(container_bbox))
                > 0.35
                * max(
                    container_bbox[2] - container_bbox[0],
                    bbox[2] - bbox[0],
                )
            ):
                continue
            line.semantic_type = None
            break


def _document_title_fonts_compatible(
    first: _LineItem,
    second: _LineItem,
) -> bool:
    """允许混排标题因主字体覆盖不足而切换字体，同时保留可靠字重屏障。"""

    if _title_fonts_compatible(first, second):
        return True
    uncertain_dominant_font = min(first.font_coverage, second.font_coverage) < 0.85
    weights_compatible = (
        first.dominant_font_weight is None
        or second.dominant_font_weight is None
        or abs(first.dominant_font_weight - second.dominant_font_weight) < 100.0
        or max(first.dominant_font_weight, second.dominant_font_weight)
        < 1.15 * min(first.dominant_font_weight, second.dominant_font_weight)
    )
    return uncertain_dominant_font and weights_compatible


def _document_title_uses_page_fallback(
    lanes: list[_TextLane],
    *,
    document_body_profile: _DocumentBodyProfile | None = None,
) -> bool:
    """判断首页标题是否依赖跨栏 1.30 倍全文正文行高兜底。"""

    title_heights = [
        _line_effective_height(line, bbox)
        for lane in lanes
        for line, bbox in lane.lines
        if line.semantic_type == "doc_title"
    ]
    column_body_heights = [
        _infer_lane_body_profile(lane).body_height
        for lane in lanes
        if not lane.is_span
    ]
    if not title_heights or not column_body_heights:
        return False
    document_body_height = (
        document_body_profile.body_height
        if document_body_profile is not None
        else statistics.median(column_body_heights)
    )
    return any(
        1.3 <= title_height / max(0.1, document_body_height) < 1.4
        for title_height in title_heights
    )


def _classify_paragraph_titles_in_lane(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    local_page_width: float,
    local_page_height: float,
    container_bboxes: list[BBox],
    *,
    document_title_bottom: float | None,
    preserve_front_matter_boundaries: bool,
    physical_gaps: dict[int, tuple[float | None, float | None]],
    grid_title_suppressions: set[int],
    document_body_profile: _DocumentBodyProfile | None,
    document_title_profile: _DocumentTitleProfile | None,
    page_index: int,
) -> None:
    """以字号、样式、留白、对齐、栏宽和容器邻接判定段落标题。"""

    lane_width = max(0.1, lane.right - lane.left)
    rows = lane.lines
    selected_indices: set[int] = set()
    front_matter_boundary = _infer_front_matter_boundary(
        lane,
        profile,
        local_page_height,
        document_title_bottom=document_title_bottom,
    )
    for index, (line, bbox) in enumerate(rows):
        if line.semantic_type is not None:
            continue
        if line.source_index in grid_title_suppressions:
            continue
        if not 0.07 * local_page_height <= _bbox_center_y(bbox) <= 0.93 * local_page_height:
            continue
        inside_front_matter = (
            front_matter_boundary is not None
            and _bbox_center_y(bbox) <= front_matter_boundary
        )
        if inside_front_matter and not preserve_front_matter_boundaries:
            continue
        if (
            document_title_bottom is not None
            and document_title_bottom <= 0.6 * local_page_height
            and _bbox_center_y(bbox) >= 0.84 * local_page_height
        ):
            # 首页底部短行通常是版本、日期等封面元数据，不属于正文标题层级。
            continue
        line_height = _line_effective_height(line, bbox)
        title_prototype = _matching_document_title_prototype(
            line,
            bbox,
            lane,
            document_body_profile,
            document_title_profile,
        )
        title_profile_size_conflict = (
            title_prototype is None
            and _line_conflicts_document_title_profile(
                line,
                bbox,
                lane,
                document_body_profile,
                document_title_profile,
            )
        )
        inside_visual_container = _line_inside_visual_container(
            bbox,
            container_bboxes,
        )
        near_visual_container = _line_near_visual_container(
            bbox,
            container_bboxes,
            profile.body_height,
        )
        if inside_visual_container or (
            near_visual_container and title_prototype is None
        ):
            continue

        recurrent_regular_font = _line_uses_document_regular_font(
            line,
            document_body_profile,
        )
        document_regular_body_candidate = (
            recurrent_regular_font
            and document_body_profile is not None
            and line_height >= 0.9 * document_body_profile.body_height
        )
        reference_body_height = profile.body_height
        if (
            profile.body_row_count < 3
            and document_body_profile is not None
        ):
            reference_body_height = document_body_profile.body_height
        if document_regular_body_candidate and document_body_profile is not None:
            reference_body_height = max(
                reference_body_height,
                document_body_profile.body_height,
            )
        height_ratio = line_height / reference_body_height
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.12 * lane_width
        left_aligned = abs(bbox[0] - lane.left) <= 0.65 * profile.body_height
        if (
            line.median_glyph_width is not None
            and bbox[2] - bbox[0] <= 1.25 * line.median_glyph_width
            and height_ratio < 1.18
        ):
            continue
        style_differs = (
            not recurrent_regular_font
            and profile.body_font is not None
            and line.font_signature is not None
            and line.font_coverage >= 0.75
            and not _font_signatures_share_family(
                line.font_signature,
                profile.body_font,
            )
        )
        low_coverage_style_differs = (
            not recurrent_regular_font
            and profile.body_font is not None
            and line.font_signature is not None
            and 0.5 <= line.font_coverage < 0.75
            and not _font_signatures_share_family(
                line.font_signature,
                profile.body_font,
            )
        )
        style_support = profile.style_support.get(line.font_signature, 1.0) if line.font_signature is not None else 1.0
        weight_emphasized = (
            profile.body_weight is not None
            and line.dominant_font_weight is not None
            and line.dominant_font_weight >= max(profile.body_weight + 100.0, 1.15 * profile.body_weight)
        )
        gap_above = _normalized_title_gap(
            rows,
            index,
            direction=-1,
            body_height=profile.body_height,
            physical_gaps=physical_gaps,
        )
        gap_below = _normalized_title_gap(
            rows,
            index,
            direction=1,
            body_height=profile.body_height,
            physical_gaps=physical_gaps,
        )
        regular_gap_ratio = profile.regular_gap / max(0.1, profile.body_height)
        gap_above_excess = max(0.0, gap_above - regular_gap_ratio)
        gap_below_excess = max(0.0, gap_below - regular_gap_ratio)
        has_spacing_signal = gap_above >= 0.35

        if _visual_row_has_body_style_sibling(rows, index):
            continue
        if _is_near_full_mixed_inline_row(
            rows,
            index,
            lane_width,
            profile,
        ):
            continue
        if (
            index > 0
            and rows[index - 1][0].semantic_type == "paragraph_title"
            and width_ratio >= 0.85
            and gap_above < 1.0
            and not _title_fonts_compatible(rows[index - 1][0], line)
        ):
            continue
        if _continues_local_body_row(
            rows,
            index,
            lane_width,
            profile,
        ):
            continue
        has_following_body_row = _has_following_body_row(
            rows,
            index,
            lane_width,
            lane.left,
            profile,
        )
        compact_local_transition = (
            low_coverage_style_differs
            and left_aligned
            and width_ratio <= 0.65
            and gap_above >= 0.1
            and gap_below >= 0.2
            and has_following_body_row
        )
        compact_text_section = (
            height_ratio < 0.9
            and centered
            and width_ratio <= 0.7
            and gap_above >= 0.35
            and gap_below >= 0.2
            and _has_following_compact_text_section(
                rows,
                index,
                lane_width,
            )
        )
        weak_regular_pitch_body_candidate = (
            document_body_profile is not None
            and 0.9
            <= line_height / document_body_profile.body_height
            <= 1.1
            and title_prototype is None
            and (not centered or width_ratio >= 0.75)
            and not weight_emphasized
            and gap_above_excess < 0.1
            and gap_below_excess < 0.1
        )
        if weak_regular_pitch_body_candidate:
            continue
        if (
            title_profile_size_conflict
            and document_body_profile is not None
            and line_height <= 1.18 * document_body_profile.body_height
            and not weight_emphasized
        ):
            continue
        prototype_promotion = (
            title_prototype is not None
            and (
                width_ratio <= 0.9
                or (bbox[2] - bbox[0]) / max(0.1, local_page_width) <= 0.8
            )
            and (centered or left_aligned)
            and (
                (
                    document_body_profile is not None
                    and line_height >= 1.18 * document_body_profile.body_height
                )
                or
                near_visual_container
                or (has_following_body_row and gap_above_excess >= 0.1)
                or gap_above_excess >= 0.35
            )
        )
        prototype_inline_heading = (
            document_body_profile is not None
            and line_height <= 1.15 * document_body_profile.body_height
            and _is_full_width_inline_heading(
                rows,
                index,
                lane_width,
                profile,
            )
        )
        if prototype_promotion and not prototype_inline_heading:
            line.semantic_type = "paragraph_title"
            selected_indices.add(index)
            continue
        if (
            document_body_profile is not None
            and 0.9
            <= line_height / document_body_profile.body_height
            <= 1.1
            and title_prototype is None
            and centered
            and not has_following_body_row
            and not compact_text_section
        ):
            continue
        if (
            height_ratio < 0.9
            and not inside_front_matter
            and not has_following_body_row
            and not compact_text_section
        ):
            continue
        if width_ratio >= 0.9 and height_ratio < 1.15 and not (
            style_differs and gap_above >= 0.65
        ):
            continue
        score = 0.0
        if style_differs or compact_local_transition:
            score += 2.0
            if compact_local_transition or style_support <= 0.2:
                score += 0.75
        if weight_emphasized:
            score += 1.0
        if height_ratio >= 1.18:
            score += 2.0
        elif height_ratio <= 0.9 and centered:
            score += 0.75
        if width_ratio <= 0.78:
            score += 0.75
        if centered and width_ratio <= 0.85:
            score += 1.25
        if gap_above >= 0.65:
            score += 1.25
        elif gap_above >= 0.35:
            score += 0.5
        if gap_below >= 0.35:
            score += 0.75
        if left_aligned:
            score += 0.25

        precisely_centered = (
            abs(
                _bbox_center_x(bbox)
                - (lane.left + lane.right) / 2.0
            )
            <= 0.05 * lane_width
        )
        centered_structural_fallback = (
            0.75 <= height_ratio <= 1.15
            and precisely_centered
            and width_ratio <= 0.7
            and not inside_front_matter
            and not (
                page_index == 0
                and _bbox_center_y(bbox) <= 0.2 * local_page_height
            )
            and 0.35 <= gap_above <= 1.5
            and 0.2 <= gap_below <= 1.5
            and has_following_body_row
            and not _is_continuous_field_row(
                rows,
                index,
                lane_width,
                profile,
            )
        )
        strong_layout_signal = (
            height_ratio >= 1.18
            or style_differs
            or compact_local_transition
            or weight_emphasized
            or centered_structural_fallback
            or compact_text_section
        )
        if (
            score >= 4.0
            and (has_spacing_signal or compact_local_transition)
            and strong_layout_signal
        ):
            if _is_full_width_inline_heading(
                rows,
                index,
                lane_width,
                profile,
            ):
                continue
            line.semantic_type = "paragraph_title"
            selected_indices.add(index)

    _unify_visual_row_title_types(
        lane,
        selected_indices,
    )
    _expand_paragraph_title_neighbors(
        lane,
        selected_indices,
        profile,
    )
    if front_matter_boundary is not None and preserve_front_matter_boundaries:
        _protect_front_matter_title_types(
            lane,
            profile,
            front_matter_boundary,
        )


def _line_uses_document_regular_font(
    line: _LineItem,
    document_body_profile: _DocumentBodyProfile | None,
) -> bool:
    """判断当前行是否使用跨页反复出现且未加粗的常规字体。"""

    return (
        document_body_profile is not None
        and line.font_signature is not None
        and line.font_coverage >= 0.5
        and any(
            _font_signatures_share_family(line.font_signature, regular_font)
            for regular_font in document_body_profile.regular_fonts
        )
    )


def _matching_document_title_prototype(
    line: _LineItem,
    bbox: BBox,
    lane: _TextLane,
    document_body_profile: _DocumentBodyProfile | None,
    document_title_profile: _DocumentTitleProfile | None,
) -> _TitleStylePrototype | None:
    """返回与当前行字体、字号、字重和栏内锚点兼容的最强标题原型。"""

    if (
        document_body_profile is None
        or document_title_profile is None
        or line.font_signature is None
        or line.font_coverage < 0.75
    ):
        return None
    font_family = _normalized_font_family(line.font_signature)
    if font_family is None:
        return None
    height_ratio = _line_effective_height(line, bbox) / max(
        0.1,
        document_body_profile.body_height,
    )
    candidates: list[_TitleStylePrototype] = []
    for prototype in document_title_profile.prototypes:
        if (
            prototype.font_family != font_family
            or prototype.font_flags != line.font_signature[1]
            or abs(height_ratio - prototype.height_ratio)
            > 0.1 * prototype.height_ratio
        ):
            continue
        if (
            prototype.weight is not None
            and line.dominant_font_weight is not None
            and abs(prototype.weight - line.dominant_font_weight) >= 100.0
            and max(prototype.weight, line.dominant_font_weight)
            >= 1.15 * min(prototype.weight, line.dominant_font_weight)
        ):
            continue
        alignment = _title_profile_alignment(
            bbox,
            lane,
            document_body_profile.body_height,
        )
        if (
            alignment is None
            or alignment[0] != prototype.alignment
            or abs(alignment[1] - prototype.anchor_offset) > 0.15
        ):
            continue
        candidates.append(prototype)
    return max(
        candidates,
        key=lambda item: (item.support_pages, item.support_count),
        default=None,
    )


def _line_inside_visual_container(
    line_bbox: BBox,
    container_bboxes: list[BBox],
) -> bool:
    """检查文本行中心是否落入视觉容器，容器内标签不得借标题原型晋升。"""

    center_x = _bbox_center_x(line_bbox)
    center_y = _bbox_center_y(line_bbox)
    return any(
        bbox[0] <= center_x <= bbox[2]
        and bbox[1] <= center_y <= bbox[3]
        for bbox in container_bboxes
    )


def _line_conflicts_document_title_profile(
    line: _LineItem,
    bbox: BBox,
    lane: _TextLane,
    document_body_profile: _DocumentBodyProfile | None,
    document_title_profile: _DocumentTitleProfile | None,
) -> bool:
    """识别字体与标题原型一致、但字号明显落入正文带的弱标题候选。"""

    if (
        document_body_profile is None
        or document_title_profile is None
        or line.font_signature is None
        or line.font_coverage < 0.75
    ):
        return False
    font_family = _normalized_font_family(line.font_signature)
    alignment = _title_profile_alignment(
        bbox,
        lane,
        document_body_profile.body_height,
    )
    if font_family is None or alignment is None:
        return False
    height_ratio = _line_effective_height(line, bbox) / max(
        0.1,
        document_body_profile.body_height,
    )
    for prototype in document_title_profile.prototypes:
        if (
            not _title_font_families_compatible(
                prototype.font_family,
                font_family,
            )
            or prototype.font_flags != line.font_signature[1]
            or prototype.alignment != alignment[0]
            or abs(prototype.anchor_offset - alignment[1]) > 0.15
        ):
            continue
        if (
            prototype.weight is not None
            and line.dominant_font_weight is not None
            and abs(prototype.weight - line.dominant_font_weight) >= 100.0
            and max(prototype.weight, line.dominant_font_weight)
            >= 1.15 * min(prototype.weight, line.dominant_font_weight)
        ):
            continue
        if abs(height_ratio - prototype.height_ratio) > 0.15 * prototype.height_ratio:
            return True
    return False


def _title_font_families_compatible(first: str, second: str) -> bool:
    """忽略已由 flags 和字重单独约束的常见字体样式后缀。"""

    if first == second:
        return True
    style_suffixes = (",bold", "bold", ",regular", "regular", ",medium", "medium")
    return any(
        first.removesuffix(suffix) == second
        or second.removesuffix(suffix) == first
        for suffix in style_suffixes
    )


def _protect_front_matter_title_types(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    front_matter_boundary: float,
) -> None:
    """把作者区误命中的标题降为正文，并仅保留原本需要跨行聚合的内部标记。"""

    front_title_indices = [
        index
        for index, (line, bbox) in enumerate(lane.lines)
        if line.semantic_type == "paragraph_title"
        and _bbox_center_y(bbox) <= front_matter_boundary
    ]
    connected_indices: set[int] = set()
    for position, first_index in enumerate(front_title_indices):
        for second_index in front_title_indices[position + 1 :]:
            if _should_connect_semantic_rows(
                lane.lines[first_index],
                lane.lines[second_index],
                lane,
                profile.regular_gap,
                [],
                [],
            ):
                connected_indices.update((first_index, second_index))
    for index in front_title_indices:
        lane.lines[index][0].semantic_type = (
            "text" if index in connected_indices else None
        )


def _visual_row_has_body_style_sibling(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
) -> bool:
    """检查同一完整视觉行是否同时包含标题样式与正文样式 run。"""

    line, _bbox = rows[index]
    if line.visual_row_id is None:
        return False
    siblings = [
        sibling
        for sibling, _sibling_bbox in rows
        if sibling is not line and sibling.visual_row_id == line.visual_row_id
    ]
    return any(not _title_fonts_compatible(line, sibling) for sibling in siblings)


def _is_near_full_mixed_inline_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别近满栏混合字体行内强调，避免把粗体条目头单独标成标题。"""

    line, bbox = rows[index]
    line_height = _line_effective_height(line, bbox)
    if (
        line.font_coverage >= 0.95
        or (bbox[2] - bbox[0]) < 0.85 * lane_width
        or not 0.85 <= line_height / profile.body_height <= 1.15
        or index + 1 >= len(rows)
    ):
        return False
    next_line, next_bbox = rows[index + 1]
    next_height = _line_effective_height(next_line, next_bbox)
    gap_below = _effective_text_row_gap(rows[index], rows[index + 1])
    return (
        next_line.semantic_type is None
        and 0.8 <= next_height / profile.body_height <= 1.2
        and gap_below <= profile.regular_gap + 0.25 * profile.body_height
    )


def _continues_local_body_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别紧随同字体满行的正文尾行，阻止全局小字号基线造成标题误判。"""

    if index <= 0:
        return False
    previous_line, previous_bbox = rows[index - 1]
    line, bbox = rows[index]
    if previous_line.semantic_type is not None:
        return False
    previous_visual_row = (
        [
            item
            for item in rows[:index]
            if item[0].visual_row_id == previous_line.visual_row_id
        ]
        if previous_line.visual_row_id is not None
        else [rows[index - 1]]
    )
    previous_style_members = [
        member
        for member, _member_bbox in previous_visual_row
        if member.font_signature is not None and member.font_coverage >= 0.5
    ]
    if (
        line.font_signature is not None
        and previous_style_members
        and not any(
            _font_signatures_share_family(member.font_signature, line.font_signature)
            and not _font_weights_conflict(member, line)
            for member in previous_style_members
        )
    ):
        return False
    previous_bbox = _bbox_union_many(
        [member_bbox for _member, member_bbox in previous_visual_row]
    )
    previous_height = statistics.median(
        _line_effective_height(member, member_bbox)
        for member, member_bbox in previous_visual_row
    )
    line_height = _line_effective_height(line, bbox)
    pair_height = max(previous_height, line_height)
    gap = _effective_text_row_gap(
        (previous_line, previous_bbox),
        rows[index],
    )
    return (
        0.8 <= line_height / previous_height <= 1.25
        and previous_bbox[2] - previous_bbox[0] >= 0.75 * lane_width
        and abs(bbox[0] - previous_bbox[0]) <= 2.0 * pair_height
        and -0.25 * pair_height
        <= gap
        <= profile.regular_gap + pair_height
    )


def _is_continuous_field_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别同缩进、同尺度且紧邻的连续字段行，阻止其借居中误差成为标题。"""

    _line, bbox = rows[index]
    line_height = _line_effective_height(*rows[index])
    if not 0.85 <= line_height / profile.body_height <= 1.15:
        return False
    for neighbor_index in (index - 1, index + 1):
        if not 0 <= neighbor_index < len(rows):
            continue
        neighbor_line, neighbor_bbox = rows[neighbor_index]
        neighbor_height = _line_effective_height(neighbor_line, neighbor_bbox)
        if (
            neighbor_line.semantic_type is None
            and 0.85 <= neighbor_height / profile.body_height <= 1.15
            and neighbor_bbox[2] - neighbor_bbox[0] >= 0.35 * lane_width
            and abs(neighbor_bbox[0] - bbox[0]) <= 0.75 * profile.body_height
            and max(
                _effective_text_row_gap(rows[neighbor_index], rows[index]),
                _effective_text_row_gap(rows[index], rows[neighbor_index]),
            )
            <= 1.5 * profile.body_height
        ):
            return True
    return False


def _is_full_width_inline_heading(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别后接常规正文续行的满栏正常字号行内标题。"""

    line, bbox = rows[index]
    line_height = _line_effective_height(line, bbox)
    if (
        not 0.85 <= line_height / profile.body_height <= 1.15
        or (bbox[2] - bbox[0]) < 0.9 * lane_width
        or index + 1 >= len(rows)
    ):
        return False

    next_line, next_bbox = rows[index + 1]
    next_height = _line_effective_height(next_line, next_bbox)
    next_uses_body_font = (
        profile.body_font is None
        or next_line.font_signature is None
        or next_line.font_coverage < 0.75
        or _font_signatures_share_family(
            next_line.font_signature,
            profile.body_font,
        )
    )
    return (
        next_line.semantic_type is None
        and next_uses_body_font
        and 0.75 <= next_height / profile.body_height <= 1.15
        and (next_bbox[2] - next_bbox[0]) >= 0.75 * lane_width
        and _effective_text_row_gap(rows[index], rows[index + 1])
        <= profile.regular_gap + 0.25 * profile.body_height
    )


def _has_following_body_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    lane_left: float,
    profile: _LaneBodyProfile,
) -> bool:
    """只检查紧邻且空间相关的下一行，禁止用远处正文支撑小字号标题。"""

    if index + 1 >= len(rows):
        return False
    current = rows[index]
    line, bbox = rows[index + 1]
    gap = _effective_text_row_gap(current, (line, bbox))
    horizontally_related = (
        _bbox_axis_overlap_ratio(current[1], bbox, axis="x") >= 0.35
        or abs(current[1][0] - bbox[0]) <= 0.75 * profile.body_height
    )
    current_centered = abs(
        _bbox_center_x(current[1]) - (lane_left + lane_width / 2.0)
    ) <= 0.12 * lane_width
    compact_left_aligned = (
        current[1][2] - current[1][0] <= 0.35 * lane_width
        and abs(current[1][0] - lane_left) <= 0.65 * profile.body_height
    )
    maximum_gap = (
        1.5 if current_centered else 1.25 if compact_left_aligned else 0.75
    ) * profile.body_height
    height_ratio = _line_effective_height(line, bbox) / profile.body_height
    width_ratio = (bbox[2] - bbox[0]) / lane_width
    return (
        line.semantic_type is None
        and horizontally_related
        and -0.25 * profile.body_height <= gap <= maximum_gap
        and 0.9 <= height_ratio <= 1.2
        and width_ratio >= 0.45
    )


def _has_following_compact_text_section(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
) -> bool:
    """检查小字号候选后是否紧接三行同尺度的宽文本区段。"""

    candidate = rows[index]
    candidate_height = _line_effective_height(*candidate)
    following_rows = rows[index + 1 : index + 4]
    if len(following_rows) < 3:
        return False

    previous = candidate
    for current in following_rows:
        line, bbox = current
        height_ratio = _line_effective_height(line, bbox) / candidate_height
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        gap = _effective_text_row_gap(previous, current)
        if (
            line.semantic_type is not None
            or not 0.75 <= height_ratio <= 1.25
            or width_ratio < 0.45
            or not -0.25 * candidate_height <= gap <= 0.75 * candidate_height
        ):
            return False
        previous = current
    return True


def _unify_visual_row_title_types(
    lane: _TextLane,
    selected_indices: set[int],
) -> None:
    """按完整 visual_row_id 统一标题类型：同样式整行晋升，混合样式整行降级。"""

    row_indices: dict[int, list[int]] = {}
    for index, (line, _bbox) in enumerate(lane.lines):
        if line.visual_row_id is not None:
            row_indices.setdefault(line.visual_row_id, []).append(index)
    for indices in row_indices.values():
        title_indices = [
            index
            for index in indices
            if lane.lines[index][0].semantic_type == "paragraph_title"
        ]
        if not title_indices:
            continue
        anchor_line, anchor_bbox = lane.lines[title_indices[0]]
        anchor_height = _line_effective_height(anchor_line, anchor_bbox)
        compatible = all(
            0.75
            <= _line_effective_height(lane.lines[index][0], lane.lines[index][1])
            / anchor_height
            <= 1.25
            and _title_fonts_compatible(anchor_line, lane.lines[index][0])
            for index in indices
        )
        if compatible:
            for index in indices:
                lane.lines[index][0].semantic_type = "paragraph_title"
                selected_indices.add(index)
            continue
        for index in title_indices:
            lane.lines[index][0].semantic_type = None
            selected_indices.discard(index)


def _infer_front_matter_boundary(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    local_page_height: float,
    *,
    document_title_bottom: float | None,
) -> float | None:
    """用短行后衔接满栏正文的几何转折点界定首页作者等前置信息。"""

    if document_title_bottom is None:
        return None
    default_boundary = max(
        document_title_bottom + 2.5 * profile.body_height,
        0.39 * local_page_height,
    )
    lane_width = max(0.1, lane.right - lane.left)
    for index, (line, bbox) in enumerate(lane.lines[:-1]):
        if bbox[1] <= document_title_bottom:
            continue
        next_line, next_bbox = lane.lines[index + 1]
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        next_width_ratio = (next_bbox[2] - next_bbox[0]) / lane_width
        left_aligned = abs(bbox[0] - lane.left) <= 0.65 * profile.body_height
        centered = abs(
            _bbox_center_x(bbox) - (lane.left + lane.right) / 2.0
        ) <= 0.12 * lane_width
        next_height_ratio = _line_effective_height(next_line, next_bbox) / profile.body_height
        gap_below = max(0.0, _effective_text_row_gap((line, bbox), (next_line, next_bbox)))
        if (
            width_ratio <= 0.35
            and next_width_ratio >= 0.75
            and (left_aligned or centered)
            and 0.75 <= next_height_ratio <= 1.35
            and gap_below <= 1.5 * profile.body_height
        ):
            return min(default_boundary, bbox[1] - 0.1)
    return default_boundary


def _normalized_title_gap(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    *,
    direction: Literal[-1, 1],
    body_height: float,
    physical_gaps: dict[int, tuple[float | None, float | None]],
) -> float:
    """返回栏内邻行和页面最近物理邻行中较小的归一化净空。"""

    neighbor_index = index + direction
    if 0 <= neighbor_index < len(rows):
        if direction < 0:
            lane_gap = _effective_text_row_gap(rows[neighbor_index], rows[index])
        else:
            lane_gap = _effective_text_row_gap(rows[index], rows[neighbor_index])
    else:
        lane_gap = 0.5 * body_height
    physical_pair = physical_gaps.get(rows[index][0].source_index, (None, None))
    physical_gap = physical_pair[0 if direction < 0 else 1]
    gap = lane_gap if physical_gap is None else min(lane_gap, physical_gap)
    return max(0.0, gap) / max(0.1, body_height)


def _line_near_visual_container(
    line_bbox: BBox,
    container_bboxes: list[BBox],
    body_height: float,
) -> bool:
    """检查短行是否紧邻图、表等容器，以纯几何方式抑制 caption 误判。"""

    for container_bbox in container_bboxes:
        if _bbox_axis_overlap_ratio(line_bbox, container_bbox, axis="x") < 0.35:
            continue
        container_width = max(0.1, container_bbox[2] - container_bbox[0])
        if line_bbox[2] - line_bbox[0] > 0.8 * container_width:
            continue
        vertical_gap = max(line_bbox[1] - container_bbox[3], container_bbox[1] - line_bbox[3], 0.0)
        if vertical_gap <= 2.0 * body_height:
            return True
    return False


def _expand_paragraph_title_neighbors(
    lane: _TextLane,
    selected_indices: set[int],
    profile: _LaneBodyProfile,
) -> None:
    """用相邻行的字体、尺寸、对齐和紧凑净空补齐折行段落标题。"""

    if not selected_indices:
        return
    rows = lane.lines
    lane_width = max(0.1, lane.right - lane.left)
    pending = list(selected_indices)
    while pending:
        selected_index = pending.pop()
        selected_line, selected_bbox = rows[selected_index]
        selected_height = _line_effective_height(selected_line, selected_bbox)
        for candidate_index in (selected_index - 1, selected_index + 1):
            if not 0 <= candidate_index < len(rows) or candidate_index in selected_indices:
                continue
            candidate_line, candidate_bbox = rows[candidate_index]
            if candidate_line.semantic_type is not None:
                continue
            candidate_height = _line_effective_height(candidate_line, candidate_bbox)
            if not 0.75 <= candidate_height / selected_height <= 1.25:
                continue
            if candidate_bbox[2] - candidate_bbox[0] > 0.8 * lane_width:
                continue
            if (
                selected_line.font_signature is not None
                and candidate_line.font_signature is not None
                and selected_line.font_signature != candidate_line.font_signature
            ):
                continue
            if not _title_fonts_compatible(selected_line, candidate_line):
                continue
            vertical_gap = max(
                candidate_bbox[1] - selected_bbox[3],
                selected_bbox[1] - candidate_bbox[3],
                0.0,
            )
            centers_align = abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(selected_bbox)) <= 0.15 * lane_width
            left_edges_align = abs(candidate_bbox[0] - selected_bbox[0]) <= 0.75 * profile.body_height
            wrapped_indent = abs(candidate_bbox[0] - selected_bbox[0]) <= 2.0 * profile.body_height
            if vertical_gap > 0.65 * profile.body_height or not (
                centers_align or left_edges_align or wrapped_indent
            ):
                continue
            candidate_line.semantic_type = "paragraph_title"
            selected_indices.add(candidate_index)
            pending.append(candidate_index)
