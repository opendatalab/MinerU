# Copyright (c) Opendatalab. All rights reserved.
"""使用既有页面分类探测构建全文标题原型。"""

from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Literal

from ..geometry import _rotate_bbox_to_upright
from ..line_layout import (
    _infer_text_lanes,
    _line_effective_height,
    _normalized_font_family,
)
from ..models import (
    _DocumentBodyProfile,
    _DocumentTitleProfile,
    _PreparedPage,
    _TitleStylePrototype,
)
from .common import _line_inside_visual_container
from .page_titles import _classify_page_titles
from .prototype import _title_profile_alignment, _title_profile_seed_matches_cluster


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
            block["bbox"] for block in prepared.fixed_blocks if not isinstance(block.get("_inline_visual_row_id"), int)
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
            median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in geometry)
            local_page_width = prepared.page_size[1] if angle in {90, 270} else prepared.page_size[0]
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
                    line_height_ratio = _line_effective_height(line, bbox) / document_body_profile.body_height
                    large_unresolved_seed = (
                        page_index > 0
                        and line.semantic_type is None
                        and line_height_ratio >= 1.3
                        and bbox[2] - bbox[0] <= 0.8 * local_page_width
                        and not _line_inside_visual_container(
                            bbox,
                            local_container_bboxes,
                        )
                    )
                    if (
                        (line.semantic_type != "paragraph_title" and not large_unresolved_seed)
                        or line.font_signature is None
                        or line.font_coverage < 0.75
                    ):
                        continue
                    row_identity = line.visual_row_id if line.visual_row_id is not None else line.source_index
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
            (cluster for cluster in clusters if _title_profile_seed_matches_cluster(seed, cluster)),
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


__all__ = ["_infer_document_title_profile"]
