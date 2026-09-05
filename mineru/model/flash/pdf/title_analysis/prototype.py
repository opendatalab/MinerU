# Copyright (c) Opendatalab. All rights reserved.
"""匹配标题原型的字体、尺度和对齐特征。"""

from __future__ import annotations

import statistics
from typing import Literal

from .....types import BBox
from ..geometry import _bbox_center_x
from ..line_layout import (
    _line_effective_height,
    _normalized_font_family,
)
from ..models import (
    _DocumentBodyProfile,
    _DocumentTitleProfile,
    _LineItem,
    _TextLane,
    _TitleStylePrototype,
)


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
        or max(seed[3], statistics.median(weights)) < 1.15 * min(seed[3], statistics.median(weights))
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
    center_offset = (_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) / lane_width
    left_offset = (bbox[0] - lane.left) / lane_width
    if abs(center_offset) <= 0.12 and bbox[2] - bbox[0] <= 0.85 * lane_width:
        return "center", center_offset
    if abs(bbox[0] - lane.left) <= 3.0 * body_height:
        return "left", left_offset
    return None


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
            or abs(height_ratio - prototype.height_ratio) > 0.1 * prototype.height_ratio
        ):
            continue
        if (
            prototype.weight is not None
            and line.dominant_font_weight is not None
            and abs(prototype.weight - line.dominant_font_weight) >= 100.0
            and max(prototype.weight, line.dominant_font_weight) >= 1.15 * min(prototype.weight, line.dominant_font_weight)
        ):
            continue
        alignment = _title_profile_alignment(
            bbox,
            lane,
            document_body_profile.body_height,
        )
        if alignment is None or alignment[0] != prototype.alignment or abs(alignment[1] - prototype.anchor_offset) > 0.15:
            continue
        candidates.append(prototype)
    return max(
        candidates,
        key=lambda item: (item.support_pages, item.support_count),
        default=None,
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
            and max(prototype.weight, line.dominant_font_weight) >= 1.15 * min(prototype.weight, line.dominant_font_weight)
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
    return any(first.removesuffix(suffix) == second or second.removesuffix(suffix) == first for suffix in style_suffixes)


__all__ = [
    "_title_profile_seed_matches_cluster",
    "_title_profile_alignment",
    "_matching_document_title_prototype",
    "_line_conflicts_document_title_profile",
    "_title_font_families_compatible",
]
