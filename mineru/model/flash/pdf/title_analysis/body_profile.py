# Copyright (c) Opendatalab. All rights reserved.
"""统计全文和栏内正文排版基线。"""

from __future__ import annotations

import statistics

from ..geometry import _rotate_bbox_to_upright
from ..inline.types import PDF_FONT_FORCE_BOLD_FLAG, PDF_FONT_ITALIC_FLAG
from ..line_layout import (
    _estimate_lane_gap,
    _font_signatures_share_family,
    _line_canonical_style_scale,
    _line_effective_height,
)
from ..models import (
    _DocumentBodyProfile,
    _LaneBodyProfile,
    _LineItem,
    _PreparedPage,
    _TextLane,
)


def _infer_document_body_profile(
    prepared_pages: list[_PreparedPage],
    *,
    use_canonical_scale: bool = False,
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
            local_page_width = prepared.page_size[1] if line.angle in {90, 270} else prepared.page_size[0]
            height = (
                _line_canonical_style_scale(line, local_bbox)
                if use_canonical_scale
                else _line_effective_height(line, local_bbox)
            )
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
    cross_page_clusters = [cluster for cluster in height_clusters if len({item[1] for item in cluster}) >= 2]
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
        has_style_scale_repairs=any(
            line.style_scale_repaired for prepared in prepared_pages for line in prepared.remaining_lines
        ),
    )


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


__all__ = [
    "_infer_document_body_profile",
    "_document_font_is_regular",
    "_infer_lane_body_profile",
    "_line_uses_document_regular_font",
]
