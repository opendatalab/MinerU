# Copyright (c) Opendatalab. All rights reserved.
"""提供标题分类共享的几何与文本规则。"""

from __future__ import annotations

import re

from .....types import BBox
from ..geometry import _bbox_axis_overlap_ratio, _bbox_center_x, _bbox_center_y
from ..line_layout import (
    _effective_text_row_gap,
)
from ..models import (
    _LineItem,
)

_NUMBERED_SECTION_TITLE_RE = re.compile(
    r"^(?P<number>\d+(?:\s*\.\s*\d+)*)\s+(?P<label>\S.*)$",
)


_SECTION_NUMBER_ONLY_RE = re.compile(
    r"^\d+(?:\s*\.\s*\d+)*\.?$",
)


_SECTION_TITLE_TERMINAL_RE = re.compile(
    r"[.!?。！？:：;；,，]$",
)


_UNNUMBERED_SECTION_HEADING_RE = re.compile(
    r"^(?:introduction|references?|bibliography|acknowledg(?:e)?ments?|引言|绪论|参考文献|参考资料)$",
    re.IGNORECASE,
)


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
            if line.visual_row_id is not None and other_line.visual_row_id == line.visual_row_id:
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


def _line_inside_visual_container(
    line_bbox: BBox,
    container_bboxes: list[BBox],
) -> bool:
    """检查文本行中心是否落入视觉容器，容器内标签不得借标题原型晋升。"""

    center_x = _bbox_center_x(line_bbox)
    center_y = _bbox_center_y(line_bbox)
    return any(bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3] for bbox in container_bboxes)


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


__all__ = [
    "_NUMBERED_SECTION_TITLE_RE",
    "_SECTION_NUMBER_ONLY_RE",
    "_SECTION_TITLE_TERMINAL_RE",
    "_UNNUMBERED_SECTION_HEADING_RE",
    "_build_physical_title_gap_map",
    "_line_inside_visual_container",
    "_line_near_visual_container",
]
