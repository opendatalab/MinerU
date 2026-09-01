# Copyright (c) Opendatalab. All rights reserved.

"""Flash 原生 PDF 提取使用的纯几何工具。"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence


from ....types import BBox

from .models import (
    _AxisLine,
    _LocalAxisLine,
)
def _horizontal_bbox_gap(first_bbox: BBox, second_bbox: BBox) -> float:
    """返回两个局部 bbox 在 x 轴上的无方向净空，重叠时为零。"""

    return max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)


def _transform_axis_lines(
    lines: list[_AxisLine],
    page_size: tuple[float, float],
    angle: int,
) -> list[_LocalAxisLine]:
    """将原页面横竖线转入当前文本方向的局部坐标。"""

    output: list[_LocalAxisLine] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        orientation: Literal["horizontal", "vertical"] = (
            "horizontal" if local_bbox[2] - local_bbox[0] >= local_bbox[3] - local_bbox[1] else "vertical"
        )
        output.append(
            _LocalAxisLine(
                bbox=local_bbox,
                original_bbox=line.bbox,
                orientation=orientation,
                width=line.width,
            )
        )
    return output


def _bbox_overlap_in_first(first: BBox, second: BBox) -> float:
    """计算交集面积占第一个 bbox 面积的比例。"""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = _bbox_area(first)
    return width * height / first_area if first_area > 0 else 0.0


def _bbox_area(bbox: BBox) -> float:
    """返回合法 bbox 的非负面积。"""

    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _normalize_bbox_to_unit(
    bbox: BBox,
    page_size: tuple[float, float],
) -> list[float]:
    """将绝对 bbox 归一化到 0-1 单位区间，并保证舍入后的宽高至少各占一个刻度。"""

    page_width, page_height = page_size
    ticks = [
        max(0, min(1000, int(round(bbox[0] / page_width * 1000)))),
        max(0, min(1000, int(round(bbox[1] / page_height * 1000)))),
        max(0, min(1000, int(round(bbox[2] / page_width * 1000)))),
        max(0, min(1000, int(round(bbox[3] / page_height * 1000)))),
    ]
    if ticks[2] <= ticks[0]:
        if ticks[0] < 1000:
            ticks[2] = ticks[0] + 1
        else:
            ticks[0] = max(0, ticks[2] - 1)
    if ticks[3] <= ticks[1]:
        if ticks[1] < 1000:
            ticks[3] = ticks[1] + 1
        else:
            ticks[1] = max(0, ticks[3] - 1)
    return [tick / 1000 for tick in ticks]


def _rotate_bbox_to_upright(
    bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
) -> BBox:
    """将页面 bbox 转到当前文本方向的正向局部坐标。"""

    page_width, page_height = page_size
    x0, y0, x1, y1 = bbox
    if angle == 270:
        return (page_height - y1, x0, page_height - y0, x1)
    if angle == 90:
        return (y0, page_width - x1, y1, page_width - x0)
    if angle == 180:
        return (page_width - x1, page_height - y1, page_width - x0, page_height - y0)
    return bbox


def _rotate_bbox_from_upright(
    bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
) -> BBox:
    """将正向局部 bbox 逆变换回 PDF 页面坐标。"""

    page_width, page_height = page_size
    x0, y0, x1, y1 = bbox
    if angle == 270:
        return (y0, page_height - x1, y1, page_height - x0)
    if angle == 90:
        return (page_width - y1, x0, page_width - y0, x1)
    if angle == 180:
        return (page_width - x1, page_height - y1, page_width - x0, page_height - y0)
    return bbox


def _coerce_bbox(value: Any) -> BBox | None:
    """将任意四元 bbox 规范成非退化浮点坐标。"""

    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _clip_bbox(
    bbox: BBox | None,
    page_size: tuple[float, float],
) -> BBox | None:
    """将 bbox 裁剪到页面范围，退化框返回 None。"""

    if bbox is None:
        return None
    page_width, page_height = page_size
    return _coerce_bbox(
        (
            max(0.0, min(page_width, bbox[0])),
            max(0.0, min(page_height, bbox[1])),
            max(0.0, min(page_width, bbox[2])),
            max(0.0, min(page_height, bbox[3])),
        )
    )


def _bbox_union(first: BBox, second: BBox) -> BBox:
    """返回两个 bbox 的外接并集框。"""

    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _bbox_union_many(bboxes: Sequence[BBox]) -> BBox:
    """返回非空 bbox 序列的外接并集框。"""

    if not bboxes:
        raise ValueError("bbox sequence must not be empty")
    result = bboxes[0]
    for bbox in bboxes[1:]:
        result = _bbox_union(result, bbox)
    return result


def _expand_bbox(bbox: BBox, margin: float) -> BBox:
    """向四周扩展 bbox，仅供几何容差判定使用。"""

    return (
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin,
    )


def _bbox_center_x(bbox: BBox) -> float:
    """返回 bbox 的水平中心。"""

    return (bbox[0] + bbox[2]) / 2.0


def _bbox_center_y(bbox: BBox) -> float:
    """返回 bbox 的垂直中心。"""

    return (bbox[1] + bbox[3]) / 2.0


def _bbox_intersects(first: BBox, second: BBox) -> bool:
    """检查两个 bbox 是否存在正面积交叠。"""

    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])


def _bbox_distance(first: BBox, second: BBox) -> float:
    """返回两个 bbox 的欧氏净空距离，交叠或相接时为零。"""

    horizontal_gap = max(first[0] - second[2], second[0] - first[2], 0.0)
    vertical_gap = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(horizontal_gap, vertical_gap)


def _bbox_overlap_in_smaller(first: BBox, second: BBox) -> float:
    """计算交集面积占较小 bbox 面积的比例。"""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _bbox_axis_overlap_ratio(
    first: BBox,
    second: BBox,
    *,
    axis: Literal["x", "y"],
) -> float:
    """计算指定轴上的交叠长度占较短轴长的比例。"""

    if axis == "x":
        first_start, first_end = first[0], first[2]
        second_start, second_end = second[0], second[2]
    else:
        first_start, first_end = first[1], first[3]
        second_start, second_end = second[1], second[3]
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    shorter = min(first_end - first_start, second_end - second_start)
    return overlap / shorter if shorter > 0 else 0.0


def _point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    """检查点是否位于 bbox 内部或边界上。"""

    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
