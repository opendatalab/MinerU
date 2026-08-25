# Copyright (c) Opendatalab. All rights reserved.
"""Native PDF 表格结构恢复使用的局部坐标与聚类原语。"""

from __future__ import annotations

from collections.abc import Iterable

from .....types import BBox


def normalize_bbox(value: object) -> BBox | None:
    """把任意四元组规范为有效浮点 bbox，异常或退化框返回空。"""

    try:
        x0, y0, x1, y1 = [float(item) for item in value]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def normalize_angle(value: object) -> int:
    """把输入角度限制为表格流程支持的四个标准方向。"""

    try:
        angle = int(float(value or 0)) % 360
    except (TypeError, ValueError):
        return 0
    return angle if angle in {0, 90, 180, 270} else 0


def bbox_area(bbox: BBox) -> float:
    """返回 bbox 的非负面积。"""

    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0,
        float(bbox[3]) - float(bbox[1]),
    )


def bbox_union(bboxes: Iterable[BBox]) -> BBox:
    """返回一组有效 bbox 的最小外接框。"""

    items = list(bboxes)
    if not items:
        raise ValueError("bbox union requires at least one bbox")
    return (
        min(item[0] for item in items),
        min(item[1] for item in items),
        max(item[2] for item in items),
        max(item[3] for item in items),
    )


def bbox_intersection(first: BBox, second: BBox) -> BBox | None:
    """返回两个 bbox 的有效交集，无交集时返回空。"""

    intersection = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return intersection if intersection[2] > intersection[0] and intersection[3] > intersection[1] else None


def bbox_overlap_ratio(inner: BBox, outer: BBox) -> float:
    """返回 inner 面积被 outer 覆盖的比例。"""

    area = bbox_area(inner)
    intersection = bbox_intersection(inner, outer)
    if area <= 0 or intersection is None:
        return 0.0
    return min(1.0, bbox_area(intersection) / area)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    """返回 bbox 的中心坐标。"""

    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def rotate_local_bbox(
    bbox: BBox,
    width: float,
    height: float,
    angle: int,
) -> BBox:
    """把表格裁剪框内 bbox 转换到正向表格局部坐标。"""

    x0, y0, x1, y1 = bbox
    if angle == 270:
        return height - y1, x0, height - y0, x1
    if angle == 90:
        return y0, width - x1, y1, width - x0
    if angle == 180:
        return width - x1, height - y1, width - x0, height - y0
    return bbox


def page_bbox_to_table_local(
    bbox: BBox,
    table_bbox: BBox,
    angle: int,
) -> BBox | None:
    """裁剪页面 bbox 并转换到正向表格局部坐标。"""

    clipped = bbox_intersection(bbox, table_bbox)
    if clipped is None:
        return None
    width = table_bbox[2] - table_bbox[0]
    height = table_bbox[3] - table_bbox[1]
    relative = (
        clipped[0] - table_bbox[0],
        clipped[1] - table_bbox[1],
        clipped[2] - table_bbox[0],
        clipped[3] - table_bbox[1],
    )
    return rotate_local_bbox(relative, width, height, angle)


def table_local_size(table_bbox: BBox, angle: int) -> tuple[float, float]:
    """返回旋转到正向后的表格局部宽高。"""

    width = table_bbox[2] - table_bbox[0]
    height = table_bbox[3] - table_bbox[1]
    return (height, width) if angle in {90, 270} else (width, height)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """把浮点值限制在闭区间内。"""

    return max(minimum, min(maximum, value))


def cluster_positions(values: Iterable[float], tolerance: float) -> list[float]:
    """按相邻距离聚类一维坐标，并返回各簇均值。"""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return []
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        current_mean = sum(clusters[-1]) / len(clusters[-1])
        if abs(value - current_mean) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def covered_interval_ratio(
    intervals: Iterable[tuple[float, float]],
    start: float,
    end: float,
) -> float:
    """返回若干区间在目标区间上的并集覆盖比例。"""

    if end <= start:
        return 0.0
    clipped = sorted(
        (max(start, min(first, second)), min(end, max(first, second)))
        for first, second in intervals
        if min(end, max(first, second)) > max(start, min(first, second))
    )
    if not clipped:
        return 0.0
    covered = 0.0
    current_start, current_end = clipped[0]
    for item_start, item_end in clipped[1:]:
        if item_start <= current_end:
            current_end = max(current_end, item_end)
            continue
        covered += current_end - current_start
        current_start, current_end = item_start, item_end
    covered += current_end - current_start
    return min(1.0, covered / (end - start))


__all__ = [
    "bbox_area",
    "bbox_center",
    "bbox_intersection",
    "bbox_overlap_ratio",
    "bbox_union",
    "clamp",
    "cluster_positions",
    "covered_interval_ratio",
    "normalize_angle",
    "normalize_bbox",
    "page_bbox_to_table_local",
    "table_local_size",
]
