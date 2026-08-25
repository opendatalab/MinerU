# Copyright (c) Opendatalab. All rights reserved.
import math

import numpy as np

from ..types import BBox, IntBBox


def normalize_to_int_bbox(
    box: list[list[float]] | list[float] | BBox | None,
    image_size: tuple[int, int] | None = None,
) -> IntBBox | None:
    if box is None:
        return None

    arr = np.asarray(box, dtype=np.float64)
    if arr.size == 0:
        return None

    if arr.ndim == 2 and arr.shape[-1] == 2:
        xs = arr[:, 0]
        ys = arr[:, 1]
        xmin = float(np.min(xs))
        ymin = float(np.min(ys))
        xmax = float(np.max(xs))
        ymax = float(np.max(ys))
    else:
        flat = arr.reshape(-1)
        if flat.size == 4:
            xmin, ymin, xmax, ymax = [float(v) for v in flat]
        elif flat.size >= 8:
            xs = flat[0::2]
            ys = flat[1::2]
            xmin = float(np.min(xs))
            ymin = float(np.min(ys))
            xmax = float(np.max(xs))
            ymax = float(np.max(ys))
        else:
            return None

    xmin = math.floor(xmin)
    ymin = math.floor(ymin)
    xmax = math.ceil(xmax)
    ymax = math.ceil(ymax)

    if image_size is not None:
        height, width = image_size
        xmin = max(0, min(int(width), xmin))
        ymin = max(0, min(int(height), ymin))
        xmax = max(0, min(int(width), xmax))
        ymax = max(0, min(int(height), ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    return (int(xmin), int(ymin), int(xmax), int(ymax))


def bbox_relative_pos(bbox1: BBox, bbox2: BBox) -> tuple[bool, bool, bool, bool]:
    """返回 bbox1 相对 bbox2 的左、右、下、上位置关系。"""
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    return left, right, bottom, top


def bbox_distance(bbox1: BBox, bbox2: BBox) -> float:
    """计算两个不相交矩形之间的最短欧氏距离。"""
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2
    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)
    if top and left:
        return math.hypot(x1 - x2b, y1b - y2)
    if left and bottom:
        return math.hypot(x1 - x2b, y1 - y2b)
    if bottom and right:
        return math.hypot(x1b - x2, y1 - y2b)
    if right and top:
        return math.hypot(x1b - x2, y1b - y2)
    if left:
        return x1 - x2b
    if right:
        return x2 - x1b
    if bottom:
        return y1 - y2b
    if top:
        return y2 - y1b
    return 0.0


def bbox_center_distance(bbox1: BBox, bbox2: BBox) -> float:
    """计算两个矩形中心点之间的欧氏距离。"""
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    return math.hypot(center1[0] - center2[0], center1[1] - center2[1])


def calculate_overlap_area_2_minbox_area_ratio(bbox1: BBox, bbox2: BBox) -> float:
    """计算交集面积占两个矩形较小面积的比例。"""
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    overlap = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0.0, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    area2 = max(0.0, (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
    min_area = min(area1, area2)
    return overlap / min_area if min_area > 0 else 0.0


def calculate_overlap_area_in_bbox1_area_ratio(bbox1: BBox, bbox2: BBox) -> float:
    """计算两个矩形交集面积占 bbox1 面积的比例。"""
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    overlap = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0.0, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    return overlap / area1 if area1 > 0 else 0.0
