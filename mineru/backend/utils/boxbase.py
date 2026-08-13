# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import math

from ...types import BBox


def bbox_relative_pos(bbox1: BBox, bbox2: BBox) -> tuple[bool, bool, bool, bool]:
    """判断两个矩形框的相对位置关系.

    Args:
        bbox1: 一个四元组，表示第一个矩形框的左上角和右下角的坐标，格式为(x1, y1, x1b, y1b)
        bbox2: 一个四元组，表示第二个矩形框的左上角和右下角的坐标，格式为(x2, y2, x2b, y2b)

    Returns:
        一个四元组，表示矩形框1相对于矩形框2的位置关系，格式为(left, right, bottom, top)
        其中，left表示矩形框1是否在矩形框2的左侧，right表示矩形框1是否在矩形框2的右侧，
        bottom表示矩形框1是否在矩形框2的下方，top表示矩形框1是否在矩形框2的上方
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    return left, right, bottom, top


def bbox_distance(bbox1: BBox, bbox2: BBox) -> float:
    """计算两个矩形框的距离。

    Args:
        bbox1 (tuple): 第一个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (tuple): 第二个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。

    Returns:
        float: 矩形框之间的距离。
    """

    def dist(point1: tuple[float, float], point2: tuple[float, float]) -> float:  # noqa: ANN202
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)

    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    return 0.0


def bbox_center_distance(bbox1: BBox, bbox2: BBox) -> float:
    """计算两个矩形框中心点之间的欧氏距离。

    Args:
        bbox1 (tuple): 第一个矩形框的坐标，格式为 (x1, y1, x2, y2)
        bbox2 (tuple): 第二个矩形框的坐标，格式为 (x1, y1, x2, y2)

    Returns:
        float: 两个矩形框中心点之间的距离
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    # 计算中心点
    center1_x = (x1 + x1b) / 2
    center1_y = (y1 + y1b) / 2
    center2_x = (x2 + x2b) / 2
    center2_y = (y2 + y2b) / 2

    # 计算欧氏距离
    return math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


def calculate_overlap_area_2_minbox_area_ratio(bbox1: BBox, bbox2: BBox) -> float:
    """计算box1和box2的重叠面积占最小面积的box的比例."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    min_box_area = min([(bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]), (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])])
    if min_box_area == 0:
        return 0
    else:
        return intersection_area / min_box_area


def calculate_overlap_area_in_bbox1_area_ratio(bbox1: BBox, bbox2: BBox) -> float:
    """计算box1和box2的重叠面积占bbox1的比例."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    if bbox1_area == 0:
        return 0
    else:
        return intersection_area / bbox1_area
