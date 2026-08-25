# Copyright (c) Opendatalab. All rights reserved.
"""OCR 文本框排序、重叠判断与检测框几何变换。"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...types import BBox

LINE_WIDTH_TO_HEIGHT_RATIO_THRESHOLD = 4


def merge_spans_to_line(spans: list[dict[str, Any]], threshold: float = 0.6) -> list[list[dict[str, Any]]]:
    """按纵向重叠把 OCR span 聚合为文本行。"""
    if not spans:
        return []
    spans.sort(key=lambda span: span["bbox"][1])
    lines: list[list[dict[str, Any]]] = []
    current_line = [spans[0]]
    for span in spans[1:]:
        if _is_overlaps_y_exceeds_threshold(span["bbox"], current_line[-1]["bbox"], threshold):
            current_line.append(span)
        else:
            lines.append(current_line)
            current_line = [span]
    lines.append(current_line)
    return lines


def _is_overlaps_y_exceeds_threshold(
    bbox1: BBox,
    bbox2: BBox,
    overlap_ratio_threshold: float = 0.8,
) -> bool:
    """判断两个框在纵轴上的重叠是否超过较小高度的阈值。"""
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2
    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    min_height = min(y1_1 - y0_1, y1_2 - y0_2)
    return (overlap / min_height) > overlap_ratio_threshold if min_height > 0 else False


def _is_overlaps_x_exceeds_threshold(
    bbox1: BBox,
    bbox2: BBox,
    overlap_ratio_threshold: float = 0.8,
) -> bool:
    """判断两个框在横轴上的重叠是否超过较小宽度的阈值。"""
    x0_1, _, x1_1, _ = bbox1
    x0_2, _, x1_2, _ = bbox2
    overlap = max(0, min(x1_1, x1_2) - max(x0_1, x0_2))
    min_width = min(x1_1 - x0_1, x1_2 - x0_2)
    return (overlap / min_width) > overlap_ratio_threshold if min_width > 0 else False


def sorted_boxes(dt_boxes: list[Any]) -> list[Any]:
    """按从上到下、同行从左到右稳定排序检测框。"""
    boxes = sorted(dt_boxes, key=lambda box: (box[0][1], box[0][0]))
    for index in range(len(boxes) - 1):
        for previous_index in range(index, -1, -1):
            if abs(boxes[previous_index + 1][0][1] - boxes[previous_index][0][1]) < 10 and (
                boxes[previous_index + 1][0][0] < boxes[previous_index][0][0]
            ):
                boxes[previous_index], boxes[previous_index + 1] = boxes[previous_index + 1], boxes[previous_index]
            else:
                break
    return boxes


def bbox_to_points(bbox: BBox) -> np.ndarray:
    """将四元组 bbox 转换为顺时针四点数组。"""
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype("float32")


def points_to_bbox(points: np.ndarray) -> BBox:
    """将顺时针四点数组转换为四元组 bbox。"""
    x0, y0 = points[0]
    x1, _ = points[1]
    _, y1 = points[2]
    return x0, y0, x1, y1


def merge_intervals(intervals: list[list[float]]) -> list[list[float]]:
    """合并相互重叠的闭区间。"""
    intervals.sort(key=lambda interval: interval[0])
    merged: list[list[float]] = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def remove_intervals(original: list[float], masks: list[list[float]]) -> list[list[float]]:
    """从一个闭区间中移除全部遮罩区间。"""
    result: list[list[float]] = []
    original_start, original_end = original
    for mask_start, mask_end in merge_intervals(masks):
        if mask_start > original_end or mask_end < original_start:
            continue
        if original_start < mask_start:
            result.append([original_start, mask_start - 1])
        original_start = max(mask_end + 1, original_start)
    if original_start <= original_end:
        result.append([original_start, original_end])
    return result


def update_det_boxes(dt_boxes: list[Any], mfd_res: list[dict[str, Any]]) -> list[Any]:
    """从横向 OCR 检测框中裁除公式检测区间。"""
    new_dt_boxes: list[Any] = []
    angle_boxes: list[Any] = []
    for text_box in dt_boxes:
        if calculate_is_angle(text_box):
            angle_boxes.append(text_box)
            continue
        text_bbox = points_to_bbox(text_box)
        masks = [
            [mf_box["bbox"][0], mf_box["bbox"][2]]
            for mf_box in mfd_res
            if _is_overlaps_y_exceeds_threshold(text_bbox, mf_box["bbox"])
        ]
        for remaining in remove_intervals([text_bbox[0], text_bbox[2]], masks):
            new_dt_boxes.append(bbox_to_points((remaining[0], text_bbox[1], remaining[1], text_bbox[3])))
    new_dt_boxes.extend(angle_boxes)
    return new_dt_boxes


def merge_overlapping_spans(spans: list[BBox]) -> list[BBox]:
    """合并同一行中横向重叠的 bbox。"""
    if not spans:
        return []
    spans.sort(key=lambda span: span[0])
    merged: list[BBox] = []
    for span in spans:
        x1, y1, x2, y2 = span
        if not merged or merged[-1][2] < x1:
            merged.append(span)
            continue
        previous = merged.pop()
        merged.append((min(previous[0], x1), min(previous[1], y1), max(previous[2], x2), max(previous[3], y2)))
    return merged


def merge_det_boxes(dt_boxes: list[Any]) -> list[Any]:
    """把同一横向文本行中的 OCR 检测框合并为较大区域。"""
    horizontal_boxes: list[dict[str, Any]] = []
    angle_boxes: list[Any] = []
    for text_box in dt_boxes:
        text_bbox = points_to_bbox(text_box)
        if calculate_is_angle(text_box):
            angle_boxes.append(text_box)
        else:
            horizontal_boxes.append({"bbox": text_bbox})

    new_dt_boxes: list[Any] = []
    for line in merge_spans_to_line(horizontal_boxes):
        line_bboxes = [span["bbox"] for span in line]
        min_x = min(bbox[0] for bbox in line_bboxes)
        max_x = max(bbox[2] for bbox in line_bboxes)
        min_y = min(bbox[1] for bbox in line_bboxes)
        max_y = max(bbox[3] for bbox in line_bboxes)
        if max_x - min_x > (max_y - min_y) * LINE_WIDTH_TO_HEIGHT_RATIO_THRESHOLD:
            line_bboxes = merge_overlapping_spans(line_bboxes)
        new_dt_boxes.extend(bbox_to_points(bbox) for bbox in line_bboxes)
    new_dt_boxes.extend(angle_boxes)
    return new_dt_boxes


def calculate_is_angle(poly: list[Any]) -> bool:
    """判断四点检测框是否明显偏离水平矩形。"""
    p1, p2, p3, p4 = poly
    height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
    return not 0.8 * height <= (p3[1] - p1[1]) <= 1.2 * height


def is_bbox_aligned_rect(points: np.ndarray) -> bool:
    """判断四点数组是否为轴对齐矩形。"""
    return len(np.unique(points[:, 0])) == 2 and len(np.unique(points[:, 1])) == 2


__all__ = [
    "_is_overlaps_x_exceeds_threshold",
    "_is_overlaps_y_exceeds_threshold",
    "bbox_to_points",
    "calculate_is_angle",
    "is_bbox_aligned_rect",
    "merge_det_boxes",
    "merge_intervals",
    "merge_overlapping_spans",
    "merge_spans_to_line",
    "points_to_bbox",
    "remove_intervals",
    "sorted_boxes",
    "update_det_boxes",
]
