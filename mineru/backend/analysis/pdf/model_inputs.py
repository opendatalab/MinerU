# Copyright (c) Opendatalab. All rights reserved.
"""模型输入坐标、文本字段及表格 OCR token 的适配规则。"""

from __future__ import annotations

from typing import Any

from docvortex.geometry import bbox_center as _table_bbox_center
from docvortex.geometry import convert_bbox

from ....types import BBox


def _bbox_to_pixel_bbox(bbox: BBox | None, page_size: tuple[int, int]) -> BBox | None:
    """在模型输入边界解释归一化或像素框，再调用共享的显式坐标转换。"""
    if bbox is None or len(bbox) != 4:
        return None
    try:
        coordinates = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    space = "unit" if all(0.0 <= value <= 1.0 for value in coordinates) else "pixel"
    return convert_bbox(coordinates, source_space=space, target_space="pixel", page_size=page_size)


def _normalize_layout_bbox_to_unit(bbox: BBox | None, page_size: tuple[int, int]) -> list[float] | None:
    """将 layout 像素 bbox 归一化为 VLM ContentBlock 需要的 0-1 坐标。"""
    pixel_bbox = _bbox_to_pixel_bbox(bbox, page_size)
    if pixel_bbox is None:
        return None

    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return None

    x0, y0, x1, y1 = pixel_bbox
    unit_bbox = [
        round(max(0.0, min(1.0, float(x0) / page_width)), 3),
        round(max(0.0, min(1.0, float(y0) / page_height)), 3),
        round(max(0.0, min(1.0, float(x1) / page_width)), 3),
        round(max(0.0, min(1.0, float(y1) / page_height)), 3),
    ]
    if unit_bbox[2] <= unit_bbox[0] or unit_bbox[3] <= unit_bbox[1]:
        return None
    return unit_bbox


def _normalize_medium_content(value: Any) -> str:
    """将 medium 本地模型输出的文本字段规范成 Hybrid block 可消费的字符串。"""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return value.strip()
    return ""


def _get_medium_table_virtual_image_bbox(
    bbox: BBox,
    image_size: tuple[int, int],
    box_size: float = 10.0,
) -> BBox:
    """在图片中心生成小 OCR token 框，避免图片大框干扰单元格匹配。"""
    image_width, image_height = image_size
    center_x, center_y = _table_bbox_center(bbox)
    half_size = box_size / 2.0
    return (
        max(0.0, center_x - half_size),
        max(0.0, center_y - half_size),
        min(float(image_width), center_x + half_size),
        min(float(image_height), center_y + half_size),
    )


def _sidecar_bbox_to_page_bbox(
    bbox: BBox | None,
    page_size: tuple[float, float],
    render_scale: float,
) -> BBox | None:
    """在 OCR/公式 sidecar 边界解释输入单位，并显式转换到裁剪后的页面 point。"""
    if bbox is None or len(bbox) != 4:
        return None
    try:
        coordinates = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    space = "unit" if all(0.0 <= value <= 1.0 for value in coordinates) else "pixel"
    return convert_bbox(
        coordinates, source_space=space, target_space="point", page_size=page_size, render_scale=render_scale, clip=True
    )
