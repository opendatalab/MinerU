# Copyright (c) Opendatalab. All rights reserved.
"""PDF 页面坐标、方向和裁图使用的无状态几何原语。"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from loguru import logger

from ....types import BBox
from ....utils.geometry import normalize_to_int_bbox


def _normalize_page_size(page_image: Any) -> tuple[int, int]:
    """从PIL或numpy图像中读取页面宽高，供归一化bbox还原为像素bbox。"""
    if hasattr(page_image, "size"):
        return page_image.size

    height, width = page_image.shape[:2]
    return width, height


def _bbox_to_pixel_bbox(bbox: BBox | None, page_size: tuple[int, int]) -> BBox | None:
    """将归一化或像素bbox统一成像素bbox，异常bbox返回None。"""
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    width, height = page_size
    if all(0.0 <= value <= 1.0 for value in [x0, y0, x1, y1]):
        x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height

    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


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


def _medium_bbox_to_quad(bbox: list[float] | tuple[float, ...]) -> np.ndarray:
    """将普通 bbox 转为表格模型 OCR token 使用的四点框。"""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _normalize_medium_content(value: Any) -> str:
    """将 medium 本地模型输出的文本字段规范成 Hybrid block 可消费的字符串。"""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return value.strip()
    return ""


def _table_bbox_center(bbox: BBox) -> tuple[float, float]:
    """计算 bbox 中心点，用于判断图片或公式应归属哪个表格。"""
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def _normalize_visual_block_angle(angle: Any) -> int:
    """规范视觉块角度为 0/90/180/270，无法识别的角度按 0 处理。"""
    try:
        normalized_angle = int(float(angle or 0)) % 360
    except (TypeError, ValueError):
        logger.warning(f"Unsupported visual block angle: {angle}, using 0")
        return 0
    if normalized_angle not in {0, 90, 180, 270}:
        logger.warning(f"Unsupported visual block angle: {angle}, using 0")
        return 0
    return normalized_angle


def _rotate_visual_block_image_to_upright(image: np.ndarray, angle: int) -> np.ndarray:
    """按 layout 视觉块角度把裁图旋转至正向，角度语义与方向分类模型保持一致。"""
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def _rotate_medium_table_bbox(
    bbox: BBox,
    image_width: float,
    image_height: float,
    angle: int,
) -> BBox:
    """把原表格裁图中的 bbox 同步转换到旋转后裁图坐标系。"""
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if angle == 270:
        # 顺时针旋转 90 度后，新 x 轴来自原 y 轴的反方向。
        return (image_height - y1, x0, image_height - y0, x1)
    if angle == 90:
        # 逆时针旋转 90 度后，新 y 轴来自原 x 轴的反方向。
        return (y0, image_width - x1, y1, image_width - x0)
    if angle == 180:
        return (image_width - x1, image_height - y1, image_width - x0, image_height - y0)
    return (x0, y0, x1, y1)


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


def _encode_page_crop_as_jpeg_data_uri(
    np_image: np.ndarray,
    page_bbox: BBox,
    angle: int,
) -> str:
    """从页面原图按像素框裁剪，按视觉块方向回正后编码为 JPEG data URI。"""
    image_h, image_w = np_image.shape[:2]
    image_bbox = normalize_to_int_bbox(page_bbox, image_size=(image_h, image_w))
    if image_bbox is None:
        return ""
    x0, y0, x1, y1 = image_bbox
    crop_rgb = np_image[y0:y1, x0:x1].copy()
    if crop_rgb.size == 0:
        return ""

    crop_rgb = _rotate_visual_block_image_to_upright(crop_rgb, angle)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", crop_bgr)
    if not success:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def _sidecar_bbox_to_page_bbox(
    bbox: BBox | None,
    page_size: tuple[float, float],
    render_scale: float,
) -> BBox | None:
    """将公式或 OCR sidecar bbox 转为 PDF point 坐标，供原生字符匹配和组行。"""
    if bbox is None or len(bbox) != 4 or render_scale <= 0:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return None
    if all(0.0 <= value <= 1.0 for value in [x0, y0, x1, y1]):
        x0, y0, x1, y1 = x0 * page_width, y0 * page_height, x1 * page_width, y1 * page_height
    else:
        x0, y0, x1, y1 = (
            x0 / render_scale,
            y0 / render_scale,
            x1 / render_scale,
            y1 / render_scale,
        )

    left, right = sorted([max(0.0, min(page_width, x0)), max(0.0, min(page_width, x1))])
    top, bottom = sorted([max(0.0, min(page_height, y0)), max(0.0, min(page_height, y1))])
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)
