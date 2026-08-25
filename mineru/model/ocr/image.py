# Copyright (c) Opendatalab. All rights reserved.
"""OCR 输入图像解码、遮罩与旋转裁剪。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...utils.geometry import normalize_to_int_bbox
from .geometry import is_bbox_aligned_rect

TEXT_REC_ROTATE_RATIO = 1.5


def img_decode(content: bytes) -> Any:
    """把编码图片字节解码为 OpenCV 数组。"""
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def check_img(img: bytes | np.ndarray) -> Any:
    """统一字节与灰度图输入为 OCR 可处理的图像数组。"""
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def alpha_to_color(img: np.ndarray, alpha_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """把带透明通道的图像合成到指定纯色背景。"""
    if len(img.shape) == 3 and img.shape[2] == 4:
        blue, green, red, alpha_channel = cv2.split(img)
        alpha = alpha_channel / 255
        red = (alpha_color[0] * (1 - alpha) + red * alpha).astype(np.uint8)
        green = (alpha_color[1] * (1 - alpha) + green * alpha).astype(np.uint8)
        blue = (alpha_color[2] * (1 - alpha) + blue * alpha).astype(np.uint8)
        img = cv2.merge((blue, green, red))
    return img


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """将 OCR 输入图像统一合成到白色背景。"""
    return alpha_to_color(image, (255, 255, 255))


def mask_formula_regions_for_ocr_det(
    bgr_image: np.ndarray,
    mask_boxes: list[dict[str, Any]] | None,
) -> np.ndarray:
    """将公式区域涂白后再做 OCR 检测，避免公式框干扰文本检测。"""
    if not mask_boxes:
        return bgr_image

    masked_image = bgr_image.copy()
    image_h, image_w = masked_image.shape[:2]
    for mask_box in mask_boxes:
        bbox = mask_box.get("bbox")
        if bbox is None:
            continue
        int_bbox = normalize_to_int_bbox(bbox, image_size=(image_h, image_w))
        if int_bbox is None:
            continue
        x0, y0, x1, y1 = int_bbox
        masked_image[y0:y1, x0:x1] = 255
    return masked_image


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """按四边形透视裁剪文本区域，并自动转正高纵横比结果。"""
    assert len(points) == 4, "shape of points must be 4*2"
    if is_bbox_aligned_rect(points):
        xmin = int(np.min(points[:, 0]))
        xmax = int(np.max(points[:, 0]))
        ymin = int(np.min(points[:, 1]))
        ymax = int(np.max(points[:, 1]))
        new_img = img[ymin:ymax, xmin:xmax].copy()
        if new_img.shape[0] > 0 and new_img.shape[1] > 0:
            return new_img

    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        matrix,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[:2]
    if dst_img_height / dst_img_width >= TEXT_REC_ROTATE_RATIO:
        dst_img = np.rot90(dst_img)
    return dst_img


def rotate_vertical_crop_if_needed(
    crop_img: np.ndarray | None,
    rotate_ratio: float = TEXT_REC_ROTATE_RATIO,
) -> np.ndarray | None:
    """在裁图明显竖长时旋转九十度供文字识别。"""
    if crop_img is None or crop_img.size == 0:
        return crop_img
    crop_height, crop_width = crop_img.shape[:2]
    if crop_width == 0:
        return crop_img
    if crop_height / crop_width >= rotate_ratio:
        return np.rot90(crop_img)
    return crop_img


def get_rotate_crop_image_for_text_rec(img: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """裁剪四边形文本区域并应用文字识别方向规则。"""
    return rotate_vertical_crop_if_needed(get_rotate_crop_image(img, points))


__all__ = [
    "TEXT_REC_ROTATE_RATIO",
    "alpha_to_color",
    "check_img",
    "get_rotate_crop_image",
    "get_rotate_crop_image_for_text_rec",
    "img_decode",
    "mask_formula_regions_for_ocr_det",
    "preprocess_image",
    "rotate_vertical_crop_if_needed",
]
