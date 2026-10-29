# Copyright (c) Opendatalab. All rights reserved.
"""OCR 输入图像解码、遮罩与旋转裁剪。"""

from __future__ import annotations
from docvortex.image import decode_image, perspective_matrix, resize_image, warp_image


from typing import Any

import numpy as np
from docvortex.geometry import normalize_to_int_bbox

from .geometry import is_bbox_aligned_rect

TEXT_REC_ROTATE_RATIO = 1.5


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """为模型输入交换 RGB/BGR 通道，忽略 alpha 并返回独立连续数组。"""
    if img.dtype not in (np.uint8, np.uint16, np.float32) or img.size == 0:
        raise ValueError("Model input expects a non-empty uint8, uint16 or float32 image")
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        return img[:, :, 2::-1].copy(order="C")
    raise ValueError(f"Unsupported model input image shape: {img.shape}")


def img_decode(content: bytes) -> Any:
    """通过共享图像接口解码字节，保留 BGR、alpha 和原始位深。"""

    return decode_image(content)


def check_img(img: bytes | np.ndarray) -> Any:
    """统一字节与灰度图输入为 OCR 可处理的图像数组。"""
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    return img


def alpha_to_color(img: np.ndarray, alpha_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """把带透明通道的图像合成到指定纯色背景。"""
    if len(img.shape) == 3 and img.shape[2] == 4:
        blue, green, red, alpha_channel = (img[:, :, index] for index in range(4))
        alpha = alpha_channel / 255
        red = (alpha_color[0] * (1 - alpha) + red * alpha).astype(np.uint8)
        green = (alpha_color[1] * (1 - alpha) + green * alpha).astype(np.uint8)
        blue = (alpha_color[2] * (1 - alpha) + blue * alpha).astype(np.uint8)
        img = np.stack((blue, green, red), axis=2)
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


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """裁剪四边形，退化框返回 None，透视裁图保留既有的高纵横比转正规则。"""
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    # 在轴对齐快捷路径之前过滤，避免跨整数边界的亚像素框被切成一列或一行像素。
    if img_crop_width < 1 or img_crop_height < 1:
        return None
    if is_bbox_aligned_rect(points):
        xmin = int(np.min(points[:, 0]))
        xmax = int(np.max(points[:, 0]))
        ymin = int(np.min(points[:, 1]))
        ymax = int(np.max(points[:, 1]))
        new_img = img[ymin:ymax, xmin:xmax].copy()
        if new_img.shape[0] > 0 and new_img.shape[1] > 0:
            return new_img

    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    # 四边形裁图调用共享数值内核，轴对齐区域继续直接复制。

    matrix = perspective_matrix(points, pts_std)
    dst_img = warp_image(img, matrix, (img_crop_width, img_crop_height), border="replicate", interpolation="cubic")
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


def resize_text_recognition_image(
    img: np.ndarray,
    max_wh_ratio: float,
    image_shape: tuple[int, int, int],
    min_width: int = 16,
    max_width: int = 2560,
) -> np.ndarray:
    """共用 OCR 识别缩放、宽度边界和归一化，保持现有 Torch 浮点计算顺序。"""
    channels, height, width = image_shape
    if img.ndim != 3 or img.shape[2] != channels or min(img.shape[:2]) <= 0:
        raise ValueError("OCR recognition expects a non-empty HWC image")
    ratio = max(max_wh_ratio, width / height)
    width = max(min(int(height * ratio), max_width), min_width)
    resized_width = min(width, max(int(np.ceil(height * img.shape[1] / img.shape[0])), min_width))

    resized = resize_image(img, (resized_width, height)) / 127.5 - 1
    padded = np.zeros((channels, height, width), dtype=np.float32)
    padded[:, :, :resized_width] = resized.transpose(2, 0, 1)
    return padded


def get_rotate_crop_image_for_text_rec(img: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """复用基础裁图的退化检查，并对有效裁片应用文字识别方向规则。"""
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
    "resize_text_recognition_image",
    "rgb_to_bgr",
    "rotate_vertical_crop_if_needed",
]
