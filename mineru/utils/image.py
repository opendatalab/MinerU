# Copyright (c) Opendatalab. All rights reserved.
"""跨模型共享的轻量图像统计与裁剪原语。"""

import cv2
import numpy as np
from PIL import Image

from ..types import BBox
from .geometry import normalize_to_int_bbox


def calculate_contrast(img: np.ndarray, img_mode: str) -> float:
    """
    计算给定图像的对比度。
    :param img: 图像，类型为numpy.ndarray
    :Param img_mode = 图像的色彩通道，'rgb' 或 'bgr'
    :return: 图像的对比度值
    """
    if img_mode == "rgb":
        # 将RGB图像转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_mode == "bgr":
        # 将BGR图像转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image mode. Please provide 'rgb' or 'bgr'.")

    # 计算均值和标准差
    mean_value = np.mean(gray_img)
    std_dev = np.std(gray_img)
    # 对比度定义为标准差除以平均值（加上小常数避免除零错误）
    contrast = std_dev / (mean_value + 1e-6)
    # logger.debug(f"contrast: {contrast}")
    return round(float(contrast), 2)


def crop_pil_image(bbox: BBox, image: Image.Image, scale: float = 2.0) -> Image.Image:
    """按缩放后的 bbox 裁剪 Pillow 图像。"""
    scaled_bbox = normalize_to_int_bbox([float(value) * scale for value in bbox])
    if scaled_bbox is None:
        return image.crop((0, 0, 0, 0))
    return image.crop(tuple(scaled_bbox))


__all__ = ["calculate_contrast", "crop_pil_image"]
