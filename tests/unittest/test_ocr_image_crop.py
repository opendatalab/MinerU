"""退化文本检测框的旋转裁剪：窄框不再抛错导致整本失败，而是返回 None 供调用方跳过该行。"""

from __future__ import annotations

import numpy as np
import pytest

from mineru.model.ocr.image import get_rotate_crop_image, get_rotate_crop_image_for_text_rec


def _image() -> np.ndarray:
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.mark.parametrize(
    "points",
    [
        # 宽取整后为 0 的窄框（issue 报告的场景）。
        np.array([[10, 10], [10.4, 10], [10.4, 60], [10, 60]], dtype=np.float32),
        # 高取整后为 0 的窄框。
        np.array([[10, 10], [60, 10], [60, 10.4], [10, 10.4]], dtype=np.float32),
        # 轴对齐零宽框：直裁为空后不再落入透视分支抛错。
        np.array([[10, 10], [10, 10], [10, 60], [10, 60]], dtype=np.float32),
        # 轴对齐零高框。
        np.array([[10, 10], [60, 10], [60, 10], [10, 10]], dtype=np.float32),
    ],
)
def test_degenerate_quad_returns_none(points: np.ndarray) -> None:
    """退化四边形返回 None，而不是触发 Unsupported warp settings。"""
    image = _image()
    assert get_rotate_crop_image(image, points) is None
    assert get_rotate_crop_image_for_text_rec(image, points) is None


def test_normal_quad_still_crops() -> None:
    """正常四边形不受退化守卫影响。"""
    points = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=np.float32)
    crop = get_rotate_crop_image(_image(), points)
    assert crop is not None
    assert crop.shape == (50, 50, 3)


def test_one_pixel_quad_still_crops() -> None:
    """宽恰好为 1 像素的框仍正常裁剪，守卫只拦取整后为 0 的情况。"""
    points = np.array([[10, 10], [11, 10], [11, 60], [10, 60]], dtype=np.float32)
    crop = get_rotate_crop_image(_image(), points)
    assert crop is not None
    assert crop.shape == (50, 1, 3)
