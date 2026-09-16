"""验证工具归位后模型输入与推理平台策略保持原行为。"""

from __future__ import annotations

import numpy as np
import pytest
from docvortex.assets import image_size

from mineru.backend.analysis.pdf.model_inputs import (
    _bbox_to_pixel_bbox,
    _get_medium_table_virtual_image_bbox,
    _normalize_layout_bbox_to_unit,
    _normalize_medium_content,
    _sidecar_bbox_to_page_bbox,
)
from mineru.model.runtime import platform


@pytest.mark.parametrize(
    ("bbox", "pixels", "points"),
    [
        ((0.1, 0.2, 0.9, 0.8), (10, 40, 90, 160), (10, 40, 90, 160)),
        ((10, 40, 90, 160), (10, 40, 90, 160), (5, 20, 45, 80)),
        ((90, 160, 10, 40), (10, 40, 90, 160), (5, 20, 45, 80)),
        ((-10, -20, 300, 500), (-10, -20, 300, 500), (0, 0, 100, 200)),
    ],
)
def test_model_coordinate_interpretation(bbox: tuple, pixels: tuple, points: tuple) -> None:
    """归一化判断留在模型边界，像素和 sidecar 的解释不被混为一谈。"""
    assert _bbox_to_pixel_bbox(bbox, (100, 200)) == pytest.approx(pixels)
    assert _sidecar_bbox_to_page_bbox(bbox, (100, 200), 2) == pytest.approx(points)


def test_model_precision_and_virtual_token() -> None:
    """保持模型三位小数精度、token 尺寸与图像边界裁剪。"""
    assert _normalize_layout_bbox_to_unit((10, 20, 90, 100), (300, 600)) == [0.033, 0.033, 0.3, 0.167]
    assert _get_medium_table_virtual_image_bbox((0, 0, 2, 2), (100, 200)) == (0, 0, 6, 6)
    assert _get_medium_table_virtual_image_bbox((20, 30, 40, 50), (100, 200)) == (25, 35, 35, 45)
    assert _bbox_to_pixel_bbox((1, 1, 1, 2), (100, 200)) is None
    assert _sidecar_bbox_to_page_bbox((1, 1, 2, 2), (100, 200), 0) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  x  ", "x"),
        ([" a ", "", "b"], " a \nb"),
        (None, ""),
        (42, ""),
    ],
)
def test_medium_model_text(value: object, expected: str) -> None:
    """保留字符串与列表输出的不同清洗约定。"""
    assert _normalize_medium_content(value) == expected


def test_numpy_image_dimensions() -> None:
    """图像尺寸读取返回宽高，不把 NumPy size 标量作为图像宽高。"""
    assert image_size(np.zeros((20, 30, 3), dtype=np.uint8)) == (30, 20)


@pytest.mark.parametrize(
    ("system", "machine", "version", "supported"),
    [
        ("Darwin", "arm64", "14.0", True),
        ("Darwin", "arm64", "13.5", False),
        ("Darwin", "x86_64", "14.0", False),
        ("Linux", "aarch64", "14.0", False),
        ("Windows", "AMD64", "", False),
    ],
)
def test_inference_platform_threshold(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    version: str,
    supported: bool,
) -> None:
    """平台门槛由宿主显式给定，保留操作系统与 CPU 双重约束。"""
    monkeypatch.setattr(platform.platform, "system", lambda: system)
    monkeypatch.setattr(platform.platform, "machine", lambda: machine)
    monkeypatch.setattr(platform.platform, "mac_ver", lambda: (version, (), ""))
    assert platform.is_mac_os_version_supported("14.0") is supported
    assert platform.is_windows_environment() is (system == "Windows")
    assert platform.is_linux_environment() is (system == "Linux")
