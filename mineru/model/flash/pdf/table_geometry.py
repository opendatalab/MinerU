# Copyright (c) Opendatalab. All rights reserved.
"""提供表格恢复与文本投影共享的局部几何，不改变各业务层校验策略。"""

from __future__ import annotations

from ....types import BBox


def normalize_bbox(value: object) -> BBox | None:
    """把任意四元组规范为有效浮点 bbox，异常或退化框返回空。"""

    try:
        x0, y0, x1, y1 = [float(item) for item in value]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def rotate_local_bbox(
    bbox: BBox,
    width: float,
    height: float,
    angle: int,
) -> BBox:
    """把表格裁剪框内 bbox 转换到正向表格局部坐标。"""

    x0, y0, x1, y1 = bbox
    if angle == 270:
        return height - y1, x0, height - y0, x1
    if angle == 90:
        return y0, width - x1, y1, width - x0
    if angle == 180:
        return width - x1, height - y1, width - x0, height - y0
    return bbox


__all__ = ["normalize_bbox", "rotate_local_bbox"]
