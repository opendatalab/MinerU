# Copyright (c) Opendatalab. All rights reserved.
"""OFD 毫米坐标、仿射矩阵与 bbox 工具。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ....types import BBox
from .constants import MM_TO_POINTS

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class Affine:
    """保存 OFD 六参数仿射矩阵。"""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def apply(self, point: Point) -> Point:
        """把一个局部点变换到目标坐标空间。"""
        x, y = point
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)

    def compose(self, inner: Affine) -> Affine:
        """返回先执行 inner、再执行当前矩阵的组合结果。"""
        return Affine(
            a=self.a * inner.a + self.c * inner.b,
            b=self.b * inner.a + self.d * inner.b,
            c=self.a * inner.c + self.c * inner.d,
            d=self.b * inner.c + self.d * inner.d,
            e=self.a * inner.e + self.c * inner.f + self.e,
            f=self.b * inner.e + self.d * inner.f + self.f,
        )

    @classmethod
    def translation(cls, x: float, y: float) -> Affine:
        """构造只包含平移的矩阵。"""
        return cls(e=x, f=y)

    @classmethod
    def rotation(cls, angle: float) -> Affine:
        """构造绕局部原点旋转指定角度的矩阵。"""
        radians = math.radians(angle)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        return cls(a=cosine, b=sine, c=-sine, d=cosine)


def parse_numbers(value: str | None, *, expected: int | None = None) -> tuple[float, ...] | None:
    """把空白分隔的有限数值解析为元组。"""
    if not value:
        return None
    try:
        numbers = tuple(float(item) for item in value.replace(",", " ").split())
    except (TypeError, ValueError):
        return None
    if expected is not None and len(numbers) != expected:
        return None
    if not all(math.isfinite(item) for item in numbers):
        return None
    return numbers


def parse_st_box(value: str | None) -> BBox | None:
    """把 OFD 的 x/y/width/height 转换为 x0/y0/x1/y1。"""
    numbers = parse_numbers(value, expected=4)
    if numbers is None:
        return None
    x, y, width, height = numbers
    if width <= 0 or height <= 0:
        return None
    return (x, y, x + width, y + height)


def parse_affine(value: str | None) -> Affine:
    """解析可选 CTM，缺失或非法时返回单位矩阵。"""
    numbers = parse_numbers(value, expected=6)
    return Affine(*numbers) if numbers is not None else Affine()


def rect_quad(bbox: BBox) -> Quad:
    """把轴对齐矩形转换为顺时针四点。"""
    x0, y0, x1, y1 = bbox
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def quad_bbox(points: Iterable[Point]) -> BBox | None:
    """计算一组有限点的轴对齐外接框。"""
    materialized = list(points)
    if not materialized or not all(math.isfinite(value) for point in materialized for value in point):
        return None
    xs = [point[0] for point in materialized]
    ys = [point[1] for point in materialized]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def transform_quad(quad: Quad, transform: Affine) -> Quad:
    """把四点按给定仿射矩阵变换。"""
    return tuple(transform.apply(point) for point in quad)  # type: ignore[return-value]


def transform_bbox(bbox: BBox, transform: Affine) -> BBox | None:
    """变换矩形四角并返回目标空间 AABB。"""
    return quad_bbox(transform_quad(rect_quad(bbox), transform))


def bbox_union(bboxes: Iterable[BBox]) -> BBox | None:
    """合并全部有效 bbox。"""
    materialized = list(bboxes)
    if not materialized:
        return None
    return (
        min(item[0] for item in materialized),
        min(item[1] for item in materialized),
        max(item[2] for item in materialized),
        max(item[3] for item in materialized),
    )


def bbox_intersection(first: BBox, second: BBox) -> BBox | None:
    """返回两个 bbox 的非退化交集。"""
    bbox = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def bbox_center(bbox: BBox) -> Point:
    """返回 bbox 中心点。"""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox: BBox) -> float:
    """返回非负 bbox 面积。"""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_overlap_ratio(first: BBox, second: BBox) -> float:
    """返回交集相对较小 bbox 的面积比例。"""
    intersection = bbox_intersection(first, second)
    smaller = min(bbox_area(first), bbox_area(second))
    return bbox_area(intersection) / smaller if intersection is not None and smaller > 0 else 0.0


def transform_angle(transform: Affine) -> float:
    """返回局部 X 轴经过矩阵后的页面角度。"""
    return math.degrees(math.atan2(transform.b, transform.a)) % 360.0


def canonical_angle(angle: float, *, tolerance: float = 5.0) -> int:
    """把接近直角的角度收敛为 0/90/180/270。"""
    normalized = angle % 360.0
    nearest = min((0, 90, 180, 270, 360), key=lambda item: abs(normalized - item))
    if abs(normalized - nearest) <= tolerance:
        return nearest % 360
    return int(round(normalized)) % 360


def bbox_to_points(bbox: BBox) -> list[float]:
    """把毫米 bbox 转换为供共享 XYCut 使用的 point 坐标。"""
    return [value * MM_TO_POINTS for value in bbox]


def normalize_bbox(bbox: BBox, physical_box: BBox) -> list[float] | None:
    """把页面 bbox 裁剪并归一化到 PhysicalBox。"""
    clipped = bbox_intersection(bbox, physical_box)
    if clipped is None:
        return None
    width = physical_box[2] - physical_box[0]
    height = physical_box[3] - physical_box[1]
    if width <= 0 or height <= 0:
        return None
    values = [
        round((clipped[0] - physical_box[0]) / width, 3),
        round((clipped[1] - physical_box[1]) / height, 3),
        round((clipped[2] - physical_box[0]) / width, 3),
        round((clipped[3] - physical_box[1]) / height, 3),
    ]
    values = [max(0.0, min(1.0, value)) for value in values]
    return values if values[2] > values[0] and values[3] > values[1] else None


__all__ = [
    "Affine",
    "Point",
    "Quad",
    "bbox_area",
    "bbox_center",
    "bbox_intersection",
    "bbox_overlap_ratio",
    "bbox_to_points",
    "bbox_union",
    "canonical_angle",
    "normalize_bbox",
    "parse_affine",
    "parse_numbers",
    "parse_st_box",
    "quad_bbox",
    "rect_quad",
    "transform_angle",
    "transform_bbox",
    "transform_quad",
]
