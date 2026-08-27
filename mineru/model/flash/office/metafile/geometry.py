# Copyright (c) Opendatalab. All rights reserved.
"""WMF/EMF 跨后端复用的路径与几何计算。"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, pi, sin
from typing import Iterable

from .limits import MAX_FLATTENED_POINTS
from .models import GraphicsPath, Matrix, MetafileResourceLimitError, PathSegment, Point, Rect


_ELLIPSE_KAPPA = 0.5522847498307936
_DEFAULT_FLATNESS = 0.25
_MAX_CUBIC_DEPTH = 12


@dataclass(slots=True)
class FlattenBudget:
    """累计单次文档渲染产生的折线点数并执行固定预算。"""

    limit: int = MAX_FLATTENED_POINTS
    used: int = 0

    def charge(self, count: int = 1) -> None:
        """累计离散点数，超过固定预算时终止渲染。"""
        self.used += count
        if self.used > self.limit:
            raise MetafileResourceLimitError(f"metafile exceeds max_flattened_points={self.limit}")


class PathBuilder:
    """以可变列表构造最终不可变的 GraphicsPath。"""

    def __init__(self) -> None:
        """创建空路径并清除当前点与子路径起点。"""
        self._segments: list[PathSegment] = []
        self.current: Point | None = None
        self.subpath_start: Point | None = None
        self._figure_closed = False

    def move_to(self, point: Point) -> None:
        """移动当前点并开始新的子路径。"""
        self._segments.append(PathSegment("M", (point,)))
        self.current = point
        self.subpath_start = point
        self._figure_closed = False

    def line_to(self, point: Point) -> None:
        """从当前点追加直线；空路径会先建立起点。"""
        if self.current is None:
            self.move_to(point)
            return
        if self._figure_closed:
            self.move_to(self.current)
        self._segments.append(PathSegment("L", (point,)))
        self.current = point

    def cubic_to(self, control1: Point, control2: Point, endpoint: Point) -> None:
        """从当前点追加三次贝塞尔曲线。"""
        if self.current is None:
            self.move_to(endpoint)
            return
        if self._figure_closed:
            self.move_to(self.current)
        self._segments.append(PathSegment("C", (control1, control2, endpoint)))
        self.current = endpoint

    def close(self) -> None:
        """闭合当前子路径并把当前点恢复到其起点。"""
        if self.current is None or self.subpath_start is None or self._figure_closed:
            return
        self._segments.append(PathSegment("Z"))
        self.current = self.subpath_start
        self._figure_closed = True

    @property
    def figure_open(self) -> bool:
        """返回当前是否存在可由 CloseFigure 闭合的开放 figure。"""
        return self.current is not None and self.subpath_start is not None and not self._figure_closed

    def extend(self, path: GraphicsPath) -> None:
        """按顺序追加另一条路径的全部片段。"""
        for segment in path.segments:
            if segment.verb == "M":
                self.move_to(segment.points[0])
            elif segment.verb == "L":
                self.line_to(segment.points[0])
            elif segment.verb == "C":
                self.cubic_to(*segment.points)
            elif segment.verb == "Z":
                self.close()

    def build(self) -> GraphicsPath:
        """冻结并返回当前路径片段。"""
        return GraphicsPath(tuple(self._segments))

    def clear(self) -> None:
        """清空全部路径片段与当前点。"""
        self._segments.clear()
        self.current = None
        self.subpath_start = None
        self._figure_closed = False


def colorref_to_rgb(value: int) -> tuple[int, int, int]:
    """把 GDI COLORREF 的 BGR 字节序转换为 RGB。"""
    return value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF


def transform_path(path: GraphicsPath, matrix: Matrix) -> GraphicsPath:
    """把路径中的所有控制点应用到同一仿射变换。"""
    segments: list[PathSegment] = []
    for segment in path.segments:
        segments.append(PathSegment(segment.verb, tuple(matrix.transform_point(point) for point in segment.points)))
    return GraphicsPath(tuple(segments))


def rectangle_path(rect: Rect) -> GraphicsPath:
    """把矩形转换为闭合路径。"""
    normalized = rect.normalized()
    builder = PathBuilder()
    builder.move_to((normalized.left, normalized.top))
    builder.line_to((normalized.right, normalized.top))
    builder.line_to((normalized.right, normalized.bottom))
    builder.line_to((normalized.left, normalized.bottom))
    builder.close()
    return builder.build()


def round_rectangle_path(rect: Rect, radius_x: float, radius_y: float) -> GraphicsPath:
    """把圆角矩形转换为包含三次曲线的闭合路径。"""
    normalized = rect.normalized()
    radius_x = min(abs(radius_x), abs(normalized.width) / 2.0)
    radius_y = min(abs(radius_y), abs(normalized.height) / 2.0)
    if radius_x <= 0.0 or radius_y <= 0.0:
        return rectangle_path(normalized)
    kx = radius_x * _ELLIPSE_KAPPA
    ky = radius_y * _ELLIPSE_KAPPA
    left, top, right, bottom = normalized.left, normalized.top, normalized.right, normalized.bottom
    builder = PathBuilder()
    builder.move_to((left + radius_x, top))
    builder.line_to((right - radius_x, top))
    builder.cubic_to((right - radius_x + kx, top), (right, top + radius_y - ky), (right, top + radius_y))
    builder.line_to((right, bottom - radius_y))
    builder.cubic_to((right, bottom - radius_y + ky), (right - radius_x + kx, bottom), (right - radius_x, bottom))
    builder.line_to((left + radius_x, bottom))
    builder.cubic_to((left + radius_x - kx, bottom), (left, bottom - radius_y + ky), (left, bottom - radius_y))
    builder.line_to((left, top + radius_y))
    builder.cubic_to((left, top + radius_y - ky), (left + radius_x - kx, top), (left + radius_x, top))
    builder.close()
    return builder.build()


def ellipse_path(rect: Rect) -> GraphicsPath:
    """把椭圆转换为四段三次贝塞尔曲线。"""
    normalized = rect.normalized()
    center_x = (normalized.left + normalized.right) / 2.0
    center_y = (normalized.top + normalized.bottom) / 2.0
    radius_x = normalized.width / 2.0
    radius_y = normalized.height / 2.0
    if radius_x <= 0.0 or radius_y <= 0.0:
        return GraphicsPath(())
    kx = radius_x * _ELLIPSE_KAPPA
    ky = radius_y * _ELLIPSE_KAPPA
    builder = PathBuilder()
    builder.move_to((center_x + radius_x, center_y))
    builder.cubic_to(
        (center_x + radius_x, center_y + ky),
        (center_x + kx, center_y + radius_y),
        (center_x, center_y + radius_y),
    )
    builder.cubic_to(
        (center_x - kx, center_y + radius_y),
        (center_x - radius_x, center_y + ky),
        (center_x - radius_x, center_y),
    )
    builder.cubic_to(
        (center_x - radius_x, center_y - ky),
        (center_x - kx, center_y - radius_y),
        (center_x, center_y - radius_y),
    )
    builder.cubic_to(
        (center_x + kx, center_y - radius_y),
        (center_x + radius_x, center_y - ky),
        (center_x + radius_x, center_y),
    )
    builder.close()
    return builder.build()


def arc_path(
    rect: Rect,
    start: Point,
    end: Point,
    *,
    direction: int,
    close_mode: str = "open",
) -> GraphicsPath:
    """按 GDI 椭圆边界与方向生成 arc、pie 或 chord 路径。"""
    normalized = rect.normalized()
    center_x = (normalized.left + normalized.right) / 2.0
    center_y = (normalized.top + normalized.bottom) / 2.0
    radius_x = normalized.width / 2.0
    radius_y = normalized.height / 2.0
    if radius_x <= 0.0 or radius_y <= 0.0:
        return GraphicsPath(())
    start_angle = atan2((start[1] - center_y) / radius_y, (start[0] - center_x) / radius_x)
    end_angle = atan2((end[1] - center_y) / radius_y, (end[0] - center_x) / radius_x)
    if direction == 1:
        while end_angle >= start_angle:
            end_angle -= 2.0 * pi
    else:
        while end_angle <= start_angle:
            end_angle += 2.0 * pi
    sweep = end_angle - start_angle
    steps = max(4, min(256, int(ceil(abs(sweep) / (pi / 24.0)))))
    points = [
        (
            center_x + radius_x * cos(start_angle + sweep * index / steps),
            center_y + radius_y * sin(start_angle + sweep * index / steps),
        )
        for index in range(steps + 1)
    ]
    builder = PathBuilder()
    if close_mode == "pie":
        builder.move_to((center_x, center_y))
        builder.line_to(points[0])
    else:
        builder.move_to(points[0])
    for point in points[1:]:
        builder.line_to(point)
    if close_mode in {"pie", "chord"}:
        builder.close()
    return builder.build()


def path_bounds(path: GraphicsPath) -> Rect | None:
    """返回包含路径全部端点和控制点的保守包围盒。"""
    points = [point for segment in path.segments for point in segment.points]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return Rect(min(xs), min(ys), max(xs), max(ys))


def close_open_subpaths(path: GraphicsPath) -> GraphicsPath:
    """为 StrokeAndFillPath 显式闭合路径中的全部开放 figure。"""
    segments: list[PathSegment] = []
    figure_open = False
    for segment in path.segments:
        if segment.verb == "M":
            if figure_open:
                segments.append(PathSegment("Z"))
            figure_open = True
        elif segment.verb == "Z":
            figure_open = False
        segments.append(segment)
    if figure_open:
        segments.append(PathSegment("Z"))
    return GraphicsPath(tuple(segments))


def union_rectangles(rectangles: Iterable[Rect | None]) -> Rect | None:
    """合并所有非空矩形并返回整体包围盒。"""
    present = [rectangle.normalized() for rectangle in rectangles if rectangle is not None]
    if not present:
        return None
    return Rect(
        min(rectangle.left for rectangle in present),
        min(rectangle.top for rectangle in present),
        max(rectangle.right for rectangle in present),
        max(rectangle.bottom for rectangle in present),
    )


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    """返回点到无限直线的距离，退化直线按点距处理。"""
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = hypot(delta_x, delta_y)
    if length <= 1e-12:
        return hypot(point[0] - start[0], point[1] - start[1])
    return abs(delta_y * point[0] - delta_x * point[1] + end[0] * start[1] - end[1] * start[0]) / length


def _split_cubic(
    start: Point,
    control1: Point,
    control2: Point,
    endpoint: Point,
) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    """按 de Casteljau 中点算法把三次贝塞尔曲线二分。"""
    first = ((start[0] + control1[0]) / 2.0, (start[1] + control1[1]) / 2.0)
    second = ((control1[0] + control2[0]) / 2.0, (control1[1] + control2[1]) / 2.0)
    third = ((control2[0] + endpoint[0]) / 2.0, (control2[1] + endpoint[1]) / 2.0)
    fourth = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    fifth = ((second[0] + third[0]) / 2.0, (second[1] + third[1]) / 2.0)
    midpoint = ((fourth[0] + fifth[0]) / 2.0, (fourth[1] + fifth[1]) / 2.0)
    return (start, first, fourth, midpoint), (midpoint, fifth, third, endpoint)


def _flatten_cubic(
    start: Point,
    control1: Point,
    control2: Point,
    endpoint: Point,
    *,
    flatness: float,
    max_depth: int,
    budget: FlattenBudget,
) -> list[Point]:
    """按输出空间误差自适应离散单条三次贝塞尔曲线。"""
    result: list[Point] = []
    stack: list[tuple[Point, Point, Point, Point, int]] = [(start, control1, control2, endpoint, 0)]
    while stack:
        current_start, current_control1, current_control2, current_endpoint, depth = stack.pop()
        error = max(
            _point_line_distance(current_control1, current_start, current_endpoint),
            _point_line_distance(current_control2, current_start, current_endpoint),
        )
        if error <= flatness or depth >= max_depth:
            budget.charge()
            result.append(current_endpoint)
            continue
        left, right = _split_cubic(current_start, current_control1, current_control2, current_endpoint)
        stack.append((*right, depth + 1))
        stack.append((*left, depth + 1))
    return result


def flatten_path(
    path: GraphicsPath,
    *,
    flatness: float = _DEFAULT_FLATNESS,
    max_depth: int = _MAX_CUBIC_DEPTH,
    budget: FlattenBudget | None = None,
) -> list[tuple[list[Point], bool]]:
    """把曲线路径按输出像素误差离散成 Pillow 可绘制的折线子路径。"""
    if flatness <= 0.0:
        raise ValueError("flatness must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must not be negative")
    active_budget = budget or FlattenBudget()
    subpaths: list[tuple[list[Point], bool]] = []
    current_points: list[Point] = []
    current_point: Point | None = None
    closed = False
    for segment in path.segments:
        if segment.verb == "M":
            if current_points:
                subpaths.append((current_points, closed))
            current_points = [segment.points[0]]
            active_budget.charge()
            current_point = segment.points[0]
            closed = False
        elif segment.verb == "L":
            if current_point is None:
                current_points = [segment.points[0]]
            else:
                current_points.append(segment.points[0])
            active_budget.charge()
            current_point = segment.points[0]
        elif segment.verb == "C" and current_point is not None:
            control1, control2, endpoint = segment.points
            current_points.extend(
                _flatten_cubic(
                    current_point,
                    control1,
                    control2,
                    endpoint,
                    flatness=flatness,
                    max_depth=max_depth,
                    budget=active_budget,
                )
            )
            current_point = endpoint
        elif segment.verb == "Z":
            closed = True
            if current_points and current_points[-1] != current_points[0]:
                current_points.append(current_points[0])
                active_budget.charge()
    if current_points:
        subpaths.append((current_points, closed))
    return subpaths


def vector_length(vector: Point) -> float:
    """返回二维向量的欧氏长度。"""
    return hypot(vector[0], vector[1])


__all__ = [
    "FlattenBudget",
    "PathBuilder",
    "arc_path",
    "close_open_subpaths",
    "colorref_to_rgb",
    "ellipse_path",
    "flatten_path",
    "path_bounds",
    "rectangle_path",
    "round_rectangle_path",
    "transform_path",
    "union_rectangles",
    "vector_length",
]
