# Copyright (c) Opendatalab. All rights reserved.
"""基于 OFD 原生横竖线与文字框恢复高置信全线表。"""

from __future__ import annotations

from dataclasses import dataclass

from ....types import BBox
from .geometry import bbox_area, bbox_center, bbox_intersection
from .models import AxisLine, TextLine
from .text import format_line_html

_COORD_TOLERANCE = 0.6
_INTERSECTION_TOLERANCE = 0.8
_MAX_GRID_CELLS = 10_000
_MAX_TABLE_AXIS_LINES = 2_000


@dataclass(frozen=True, slots=True)
class OfdTableRegion:
    """保存一个已物化 OFD 表格及其消耗的文字行。"""

    bbox: BBox
    html: str
    paint_order: int
    consumed_line_ids: frozenset[int]


def _line_coord(line: AxisLine) -> float:
    """返回轴向线段在垂直方向上的中心坐标。"""
    if line.orientation == "horizontal":
        return (line.bbox[1] + line.bbox[3]) / 2.0
    return (line.bbox[0] + line.bbox[2]) / 2.0


def _line_interval(line: AxisLine) -> tuple[float, float]:
    """返回轴向线段沿自身方向的区间。"""
    return (line.bbox[0], line.bbox[2]) if line.orientation == "horizontal" else (line.bbox[1], line.bbox[3])


def _touches(first: AxisLine, second: AxisLine) -> bool:
    """判断两条轴向线段是否相交或共线连接。"""
    if first.orientation == second.orientation:
        if abs(_line_coord(first) - _line_coord(second)) > _COORD_TOLERANCE:
            return False
        first_interval = _line_interval(first)
        second_interval = _line_interval(second)
        return min(first_interval[1], second_interval[1]) + _INTERSECTION_TOLERANCE >= max(
            first_interval[0], second_interval[0]
        )
    horizontal = first if first.orientation == "horizontal" else second
    vertical = second if first.orientation == "horizontal" else first
    x = _line_coord(vertical)
    y = _line_coord(horizontal)
    return (
        horizontal.bbox[0] - _INTERSECTION_TOLERANCE <= x <= horizontal.bbox[2] + _INTERSECTION_TOLERANCE
        and vertical.bbox[1] - _INTERSECTION_TOLERANCE <= y <= vertical.bbox[3] + _INTERSECTION_TOLERANCE
    )


def _components(lines: list[AxisLine]) -> list[list[AxisLine]]:
    """按线段相交关系构造确定性的连通分量。"""
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找并压缩一个并查集根节点。"""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        """合并两个线段分量。"""
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    for first_index, first in enumerate(lines):
        for second_index in range(first_index + 1, len(lines)):
            if _touches(first, lines[second_index]):
                union(first_index, second_index)
    grouped: dict[int, list[AxisLine]] = {}
    for index, line in enumerate(lines):
        grouped.setdefault(find(index), []).append(line)
    return list(grouped.values())


def _cluster_coordinates(values: list[float]) -> list[float]:
    """把近邻线坐标聚合为稳定网格轨道。"""
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or value - clusters[-1][-1] > _COORD_TOLERANCE:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _covers_interval(lines: list[AxisLine], start: float, end: float) -> bool:
    """判断共线片段是否基本覆盖给定区间。"""
    intervals = sorted(_line_interval(line) for line in lines)
    cursor = start
    for left, right in intervals:
        if right < cursor - _INTERSECTION_TOLERANCE:
            continue
        if left > cursor + _INTERSECTION_TOLERANCE:
            return False
        cursor = max(cursor, right)
        if cursor >= end - _INTERSECTION_TOLERANCE:
            return True
    return cursor >= end - _INTERSECTION_TOLERANCE


def _component_grid(component: list[AxisLine]) -> tuple[list[float], list[float], BBox] | None:
    """从线段分量恢复有完整外框的网格轨道。"""
    horizontal = [line for line in component if line.orientation == "horizontal"]
    vertical = [line for line in component if line.orientation == "vertical"]
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    xs = _cluster_coordinates([_line_coord(line) for line in vertical])
    ys = _cluster_coordinates([_line_coord(line) for line in horizontal])
    if len(xs) < 2 or len(ys) < 2 or (len(xs) - 1) * (len(ys) - 1) > _MAX_GRID_CELLS:
        return None
    left, right = xs[0], xs[-1]
    top, bottom = ys[0], ys[-1]
    outer_horizontal = [line for line in horizontal if abs(_line_coord(line) - top) <= _COORD_TOLERANCE]
    outer_horizontal += [line for line in horizontal if abs(_line_coord(line) - bottom) <= _COORD_TOLERANCE]
    outer_vertical = [line for line in vertical if abs(_line_coord(line) - left) <= _COORD_TOLERANCE]
    outer_vertical += [line for line in vertical if abs(_line_coord(line) - right) <= _COORD_TOLERANCE]
    top_ok = _covers_interval(
        [line for line in outer_horizontal if abs(_line_coord(line) - top) <= _COORD_TOLERANCE],
        left,
        right,
    )
    bottom_ok = _covers_interval(
        [line for line in outer_horizontal if abs(_line_coord(line) - bottom) <= _COORD_TOLERANCE], left, right
    )
    left_ok = _covers_interval(
        [line for line in outer_vertical if abs(_line_coord(line) - left) <= _COORD_TOLERANCE],
        top,
        bottom,
    )
    right_ok = _covers_interval(
        [line for line in outer_vertical if abs(_line_coord(line) - right) <= _COORD_TOLERANCE], top, bottom
    )
    if not (top_ok and bottom_ok and left_ok and right_ok):
        return None
    return xs, ys, (left, top, right, bottom)


def _cell_index(values: list[float], coordinate: float) -> int | None:
    """返回坐标所在的相邻轨道区间索引。"""
    for index, (start, end) in enumerate(zip(values, values[1:], strict=True)):
        if start - _COORD_TOLERANCE <= coordinate <= end + _COORD_TOLERANCE:
            return index
    return None


def _serialize_grid(xs: list[float], ys: list[float], lines: list[TextLine]) -> str:
    """把网格内文字按单元格序列化为安全 HTML。"""
    cells: dict[tuple[int, int], list[TextLine]] = {}
    for line in lines:
        center_x, center_y = bbox_center(line.bbox)
        column = _cell_index(xs, center_x)
        row = _cell_index(ys, center_y)
        if row is None or column is None:
            continue
        cells.setdefault((row, column), []).append(line)
    output = ["<table>"]
    for row in range(len(ys) - 1):
        output.append("<tr>")
        for column in range(len(xs) - 1):
            items = sorted(cells.get((row, column), []), key=lambda item: (item.bbox[1], item.bbox[0], item.paint_order))
            content = "<br>".join(format_line_html(item.text, item.styles) for item in items if item.text.strip())
            output.append(f"<td>{content}</td>")
        output.append("</tr>")
    output.append("</table>")
    return "".join(output)


def recover_tables(axis_lines: list[AxisLine], text_lines: list[TextLine]) -> list[OfdTableRegion]:
    """从页面轴向线段中恢复互不重叠的高置信表格。"""
    if len(axis_lines) < 4 or len(text_lines) < 2 or len(axis_lines) > _MAX_TABLE_AXIS_LINES:
        return []
    candidates: list[OfdTableRegion] = []
    for component in _components(axis_lines):
        grid = _component_grid(component)
        if grid is None:
            continue
        xs, ys, bbox = grid
        contained: list[TextLine] = []
        for line in text_lines:
            center_x, center_y = bbox_center(line.bbox)
            if (
                bbox_intersection(line.bbox, bbox) is not None
                and bbox[0] <= center_x <= bbox[2]
                and bbox[1] <= center_y <= bbox[3]
            ):
                contained.append(line)
        if len(contained) < 2:
            continue
        candidates.append(
            OfdTableRegion(
                bbox=bbox,
                html=_serialize_grid(xs, ys, contained),
                paint_order=min((line.paint_order for line in component), default=0),
                consumed_line_ids=frozenset(id(line) for line in contained),
            )
        )
    selected: list[OfdTableRegion] = []
    for candidate in sorted(candidates, key=lambda item: (-bbox_area(item.bbox), item.bbox[1], item.bbox[0])):
        if any(bbox_intersection(candidate.bbox, existing.bbox) is not None for existing in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item.bbox[1], item.bbox[0], item.paint_order))


__all__ = ["OfdTableRegion", "recover_tables"]
