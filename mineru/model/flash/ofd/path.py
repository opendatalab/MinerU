# Copyright (c) Opendatalab. All rights reserved.
"""解析 OFD PathObject 并提取表格可用的轴向线段。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from loguru import logger
from lxml import etree  # type: ignore[reportMissingImports]

from ....types import BBox
from .constants import MAX_PATH_COMMANDS
from .errors import OfdResourceLimitError
from .geometry import Affine, bbox_intersection, parse_affine, parse_st_box, transform_bbox
from .models import AxisLine
from .package import element_text, first_descendant, parse_int

_TOKEN_RE = re.compile(r"CM|[SMLQBAC]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(slots=True)
class OfdPathBudget:
    """累计限制紧缩路径命令数量。"""

    command_count: int = 0

    def charge(self, count: int) -> None:
        """累计本次路径命令并在超限时失败。"""
        self.command_count += count
        if self.command_count > MAX_PATH_COMMANDS:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_path_commands={MAX_PATH_COMMANDS}")


def _finite_float(value: str) -> float | None:
    """把路径 token 转换为有限浮点数。"""
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _tokenize(value: str) -> list[str]:
    """把紧缩路径文本拆成命令和数值 token。"""
    return _TOKEN_RE.findall(value)


def _line_bbox(first: tuple[float, float], second: tuple[float, float], width: float) -> tuple[BBox, str] | None:
    """把近似水平或垂直线段转换为非退化 bbox。"""
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    tolerance = max(0.2, 0.02 * length)
    thickness = max(width, 0.1)
    if abs(dy) <= tolerance:
        return (
            (
                min(first[0], second[0]),
                min(first[1], second[1]) - thickness / 2,
                max(first[0], second[0]),
                max(first[1], second[1]) + thickness / 2,
            ),
            "horizontal",
        )
    if abs(dx) <= tolerance:
        return (
            (
                min(first[0], second[0]) - thickness / 2,
                min(first[1], second[1]),
                max(first[0], second[0]) + thickness / 2,
                max(first[1], second[1]),
            ),
            "vertical",
        )
    return None


def _segments(value: str, transform: Affine, budget: OfdPathBudget) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """提取 S/M/L/CM 产生的直线段，曲线命令只推进当前位置。"""
    tokens = _tokenize(value)
    command_count = sum(token in {"S", "M", "L", "Q", "B", "A", "C", "CM"} for token in tokens)
    budget.charge(command_count)
    result: list[tuple[tuple[float, float], tuple[float, float]]] = []
    index = 0
    current: tuple[float, float] | None = None
    subpath_start: tuple[float, float] | None = None
    command: str | None = None
    arity = {"S": 2, "M": 2, "L": 2, "Q": 4, "B": 6, "A": 7, "CM": 2}
    while index < len(tokens):
        token = tokens[index]
        if token in {"S", "M", "L", "Q", "B", "A", "C", "CM"}:
            command = token
            index += 1
            if command == "C":
                if current is not None and subpath_start is not None and current != subpath_start:
                    result.append((transform.apply(current), transform.apply(subpath_start)))
                current = subpath_start
                continue
        if command is None or command == "C":
            return []
        needed = arity[command]
        if index + needed > len(tokens):
            break
        values = [_finite_float(item) for item in tokens[index : index + needed]]
        if any(item is None for item in values):
            index += needed
            continue
        numbers = [float(item) for item in values if item is not None]
        endpoint = (numbers[-2], numbers[-1])
        if command in {"S", "M"}:
            current = endpoint
            subpath_start = endpoint
        elif command in {"L", "CM"}:
            if current is not None:
                result.append((transform.apply(current), transform.apply(endpoint)))
            current = endpoint
        else:
            current = endpoint
        index += needed
    return result


def build_axis_lines(
    path_object: etree._Element,
    *,
    parent_transform: Affine,
    parent_clip: BBox,
    paint_order: int,
    template_id: int | None,
    budget: OfdPathBudget,
    resolved_style: dict[str, str] | None = None,
) -> list[AxisLine]:
    """从一个 PathObject 提取可见轴向线段。"""
    style = resolved_style or {}
    if (style.get("Visible") or path_object.get("Visible") or "true").casefold() in {"false", "0"}:
        return []
    if (style.get("Alpha") or path_object.get("Alpha") or "255").strip() == "0":
        return []
    boundary = parse_st_box(path_object.get("Boundary"))
    if boundary is None:
        return []
    boundary_page = transform_bbox(boundary, parent_transform)
    if boundary_page is None or bbox_intersection(boundary_page, parent_clip) is None:
        return []
    try:
        width = max(0.1, float(style.get("LineWidth") or path_object.get("LineWidth") or 0.353))
    except ValueError:
        width = 0.353
    object_transform = parent_transform.compose(Affine.translation(boundary[0], boundary[1])).compose(
        parse_affine(path_object.get("CTM"))
    )
    extracted: list[AxisLine] = []
    data = first_descendant(path_object, "AbbreviatedData")
    if data is not None:
        for first, second in _segments(element_text(data), object_transform, budget):
            line = _line_bbox(first, second, width)
            if line is None:
                continue
            line_bbox, orientation = line
            clipped = bbox_intersection(line_bbox, parent_clip)
            if clipped is not None:
                extracted.append(
                    AxisLine(
                        bbox=clipped,
                        orientation=orientation,
                        width=width,
                        paint_order=paint_order,
                        template_id=template_id,
                    )
                )
    boundary_width = boundary_page[2] - boundary_page[0]
    boundary_height = boundary_page[3] - boundary_page[1]
    if not extracted and max(boundary_width, boundary_height) >= 5 * max(min(boundary_width, boundary_height), 0.01):
        orientation = "horizontal" if boundary_width >= boundary_height else "vertical"
        clipped = bbox_intersection(boundary_page, parent_clip)
        if clipped is not None:
            extracted.append(
                AxisLine(
                    bbox=clipped,
                    orientation=orientation,
                    width=width,
                    paint_order=paint_order,
                    template_id=template_id,
                )
            )
    if not extracted and parse_int(path_object.get("ID")) is None:
        logger.debug("OFD_PATH_SKIPPED: path without stable ID produced no axis lines")
    return extracted


__all__ = ["OfdPathBudget", "build_axis_lines"]
