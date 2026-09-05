# Copyright (c) Opendatalab. All rights reserved.
"""共享稀疏表格的坐标聚类和局部物理规则转换，保留两路候选策略。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .contracts import NativeTableInput
from .geometry import normalize_angle, normalize_bbox, page_bbox_to_table_local


@dataclass(frozen=True, slots=True)
class _LocalRule:
    """保存转换到正向表格坐标后的细线区间。"""

    orientation: str
    coordinate: float
    start: float
    end: float
    width: float


def cluster_members(
    values: list[float],
    tolerance: float,
) -> list[tuple[float, tuple[float, ...]]]:
    """按一维距离聚类坐标并保留每簇原始成员。"""

    if not values:
        return []
    clusters: list[list[float]] = [[value] for value in sorted(values)[:1]]
    for value in sorted(values)[1:]:
        center = float(statistics.median(clusters[-1]))
        if abs(value - center) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [(float(statistics.median(cluster)), tuple(cluster)) for cluster in clusters]


def _local_rules(
    table_input: NativeTableInput,
    width: float,
    height: float,
) -> tuple[_LocalRule, ...]:
    """把页面 drawing 线转换为按局部长轴重新判向的规则区间。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return ()
    angle = normalize_angle(table_input.angle)
    output: list[_LocalRule] = []
    for rule in table_input.drawing_lines:
        bbox = normalize_bbox(rule.bbox)
        if bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(bbox, table_bbox, angle)
        if local_bbox is None:
            continue
        local_width = local_bbox[2] - local_bbox[0]
        local_height = local_bbox[3] - local_bbox[1]
        if local_width >= max(1.0, 3.0 * max(local_height, 0.1)):
            output.append(
                _LocalRule(
                    orientation="horizontal",
                    coordinate=(local_bbox[1] + local_bbox[3]) / 2.0,
                    start=max(0.0, local_bbox[0]),
                    end=min(width, local_bbox[2]),
                    width=max(rule.width, local_height),
                )
            )
        elif local_height >= max(1.0, 3.0 * max(local_width, 0.1)):
            output.append(
                _LocalRule(
                    orientation="vertical",
                    coordinate=(local_bbox[0] + local_bbox[2]) / 2.0,
                    start=max(0.0, local_bbox[1]),
                    end=min(height, local_bbox[3]),
                    width=max(rule.width, local_width),
                )
            )
    return tuple(output)


__all__ = ["_LocalRule", "cluster_members", "_local_rules"]
