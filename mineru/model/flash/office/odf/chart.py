# Copyright (c) Opendatalab. All rights reserved.
"""恢复 ODF 嵌入图表的预览与源数据表。"""

from __future__ import annotations

from collections.abc import Callable

from lxml import etree  # type: ignore[reportMissingImports]

from .....types import BlockType
from .constants import qname
from .models import TableGrid
from .table import crop_table_grid, parse_cell_range_bounds, parse_table_grid, table_grid_to_html, union_bounds


def _chart_range_bounds(chart: etree._Element) -> tuple[int, int, int, int] | None:
    """收集 chart series、label 和 categories 的精确单元格引用范围。"""
    values: list[tuple[int, int, int, int]] = []
    attribute_names = {
        qname("chart", "values-cell-range-address"),
        qname("chart", "label-cell-address"),
        qname("table", "cell-range-address"),
    }
    for element in chart.iter():
        for attribute_name in attribute_names:
            if bounds := parse_cell_range_bounds(element.get(attribute_name, "")):
                values.append(bounds)
    return union_bounds(values)


def _single_nonempty_grid(grids: list[TableGrid]) -> TableGrid | None:
    """仅在对象内存在唯一非空表格时返回安全回退候选。"""
    nonempty = [grid for grid in grids if grid.rows]
    return nonempty[0] if len(nonempty) == 1 else None


def parse_chart_block(
    object_root: etree._Element,
    *,
    render_cell: Callable[[etree._Element], str],
    preview_data_uri: str | None,
) -> dict | None:
    """按精确引用优先、唯一表回退的规则构造图表 raw block。"""
    chart = next(object_root.iter(qname("chart", "chart")), None)
    if chart is None:
        return None
    grids = [parse_table_grid(table, render_cell) for table in object_root.iter(qname("table", "table"))]
    selected: TableGrid | None = None
    if grids and (bounds := _chart_range_bounds(chart)) is not None:
        selected = crop_table_grid(grids[0], bounds)
        if not selected.rows:
            selected = None
    if selected is None:
        selected = _single_nonempty_grid(grids)
    if selected is None:
        return None
    content = table_grid_to_html(selected)
    if not content:
        return None
    block: dict = {"type": BlockType.CHART, "content": content}
    if preview_data_uri:
        block["image_base64"] = preview_data_uri
    return block


__all__ = ["parse_chart_block"]
