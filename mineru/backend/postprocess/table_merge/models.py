# Copyright (c) Opendatalab. All rights reserved.
"""跨页表格合并使用的内部状态模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

MAX_HEADER_ROWS = 5

BlockDict: TypeAlias = dict[str, Any]
PageInfoDict: TypeAlias = dict[str, Any]
CalculationBBox: TypeAlias = tuple[int, int, int, int]


@dataclass
class RowMetrics:
    """记录单行的有效列、实际列和视觉列指标。"""

    row_idx: int
    effective_cols: int
    actual_cols: int
    visual_cols: int


@dataclass
class RowSignature:
    """记录表头行的列结构与规范化文本签名。"""

    effective_cols: int
    colspans: tuple[int, ...]
    rowspans: tuple[int, ...]
    normalized_texts: tuple[str, ...]
    display_texts: tuple[str, ...]

    @property
    def cell_count(self) -> int:
        """返回签名中的显式单元格数量。"""
        return len(self.colspans)


@dataclass
class RenderedCellSegment:
    """记录一个渲染单元格覆盖的视觉列区间。"""

    text: str
    start_col: int
    end_col: int


@dataclass
class RowScanResult:
    """封装一次 HTML 行扫描得到的列指标与跨行占位。"""

    row_effective_cols: list[int]
    row_metrics: list[RowMetrics]
    total_cols: int
    last_nonempty_row_metrics: RowMetrics | None
    tail_occupied: dict[int, set[int]]


@dataclass
class TableMergeState:
    """缓存单张表格的 block 所有者、HTML 树和结构指标。"""

    owner_block: BlockDict | None
    body_block: BlockDict | None
    soup: Any
    tbody: Any
    rows: list[Any]
    total_cols: int
    front_header_info: list[RowSignature]
    front_first_data_row_metrics: dict[int, RowMetrics]
    last_data_row_metrics: RowMetrics | None
    row_effective_cols: list[int]
    tail_occupied: dict[int, set[int]]
    dirty: bool = False
