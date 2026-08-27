# Copyright (c) Opendatalab. All rights reserved.
"""工作表投影阶段使用的中立内部数据模型。"""

from __future__ import annotations

from typing import Annotated, Any, TypeAlias

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt
from pydantic.dataclasses import dataclass


CellPosition: TypeAlias = tuple[int, int]
OptionalCellPosition: TypeAlias = tuple[int | None, int | None]
AnchoredBlock: TypeAlias = tuple[CellPosition, int, dict[str, Any]]
FormulaMap: TypeAlias = dict[CellPosition, list[str]]


@dataclass
class DataRegion:
    """表示工作表中非空单元格的 1-based 边界矩形区域。"""

    min_row: Annotated[PositiveInt, Field(description="Smallest row index (1-based index).")]
    max_row: Annotated[PositiveInt, Field(description="Largest row index (1-based index).")]
    min_col: Annotated[PositiveInt, Field(description="Smallest column index (1-based index).")]
    max_col: Annotated[PositiveInt, Field(description="Largest column index (1-based index).")]

    def width(self) -> PositiveInt:
        """返回数据区域的列数。"""
        return self.max_col - self.min_col + 1

    def height(self) -> PositiveInt:
        """返回数据区域的行数。"""
        return self.max_row - self.min_row + 1


class ExcelCell(BaseModel):
    """表示已经完成文本、媒体与公式物化的工作表单元格。"""

    row: int
    col: int
    text: str
    row_span: int
    col_span: int
    styles: dict[str, Any] = Field(default_factory=dict)
    media: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    text_is_html: bool = False
    source_row: int | None = None
    source_col: int | None = None


class ExcelTable(BaseModel):
    """表示具有显示坐标和源工作表锚点的矩形表格。"""

    anchor: tuple[NonNegativeInt, NonNegativeInt]
    num_rows: int
    num_cols: int
    data: list[ExcelCell]


@dataclass(frozen=True, slots=True)
class SheetImage:
    """表示绑定到工作表 cell anchor 的图片或图片公式。"""

    anchor: OptionalCellPosition
    image_base64: str | None = None
    latex: str | None = None
    order: int = 0


__all__ = [
    "AnchoredBlock",
    "CellPosition",
    "DataRegion",
    "ExcelCell",
    "ExcelTable",
    "FormulaMap",
    "OptionalCellPosition",
    "SheetImage",
]
