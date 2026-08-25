# Copyright (c) Opendatalab. All rights reserved.
"""面向已有 table bbox 的 Native PDF 表格结构恢复公共内部入口。"""

from .contracts import (
    NativeTableCell,
    NativeTableInput,
    NativeTableRectangle,
    NativeTableResult,
    NativeTableRule,
)
from .engine import (
    coerce_native_table_rectangles,
    coerce_native_table_rules,
    recover_native_pdf_table,
)

__all__ = [
    "NativeTableCell",
    "NativeTableInput",
    "NativeTableRectangle",
    "NativeTableResult",
    "NativeTableRule",
    "coerce_native_table_rectangles",
    "coerce_native_table_rules",
    "recover_native_pdf_table",
]
