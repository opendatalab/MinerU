# Copyright (c) Opendatalab. All rights reserved.
"""跨页表格结构检测和内容合并的稳定入口。"""

from .content import merge_table_content
from .document import merge_table
from .html import (
    build_row_rendered_cell_segments,
    build_table_state_from_html,
    calculate_row_rendered_segments,
)
from .structure import can_merge_by_structure, detect_table_headers

__all__ = [
    "merge_table",
    "merge_table_content",
    "build_table_state_from_html",
    "build_row_rendered_cell_segments",
    "can_merge_by_structure",
    "calculate_row_rendered_segments",
    "detect_table_headers",
]
