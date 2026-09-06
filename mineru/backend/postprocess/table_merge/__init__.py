# Copyright (c) Opendatalab. All rights reserved.
"""共享跨页表格内容操作的稳定入口。"""

from docvortex.content.table import (
    merge_table,
    merge_table_content,
    build_table_state_from_html,
    build_row_rendered_cell_segments,
    can_merge_by_structure,
    calculate_row_rendered_segments,
    detect_table_headers,
)

__all__ = [
    "merge_table",
    "merge_table_content",
    "build_table_state_from_html",
    "build_row_rendered_cell_segments",
    "can_merge_by_structure",
    "calculate_row_rendered_segments",
    "detect_table_headers",
]
