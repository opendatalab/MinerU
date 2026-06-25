# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations


def resolve_output_page_idx(physical_page_idx: int, page_index_map: list[int] | None = None) -> int:
    """根据物理页序解析输出页号，确保轻量解析路径不依赖 pipeline 工具模块。"""
    if page_index_map is None:
        return physical_page_idx
    if physical_page_idx < 0 or physical_page_idx >= len(page_index_map):
        raise ValueError(
            f"page_index_map does not cover physical page index {physical_page_idx}: "
            f"map length={len(page_index_map)}"
        )
    return page_index_map[physical_page_idx]
