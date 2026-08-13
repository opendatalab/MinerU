# Copyright (c) Opendatalab. All rights reserved.
"""raw model-list 到严格 PageInfo 列表的唯一转换边界。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import TypeAdapter

from mineru.backend.postprocess.lists import fix_office_paragraph_titles
from mineru.backend.postprocess.page_blocks import process_page_blocks
from mineru.backend.postprocess.paragraphs import merge_para_text_blocks
from mineru.backend.postprocess.table_merge import merge_table
from mineru.types import PageInfo

PAGE_INFO_LIST_ADAPTER = TypeAdapter(list[PageInfo])


def _validate_page_index_map(
    page_count: int,
    page_index_map: list[int] | None,
) -> list[int]:
    """校验实际页号映射，禁止 zip 截断、重复页号或逆序页号。"""
    if page_index_map is None:
        return list(range(page_count))
    if len(page_index_map) != page_count:
        raise ValueError(f"page_index_map length mismatch: pages={page_count}, mapping={len(page_index_map)}")
    if any(isinstance(page_idx, bool) or not isinstance(page_idx, int) or page_idx < 0 for page_idx in page_index_map):
        raise ValueError("page_index_map values must be non-negative integers")
    if len(page_index_map) != len(set(page_index_map)):
        raise ValueError("page_index_map values must be unique")
    if any(current <= previous for previous, current in zip(page_index_map, page_index_map[1:])):
        raise ValueError("page_index_map values must preserve strictly increasing order")
    return list(page_index_map)


def _document_uses_bbox(model_list: list[list[dict[str, Any]]]) -> bool:
    """按整份文档是否出现 bbox 判定 PDF/Office，避免空白首页误判。"""
    return any(
        block.get("bbox") is not None for page_model_list in model_list for block in page_model_list if isinstance(block, dict)
    )


def _remove_private_block_metadata(block: dict[str, Any]) -> None:
    """递归清除对象化边界之前仅供 Analyze 计算使用的临时字段。"""
    for field_name in ("lines", "_lines", "angle", "score", "label"):
        block.pop(field_name, None)
    content = block.get("content")
    if isinstance(content, list):
        for child in content:
            if isinstance(child, dict):
                _remove_private_block_metadata(child)


def _blocks_to_raw_page_info(
    page_model_list: list[dict[str, Any]],
    *,
    page_idx: int,
    use_bbox: bool,
) -> dict[str, Any]:
    """运行单页后处理流水线并保留 raw dict，供跨页处理继续消费。"""
    page_blocks = process_page_blocks(page_model_list, use_bbox=use_bbox)
    page_blocks.sort(key=lambda block: block["index"])
    return {"page_idx": page_idx, "blocks": page_blocks}


def blocks_to_page_info(
    page_model_list: list[dict[str, Any]],
    *,
    page_idx: int = 0,
    use_bbox: bool | None = None,
) -> PageInfo:
    """无副作用地把单页 raw blocks 转换为严格 PageInfo 对象。"""
    copied_blocks = deepcopy(page_model_list)
    resolved_use_bbox = _document_uses_bbox([copied_blocks]) if use_bbox is None else use_bbox
    if not resolved_use_bbox:
        fix_office_paragraph_titles([copied_blocks])
    raw_page = _blocks_to_raw_page_info(
        copied_blocks,
        page_idx=page_idx,
        use_bbox=resolved_use_bbox,
    )
    for block in raw_page["blocks"]:
        _remove_private_block_metadata(block)
    return PageInfo.model_validate(raw_page)


def model_list_to_pages(
    model_list: list[list[dict[str, Any]]],
    page_index_map: list[int] | None = None,
) -> list[PageInfo]:
    """在 raw 后处理结束后，一次性构造严格且可递归序列化的 PageInfo。"""
    page_indices = _validate_page_index_map(len(model_list), page_index_map)
    copied_model_list = deepcopy(model_list)
    use_bbox = _document_uses_bbox(copied_model_list)
    if not use_bbox:
        fix_office_paragraph_titles(copied_model_list)

    raw_pages = [
        _blocks_to_raw_page_info(
            page_model_list,
            page_idx=page_idx,
            use_bbox=use_bbox,
        )
        for page_model_list, page_idx in zip(copied_model_list, page_indices, strict=True)
    ]
    if use_bbox:
        merge_para_text_blocks(raw_pages)
        merge_table(raw_pages)

    for page in raw_pages:
        for block in page["blocks"]:
            _remove_private_block_metadata(block)
    return PAGE_INFO_LIST_ADAPTER.validate_python(raw_pages)
