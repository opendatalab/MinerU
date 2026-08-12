# Copyright (c) Opendatalab. All rights reserved.
from copy import deepcopy
from typing import Any

from mineru.backend.magic_model import MagicModel
from mineru.backend.utils.para_block_utils import merge_para_text_blocks


def blocks_to_page_info(
    page_model_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """将blocks转换为页面信息"""
    # Middle JSON 转换允许重写块结构，但不能污染调用方保留的原始 model_list。
    magic_model = MagicModel(
        deepcopy(page_model_list),
    )

    page_blocks = magic_model.blocks
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x["index"])

    page_info = {
        "blocks": page_blocks,
    }
    return page_info


def model_list_to_pages(
    model_list: list[list[dict[str, Any]]],
    page_index_map: list[int] | None = None,
) -> list[dict[str, Any]]:
    """将 model_list 转换为 pages"""
    if page_index_map is None:
        page_index_map = list(range(len(model_list)))
    pages: list[dict[str, Any]] = []
    for page_model_list, page_index in zip(model_list, page_index_map):
        page_info = blocks_to_page_info(
            page_model_list,
        )
        page_info["page_idx"] = page_index
        pages.append(page_info)

    # PDF block 一定带 bbox；扫描全部页面，避免首页为空时误判为 Office。
    has_bbox_block = any(
        isinstance(block, dict) and "bbox" in block
        for page_info in pages
        for block in page_info.get("blocks", [])
    )
    if has_bbox_block:
        merge_para_text_blocks(pages)
    return pages
