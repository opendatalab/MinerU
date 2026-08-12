# Copyright (c) Opendatalab. All rights reserved.
from copy import deepcopy
from typing import Any

from mineru.backend.magic_model import MagicModel


def model_list_to_pages(
    model_list:  list[list[dict[str, Any]]],
    page_index_map: list[int] | None = None,
):
    """将 model_list 转换为 pages"""
    if page_index_map is None:
        page_index_map = list(range(len(model_list)))
    pages: list[list[dict[str, Any]]] = []
    for page_model_list, page_index in zip(model_list, page_index_map):
        page_info = blocks_to_page_info(
            page_model_list,
        )
        page_info["page_idx"] = page_index
        pages.append(page_info)
    return pages


def blocks_to_page_info(
    page_model_list: list[dict[str, Any]],
):
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


# def finalize_middle_json_from_preproc(pages: list[PageInfo], effort: str = DEFAULT_HYBRID_EFFORT) -> None:
#     """从 Hybrid preproc_blocks 执行完整 finalize，供服务端完整路径和客户端复用。"""
#     effort = validate_effort(effort)
#     build_para_blocks_from_preproc(pages)
#     merge_para_text_blocks(
#         pages,
#         auto_merge_by_det=True,
#         auto_merge_vertical_by_det=effort in {LOCAL_HYBRID_EFFORT, LAYOUT_HYBRID_EFFORT},
#     )
#
#     cross_page_table_merge(pages)
#
#     apply_title_leveling_to_pdf_info(pages)
#     cleanup_internal_para_block_metadata(pages)
#
#
# def finalize_middle_json(
#     pages: list[PageInfo],
#     effort: str = DEFAULT_HYBRID_EFFORT,
# ) -> None:
#     finalize_middle_json_from_preproc(pages, effort=effort)
