# Copyright (c) Opendatalab. All rights reserved.

from __future__ import annotations

from typing import Any

from ...types import BlockType, PageInfo
from ...utils.image_payload import ImagePayloadCache
from ...utils.pdf_document import PDFPage
from ...utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from ...utils.backend_options import DEFAULT_HYBRID_EFFORT, LAYOUT_HYBRID_EFFORT, LOCAL_HYBRID_EFFORT, validate_effort
from ..utils.para_block_utils import (
    build_para_blocks_from_preproc,
    cleanup_internal_para_block_metadata,
    merge_para_text_blocks,
)
from ..utils.runtime_utils import cross_page_table_merge
from ..utils.visual_span_utils import cut_visual_spans_in_blocks
from .magic_model import MagicModel

def blocks_to_page_info(
    page_model_list: list[dict[str, Any]],
    image_dict: dict[str, Any],
    pdf_page: PDFPage,
    page_index: int,
    image_cache: ImagePayloadCache | None = None,
) -> PageInfo:
    """将blocks转换为页面信息"""
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_size = getattr(pdf_page, "size", None)
    if page_size is None and hasattr(pdf_page, "get_size"):
        page_size = pdf_page.get_size()
    width, height = map(int, page_size)

    magic_model = MagicModel(
        page_model_list,
        width,
        height,
    )
    image_blocks = magic_model.get_image_blocks()
    table_blocks = magic_model.get_table_blocks()
    chart_blocks = magic_model.get_chart_blocks()
    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    code_blocks = magic_model.get_code_blocks()
    ref_text_blocks = magic_model.get_ref_text_blocks()
    phonetic_blocks = magic_model.get_phonetic_blocks()
    list_blocks = magic_model.get_list_blocks()

    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    page_blocks = []
    page_blocks.extend(
        [
            *image_blocks,
            *table_blocks,
            *chart_blocks,
            *code_blocks,
            *ref_text_blocks,
            *phonetic_blocks,
            *title_blocks,
            *text_blocks,
            *interline_equation_blocks,
            *list_blocks,
        ]
    )
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x.index)

    cut_visual_spans_in_blocks(
        [*page_blocks, *discarded_blocks],
        page_pil_img,
        page_index,
        scale=scale,
        image_cache=image_cache,
    )

    page_info = PageInfo(
        blocks=[*page_blocks, *discarded_blocks],
        page_idx=page_index,
        _backend="hybrid",
    )
    return page_info


def finalize_middle_json_from_preproc(pages: list[PageInfo], effort: str = DEFAULT_HYBRID_EFFORT) -> None:
    """从 Hybrid preproc_blocks 执行完整 finalize，供服务端完整路径和客户端复用。"""
    effort = validate_effort(effort)
    build_para_blocks_from_preproc(pages)
    merge_para_text_blocks(
        pages,
        auto_merge_by_det=True,
        auto_merge_vertical_by_det=effort in {LOCAL_HYBRID_EFFORT, LAYOUT_HYBRID_EFFORT},
    )

    cross_page_table_merge(pages)

    apply_title_leveling_to_pdf_info(pages)
    cleanup_internal_para_block_metadata(pages)


def finalize_middle_json(
    pages: list[PageInfo],
    effort: str = DEFAULT_HYBRID_EFFORT,
) -> None:
    finalize_middle_json_from_preproc(pages, effort=effort)
