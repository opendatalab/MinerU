# Copyright (c) Opendatalab. All rights reserved.

from __future__ import annotations

import os
from typing import Any

from ...types import Block, BlockType, PageInfo
from ...utils.config_reader import get_table_enable
from ...utils.hash_utils import bytes_md5
from ...utils.image_payload import ImagePayloadCache
from ...utils.pdf_document import PDFPage
from ...utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from ..pipeline.model_init import MineruHybridModel
from ..utils.formula_number import optimize_hybrid_formula_number_blocks
from ..utils.middle_json_utils import apply_post_ocr
from ..utils.para_block_utils import (
    build_para_blocks_from_preproc,
    cleanup_internal_para_block_metadata,
    merge_para_text_blocks,
)
from ..utils.runtime_utils import cross_page_table_merge
from ..utils.visual_span_utils import cut_visual_spans_in_blocks
from .hybrid_magic_model import MagicModel


def _resolve_title_line_avg_height(title_block: Block) -> int:
    """解析标题平均行高：优先复用 Hybrid OCR det 行提示，再回退到原始行或块高。"""
    for lines in [title_block._ocr_det_lines, title_block.lines]:
        line_heights = []
        for line in lines:
            bbox = line.bbox
            if not bbox or len(bbox) < 4:
                continue
            line_height = bbox[3] - bbox[1]
            if line_height > 0:
                line_heights.append(line_height)
        if line_heights:
            return round(sum(line_heights) / len(line_heights))
    return int(title_block.bbox[3] - title_block.bbox[1])


def blocks_to_page_info(
    page_model_list: list[dict[str, Any]],
    image_dict: dict[str, Any],
    pdf_page: PDFPage,
    page_index: int,
    _ocr_enable: bool,
    _vlm_ocr_enable: bool,
    image_cache: ImagePayloadCache | None = None,
) -> PageInfo:
    """将blocks转换为页面信息"""

    page_model_list = optimize_hybrid_formula_number_blocks(page_model_list)
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    page_size = getattr(pdf_page, "size", None)
    if page_size is None and hasattr(pdf_page, "get_size"):
        page_size = pdf_page.get_size()
    width, height = map(int, page_size)

    magic_model = MagicModel(
        page_model_list,
        pdf_page,
        scale,
        page_pil_img,
        width,
        height,
        _ocr_enable,
        _vlm_ocr_enable,
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

    # 标题平均行高是 finalized middle json 的稳定字段，供服务端和客户端标题分级共用。
    for title_block in title_blocks:
        title_block._line_avg_height = _resolve_title_line_avg_height(title_block)

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
        page_img_md5,
        page_index,
        scale=scale,
        image_cache=image_cache,
    )

    page_info = PageInfo(
        preproc_blocks=page_blocks,
        discarded_blocks=discarded_blocks,
        page_size=(width, height),
        page_idx=page_index,
        _backend="hybrid",
    )
    return page_info


def _apply_post_ocr(pages: list[PageInfo], hybrid_pipeline_model: MineruHybridModel) -> None:
    apply_post_ocr(pages, hybrid_pipeline_model.ocr_model)


def _normalize_split_title_blocks(pages: list[PageInfo]) -> None:
    """将Hybrid内部拆分标题统一为输出层通用title，并补齐默认标题层级。"""
    title_type_to_level = {
        BlockType.DOC_TITLE: 1,
        BlockType.PARAGRAPH_TITLE: 2,
    }
    for page_info in pages:
        for blocks in [page_info.preproc_blocks, page_info.para_blocks]:
            for block in blocks:
                title_level = title_type_to_level.get(block.type)
                if title_level is None:
                    continue
                block.type = BlockType.TITLE
                block.level = title_level


def apply_server_side_postprocess(
    pages: list[PageInfo], hybrid_pipeline_model: MineruHybridModel, _ocr_enable: bool, _vlm_ocr_enable: bool
) -> None:
    """执行 Hybrid 只能在服务端完成的 post-OCR，避免客户端依赖 pipeline OCR 模型。"""
    if not (_vlm_ocr_enable or _ocr_enable):
        _apply_post_ocr(pages, hybrid_pipeline_model)


def finalize_middle_json_from_preproc(pages: list[PageInfo]) -> None:
    """从 Hybrid preproc_blocks 执行完整 finalize，供服务端完整路径和客户端复用。"""
    build_para_blocks_from_preproc(pages)
    merge_para_text_blocks(pages, auto_merge_by_det=True)

    table_enable = get_table_enable(os.getenv("MINERU_VLM_TABLE_ENABLE", "True").lower() == "true")
    if table_enable:
        cross_page_table_merge(pages)

    apply_title_leveling_to_pdf_info(pages)
    _normalize_split_title_blocks(pages)
    cleanup_internal_para_block_metadata(pages)


def finalize_middle_json(
    pages: list[PageInfo], hybrid_pipeline_model: MineruHybridModel, _ocr_enable: bool, _vlm_ocr_enable: bool
) -> None:
    """保持旧入口语义：服务端先做必要 post-OCR，再执行完整 finalize。"""
    apply_server_side_postprocess(pages, hybrid_pipeline_model, _ocr_enable, _vlm_ocr_enable)
    finalize_middle_json_from_preproc(pages)
