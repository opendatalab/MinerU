# Copyright (c) Opendatalab. All rights reserved.

import os
from typing import Any

from mineru.backend.utils.html_image_utils import replace_inline_table_images
from mineru.backend.utils.para_block_utils import (
    OCR_DET_LINES_KEY,
    build_para_blocks_from_preproc,
    cleanup_internal_para_block_metadata,
    merge_para_text_blocks,
)
from mineru.backend.hybrid.hybrid_magic_model import MagicModel
from mineru.backend.utils.runtime_utils import cross_page_table_merge
from mineru.utils.config_reader import get_table_enable
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.hash_utils import bytes_md5
from mineru.utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from mineru.utils.pdfium_guard import pdfium_guard
from mineru.types import PageInfo, block_from_dict
from mineru.version import __version__


def _resolve_title_line_avg_height(title_block):
    """解析标题平均行高：优先复用 Hybrid OCR det 行提示，再回退到原始行或块高。"""
    for lines_key in [OCR_DET_LINES_KEY, "lines"]:
        line_heights = []
        for line in title_block.get(lines_key, []):
            bbox = line.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            line_height = bbox[3] - bbox[1]
            if line_height > 0:
                line_heights.append(line_height)
        if line_heights:
            return round(sum(line_heights) / len(line_heights))

    bbox = title_block.get("bbox", [0, 0, 0, 0])
    if len(bbox) >= 4:
        return bbox[3] - bbox[1]
    return 0


def blocks_to_page_info(
        page_model_list: list,
        image_dict: dict,
        page: Any,
        image_writer: Any,
        page_index: int,
        _ocr_enable: bool,
        _vlm_ocr_enable: bool,
) -> PageInfo:
    """将blocks转换为页面信息"""

    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    with pdfium_guard():
        width, height = map(int, page.get_size())

    magic_model = MagicModel(
        page_model_list,
        page,
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
        title_block['line_avg_height'] = _resolve_title_line_avg_height(title_block)

    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    all_spans = magic_model.get_all_spans()
    # 对image/table/chart/interline_equation的span截图
    for span in all_spans:
        if span["type"] in [ContentType.IMAGE, ContentType.TABLE, ContentType.CHART, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    replace_inline_table_images(table_blocks, image_writer, page_index)

    page_blocks = []
    page_blocks.extend([
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
    ])
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x["index"])

    page_blocks = [block_from_dict(b) for b in page_blocks]
    discarded_blocks = [block_from_dict(b) for b in discarded_blocks]

    page_info = PageInfo(
        preproc_blocks=page_blocks,
        discarded_blocks=discarded_blocks,
        page_size=[width, height],
        page_idx=page_index,
    )
    return page_info


def _apply_post_ocr(pdf_info_list, hybrid_pipeline_model):
    from mineru.backend.utils.middle_json_utils import apply_post_ocr

    apply_post_ocr(pdf_info_list, hybrid_pipeline_model.ocr_model)


def _normalize_split_title_blocks(pdf_info_list):
    """将Hybrid内部拆分标题统一为输出层通用title，并补齐默认标题层级。"""
    title_type_to_level = {
        BlockType.DOC_TITLE: 1,
        BlockType.PARAGRAPH_TITLE: 2,
    }
    for page_info in pdf_info_list:
        for block_key in ["preproc_blocks", "para_blocks"]:
            for block in page_info.get(block_key, []):
                title_level = title_type_to_level.get(block.get("type"))
                if title_level is None:
                    continue
                block["type"] = BlockType.TITLE
                block["level"] = title_level


def init_middle_json(_ocr_enable, _vlm_ocr_enable):
    return {
        "pdf_info": [],
        "_backend": "hybrid",
        "_ocr_enable": _ocr_enable,
        "_vlm_ocr_enable": _vlm_ocr_enable,
        "_version_name": __version__
    }


def apply_server_side_postprocess(
    pdf_info_list,
    hybrid_pipeline_model,
    _ocr_enable,
    _vlm_ocr_enable,
):
    """执行 Hybrid 只能在服务端完成的 post-OCR，避免客户端依赖 pipeline OCR 模型。"""
    if not (_vlm_ocr_enable or _ocr_enable):
        _apply_post_ocr(pdf_info_list, hybrid_pipeline_model)


def finalize_middle_json_from_preproc(pdf_info_list):
    """从 Hybrid preproc_blocks 执行完整 finalize，供服务端完整路径和客户端复用。"""
    build_para_blocks_from_preproc(pdf_info_list)
    merge_para_text_blocks(
        pdf_info_list,
        auto_merge_by_det=True,
    )

    table_enable = get_table_enable(os.getenv('MINERU_VLM_TABLE_ENABLE', 'True').lower() == 'true')
    if table_enable:
        cross_page_table_merge(pdf_info_list)

    apply_title_leveling_to_pdf_info(pdf_info_list)
    _normalize_split_title_blocks(pdf_info_list)
    cleanup_internal_para_block_metadata(pdf_info_list)


def finalize_middle_json(
    pdf_info_list,
    hybrid_pipeline_model,
    _ocr_enable,
    _vlm_ocr_enable,
):
    """保持旧入口语义：服务端先做必要 post-OCR，再执行完整 finalize。"""
    apply_server_side_postprocess(
        pdf_info_list,
        hybrid_pipeline_model,
        _ocr_enable,
        _vlm_ocr_enable,
    )
    finalize_middle_json_from_preproc(pdf_info_list)


def result_to_middle_json(
        model_list,
        images_list,
        pdf_doc,
        image_writer,
        _ocr_enable,
        _vlm_ocr_enable,
        hybrid_pipeline_model,
):
    from mineru.backend.utils.middle_json_utils import build_middle_json

    return build_middle_json(
        model_list, images_list, pdf_doc, image_writer,
        init_fn=init_middle_json,
        page_cvt_fn=blocks_to_page_info,
        finalize_fn=finalize_middle_json,
        _ocr_enable=_ocr_enable,
        _vlm_ocr_enable=_vlm_ocr_enable,
        hybrid_pipeline_model=hybrid_pipeline_model,
    )
