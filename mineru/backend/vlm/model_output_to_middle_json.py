# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import os
from typing import Any

from mineru_vl_utils.structs import ExtractResult

from ...data.data_reader_writer import DataWriter
from ...types import PageInfo
from ...utils.config_reader import get_table_enable
from ...utils.cut_image import cut_image_and_table
from ...utils.enum_class import ContentType
from ...utils.hash_utils import bytes_md5
from ...utils.pdfium_guard import pdfium_guard
from ...utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from ...version import __version__
from ..utils.html_image_utils import replace_inline_table_images
from ..utils.middle_json_utils import build_middle_json
from ..utils.para_block_utils import (
    build_para_blocks_from_preproc,
    cleanup_internal_para_block_metadata,
    merge_para_text_blocks,
)
from ..utils.runtime_utils import cross_page_table_merge
from .vlm_magic_model import MagicModel


def blocks_to_page_info(
    page_blocks: ExtractResult,
    image_dict: dict[str, Any],
    page: Any,
    image_writer: DataWriter | None,
    page_index: int,
) -> PageInfo:
    """将blocks转换为页面信息"""

    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    with pdfium_guard():
        width, height = map(int, page.get_size())

    magic_model = MagicModel(page_blocks, width, height)
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

    all_spans = magic_model.get_all_spans()
    # 对image/table/chart/interline_equation的span截图
    for span in all_spans:
        if span.type in [ContentType.IMAGE, ContentType.TABLE, ContentType.CHART, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    replace_inline_table_images(table_blocks, image_writer, page_index)

    blocks = [
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

    # 对page_blocks根据index的值进行排序
    blocks.sort(key=lambda x: x.index)

    page_info = PageInfo(
        preproc_blocks=blocks,
        discarded_blocks=discarded_blocks,
        page_size=[width, height],
        page_idx=page_index,
    )
    return page_info


def init_middle_json() -> dict[str, Any]:
    return {"pdf_info": [], "_backend": "vlm", "_version_name": __version__}


def finalize_middle_json(pdf_info_list: list[PageInfo]) -> None:
    """从 VLM preproc_blocks 执行完整 finalize，客户端和服务端完整路径共用。"""
    build_para_blocks_from_preproc(pdf_info_list)
    merge_para_text_blocks(pdf_info_list)

    table_enable = get_table_enable(os.getenv("MINERU_VLM_TABLE_ENABLE", "True").lower() == "true")
    if table_enable:
        cross_page_table_merge(pdf_info_list)

    apply_title_leveling_to_pdf_info(pdf_info_list)

    cleanup_internal_para_block_metadata(pdf_info_list)


def result_to_middle_json(
    model_output_blocks_list: list[list[dict[str, Any]]],
    images_list: list[dict[str, Any]],
    pdf_doc: Any,
    image_writer: DataWriter,
) -> dict[str, Any]:
    return build_middle_json(
        model_output_blocks_list,
        images_list,
        pdf_doc,
        image_writer,
        init_fn=init_middle_json,
        page_cvt_fn=blocks_to_page_info,
        finalize_fn=finalize_middle_json,
    )
