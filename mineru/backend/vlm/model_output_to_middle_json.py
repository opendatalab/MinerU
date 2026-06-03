# Copyright (c) Opendatalab. All rights reserved.
import os
from typing import Any

from mineru.backend.utils.html_image_utils import replace_inline_table_images
from mineru.backend.utils.para_block_utils import (
    build_para_blocks_from_preproc,
    cleanup_internal_para_block_metadata,
    merge_para_text_blocks,
)
from mineru.backend.utils.runtime_utils import cross_page_table_merge
from mineru.backend.vlm.vlm_magic_model import MagicModel
from mineru.utils.config_reader import get_table_enable
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType
from mineru.utils.hash_utils import bytes_md5
from mineru.utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from mineru.utils.pdfium_guard import pdfium_guard
from mineru.parser.types import PageInfo
from mineru.version import __version__


def blocks_to_page_info(page_blocks: list, image_dict: dict, page: Any, image_writer: Any, page_index: int) -> PageInfo:
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

    page_info = PageInfo(
        preproc_blocks=page_blocks,
        discarded_blocks=discarded_blocks,
        page_size=[width, height],
        page_idx=page_index,
    )
    return page_info


def init_middle_json():
    return {"pdf_info": [], "_backend": "vlm", "_version_name": __version__}


def finalize_middle_json(pdf_info_list):
    """从 VLM preproc_blocks 执行完整 finalize，客户端和服务端完整路径共用。"""
    build_para_blocks_from_preproc(pdf_info_list)
    merge_para_text_blocks(pdf_info_list)

    table_enable = get_table_enable(os.getenv('MINERU_VLM_TABLE_ENABLE', 'True').lower() == 'true')
    if table_enable:
        cross_page_table_merge(pdf_info_list)

    apply_title_leveling_to_pdf_info(pdf_info_list)

    cleanup_internal_para_block_metadata(pdf_info_list)


def result_to_middle_json(model_output_blocks_list, images_list, pdf_doc, image_writer):
    from mineru.backend.utils.middle_json_utils import build_middle_json

    return build_middle_json(
        model_output_blocks_list, images_list, pdf_doc, image_writer,
        init_fn=init_middle_json,
        page_cvt_fn=blocks_to_page_info,
        finalize_fn=finalize_middle_json,
    )
