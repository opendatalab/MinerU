import copy
import os
import statistics
import time
from typing import List

import torch
from loguru import logger

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.dataset import Dataset, PageableData
from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.commons import fitz, get_delta_time
from magic_pdf.libs.config_reader import get_local_layoutreader_model_dir
from magic_pdf.libs.convert_utils import dict_to_list
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.libs.local_math import float_equal
from magic_pdf.libs.ocr_content_type import ContentType, BlockType
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.para.para_split_v3 import para_split
from magic_pdf.pre_proc.citationmarker_remove import remove_citation_marker
from magic_pdf.pre_proc.construct_page_dict import \
    ocr_construct_page_component_v2
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.equations_replace import (
    combine_chars_to_pymudict, remove_chars_in_text_blocks,
    replace_equations_in_textblock)
from magic_pdf.pre_proc.ocr_detect_all_bboxes import \
    ocr_prepare_bboxes_for_layout_split_v2
from magic_pdf.pre_proc.ocr_dict_merge import (fill_spans_in_blocks,
                                               fix_block_spans,
                                               fix_discarded_block, fix_block_spans_v2)
from magic_pdf.pre_proc.ocr_span_list_modify import (
    get_qa_need_list_v2, remove_overlaps_low_confidence_spans,
    remove_overlaps_min_spans)
from magic_pdf.pre_proc.resolve_bbox_conflict import \
    check_useful_block_horizontal_overlap

# ... (rest of the imports and functions remain the same)

def pdf_parse_union(
    dataset: Dataset,
    model_list,
    imageWriter,
    parse_mode,
    start_page_id=0,
    end_page_id=None,
    debug_mode=False,
):
    pdf_bytes_md5 = compute_md5(dataset.data_bits())

    """初始化空的pdf_info_dict"""
    pdf_info_dict = {}

    """用model_list和docs对象初始化magic_model"""
    magic_model = MagicModel(model_list, dataset)

    """根据输入的起始范围解析pdf"""
    # end_page_id = end_page_id if end_page_id else len(pdf_docs) - 1
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else len(dataset) - 1
    )

    if end_page_id > len(dataset) - 1:
        logger.warning('end_page_id is out of range, use pdf_docs length')
        end_page_id = len(dataset) - 1

    """初始化启动时间"""
    start_time = time.time()

    for page_id, page in enumerate(dataset):
        """debug时输出每页解析的耗时."""
        if debug_mode:
            time_now = time.time()
            logger.info(
                f'page_id: {page_id}, last_page_cost_time: {get_delta_time(start_time)}'
            )
            start_time = time_now

        """解析pdf中的每一页"""
        if start_page_id <= page_id <= end_page_id:
            try:
                page_info = parse_page_core(
                    page, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode
                )
            except Exception as e:
                logger.error(f"Error parsing page {page_id}: {str(e)}")
                page_info = page.get_page_info()
                page_w = page_info.w
                page_h = page_info.h
                page_info = ocr_construct_page_component_v2(
                    [], [], page_id, page_w, page_h, [], [], [], [], [], True, f'Error: {str(e)}'
                )
        else:
            page_info = page.get_page_info()
            page_w = page_info.w
            page_h = page_info.h
            page_info = ocr_construct_page_component_v2(
                [], [], page_id, page_w, page_h, [], [], [], [], [], True, 'skip page'
            )
        pdf_info_dict[f'page_{page_id}'] = page_info

    """分段"""
    para_split(pdf_info_dict, debug_mode=debug_mode)

    """dict转list"""
    pdf_info_list = dict_to_list(pdf_info_dict)
    new_pdf_info_dict = {
        'pdf_info': pdf_info_list,
    }

    clean_memory()

    return new_pdf_info_dict


if __name__ == '__main__':
    pass
