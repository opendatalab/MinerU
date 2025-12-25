#  Copyright (c) Opendatalab. All rights reserved.

import os
import time

import cv2
import numpy as np
from loguru import logger

from mineru.backend.hybrid.hybrid_magic_model import MagicModel
from mineru.backend.utils import cross_page_table_merge
from mineru.utils.config_reader import get_table_enable, get_llm_aided_config
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType
from mineru.utils.hash_utils import bytes_md5
from mineru.utils.ocr_utils import OcrConfidence
from mineru.utils.pdf_image_tools import get_crop_img
from mineru.version import __version__


heading_level_import_success = False
llm_aided_config = get_llm_aided_config()
if llm_aided_config:
    title_aided_config = llm_aided_config.get('title_aided', {})
    if title_aided_config.get('enable', False):
        try:
            from mineru.utils.llm_aided import llm_aided_title
            from mineru.backend.pipeline.model_init import AtomModelSingleton
            heading_level_import_success = True
        except Exception as e:
            logger.warning("The heading level feature cannot be used. If you need to use the heading level feature, "
                            "please execute `pip install mineru[core]` to install the required packages.")


def blocks_to_page_info(
        page_blocks,
        page_inline_formula,
        page_ocr_res,
        image_dict,
        page,
        image_writer,
        page_index,
        _ocr_enable,
        _vlm_ocr_enable,
) -> dict:
    """将blocks转换为页面信息"""

    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    width, height = map(int, page.get_size())

    magic_model = MagicModel(
        page_blocks,
        page_inline_formula,
        page_ocr_res,
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
    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    code_blocks = magic_model.get_code_blocks()
    ref_text_blocks = magic_model.get_ref_text_blocks()
    phonetic_blocks = magic_model.get_phonetic_blocks()
    list_blocks = magic_model.get_list_blocks()

    # 如果有标题优化需求，计算标题的平均行高
    if heading_level_import_success:
        if _vlm_ocr_enable:  # vlm_ocr导致没有line信息，需要重新det获取平均行高
            atom_model_manager = AtomModelSingleton()
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name='ocr',
                ocr_show_log=False,
                det_db_box_thresh=0.3,
                lang='ch_lite'
            )
            for title_block in title_blocks:
                title_pil_img = get_crop_img(title_block['bbox'], page_pil_img, scale)
                title_np_img = np.array(title_pil_img)
                # 给title_pil_img添加上下左右各50像素白边padding
                title_np_img = cv2.copyMakeBorder(
                    title_np_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255]
                )
                title_img = cv2.cvtColor(title_np_img, cv2.COLOR_RGB2BGR)
                ocr_det_res = ocr_model.ocr(title_img, rec=False)[0]
                if len(ocr_det_res) > 0:
                    # 计算所有res的平均高度
                    avg_height = np.mean([box[2][1] - box[0][1] for box in ocr_det_res])
                    title_block['line_avg_height'] = round(avg_height/scale)
        else:  # 有line信息，直接计算平均行高
            for title_block in title_blocks:
                lines = title_block.get('lines', [])
                if lines:
                    # 使用列表推导式和内置函数,一次性计算平均高度
                    avg_height = sum(line['bbox'][3] - line['bbox'][1] for line in lines) / len(lines)
                    title_block['line_avg_height'] = round(avg_height)
                else:
                    title_block['line_avg_height'] = title_block['bbox'][3] - title_block['bbox'][1]

    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    all_spans = magic_model.get_all_spans()
    # 对image/table/interline_equation的span截图
    for span in all_spans:
        if span["type"] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    page_blocks = []
    page_blocks.extend([
        *image_blocks,
        *table_blocks,
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

    page_info = {"para_blocks": page_blocks, "discarded_blocks": discarded_blocks, "page_size": [width, height], "page_idx": page_index}
    return page_info


def result_to_middle_json(
        model_output_blocks_list,
        inline_formula_list,
        ocr_res_list,
        images_list,
        pdf_doc,
        image_writer,
        _ocr_enable,
        _vlm_ocr_enable,
        hybrid_pipeline_model,
):
    middle_json = {
        "pdf_info": [],
        "_backend": "hybrid",
        "_ocr_enable": _ocr_enable,
        "_vlm_ocr_enable": _vlm_ocr_enable,
        "_version_name": __version__
    }

    for index, (page_blocks, page_inline_formula, page_ocr_res) in enumerate(zip(model_output_blocks_list, inline_formula_list, ocr_res_list)):
        page = pdf_doc[index]
        image_dict = images_list[index]
        page_info = blocks_to_page_info(
            page_blocks, page_inline_formula, page_ocr_res,
            image_dict, page, image_writer, index,
            _ocr_enable, _vlm_ocr_enable
        )
        middle_json["pdf_info"].append(page_info)

    if not (_vlm_ocr_enable or _ocr_enable):
        """后置ocr处理"""
        need_ocr_list = []
        img_crop_list = []
        text_block_list = []
        for page_info in middle_json["pdf_info"]:
            for block in page_info['para_blocks']:
                if block['type'] in ['table', 'image', 'list', 'code']:
                    for sub_block in block['blocks']:
                        if not sub_block['type'].endswith('body'):
                            text_block_list.append(sub_block)
                elif block['type'] in ['text', 'title', 'ref_text']:
                    text_block_list.append(block)
            for block in page_info['discarded_blocks']:
                text_block_list.append(block)
        for block in text_block_list:
            for line in block['lines']:
                for span in line['spans']:
                    if 'np_img' in span:
                        need_ocr_list.append(span)
                        img_crop_list.append(span['np_img'])
                        span.pop('np_img')
        if len(img_crop_list) > 0:
            ocr_res_list = hybrid_pipeline_model.ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
            assert len(ocr_res_list) == len(
                need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
            for index, span in enumerate(need_ocr_list):
                ocr_text, ocr_score = ocr_res_list[index]
                if ocr_score > OcrConfidence.min_confidence:
                    span['content'] = ocr_text
                    span['score'] = float(f"{ocr_score:.3f}")
                else:
                    span['content'] = ''
                    span['score'] = 0.0

    """表格跨页合并"""
    table_enable = get_table_enable(os.getenv('MINERU_VLM_TABLE_ENABLE', 'True').lower() == 'true')
    if table_enable:
        cross_page_table_merge(middle_json["pdf_info"])

    """llm优化标题分级"""
    if heading_level_import_success:
        llm_aided_title_start_time = time.time()
        llm_aided_title(middle_json["pdf_info"], title_aided_config)
        logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    # 关闭pdf文档
    pdf_doc.close()
    return middle_json