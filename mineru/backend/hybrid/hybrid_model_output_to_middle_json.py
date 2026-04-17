# Copyright (c) Opendatalab. All rights reserved.

import os
import time

import numpy as np
from loguru import logger
from tqdm import tqdm

from mineru.backend.utils.html_image_utils import replace_inline_table_images
from mineru.backend.utils.ocr_det_utils import (
    detect_ocr_boxes_from_padded_crop,
    get_ch_lite_ocr_det_model,
)
from mineru.backend.utils.para_block_utils import (
    annotate_hybrid_cross_page_merge_prev,
    build_para_blocks_from_preproc,
    cleanup_internal_para_block_metadata,
    edge_text_line_hints_key,
    iter_block_spans,
    merge_para_text_blocks,
)
from mineru.backend.hybrid.hybrid_magic_model import MagicModel
from mineru.backend.utils.runtime_utils import cross_page_table_merge
from mineru.utils.config_reader import get_table_enable, get_llm_aided_config
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType
from mineru.utils.hash_utils import bytes_md5
from mineru.utils.ocr_utils import OcrConfidence, rotate_vertical_crop_if_needed
from mineru.utils.pdfium_guard import close_pdfium_document, pdfium_guard
from mineru.version import __version__

from mineru.utils.llm_aided import llm_aided_title
title_aided_enable = False
llm_aided_config = get_llm_aided_config()
if llm_aided_config:
    title_aided_config = llm_aided_config.get('title_aided', {})
    title_aided_enable = title_aided_config.get('enable', False)


def blocks_to_page_info(
        page_model_list,
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

    # 如果有标题优化需求，计算标题的平均行高
    if title_aided_enable:
        if _vlm_ocr_enable:  # vlm_ocr导致没有line信息，需要重新det获取平均行高
            ocr_model = get_ch_lite_ocr_det_model()
            for title_block in title_blocks:
                ocr_det_res, _ = detect_ocr_boxes_from_padded_crop(
                    title_block.get('bbox'),
                    page_pil_img,
                    scale,
                    ocr_model=ocr_model,
                )
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

    page_info = {
        "preproc_blocks": page_blocks,
        "discarded_blocks": discarded_blocks,
        "page_size": [width, height],
        "page_idx": page_index,
    }
    if _vlm_ocr_enable:
        edge_text_line_hints = _detect_edge_text_line_hints(page_blocks, page_pil_img, scale)
        if edge_text_line_hints:
            page_info[edge_text_line_hints_key()] = edge_text_line_hints
    return page_info


def _apply_post_ocr(pdf_info_list, hybrid_pipeline_model):
    need_ocr_list = []
    img_crop_list = []
    for page_info in pdf_info_list:
        for block in page_info.get('preproc_blocks', []):
            for span in iter_block_spans(block):
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    img_crop_list.append(rotate_vertical_crop_if_needed(span['np_img']))
                    span.pop('np_img')
        for block in page_info.get('discarded_blocks', []):
            for span in iter_block_spans(block):
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    img_crop_list.append(rotate_vertical_crop_if_needed(span['np_img']))
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


def init_middle_json(_ocr_enable, _vlm_ocr_enable):
    return {
        "pdf_info": [],
        "_backend": "hybrid",
        "_ocr_enable": _ocr_enable,
        "_vlm_ocr_enable": _vlm_ocr_enable,
        "_version_name": __version__
    }


def append_page_results_to_middle_json(
    middle_json,
    model_list,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    _ocr_enable=False,
    _vlm_ocr_enable=False,
    progress_bar=None,
):
    for offset, (page_model_list, image_dict) in enumerate(
        zip(model_list, images_list)
    ):
        page_index = page_start_index + offset
        with pdfium_guard():
            page = pdf_doc[page_index]
        page_info = blocks_to_page_info(
            page_model_list,
            image_dict,
            page,
            image_writer,
            page_index,
            _ocr_enable,
            _vlm_ocr_enable,
        )
        middle_json["pdf_info"].append(page_info)
        if progress_bar is not None:
            progress_bar.update(1)


def append_page_model_list_to_middle_json(
    middle_json,
    model_list,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    _ocr_enable=False,
    _vlm_ocr_enable=False,
    progress_bar=None,
):
    append_page_results_to_middle_json(
        middle_json,
        model_list,
        images_list,
        pdf_doc,
        image_writer,
        page_start_index=page_start_index,
        _ocr_enable=_ocr_enable,
        _vlm_ocr_enable=_vlm_ocr_enable,
        progress_bar=progress_bar,
    )


def finalize_middle_json(pdf_info_list, hybrid_pipeline_model, _ocr_enable, _vlm_ocr_enable):
    if not (_vlm_ocr_enable or _ocr_enable):
        _apply_post_ocr(pdf_info_list, hybrid_pipeline_model)

    build_para_blocks_from_preproc(pdf_info_list)
    annotate_hybrid_cross_page_merge_prev(
        pdf_info_list,
        prefer_edge_line_hints=_vlm_ocr_enable,
    )
    merge_para_text_blocks(pdf_info_list, allow_cross_page=True)

    table_enable = get_table_enable(os.getenv('MINERU_VLM_TABLE_ENABLE', 'True').lower() == 'true')
    if table_enable:
        cross_page_table_merge(pdf_info_list)

    if title_aided_enable:
        llm_aided_title_start_time = time.time()
        llm_aided_title(pdf_info_list, title_aided_config)
        logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    cleanup_internal_para_block_metadata(pdf_info_list)


def _detect_edge_text_line_hints(page_blocks, page_pil_img, scale):
    text_blocks = [block for block in page_blocks if block.get("type") == "text"]
    if not text_blocks:
        return {}

    edge_blocks = {}
    edge_blocks["first"] = text_blocks[0]
    edge_blocks["last"] = text_blocks[-1]

    ocr_model = get_ch_lite_ocr_det_model()

    edge_line_hints = {}
    for edge_name, block in edge_blocks.items():
        line_bboxes = _detect_block_line_bboxes(block, page_pil_img, scale, ocr_model)
        if line_bboxes:
            edge_line_hints[edge_name] = {
                "index": block.get("index"),
                "lines": [{"bbox": bbox, "spans": []} for bbox in line_bboxes],
            }

    return edge_line_hints


def _detect_block_line_bboxes(block, page_pil_img, scale, ocr_model):
    block_bbox = block.get("bbox")
    if not block_bbox:
        return []

    ocr_det_res, padding = detect_ocr_boxes_from_padded_crop(
        block_bbox,
        page_pil_img,
        scale,
        ocr_model=ocr_model,
    )
    if not ocr_det_res:
        return []

    block_x0, block_y0 = block_bbox[0], block_bbox[1]
    line_bboxes = []
    for box in ocr_det_res:
        x_coords = [point[0] - padding for point in box]
        y_coords = [point[1] - padding for point in box]
        line_bboxes.append([
            block_x0 + min(x_coords) / scale,
            block_y0 + min(y_coords) / scale,
            block_x0 + max(x_coords) / scale,
            block_y0 + max(y_coords) / scale,
        ])

    line_bboxes.sort(key=lambda bbox: bbox[1])
    return line_bboxes


def result_to_middle_json(
        model_list,
        images_list,
        pdf_doc,
        image_writer,
        _ocr_enable,
        _vlm_ocr_enable,
        hybrid_pipeline_model,
):
    middle_json = init_middle_json(_ocr_enable, _vlm_ocr_enable)

    with tqdm(total=len(model_list), desc="Processing pages") as progress_bar:
        append_page_model_list_to_middle_json(
            middle_json,
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            _ocr_enable=_ocr_enable,
            _vlm_ocr_enable=_vlm_ocr_enable,
            progress_bar=progress_bar,
        )

    finalize_middle_json(
        middle_json["pdf_info"],
        hybrid_pipeline_model,
        _ocr_enable,
        _vlm_ocr_enable,
    )
    close_pdfium_document(pdf_doc)
    return middle_json
