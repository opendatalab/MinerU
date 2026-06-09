# Copyright (c) Opendatalab. All rights reserved.

import os

from tqdm import tqdm

from mineru.backend.utils.html_image_utils import replace_inline_table_images
from mineru.backend.utils.para_block_utils import (
    OCR_DET_LINES_KEY,
    build_para_blocks_from_preproc,
    cleanup_internal_para_block_metadata,
    iter_block_spans,
    merge_para_text_blocks,
)
from mineru.backend.hybrid.hybrid_magic_model import MagicModel
from mineru.backend.utils.runtime_utils import cross_page_table_merge
from mineru.backend.pipeline.model_init import run_ocr_inference
from mineru.utils.config_reader import get_table_enable
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.hash_utils import bytes_md5
from mineru.utils.ocr_utils import OcrConfidence, rotate_vertical_crop_if_needed
from mineru.utils.span_pre_proc import (
    _clear_post_ocr_fallback,
    _restore_post_ocr_fallback,
)
from mineru.utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from mineru.utils.pdfium_guard import close_pdfium_child, close_pdfium_document, pdfium_guard
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
        page_model_list,
        image_dict,
        page,
        image_writer,
        page_index,
        _ocr_enable,
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

    page_info = {
        "preproc_blocks": page_blocks,
        "discarded_blocks": discarded_blocks,
        "page_size": [width, height],
        "page_idx": page_index,
    }
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
        ocr_res_list = run_ocr_inference(
            hybrid_pipeline_model.ocr_model.ocr,
            img_crop_list,
            det=False,
            tqdm_enable=True,
        )[0]
        assert len(ocr_res_list) == len(
            need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
        for index, span in enumerate(need_ocr_list):
            ocr_text, ocr_score = ocr_res_list[index]
            if ocr_score > OcrConfidence.min_confidence:
                span['content'] = ocr_text
                span['score'] = float(f"{ocr_score:.3f}")
                _clear_post_ocr_fallback(span)
            elif _restore_post_ocr_fallback(span):
                continue
            else:
                span['content'] = ''
                span['score'] = 0.0


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


def init_middle_json(_ocr_enable, effort="medium"):
    """初始化 Hybrid middle json，使用公开 effort 元数据描述解析强度。"""
    return {
        "pdf_info": [],
        "_backend": "hybrid",
        "_effort": effort,
        "_ocr_enable": _ocr_enable,
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
    progress_bar=None,
):
    for offset, (page_model_list, image_dict) in enumerate(
        zip(model_list, images_list)
    ):
        page_index = page_start_index + offset
        page = None
        try:
            with pdfium_guard():
                page = pdf_doc[page_index]
            page_info = blocks_to_page_info(
                page_model_list,
                image_dict,
                page,
                image_writer,
                page_index,
                _ocr_enable,
            )
        finally:
            close_pdfium_child(page)
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
        progress_bar=progress_bar,
    )


def apply_server_side_postprocess(
    pdf_info_list,
    hybrid_pipeline_model,
    _ocr_enable,
):
    """执行 Hybrid 只能在服务端完成的 post-OCR，避免客户端依赖 pipeline OCR 模型。"""
    if not _ocr_enable:
        _apply_post_ocr(pdf_info_list, hybrid_pipeline_model)


def finalize_middle_json_from_preproc(pdf_info_list, effort="medium"):
    """从 Hybrid preproc_blocks 执行完整 finalize，供服务端完整路径和客户端复用。"""
    build_para_blocks_from_preproc(pdf_info_list)
    merge_para_text_blocks(
        pdf_info_list,
        auto_merge_by_det=True,
        auto_merge_vertical_by_det=effort == "medium",
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
    effort="medium",
):
    """服务端先做必要 post-OCR，再按公开 effort 执行完整 finalize。"""
    apply_server_side_postprocess(
        pdf_info_list,
        hybrid_pipeline_model,
        _ocr_enable,
    )
    finalize_middle_json_from_preproc(pdf_info_list, effort=effort)


def result_to_middle_json(
        model_list,
        images_list,
        pdf_doc,
        image_writer,
        _ocr_enable,
        hybrid_pipeline_model,
):
    middle_json = init_middle_json(_ocr_enable)

    with tqdm(total=len(model_list), desc="Processing pages") as progress_bar:
        append_page_model_list_to_middle_json(
            middle_json,
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            _ocr_enable=_ocr_enable,
            progress_bar=progress_bar,
        )

    finalize_middle_json(
        middle_json["pdf_info"],
        hybrid_pipeline_model,
        _ocr_enable,
    )
    close_pdfium_document(pdf_doc)
    return middle_json
