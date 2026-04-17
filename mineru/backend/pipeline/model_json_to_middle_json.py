# Copyright (c) Opendatalab. All rights reserved.
import copy
import os
import time

from loguru import logger
from tqdm import tqdm

from mineru.backend.utils.html_image_utils import replace_inline_table_images
from mineru.backend.utils.runtime_utils import cross_page_table_merge
from mineru.utils.config_reader import get_device, get_llm_aided_config
from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.backend.pipeline.para_split import para_split
from mineru.utils.char_utils import full_to_half
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.llm_aided import llm_aided_title
from mineru.utils.model_utils import clean_memory
from mineru.backend.pipeline.pipeline_magic_model import MagicModel
from mineru.utils.ocr_utils import OcrConfidence, rotate_vertical_crop_if_needed
from mineru.version import __version__
from mineru.utils.hash_utils import bytes_md5
from mineru.utils.pdfium_guard import close_pdfium_document, pdfium_guard


def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=False):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    with pdfium_guard():
        page_w, page_h = map(int, page.get_size())
    magic_model = MagicModel(
        page_model_info,
        page,
        scale,
        page_pil_img,
        page_w,
        page_h,
        ocr_enable
    )

    """从magic_model对象中获取后面会用到的区块信息"""
    preproc_blocks = magic_model.get_preproc_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    all_image_spans = magic_model.get_all_image_spans()

    # 对image/table/chart/interline_equation的span截图
    for span in all_image_spans:
        if span["type"] in [
            ContentType.IMAGE,
            ContentType.TABLE,
            ContentType.CHART,
            ContentType.SEAL,
            ContentType.INTERLINE_EQUATION
        ]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    """构造page_info"""
    replace_inline_table_images(preproc_blocks, image_writer, page_index)

    page_info = make_page_info_dict(preproc_blocks, page_index, page_w, page_h, discarded_blocks)

    return page_info


def build_page_model_info(page_layout_dets, page_index, pil_img):
    page_info_dict = {'page_no': page_index, 'width': pil_img.width, 'height': pil_img.height}
    return {'layout_dets': page_layout_dets, 'page_info': page_info_dict}


def append_page_model_infos_to_middle_json(
    middle_json,
    page_model_infos,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    ocr_enable=False,
    progress_bar=None,
):
    for offset, (page_model_info, image_dict) in enumerate(zip(page_model_infos, images_list)):
        page_index = page_start_index + offset
        with pdfium_guard():
            page = pdf_doc[page_index]
        page_info = page_model_info_to_page_info(
            copy.deepcopy(page_model_info),
            image_dict,
            page,
            image_writer,
            page_index,
            ocr_enable=ocr_enable,
        )
        if page_info is None:
            with pdfium_guard():
                page_w, page_h = map(int, pdf_doc[page_index].get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)
        if progress_bar is not None:
            progress_bar.update(1)


def append_batch_results_to_middle_json(
    middle_json,
    batch_results,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    ocr_enable=False,
    model_list=None,
    progress_bar=None,
):
    page_model_infos = []
    for offset, (image_dict, page_layout_dets) in enumerate(zip(images_list, batch_results)):
        page_index = page_start_index + offset
        page_model_info = build_page_model_info(page_layout_dets, page_index, image_dict['img_pil'])
        page_model_infos.append(page_model_info)

    if model_list is not None:
        model_list.extend(page_model_infos)

    append_page_model_infos_to_middle_json(
        middle_json,
        page_model_infos,
        images_list,
        pdf_doc,
        image_writer,
        page_start_index=page_start_index,
        ocr_enable=ocr_enable,
        progress_bar=progress_bar,
    )


def _extract_text_from_block(block):
    text_parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.TEXT:
                text_parts.append(span.get("content", ""))
    return "".join(text_parts).strip()


def _iter_block_spans(block):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            yield span

    for sub_block in block.get("blocks", []):
        yield from _iter_block_spans(sub_block)


def _normalize_formula_tag_content(tag_content):
    tag_content = full_to_half(tag_content.strip())
    if tag_content.startswith("("):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith(")"):
        tag_content = tag_content[:-1].strip()
    return tag_content


def _get_interline_equation_span(block):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.INTERLINE_EQUATION:
                return span
    return None


def _append_formula_number_tag(equation_block, formula_number_block):
    equation_span = _get_interline_equation_span(equation_block)
    tag_content = _normalize_formula_tag_content(_extract_text_from_block(formula_number_block))
    if equation_span is not None:
        formula = equation_span.get("content", "")
        equation_span["content"] = f"{formula}\\tag{{{tag_content}}}"


def _optimize_formula_number_blocks(pdf_info_list):
    for page_info in pdf_info_list:
        optimized_blocks = []
        blocks = page_info.get("preproc_blocks", [])
        for index, block in enumerate(blocks):
            if block.get("type") != BlockType.FORMULA_NUMBER:
                optimized_blocks.append(block)
                continue

            prev_block = blocks[index - 1] if index > 0 else None
            if prev_block and prev_block.get("type") == BlockType.INTERLINE_EQUATION:
                _append_formula_number_tag(prev_block, block)
                continue

            next_block = blocks[index + 1] if index + 1 < len(blocks) else None
            next_next_block = blocks[index + 2] if index + 2 < len(blocks) else None
            if (
                next_block
                and next_block.get("type") == BlockType.INTERLINE_EQUATION
                and (next_next_block is None or next_next_block.get("type") != BlockType.FORMULA_NUMBER)
            ):
                _append_formula_number_tag(next_block, block)
                continue

            block["type"] = BlockType.TEXT
            optimized_blocks.append(block)

        page_info["preproc_blocks"] = optimized_blocks


def _apply_post_ocr(pdf_info_list, lang=None):
    need_ocr_list = []
    img_crop_list = []

    for page_info in pdf_info_list:
        for block in page_info.get('preproc_blocks', []):
            for span in _iter_block_spans(block):
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    # Keep post-OCR rec aligned with the main OCR pipeline for vertical tall crops.
                    img_crop_list.append(rotate_vertical_crop_if_needed(span['np_img']))
                    span.pop('np_img')

        for block in page_info.get('discarded_blocks', []):
            for span in _iter_block_spans(block):
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    # Keep post-OCR rec aligned with the main OCR pipeline for vertical tall crops.
                    img_crop_list.append(rotate_vertical_crop_if_needed(span['np_img']))
                    span.pop('np_img')

    if len(img_crop_list) == 0:
        return

    atom_model_manager = AtomModelSingleton()
    ocr_model = atom_model_manager.get_atom_model(
        atom_model_name='ocr',
        det_db_box_thresh=0.3,
        lang=lang
    )
    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
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


def _post_block_process(pdf_info_list):
    for page_info in pdf_info_list:
        for block_key in ["preproc_blocks", "para_blocks"]:
            for block in page_info.get(block_key, []):
                block_type = block.get("type")
                if block_type == BlockType.DOC_TITLE:
                    block["type"] = BlockType.TITLE
                    block["level"] = 1
                elif block_type == BlockType.PARAGRAPH_TITLE:
                    block["type"] = BlockType.TITLE
                    block["level"] = 2
                elif block_type == BlockType.VERTICAL_TEXT:
                    block["type"] = BlockType.TEXT


def finalize_middle_json(pdf_info_list, lang=None, ocr_enable=False):
    """Apply document-level post processing once all page_info entries are ready."""
    _apply_post_ocr(pdf_info_list, lang=lang)
    _optimize_formula_number_blocks(pdf_info_list)
    para_split(pdf_info_list)
    cross_page_table_merge(pdf_info_list)

    llm_aided_config = get_llm_aided_config()
    if llm_aided_config is not None:
        title_aided_config = llm_aided_config.get('title_aided', None)
        if title_aided_config is not None and title_aided_config.get('enable', False):
            llm_aided_title_start_time = time.time()
            llm_aided_title(pdf_info_list, title_aided_config)
            logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    _post_block_process(pdf_info_list)

    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(pdf_info_list) >= 10:
        clean_memory(get_device())


def init_middle_json():
    return {"pdf_info": [], "_backend": "pipeline", "_version_name": __version__}


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False, formula_enable=None):
    middle_json = init_middle_json()
    with tqdm(total=len(model_list), desc="Processing pages") as progress_bar:
        append_page_model_infos_to_middle_json(
            middle_json,
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            ocr_enable=ocr_enable,
            progress_bar=progress_bar,
        )

    finalize_middle_json(middle_json["pdf_info"], lang=lang, ocr_enable=ocr_enable)
    close_pdfium_document(pdf_doc)
    return middle_json


def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        'discarded_blocks': discarded_blocks,
    }
    return return_dict
