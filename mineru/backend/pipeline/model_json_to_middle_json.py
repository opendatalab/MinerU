# Copyright (c) Opendatalab. All rights reserved.
import os
import time

from loguru import logger
from tqdm import tqdm

from mineru.backend.utils import cross_page_table_merge
from mineru.utils.config_reader import get_device, get_llm_aided_config, get_formula_enable
from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.backend.pipeline.para_split import para_split
from mineru.utils.char_utils import full_to_half
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.llm_aided import llm_aided_title
from mineru.utils.model_utils import clean_memory
from mineru.backend.pipeline.pipeline_magic_model import MagicModel
from mineru.utils.ocr_utils import OcrConfidence
from mineru.version import __version__
from mineru.utils.hash_utils import bytes_md5


def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=False):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
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
    page_info = make_page_info_dict(preproc_blocks, page_index, page_w, page_h, discarded_blocks)

    return page_info


def _extract_text_from_block(block):
    text_parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.TEXT:
                text_parts.append(span.get("content", ""))
    return "".join(text_parts).strip()


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


def _optimize_formula_number_blocks(pdf_info_list):
    for page_info in pdf_info_list:
        optimized_blocks = []
        for block in page_info.get("preproc_blocks", []):
            if block.get("type") != BlockType.FORMULA_NUMBER:
                optimized_blocks.append(block)
                continue

            prev_block = optimized_blocks[-1] if optimized_blocks else None
            if prev_block and prev_block.get("type") == BlockType.INTERLINE_EQUATION:
                equation_span = _get_interline_equation_span(prev_block)
                tag_content = _normalize_formula_tag_content(_extract_text_from_block(block))
                if equation_span is not None:
                    formula = equation_span.get("content", "")
                    equation_span["content"] = f"{formula}\\tag{{{tag_content}}}"
                continue

            block["type"] = BlockType.TEXT
            optimized_blocks.append(block)

        page_info["preproc_blocks"] = optimized_blocks


def _apply_post_ocr(pdf_info_list, lang=None):
    need_ocr_list = []
    img_crop_list = []
    text_block_list = []

    for page_info in pdf_info_list:
        for block in page_info['preproc_blocks']:
            if 'blocks' in block:
                for sub_block in block['blocks']:
                    if sub_block.get("type", "").endswith('caption') or sub_block.get("type", "").endswith('footnote'):
                        text_block_list.append(sub_block)
            elif block["type"] not in [BlockType.INTERLINE_EQUATION, BlockType.SEAL]:
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


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False):
    middle_json = {"pdf_info": [], "_backend":"pipeline", "_version_name": __version__}
    for page_index, page_model_info in tqdm(enumerate(model_list), total=len(model_list), desc="Processing pages"):
        page = pdf_doc[page_index]
        image_dict = images_list[page_index]
        page_info = page_model_info_to_page_info(
            page_model_info, image_dict, page, image_writer, page_index, ocr_enable=ocr_enable,
        )
        if page_info is None:
            page_w, page_h = map(int, page.get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)

    """后置ocr处理"""
    _apply_post_ocr(middle_json["pdf_info"], lang=lang)

    """formula_number优化"""
    _optimize_formula_number_blocks(middle_json["pdf_info"])

    """分段"""
    para_split(middle_json["pdf_info"])

    """表格跨页合并"""
    cross_page_table_merge(middle_json["pdf_info"])

    """llm优化"""
    llm_aided_config = get_llm_aided_config()

    if llm_aided_config is not None:
        """标题优化"""
        title_aided_config = llm_aided_config.get('title_aided', None)
        if title_aided_config is not None:
            if title_aided_config.get('enable', False):
                llm_aided_title_start_time = time.time()
                llm_aided_title(middle_json["pdf_info"], title_aided_config)
                logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    """清理内存"""
    pdf_doc.close()
    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(model_list) >= 10:
        clean_memory(get_device())

    return middle_json


def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        'discarded_blocks': discarded_blocks,
    }
    return return_dict
