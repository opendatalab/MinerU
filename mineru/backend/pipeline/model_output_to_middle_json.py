# Copyright (c) Opendatalab. All rights reserved.
from typing import Any

from mineru.backend.utils.html_image_utils import replace_inline_table_images
from mineru.backend.utils.runtime_utils import cross_page_table_merge
from mineru.backend.pipeline.model_init import (
    AtomModelSingleton,
)
from mineru.backend.pipeline.para_split import para_split
from mineru.utils.char_utils import full_to_half
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from mineru.parser.types import PageInfo, block_from_dict
from mineru.backend.pipeline.pipeline_magic_model import MagicModel
from mineru.version import __version__
from mineru.utils.hash_utils import bytes_md5
from mineru.utils.pdfium_guard import pdfium_guard


def blocks_to_page_info(page_model_info: dict, image_dict: dict, page: Any, image_writer: Any, page_index: int, ocr_enable: bool = False) -> PageInfo:
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
            ContentType.INTERLINE_EQUATION
        ]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    """构造page_info"""
    replace_inline_table_images(preproc_blocks, image_writer, page_index)

    page_info = make_page_info_dict(
        [block_from_dict(b) for b in preproc_blocks],
        page_index, page_w, page_h,
        [block_from_dict(b) for b in discarded_blocks],
    )

    return page_info


def build_page_model_info(page_layout_dets, page_index, pil_img):
    page_info_dict = {'page_no': page_index, 'width': pil_img.width, 'height': pil_img.height}
    return {'layout_dets': page_layout_dets, 'page_info': page_info_dict}


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

    from mineru.backend.utils.middle_json_utils import append_pages

    append_pages(
        middle_json,
        page_model_infos,
        images_list,
        pdf_doc,
        image_writer,
        page_cvt_fn=blocks_to_page_info,
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
    from mineru.backend.utils.middle_json_utils import apply_post_ocr

    atom_model_manager = AtomModelSingleton()
    ocr_model = atom_model_manager.get_atom_model(
        atom_model_name="ocr",
        det_db_box_thresh=0.3,
        lang=lang,
    )
    apply_post_ocr(pdf_info_list, ocr_model)


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


def apply_server_side_postprocess(pdf_info_list, lang=None):
    """执行只能在服务端完成的后处理；目前仅包含依赖 OCR 模型的 post-OCR。"""
    _apply_post_ocr(pdf_info_list, lang=lang)


def finalize_middle_json_from_preproc(pdf_info_list):
    """从 preproc_blocks 执行确定性 finalize，供服务端完整路径和客户端复用。"""
    _optimize_formula_number_blocks(pdf_info_list)
    para_split(pdf_info_list)
    cross_page_table_merge(pdf_info_list)
    apply_title_leveling_to_pdf_info(pdf_info_list)
    _post_block_process(pdf_info_list)


def finalize_middle_json(
    pdf_info_list,
    lang=None,
):
    """Apply document-level post processing once all page_info entries are ready."""
    apply_server_side_postprocess(pdf_info_list, lang=lang)
    finalize_middle_json_from_preproc(pdf_info_list)


def init_middle_json():
    return {"pdf_info": [], "_backend": "pipeline", "_version_name": __version__}


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False, formula_enable=None):
    from mineru.backend.utils.middle_json_utils import build_middle_json

    return build_middle_json(
        model_list, images_list, pdf_doc, image_writer,
        init_fn=init_middle_json,
        page_cvt_fn=blocks_to_page_info,
        finalize_fn=finalize_middle_json,
        lang=lang,
        ocr_enable=ocr_enable,
    )


def make_page_info_dict(blocks: list, page_id: int, page_w: int, page_h: int, discarded_blocks: list) -> PageInfo:
    return PageInfo(
        preproc_blocks=blocks,
        page_idx=page_id,
        page_size=[page_w, page_h],
        discarded_blocks=discarded_blocks,
    )
