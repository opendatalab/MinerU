# Copyright (c) Opendatalab. All rights reserved.
from typing import Any

from ...data.data_reader_writer import DataWriter
from ...types import Block, PageInfo, Span
from ...utils.cut_image import cut_image_and_table
from ...utils.enum_class import BlockType, ContentType
from ...utils.hash_utils import bytes_md5
from ...utils.pdfium_guard import pdfium_guard
from ...utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from ..utils.char_utils import full_to_half
from ..utils.html_image_utils import replace_inline_table_images
from ..utils.middle_json_utils import append_pages, apply_post_ocr
from ..utils.runtime_utils import cross_page_table_merge
from .model_init import AtomModelSingleton
from .para_split import para_split
from .pipeline_magic_model import MagicModel


def blocks_to_page_info(
    page_model_info: dict[str, Any],
    image_dict: dict[str, Any],
    page: Any,
    image_writer: DataWriter | None,
    page_index: int,
    ocr_enable: bool = False,
) -> PageInfo:
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    with pdfium_guard():
        page_w, page_h = map(int, page.get_size())

    magic_model = MagicModel(page_model_info, page, scale, page_pil_img, page_w, page_h, ocr_enable)

    """从magic_model对象中获取后面会用到的区块信息"""
    preproc_blocks = magic_model.get_preproc_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    all_image_spans = magic_model.get_all_image_spans()

    # 对image/table/chart/interline_equation的span截图
    for span in all_image_spans:
        if span.type in [ContentType.IMAGE, ContentType.TABLE, ContentType.CHART, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    """构造page_info"""
    replace_inline_table_images(preproc_blocks, image_writer, page_index)

    return PageInfo(
        preproc_blocks=preproc_blocks,
        page_idx=page_index,
        page_size=(page_w, page_h),
        discarded_blocks=discarded_blocks,
    )


def build_page_model_info(page_layout_dets: list[dict[str, Any]], page_index: int, pil_img: Any) -> dict[str, Any]:
    page_info_dict = {"page_no": page_index, "width": pil_img.width, "height": pil_img.height}
    return {"layout_dets": page_layout_dets, "page_info": page_info_dict}


def append_batch_results_to_middle_json(
    middle_json: list[PageInfo],
    batch_results: list[list[dict[str, Any]]],
    images_list: list[dict[str, Any]],
    pdf_doc: Any,
    image_writer: DataWriter,
    page_start_index: int = 0,
    ocr_enable: bool = False,
    model_list: list[dict[str, Any]] | None = None,
    progress_bar: Any = None,
) -> None:
    page_model_infos = []
    for offset, (image_dict, page_layout_dets) in enumerate(zip(images_list, batch_results)):
        page_index = page_start_index + offset
        page_model_info = build_page_model_info(page_layout_dets, page_index, image_dict["img_pil"])
        page_model_infos.append(page_model_info)

    if model_list is not None:
        model_list.extend(page_model_infos)

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


def _extract_text_from_block(block: Block) -> str:
    text_parts = []
    for line in block.lines:
        for span in line.spans:
            if span.type == ContentType.TEXT:
                text_parts.append(span.content)
    return "".join(text_parts).strip()


def _normalize_formula_tag_content(tag_content: str) -> str:
    tag_content = full_to_half(tag_content.strip())
    if tag_content.startswith("("):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith(")"):
        tag_content = tag_content[:-1].strip()
    return tag_content


def _get_interline_equation_span(block: Block) -> Span | None:
    for line in block.lines:
        for span in line.spans:
            if span.type == ContentType.INTERLINE_EQUATION:
                return span
    return None


def _append_formula_number_tag(equation_block: Block, formula_number_block: Block) -> None:
    equation_span = _get_interline_equation_span(equation_block)
    tag_content = _normalize_formula_tag_content(_extract_text_from_block(formula_number_block))
    if equation_span is not None:
        formula = equation_span.content
        equation_span.content = f"{formula}\\tag{{{tag_content}}}"


def _optimize_formula_number_blocks(pages: list[PageInfo]) -> None:
    for page_info in pages:
        optimized_blocks = []
        blocks = page_info.preproc_blocks
        for index, block in enumerate(blocks):
            if block.type != BlockType.FORMULA_NUMBER:
                optimized_blocks.append(block)
                continue

            prev_block = blocks[index - 1] if index > 0 else None
            if prev_block and prev_block.type == BlockType.INTERLINE_EQUATION:
                _append_formula_number_tag(prev_block, block)
                continue

            next_block = blocks[index + 1] if index + 1 < len(blocks) else None
            next_next_block = blocks[index + 2] if index + 2 < len(blocks) else None
            if (
                next_block
                and next_block.type == BlockType.INTERLINE_EQUATION
                and (next_next_block is None or next_next_block.type != BlockType.FORMULA_NUMBER)
            ):
                _append_formula_number_tag(next_block, block)
                continue

            block.type = BlockType.TEXT
            optimized_blocks.append(block)
        page_info.preproc_blocks = optimized_blocks


def _apply_post_ocr(pages: list[PageInfo], lang: str | None = None) -> None:
    atom_model_manager = AtomModelSingleton()
    ocr_model = atom_model_manager.get_atom_model(
        atom_model_name="ocr",
        det_db_box_thresh=0.3,
        lang=lang,
    )
    apply_post_ocr(pages, ocr_model)


def _post_block_process(pages: list[PageInfo]) -> None:
    for page_info in pages:
        for blocks in [page_info.preproc_blocks, page_info.para_blocks]:
            for block in blocks:
                block_type = block.type
                if block_type == BlockType.DOC_TITLE:
                    block.type = BlockType.TITLE
                    block.level = 1
                elif block_type == BlockType.PARAGRAPH_TITLE:
                    block.type = BlockType.TITLE
                    block.level = 2
                elif block_type == BlockType.VERTICAL_TEXT:
                    block.type = BlockType.TEXT


def apply_server_side_postprocess(pages: list[PageInfo], lang: str | None = None) -> None:
    """执行只能在服务端完成的后处理；目前仅包含依赖 OCR 模型的 post-OCR。"""
    _apply_post_ocr(pages, lang=lang)


def finalize_middle_json_from_preproc(pages: list[PageInfo]) -> None:
    """从 preproc_blocks 执行确定性 finalize，供服务端完整路径和客户端复用。"""
    _optimize_formula_number_blocks(pages)
    para_split(pages)
    cross_page_table_merge(pages)
    apply_title_leveling_to_pdf_info(pages)
    _post_block_process(pages)


def finalize_middle_json(pages: list[PageInfo], lang: str | None = None) -> None:
    """Apply document-level post processing once all page_info entries are ready."""
    apply_server_side_postprocess(pages, lang=lang)
    finalize_middle_json_from_preproc(pages)
