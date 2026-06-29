# Copyright (c) Opendatalab. All rights reserved.
from typing import Any

from ...types import BlockType, PageInfo
from ...utils.hash_utils import bytes_md5
from ...utils.image_payload import ImagePayloadCache
from ...utils.page_index import resolve_output_page_idx
from ...utils.pdf_document import PDFDocument, PDFPage
from ...utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from ..utils.formula_number import optimize_pipeline_formula_number_blocks
from ..utils.middle_json_utils import append_pages, apply_post_ocr
from ..utils.runtime_utils import cross_page_table_merge
from ..utils.visual_span_utils import cut_visual_spans_in_blocks
from .model_init import AtomModelSingleton
from .para_split import para_split
from .pipeline_magic_model import MagicModel


def blocks_to_page_info(
    page_model_info: dict[str, Any],
    image_dict: dict[str, Any],
    pdf_page: PDFPage,
    page_index: int,
    ocr_enable: bool = False,
    image_cache: ImagePayloadCache | None = None,
) -> PageInfo:
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    page_w, page_h = map(int, pdf_page.size)

    magic_model = MagicModel(page_model_info, pdf_page, scale, page_pil_img, page_w, page_h, ocr_enable)

    """从magic_model对象中获取后面会用到的区块信息"""
    preproc_blocks = magic_model.get_preproc_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()

    cut_visual_spans_in_blocks(
        [*preproc_blocks, *discarded_blocks],
        page_pil_img,
        page_img_md5,
        page_index,
        scale=scale,
        image_cache=image_cache,
    )

    return PageInfo(
        preproc_blocks=preproc_blocks,
        page_idx=page_index,
        page_size=(page_w, page_h),
        discarded_blocks=discarded_blocks,
        _backend="pipeline",
    )


def build_page_model_info(page_layout_dets: list[dict[str, Any]], page_index: int, pil_img: Any) -> dict[str, Any]:
    page_info_dict = {"page_no": page_index, "width": pil_img.width, "height": pil_img.height}
    return {"layout_dets": page_layout_dets, "page_info": page_info_dict}


def append_batch_results_to_middle_json(
    middle_json: list[PageInfo],
    batch_results: list[list[dict[str, Any]]],
    images_list: list[dict[str, Any]],
    pdf_doc: PDFDocument,
    page_start_index: int = 0,
    ocr_enable: bool = False,
    model_list: list[dict[str, Any]] | None = None,
    page_index_map: list[int] | None = None,
    progress_bar: Any = None,
    image_cache: ImagePayloadCache | None = None,
) -> None:
    page_model_infos = []
    for offset, (image_dict, page_layout_dets) in enumerate(zip(images_list, batch_results)):
        physical_page_idx = page_start_index + offset
        output_page_idx = resolve_output_page_idx(physical_page_idx, page_index_map)
        page_model_info = build_page_model_info(page_layout_dets, output_page_idx, image_dict["img_pil"])
        page_model_infos.append(page_model_info)

    if model_list is not None:
        model_list.extend(page_model_infos)

    append_pages(
        middle_json,
        page_model_infos,
        images_list,
        pdf_doc,
        page_cvt_fn=blocks_to_page_info,
        page_start_index=page_start_index,
        page_index_map=page_index_map,
        ocr_enable=ocr_enable,
        progress_bar=progress_bar,
        image_cache=image_cache,
    )


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
    optimize_pipeline_formula_number_blocks(pages)
    para_split(pages)
    cross_page_table_merge(pages)
    apply_title_leveling_to_pdf_info(pages)
    _post_block_process(pages)


def finalize_middle_json(pages: list[PageInfo], lang: str | None = None) -> None:
    """Apply document-level post processing once all page_info entries are ready."""
    apply_server_side_postprocess(pages, lang=lang)
    finalize_middle_json_from_preproc(pages)
