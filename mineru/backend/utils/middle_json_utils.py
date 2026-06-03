"""Shared middle_json build template for PDF backends (pipeline / vlm / hybrid).

Each backend still owns its ``init``, ``append``, and ``finalize`` helpers;
this module also provides shared post-processing helpers that were duplicated
across backends.
"""

from __future__ import annotations

from typing import Any, Callable

from tqdm import tqdm


def build_middle_json(
    model_list: list,
    images_list: list,
    pdf_doc: Any,
    image_writer: Any,
    *,
    init_fn: Callable[..., dict],
    page_cvt_fn: Callable[..., dict],
    finalize_fn: Callable[..., None],
    **kwargs: Any,
) -> dict:
    """Shared orchestration for the three PDF backends.

    Parameters
    ----------
    model_list / images_list / pdf_doc / image_writer:
        Backend-specific inputs.
    init_fn(**kwargs) -> middle_json:
        Creates the initial middle_json dict.
    page_cvt_fn(page_data, image_dict, page, image_writer, page_index, **kwargs) -> page_info:
        Converts one page's model output into a page_info dict.
    finalize_fn(pdf_info_list, **kwargs):
        Post-processing applied once all pages are ready.
    """
    from ....utils.pdfium_guard import close_pdfium_child, close_pdfium_document, pdfium_guard

    middle_json = init_fn(**kwargs)
    with tqdm(total=len(model_list), desc="Processing pages") as progress_bar:
        for offset, (page_data, image_dict) in enumerate(zip(model_list, images_list)):
            page_index = offset
            page = None
            try:
                with pdfium_guard():
                    page = pdf_doc[page_index]
                page_info = page_cvt_fn(
                    page_data, image_dict, page, image_writer, page_index, **kwargs
                )
            finally:
                close_pdfium_child(page)
            middle_json["pdf_info"].append(page_info)
            progress_bar.update(1)

    finalize_fn(middle_json["pdf_info"], **kwargs)
    close_pdfium_document(pdf_doc)
    return middle_json


def apply_post_ocr(pdf_info_list: list, ocr_model: Any) -> None:
    """Run OCR recognition on residual ``np_img`` crops inside spans.

    Common to Pipeline and Hybrid.  The caller provides the OCR model object
    (which has an ``ocr`` attribute pointing to the inference engine).
    """
    from mineru.backend.pipeline.model_init import run_ocr_rec_inference
    from mineru.utils.ocr_utils import OcrConfidence, rotate_vertical_crop_if_needed

    need_ocr_list = []
    img_crop_list = []

    for page_info in pdf_info_list:
        for blocks_key in ("preproc_blocks", "discarded_blocks"):
            for block in page_info.get(blocks_key, []):
                for span in _iter_block_spans(block):
                    if "np_img" in span:
                        need_ocr_list.append(span)
                        img_crop_list.append(rotate_vertical_crop_if_needed(span["np_img"]))
                        span.pop("np_img")

    if not img_crop_list:
        return

    ocr_res_list = run_ocr_rec_inference(
        ocr_model.ocr, img_crop_list, det=False, tqdm_enable=True
    )[0]
    assert len(ocr_res_list) == len(need_ocr_list), (
        f"ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}"
    )
    for index, span in enumerate(need_ocr_list):
        ocr_text, ocr_score = ocr_res_list[index]
        if ocr_score > OcrConfidence.min_confidence:
            span["content"] = ocr_text
            span["score"] = float(f"{ocr_score:.3f}")
        else:
            span["content"] = ""
            span["score"] = 0.0


def _iter_block_spans(block: dict):
    """Depth-first generator yielding every span in a block tree."""
    for line in block.get("lines", []):
        yield from line.get("spans", [])
    for child in block.get("blocks", []):
        yield from _iter_block_spans(child)
