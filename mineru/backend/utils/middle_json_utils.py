"""Shared middle_json build template for PDF backends (pipeline / vlm / hybrid).

Each backend still owns its ``init``, ``append``, and ``finalize`` helpers;
this module also provides shared post-processing helpers that were duplicated
across backends.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterator, TypeVar

from tqdm import tqdm

from ...data.data_reader_writer import DataWriter
from ...types import Block, PageInfo, Span
from ...utils.ocr_utils import OcrConfidence, rotate_vertical_crop_if_needed
from ...utils.pdfium_guard import close_pdfium_child, close_pdfium_document, pdfium_guard
from ..pipeline.model_init import run_ocr_rec_inference

T = TypeVar("T")


def build_middle_json(
    model_list: list[T],
    images_list: list[dict[str, Any]],
    pdf_doc: Any,
    image_writer: DataWriter,
    *,
    init_fn: Callable[..., dict[str, Any]],
    page_cvt_fn: Callable[[T, dict, Any, DataWriter | None, int], PageInfo],
    finalize_fn: Callable[..., None],
    **kwargs: Any,
) -> dict[str, Any]:
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

    middle_json = init_fn(**kwargs)
    with tqdm(total=len(model_list), desc="Processing pages") as progress_bar:
        append_pages(
            middle_json,
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            page_cvt_fn=page_cvt_fn,
            progress_bar=progress_bar,
            **kwargs,
        )

    finalize_fn(middle_json["pdf_info"], **kwargs)
    close_pdfium_document(pdf_doc)
    return middle_json


def append_pages(
    middle_json: dict[str, Any],
    model_list: list[T],
    images_list: list[dict[str, Any]],
    pdf_doc: Any,
    image_writer: DataWriter | None,
    *,
    page_cvt_fn: Callable[[T, dict, Any, DataWriter | None, int], PageInfo],
    page_start_index: int = 0,
    progress_bar: Any = None,
    **kwargs: Any,
) -> None:
    """Append per-page results to ``middle_json["pdf_info"]``.

    Shared across Pipeline / VLM / Hybrid.  The only backend-specific piece
    is ``page_cvt_fn``, which converts one page's model output into a
    page_info dict.
    """

    for offset, (page_data, image_dict) in enumerate(zip(model_list, images_list)):
        page_index = page_start_index + offset
        page = None
        try:
            with pdfium_guard():
                page = pdf_doc[page_index]
            page_info = page_cvt_fn(copy.deepcopy(page_data), image_dict, page, image_writer, page_index, **kwargs)
            if page_info is None:
                with pdfium_guard():
                    page_w, page_h = map(int, page.get_size())
                page_info = PageInfo(
                    preproc_blocks=[],
                    page_idx=page_index,
                    page_size=[page_w, page_h],
                    discarded_blocks=[],
                )
        finally:
            close_pdfium_child(page)
        middle_json["pdf_info"].append(page_info)
        if progress_bar is not None:
            progress_bar.update(1)


def apply_post_ocr(pdf_info_list: list[PageInfo], ocr_model: Any) -> None:
    """Run OCR recognition on residual ``np_img`` crops inside spans.

    Common to Pipeline and Hybrid.  The caller provides the OCR model object
    (which has an ``ocr`` attribute pointing to the inference engine).
    """

    need_ocr_list = []
    img_crop_list = []

    for page_info in pdf_info_list:
        for blocks in [page_info.preproc_blocks, page_info.discarded_blocks]:
            for block in blocks:
                for span in _iter_block_spans(block):
                    if "np_img" in span:
                        need_ocr_list.append(span)
                        img_crop_list.append(rotate_vertical_crop_if_needed(span["np_img"]))
                        span.pop("np_img")

    if not img_crop_list:
        return

    ocr_res_list = run_ocr_rec_inference(ocr_model.ocr, img_crop_list, det=False, tqdm_enable=True)[0]
    assert len(ocr_res_list) == len(need_ocr_list), f"ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}"
    for index, span in enumerate(need_ocr_list):
        ocr_text, ocr_score = ocr_res_list[index]
        if ocr_score > OcrConfidence.min_confidence:
            span["content"] = ocr_text
            span["score"] = float(f"{ocr_score:.3f}")
        else:
            span["content"] = ""
            span["score"] = 0.0


def _iter_block_spans(block: Block) -> Iterator[Span]:
    """Depth-first generator yielding every span in a block tree."""
    for line in block.lines:
        yield from line.spans
    for child in block.blocks:
        yield from _iter_block_spans(child)
