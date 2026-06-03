"""Shared middle_json build template for PDF backends (pipeline / vlm / hybrid).

Each backend still owns its ``init``, ``append``, and ``finalize`` helpers;
this module just removes the copy-pasted orchestration boilerplate.
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
    append_fn: Callable[..., None],
    finalize_fn: Callable[..., None],
    **kwargs: Any,
) -> dict:
    """Shared orchestration for the three PDF backends.

    Parameters
    ----------
    model_list / images_list / pdf_doc / image_writer:
        Backend-specific inputs forwarded to ``append_fn``.
    init_fn(**kwargs) -> middle_json:
        Creates the initial middle_json dict.
    append_fn(middle_json, model_list, images_list, pdf_doc,
              image_writer, progress_bar, **kwargs):
        Per-page processing loop.
    finalize_fn(pdf_info_list, **kwargs):
        Post-processing applied once all pages are ready.
    **kwargs:
        Backend-specific parameters forwarded to each helper.
    """
    from ....utils.pdfium_guard import close_pdfium_document

    middle_json = init_fn(**kwargs)
    with tqdm(total=len(model_list), desc="Processing pages") as progress_bar:
        append_fn(
            middle_json,
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            progress_bar=progress_bar,
            **kwargs,
        )
    finalize_fn(middle_json["pdf_info"], **kwargs)
    close_pdfium_document(pdf_doc)
    return middle_json
