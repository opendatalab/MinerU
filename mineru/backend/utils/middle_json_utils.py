"""Shared middle_json build template for current PDF backends.

Each backend still owns its ``init``, ``append``, and ``finalize`` helpers;
this module also provides shared post-processing helpers that were duplicated
across backends.
"""

from __future__ import annotations

import copy
from typing import Any, Iterator, TypeVar

from ..pdf.model_output_to_middle_json import blocks_to_page_info
from ...types import Block, PageInfo, Span
from ...utils.image_payload import ImagePayloadCache
from ...utils.page_index import resolve_output_page_idx
from ...utils.pdf_document import PDFDocument

T = TypeVar("T")


def append_pages(
    middle_json: list[PageInfo],
    model_list: list[T],
    images_list: list[dict[str, Any]],
    pdf_doc: PDFDocument,
    page_start_index: int = 0,
    page_index_map: list[int] | None = None,
    progress_bar: Any = None,
    image_cache: ImagePayloadCache | None = None,
) -> None:
    """Append per-page results to `middle_json` list.
    """
    for offset, (page_data, image_dict) in enumerate(zip(model_list, images_list)):
        physical_page_idx = page_start_index + offset
        output_page_idx = resolve_output_page_idx(physical_page_idx, page_index_map)

        pdf_page = pdf_doc[physical_page_idx]
        page_info = blocks_to_page_info(
            copy.deepcopy(page_data),
            image_dict,
            pdf_page,
            output_page_idx,
            image_cache,
        )

        if page_info is None:
            page_info = PageInfo(
                blocks=[],
                page_idx=output_page_idx,
                _backend=None,
            )

        middle_json.append(page_info)
        if progress_bar is not None:
            progress_bar.update(1)


def _iter_block_spans(block: Block) -> Iterator[Span]:
    """Depth-first generator yielding every span in a block tree."""
    for line in block.lines:
        yield from line.spans
    for child in block.blocks:
        yield from _iter_block_spans(child)
