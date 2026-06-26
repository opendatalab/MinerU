# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Any, Literal

from ..types import PageInfo
from .markdown import blocks_to_markdown
from .office.output import blocks_to_markdown as office_blocks_to_markdown
from .structured_content import block_to_structured_content, merge_adjacent_ref_text_blocks_for_content

Backend = Literal["pipeline", "vlm", "hybrid", "office"]


def _backend_from_pages(pages: list[PageInfo]) -> Backend | None:
    """Infer backend from the first page.  Returns ``None`` for empty input."""
    return pages[0]._backend if pages else None


# ------------------------------------------------------------------ #
#  Per-backend dispatch helpers for Content-List modes
#  (Markdown rendering moved to ``render/markdown.py``).
# ------------------------------------------------------------------ #


def _dispatch_make_content_list(
    backend: Backend,
    para_block: Any,
    img_bucket_path: str,
    page_idx: int,
    page_size: Any,
) -> Any:
    if backend == "office":
        from .office.output import make_blocks_to_content_list

        return make_blocks_to_content_list(para_block, img_bucket_path, page_idx)

    from .content_list import block_to_content_list

    item = block_to_content_list(para_block, img_bucket_path, page_idx, page_size)
    if item is None:
        return None
    return item.to_dict(skip_defaults=True)


def _dispatch_block_to_structured_content(
    backend: Backend,
    para_block: Any,
    img_bucket_path: str,
    page_size: Any,
) -> Any:
    if backend == "office":
        from .office.output import block_to_structured_content as office_block_to_structured_content

        return office_block_to_structured_content(para_block, img_bucket_path)

    return block_to_structured_content(para_block, img_bucket_path, page_size)


def _get_merged_para_blocks(paras_of_layout: list, paras_of_discarded: list, backend: Backend) -> list:
    """Build the para-block list for Content-List modes.

    PDF backends merge adjacent ``ref_text`` blocks before rendering so both
    legacy content_list and structured_content expose references as a list.
    """
    if backend != "office":
        return merge_adjacent_ref_text_blocks_for_content((paras_of_layout or []) + (paras_of_discarded or []))
    return (paras_of_layout or []) + (paras_of_discarded or [])


# ------------------------------------------------------------------ #
#  Public renderer functions
# ------------------------------------------------------------------ #


def render_markdown(
    pdf_info: list[PageInfo],
    img_bucket_path: str = "",
    *,
    formula_enable: bool = True,  # TODO
    table_enable: bool = True,  # TODO
    no_rich_content: bool = False,
    add_markers: bool = False,
) -> str:
    """Render pages to a single Markdown string.

    If *add_markers* is True, each page is prefixed with ``<!-- page N of M -->``.
    """
    backend = _backend_from_pages(pdf_info)
    total = len(pdf_info)
    output_md: list[str] = []
    if backend == "office":
        for page_info in pdf_info:
            page_md = office_blocks_to_markdown(
                para_blocks=page_info.para_blocks,
                img_bucket_path=img_bucket_path,
                no_rich_content=no_rich_content,
            )
            if add_markers:
                page_num = page_info.page_idx + 1
                output_md.append(f"<!-- page {page_num} of {total} -->")
            output_md.extend(page_md)
    else:  # PDF
        for page_info in pdf_info:
            page_md = blocks_to_markdown(
                para_blocks=page_info.para_blocks,
                img_bucket_path=img_bucket_path,
                table_as_image=not table_enable,
                formula_as_image=not formula_enable,
                no_rich_content=no_rich_content,
            )
            if add_markers:
                page_num = page_info.page_idx + 1
                output_md.append(f"<!-- page {page_num} of {total} -->")
            output_md.extend(page_md)
    return "\n\n".join(output_md)


def render_content_list(
    pdf_info: list[PageInfo],
    img_bucket_path: str = "",
) -> list[dict[str, Any]]:
    """Render pages to a flat Content List (V1)."""
    backend = _backend_from_pages(pdf_info)
    if backend is None:
        return []
    output_items: list[dict[str, Any]] = []
    for page_info in pdf_info:
        para_blocks = _get_merged_para_blocks(page_info.para_blocks, page_info.discarded_blocks, backend)
        if not para_blocks:
            continue
        for para_block in para_blocks:
            item = _dispatch_make_content_list(backend, para_block, img_bucket_path, page_info.page_idx, page_info.page_size)
            if item:
                output_items.append(item)
    return output_items


def render_structured_content(
    pdf_info: list[PageInfo],
    img_bucket_path: str = "",
) -> list[list[dict[str, Any]]]:
    """Render pages to per-page Structured Content."""
    backend = _backend_from_pages(pdf_info)
    if backend is None:
        return []
    output_lists: list[list[dict[str, Any]]] = []
    for page_info in pdf_info:
        para_blocks = _get_merged_para_blocks(page_info.para_blocks, page_info.discarded_blocks, backend)
        page_contents: list[dict[str, Any]] = []
        if para_blocks:
            for para_block in para_blocks:
                item = _dispatch_block_to_structured_content(backend, para_block, img_bucket_path, page_info.page_size)
                if item:
                    page_contents.append(item)
        output_lists.append(page_contents)
    return output_lists
