# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Any, Literal

from ..render.markdown import blocks_to_markdown
from ..types import PageInfo
from ..utils.enum_class import MakeMode

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
) -> Any:  # type: ignore[return]
    if backend == "pipeline":
        from ..backend.pipeline.pipeline_middle_json_mkcontent import make_blocks_to_content_list

        return make_blocks_to_content_list(para_block, img_bucket_path, page_idx, page_size)
    if backend == "office":
        from ..backend.office.mkcontent.output_builders import make_blocks_to_content_list

        return make_blocks_to_content_list(para_block, img_bucket_path, page_idx)
    # vlm / hybrid
    from ..backend.vlm.vlm_middle_json_mkcontent import make_blocks_to_content_list

    return make_blocks_to_content_list(para_block, img_bucket_path, page_idx, page_size)


def _dispatch_make_content_list_v2(
    backend: Backend,
    para_block: Any,
    img_bucket_path: str,
    page_size: Any,
) -> Any:  # type: ignore[return]
    if backend == "pipeline":
        from ..backend.pipeline.pipeline_middle_json_mkcontent import make_blocks_to_content_list_v2

        return make_blocks_to_content_list_v2(para_block, img_bucket_path, page_size)
    if backend == "office":
        from ..backend.office.mkcontent.output_builders import make_blocks_to_content_list_v2

        return make_blocks_to_content_list_v2(para_block, img_bucket_path)
    # vlm / hybrid
    from ..backend.vlm.vlm_middle_json_mkcontent import make_blocks_to_content_list_v2

    return make_blocks_to_content_list_v2(para_block, img_bucket_path, page_size)


def _get_merged_para_blocks(paras_of_layout: list, paras_of_discarded: list, backend: Backend) -> list:
    """Build the para-block list for Content-List modes.

    The pipeline backend runs ``merge_adjacent_ref_text_blocks_for_content``
    before merging layout and discarded blocks; the other backends simply
    concatenate them.
    """
    if backend == "pipeline":
        from ..backend.pipeline.pipeline_middle_json_mkcontent import merge_adjacent_ref_text_blocks_for_content

        return merge_adjacent_ref_text_blocks_for_content((paras_of_layout or []) + (paras_of_discarded or []))
    return (paras_of_layout or []) + (paras_of_discarded or [])


# ------------------------------------------------------------------ #
#  Public renderer functions
# ------------------------------------------------------------------ #


def render_markdown(
    pdf_info: list[PageInfo],
    img_bucket_path: str = "",
    *,
    formula_enable: bool = True,
    table_enable: bool = True,
    make_mode: str = MakeMode.MM_MD,
) -> str:
    """Render pages to a single Markdown string."""
    backend = _backend_from_pages(pdf_info)
    if backend is None:
        return ""
    assert backend != "office", "temp not support"
    output_md: list[str] = []
    for page_info in pdf_info:
        paras_of_layout = page_info.para_blocks
        if not paras_of_layout:
            continue
        page_md = blocks_to_markdown(
            para_blocks=paras_of_layout,
            img_bucket_path=img_bucket_path,
            table_as_image=not table_enable,
            formula_as_image=not formula_enable,
            no_rich_content=(make_mode == MakeMode.NLP_MD),
        )
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


def render_content_list_v2(
    pdf_info: list[PageInfo],
    img_bucket_path: str = "",
) -> list[list[dict[str, Any]]]:
    """Render pages to a per-page Content List (V2)."""
    backend = _backend_from_pages(pdf_info)
    if backend is None:
        return []
    output_lists: list[list[dict[str, Any]]] = []
    for page_info in pdf_info:
        para_blocks = _get_merged_para_blocks(page_info.para_blocks, page_info.discarded_blocks, backend)
        page_contents: list[dict[str, Any]] = []
        if para_blocks:
            for para_block in para_blocks:
                item = _dispatch_make_content_list_v2(backend, para_block, img_bucket_path, page_info.page_size)
                if item:
                    page_contents.append(item)
        output_lists.append(page_contents)
    return output_lists
