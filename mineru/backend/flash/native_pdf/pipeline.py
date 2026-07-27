# Copyright (c) Opendatalab. All rights reserved.

"""编排 Flash 原生 PDF 的页面准备、语义处理和输出归一化。"""

from __future__ import annotations

from typing import Any


from mineru.backend.utils.xycut_pp_sorter import sort_entries
from mineru.utils.pdf_document import PDFDocument, get_lines_from_chars

from .models import (
    _LineItem,
    _PageSource,
    _PreparedPage,
)
from .geometry import (
    _bbox_overlap_in_smaller,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _normalize_bbox_to_thousandths,
    _rotate_bbox_to_upright,
)
from .native_text import (
    _build_native_line_items,
    _get_pdf_drawing_lines,
    _median_native_glyph_width,
    _sanitize_pdf_control_text,
)
from .line_merging import (
    _merge_overlapping_inline_text_clusters,
    _merge_post_semantic_text_runs,
    _merge_same_baseline_text_lines,
    _merge_title_resolved_visual_rows,
    _restore_dense_split_visual_rows,
)
from .tables import (
    _detect_table_candidates,
    _materialize_table_blocks,
)
from .graphics import (
    _IMAGE_CONTAINER_OVERLAP_THRESHOLD,
    _build_form_image_blocks,
    _build_graphic_like_blocks,
    _build_raster_image_blocks,
    _detect_strong_graphic_bboxes,
    _form_supersedes_nested_bbox,
    _select_form_image_bboxes,
)
from .formulas import (
    _build_formula_like_blocks,
    _build_vector_formula_blocks,
)
from .auxiliary_text import (
    _classify_page_auxiliary_text,
    _classify_repeated_page_marginals,
    _classify_repeated_visual_headers,
)
from .titles import _classify_page_titles
from .text_blocks import _build_text_blocks


_TEXT_SEMANTIC_TYPES = {
    "doc_title",
    "paragraph_title",
    "header",
    "footer",
    "page_number",
    "page_footnote",
    "aside_text",
}


_OUTPUT_BLOCK_TYPES = {"text", "table", "image", "equation"} | _TEXT_SEMANTIC_TYPES


def _analyze_native_document(pdf_doc: PDFDocument) -> list[list[dict[str, Any]]]:
    """逐页读取数字 PDF，并在轻量页面上完成跨页文本类型判定。"""

    prepared_pages: list[_PreparedPage] = []
    for page_idx in range(pdf_doc.page_count):
        page_size = pdf_doc.page_size(page_idx)
        chars = pdf_doc.get_page_chars(page_idx)
        lines = _build_native_line_items(
            get_lines_from_chars(chars),
            page_size,
            page_rotation=pdf_doc.page_rotation(page_idx),
        )
        drawing_lines = _get_pdf_drawing_lines(pdf_doc, page_idx)
        source = _PageSource(
            page_size=page_size,
            lines=lines,
            chars=chars,
            drawing_lines=drawing_lines,
            image_bboxes=pdf_doc.get_page_image_bboxes(page_idx),
            form_bboxes=pdf_doc.get_page_form_bboxes(page_idx),
            path_infos=pdf_doc.get_page_path_infos(page_idx),
        )
        prepared_pages.append(_prepare_page_source(source))

    _classify_repeated_visual_headers(prepared_pages)
    _classify_repeated_page_marginals(prepared_pages)
    return [
        _finalize_prepared_page(prepared, page_index)
        for page_index, prepared in enumerate(prepared_pages)
    ]


def _prepare_page_source(source: _PageSource) -> _PreparedPage:
    """先认领视觉容器，再标注辅助文本并留下可跨页比较的轻量文本行。"""

    form_bboxes = _select_form_image_bboxes(source)
    strong_graphic_bboxes = _detect_strong_graphic_bboxes(source)
    candidates = [
        candidate
        for candidate in _detect_table_candidates(
            source,
            excluded_bboxes=strong_graphic_bboxes,
        )
        if not any(
            _form_supersedes_nested_bbox(form_bbox, candidate.bbox)
            for form_bbox in form_bboxes
        )
    ]
    table_blocks, claimed_line_indices = _materialize_table_blocks(
        source,
        candidates,
    )
    table_bboxes = [block["bbox"] for block in table_blocks]
    active_form_bboxes = [
        form_bbox
        for form_bbox in form_bboxes
        if not any(
            _bbox_overlap_in_smaller(form_bbox, table_bbox)
            >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for table_bbox in table_bboxes
        )
    ]
    form_image_blocks, claimed_form_line_indices = _build_form_image_blocks(
        source,
        active_form_bboxes,
        claimed_line_indices,
    )
    graphic_blocks, claimed_graphic_line_indices = _build_graphic_like_blocks(
        source,
        table_bboxes + active_form_bboxes,
        claimed_line_indices | claimed_form_line_indices,
        strong_graphic_bboxes,
    )
    raster_image_blocks, claimed_raster_line_indices = _build_raster_image_blocks(
        source,
        table_blocks + form_image_blocks + graphic_blocks,
        claimed_line_indices | claimed_form_line_indices | claimed_graphic_line_indices,
    )
    vector_formula_blocks, claimed_vector_number_indices = _build_vector_formula_blocks(
        source,
        table_blocks + form_image_blocks + graphic_blocks + raster_image_blocks,
        claimed_line_indices
        | claimed_form_line_indices
        | claimed_graphic_line_indices
        | claimed_raster_line_indices,
    )
    claimed_line_indices = (
        claimed_line_indices
        | claimed_form_line_indices
        | claimed_graphic_line_indices
        | claimed_raster_line_indices
        | claimed_vector_number_indices
    )
    remaining_lines = _merge_same_baseline_text_lines(
        [line for line in source.lines if line.source_index not in claimed_line_indices],
        source.page_size,
        table_bboxes,
    )
    remaining_lines = _merge_overlapping_inline_text_clusters(
        remaining_lines,
        source.page_size,
        table_bboxes,
    )
    _compact_prepared_lines(remaining_lines, source.page_size)
    prepared = _PreparedPage(
        page_size=source.page_size,
        remaining_lines=remaining_lines,
        table_bboxes=table_bboxes,
        drawing_lines=source.drawing_lines,
        fixed_blocks=(
            table_blocks
            + form_image_blocks
            + graphic_blocks
            + raster_image_blocks
            + vector_formula_blocks
        ),
    )
    _classify_page_auxiliary_text(prepared)
    return prepared


def _compact_prepared_lines(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> None:
    """缓存后续仍需的字符尺度并释放字符字典，限制跨页阶段内存占用。"""

    for line in lines:
        if line.median_glyph_width is None:
            line.median_glyph_width = _median_native_glyph_width(line, page_size)
        line.chars.clear()


def _finalize_prepared_page(
    prepared: _PreparedPage,
    page_index: int,
) -> list[dict[str, Any]]:
    """按预分类语义、公式、标题、正文的优先级完成单页文本并排序。"""

    semantic_lines = [line for line in prepared.remaining_lines if line.semantic_type is not None]
    formula_input = [line for line in prepared.remaining_lines if line.semantic_type is None]
    formula_blocks, remaining_lines = _build_formula_like_blocks(
        formula_input,
        prepared.table_bboxes,
        prepared.page_size,
    )
    remaining_lines = _restore_dense_split_visual_rows(
        remaining_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    _classify_page_titles(
        remaining_lines,
        prepared.page_size,
        page_index=page_index,
        container_bboxes=[block["bbox"] for block in prepared.fixed_blocks],
    )
    remaining_lines = _merge_title_resolved_visual_rows(
        remaining_lines,
        prepared.page_size,
    )
    remaining_lines = _merge_post_semantic_text_runs(
        remaining_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    text_blocks = _build_text_blocks(
        semantic_lines + remaining_lines,
        prepared.table_bboxes,
        prepared.page_size,
        prepared.drawing_lines,
        page_footnote_groups=prepared.page_footnote_groups,
    )
    absolute_blocks = prepared.fixed_blocks + formula_blocks + text_blocks
    sorted_blocks = _sort_blocks_with_visual_row_groups(absolute_blocks, prepared.page_size)
    return [
        normalized
        for block in sorted_blocks
        if (normalized := _normalize_output_block(block, prepared.page_size)) is not None
    ]


def _analyze_page_source(source: _PageSource) -> list[dict[str, Any]]:
    """兼容单页测试入口；单页不凭边缘位置猜测页眉、页脚或页码。"""

    if not source.lines and not source.image_bboxes:
        return []
    return _finalize_prepared_page(_prepare_page_source(source), page_index=0)


def _sort_blocks_with_visual_row_groups(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """把同一粗行拆出的单行 block 包装成虚拟项排序，再按局部 x 顺序展开。"""

    grouped_indices: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        row_id = block.get("_single_run_row_id")
        if isinstance(row_id, int):
            grouped_indices.setdefault(row_id, []).append(index)

    virtual_groups: dict[int, dict[str, Any]] = {}
    consumed_indices: set[int] = set()
    for row_id, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        members = [blocks[index] for index in indices]
        virtual_group = {
            "type": "_xycut_visual_row_group",
            "bbox": _bbox_union_many([member["bbox"] for member in members]),
            "angle": members[0].get("angle", 0),
            "content": "",
            "_members": members,
        }
        virtual_groups[row_id] = virtual_group
        consumed_indices.update(indices)

    sortable_blocks = [block for index, block in enumerate(blocks) if index not in consumed_indices]
    sortable_blocks.extend(virtual_groups.values())
    sorted_payloads = sort_entries(sortable_blocks)
    output: list[dict[str, Any]] = []
    for payload in sorted_payloads:
        members = payload.get("_members")
        if not isinstance(members, list):
            output.append(payload)
            continue
        angle = int(payload.get("angle", 0) or 0) % 360
        members.sort(
            key=lambda member: (
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[0],
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[1],
            )
        )
        output.extend(members)
    return output


def _normalize_output_block(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> dict[str, Any] | None:
    """在排序完成后将绝对 bbox 裁剪并归一化为 model_list 坐标。"""

    page_width, page_height = page_size
    bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page_size)
    if bbox is None or page_width <= 0 or page_height <= 0:
        return None
    content = block.get("content")
    if not isinstance(content, str):
        return None
    content = _sanitize_pdf_control_text(content, preserve_newlines=True)
    block_type = block.get("type")
    normalized_type = block_type if block_type in _OUTPUT_BLOCK_TYPES else "text"
    if normalized_type not in {"image", "equation"} and not content.strip():
        return None
    normalized_bbox = _normalize_bbox_to_thousandths(bbox, page_size)
    return {
        "type": normalized_type,
        "bbox": normalized_bbox,
        "angle": 0 if normalized_type == "image" else int(block.get("angle", 0) or 0) % 360,
        "content": content,
    }
