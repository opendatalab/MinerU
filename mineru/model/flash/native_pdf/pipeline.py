# Copyright (c) Opendatalab. All rights reserved.

"""编排 Flash 原生 PDF 的页面准备、语义处理和输出归一化。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any


from mineru.backend.utils.xycut_pp_sorter import sort_entries
from mineru.utils.pdf_document import PDFDocument, PDFImageInfo, get_lines_from_chars

from .models import (
    _DocumentBodyProfile,
    _DocumentTitleProfile,
    _LineItem,
    _PageSource,
    _PreparedPage,
)
from .geometry import (
    _bbox_area,
    _bbox_axis_overlap_ratio,
    _bbox_center_y,
    _bbox_overlap_in_smaller,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _normalize_bbox_to_thousandths,
    _rotate_bbox_from_upright,
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
from .index_blocks import _extract_index_blocks
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
    _split_parallel_graphic_rule_rows,
)
from .formulas import (
    _build_formula_like_blocks,
    _build_vector_formula_blocks,
)
from .code_blocks import (
    _build_code_blocks,
    _build_rule_delimited_code_blocks,
)
from .auxiliary_text import (
    _classify_deferred_image_footnotes,
    _classify_isolated_first_page_footer,
    _classify_page_footnote_trailing_footers,
    _classify_page_number_outer_companions,
    _classify_page_auxiliary_text,
    _classify_raw_page_marginals,
    _classify_rule_delimited_footers,
    _classify_rule_delimited_headers,
    _classify_split_marginal_row_companions,
    _classify_repeated_page_marginals,
    _classify_repeated_visual_headers,
    _classify_single_page_compound_headers,
)
from .titles import (
    _classify_body_height_section_titles,
    _classify_page_titles,
    _infer_document_body_profile,
    _infer_document_title_profile,
)
from .text_blocks import (
    _build_text_blocks,
    _merge_fragmented_header_blocks,
    _merge_front_matter_column_blocks,
    _merge_image_caption_text_blocks,
    _merge_internal_text_block_group,
    _merge_multiline_title_blocks,
    _merge_repeated_compact_title_continuations,
)
from .visual_annotations import _classify_and_bind_visual_annotations


_TEXT_SEMANTIC_TYPES = {
    "doc_title",
    "paragraph_title",
    "header",
    "footer",
    "page_number",
    "caption",
    "footnote",
    "page_footnote",
    "aside_text",
    "index",
}


_OUTPUT_BLOCK_TYPES = {"text", "table", "image", "equation", "code"} | _TEXT_SEMANTIC_TYPES
_LINE_METADATA_OUTPUT_TYPES = {
    "text",
    "doc_title",
    "paragraph_title",
    "caption",
    "footnote",
}
_REPEATED_RASTER_IMAGE_MIN_PAGE_AREA_RATIO = 0.08
_REPEATED_RASTER_IMAGE_MIN_DISTINCT_PAGES = 3


def _is_large_raster_image(
    image_info: PDFImageInfo,
    page_size: tuple[float, float],
) -> bool:
    """判断点阵图裁剪后面积是否达到页面面积的 8%。"""

    bbox = _coerce_bbox(image_info.bbox)
    page_area = max(0.0, page_size[0]) * max(0.0, page_size[1])
    return (
        bbox is not None
        and page_area > 0
        and _bbox_area(bbox) / page_area
        >= _REPEATED_RASTER_IMAGE_MIN_PAGE_AREA_RATIO
    )


def _detect_repeated_raster_watermark_fingerprints(
    page_image_infos: list[list[PDFImageInfo]],
    page_sizes: list[tuple[float, float]],
) -> set[str]:
    """按大图指纹统计不同页号，出现至少三页时判为跨页图片水印。"""

    page_indices_by_fingerprint: dict[str, set[int]] = {}
    for page_idx, (image_infos, page_size) in enumerate(
        zip(page_image_infos, page_sizes, strict=True)
    ):
        for image_info in image_infos:
            if image_info.fingerprint is None or not _is_large_raster_image(image_info, page_size):
                continue
            page_indices_by_fingerprint.setdefault(image_info.fingerprint, set()).add(page_idx)
    return {
        fingerprint
        for fingerprint, page_indices in page_indices_by_fingerprint.items()
        if len(page_indices) >= _REPEATED_RASTER_IMAGE_MIN_DISTINCT_PAGES
    }


def _filter_repeated_raster_watermark_bboxes(
    image_infos: list[PDFImageInfo],
    page_size: tuple[float, float],
    watermark_fingerprints: set[str],
) -> list[tuple[float, float, float, float]]:
    """仅删除命中跨页水印指纹且面积达标的 bbox，小尺寸同图继续保留。"""

    return [
        image_info.bbox
        for image_info in image_infos
        if not (
            image_info.fingerprint in watermark_fingerprints
            and _is_large_raster_image(image_info, page_size)
        )
    ]


def _analyze_native_document(pdf_doc: PDFDocument) -> list[list[dict[str, Any]]]:
    """逐页读取数字 PDF，并在轻量页面上完成跨页文本类型判定。"""

    page_sizes = [
        pdf_doc.page_size(page_idx)
        for page_idx in range(pdf_doc.page_count)
    ]
    page_image_infos = [
        pdf_doc.get_page_image_infos(page_idx)
        for page_idx in range(pdf_doc.page_count)
    ]
    page_signature_bboxes = [
        pdf_doc.get_page_signature_bboxes(page_idx)
        for page_idx in range(pdf_doc.page_count)
    ]
    watermark_fingerprints = _detect_repeated_raster_watermark_fingerprints(
        page_image_infos,
        page_sizes,
    )

    page_sources: list[_PageSource] = []
    for page_idx in range(pdf_doc.page_count):
        page_size = page_sizes[page_idx]
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
            image_bboxes=_filter_repeated_raster_watermark_bboxes(
                page_image_infos[page_idx],
                page_size,
                watermark_fingerprints,
            ),
            signature_bboxes=page_signature_bboxes[page_idx],
            form_bboxes=pdf_doc.get_page_form_bboxes(page_idx),
            path_infos=pdf_doc.get_page_path_infos(page_idx),
        )
        page_sources.append(source)

    _classify_raw_page_marginals(page_sources)
    prepared_pages = [
        _prepare_page_source(source)
        for source in page_sources
    ]

    _classify_repeated_visual_headers(prepared_pages)
    _classify_repeated_page_marginals(prepared_pages)
    _classify_split_marginal_row_companions(prepared_pages)
    _classify_single_page_compound_headers(prepared_pages)
    _classify_rule_delimited_headers(prepared_pages)
    _classify_rule_delimited_footers(prepared_pages)
    _classify_page_number_outer_companions(prepared_pages)
    _classify_page_footnote_trailing_footers(prepared_pages)
    _classify_isolated_first_page_footer(prepared_pages)
    document_body_profile = _infer_document_body_profile(prepared_pages)
    if document_body_profile is not None:
        _classify_deferred_image_footnotes(
            prepared_pages,
            document_body_profile.body_height,
        )
    document_title_profile = _infer_document_title_profile(
        prepared_pages,
        document_body_profile,
    )
    return [
        _finalize_prepared_page(
            prepared,
            page_index,
            document_body_profile=document_body_profile,
            document_title_profile=document_title_profile,
        )
        for page_index, prepared in enumerate(prepared_pages)
    ]


def _prepare_page_source(source: _PageSource) -> _PreparedPage:
    """先认领视觉容器，再标注辅助文本并留下可跨页比较的轻量文本行。"""

    protected_line_indices = {
        line.source_index
        for line in source.lines
        if line.semantic_type is not None
    }
    analysis_source = replace(
        source,
        lines=[
            line
            for line in source.lines
            if line.source_index not in protected_line_indices
        ],
    )
    form_bboxes = _select_form_image_bboxes(source)
    strong_graphic_bboxes = _detect_strong_graphic_bboxes(analysis_source)
    rule_code_blocks, claimed_rule_code_line_indices = (
        _build_rule_delimited_code_blocks(
            analysis_source,
            form_bboxes
            + strong_graphic_bboxes
            + list(source.image_bboxes)
            + list(source.signature_bboxes),
        )
    )
    rule_code_bboxes = [block["bbox"] for block in rule_code_blocks]
    candidates = [
        candidate
        for candidate in _detect_table_candidates(
            analysis_source,
            excluded_bboxes=strong_graphic_bboxes + rule_code_bboxes,
        )
        if not any(
            _form_supersedes_nested_bbox(form_bbox, candidate.bbox)
            for form_bbox in form_bboxes
        )
    ]
    # 候选检测仍避开预分类边缘文本；已确认表格物化时回到原始行，
    # 让 core_bbox 内的误标页脚可被重新认领，表格外边缘文本不会被矩形扩张带入。
    table_blocks, table_annotation_blocks, claimed_line_indices = (
        _materialize_table_blocks(
            source,
            candidates,
        )
    )
    claimed_line_indices.update(claimed_rule_code_line_indices)
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
    code_blocks, claimed_code_line_indices = _build_code_blocks(
        analysis_source,
        table_bboxes
        + active_form_bboxes
        + strong_graphic_bboxes
        + list(source.image_bboxes)
        + list(source.signature_bboxes),
        claimed_line_indices,
    )
    code_bboxes = [block["bbox"] for block in code_blocks]
    form_image_blocks, claimed_form_line_indices = _build_form_image_blocks(
        analysis_source,
        active_form_bboxes,
        claimed_line_indices | claimed_code_line_indices,
    )
    graphic_blocks, claimed_graphic_line_indices = _build_graphic_like_blocks(
        analysis_source,
        table_bboxes + active_form_bboxes + rule_code_bboxes + code_bboxes,
        claimed_line_indices | claimed_code_line_indices | claimed_form_line_indices,
        strong_graphic_bboxes,
    )
    raster_image_blocks, claimed_raster_line_indices = _build_raster_image_blocks(
        analysis_source,
        table_blocks
        + rule_code_blocks
        + code_blocks
        + form_image_blocks
        + graphic_blocks,
        claimed_line_indices
        | claimed_code_line_indices
        | claimed_form_line_indices
        | claimed_graphic_line_indices,
    )
    vector_formula_blocks, claimed_vector_number_indices = _build_vector_formula_blocks(
        analysis_source,
        table_blocks
        + rule_code_blocks
        + code_blocks
        + form_image_blocks
        + graphic_blocks
        + raster_image_blocks,
        claimed_line_indices
        | claimed_code_line_indices
        | claimed_form_line_indices
        | claimed_graphic_line_indices
        | claimed_raster_line_indices,
    )
    claimed_line_indices = (
        claimed_line_indices
        | claimed_code_line_indices
        | claimed_form_line_indices
        | claimed_graphic_line_indices
        | claimed_raster_line_indices
        | claimed_vector_number_indices
    )
    remaining_lines = _split_parallel_graphic_rule_rows(
        [
            line
            for line in source.lines
            if line.source_index not in claimed_line_indices
        ],
        source.drawing_lines,
        [
            block["bbox"]
            for block in (
                form_image_blocks
                + graphic_blocks
                + raster_image_blocks
            )
        ],
        table_bboxes,
        source.page_size,
        source_index_start=max(
            (line.source_index for line in source.lines),
            default=-1,
        )
        + 1,
    )
    remaining_lines = _merge_same_baseline_text_lines(
        remaining_lines,
        source.page_size,
        table_bboxes,
    )
    remaining_lines = _merge_overlapping_inline_text_clusters(
        remaining_lines,
        source.page_size,
        table_bboxes,
    )
    # 首轮同行合并可能补齐宿主 bbox，使相邻 hard-split 尾段具备二次闭包条件。
    remaining_lines = _merge_same_baseline_text_lines(
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
            rule_code_blocks
            + table_annotation_blocks
            + table_blocks
            + code_blocks
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
    *,
    document_body_profile: _DocumentBodyProfile | None = None,
    document_title_profile: _DocumentTitleProfile | None = None,
) -> list[dict[str, Any]]:
    """按预分类语义、公式、标题、正文的优先级完成单页文本并排序。"""

    semantic_lines = [line for line in prepared.remaining_lines if line.semantic_type is not None]
    unresolved_lines = [line for line in prepared.remaining_lines if line.semantic_type is None]
    container_bboxes = [block["bbox"] for block in prepared.fixed_blocks]
    index_blocks, unresolved_lines = _extract_index_blocks(
        unresolved_lines,
        prepared.page_size,
        container_bboxes,
        require_heading=True,
    )
    semantic_lines.extend(
        line
        for line in unresolved_lines
        if line.semantic_type is not None
    )
    formula_input = [line for line in unresolved_lines if line.semantic_type is None]
    formula_input = _restore_dense_split_visual_rows(
        formula_input,
        prepared.page_size,
        prepared.table_bboxes,
    )
    formula_input = _merge_same_baseline_text_lines(
        formula_input,
        prepared.page_size,
        prepared.table_bboxes,
    )
    formula_blocks, remaining_lines = _build_formula_like_blocks(
        formula_input,
        prepared.table_bboxes,
        prepared.page_size,
    )
    fallback_index_blocks, remaining_lines = _extract_index_blocks(
        remaining_lines,
        prepared.page_size,
        container_bboxes,
    )
    index_blocks.extend(fallback_index_blocks)
    semantic_lines.extend(
        line
        for line in remaining_lines
        if line.semantic_type is not None
    )
    remaining_lines = [
        line
        for line in remaining_lines
        if line.semantic_type is None
    ]
    title_container_bboxes = [
        block["bbox"]
        for block in prepared.fixed_blocks
        if not isinstance(block.get("_inline_visual_row_id"), int)
    ]
    caption_container_bboxes = [
        block["bbox"]
        for block in prepared.fixed_blocks
        if block.get("type") in {"image", "code"}
    ]
    _classify_body_height_section_titles(
        remaining_lines,
        prepared.page_size,
        container_bboxes=title_container_bboxes,
        document_body_profile=document_body_profile,
    )
    _classify_page_titles(
        remaining_lines,
        prepared.page_size,
        page_index=page_index,
        container_bboxes=title_container_bboxes,
        caption_container_bboxes=caption_container_bboxes,
        document_body_profile=document_body_profile,
        document_title_profile=document_title_profile,
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
    text_blocks = _merge_multiline_title_blocks(text_blocks)
    text_blocks = _merge_front_matter_column_blocks(
        text_blocks,
        prepared.page_size,
        page_index=page_index,
    )
    text_blocks = _merge_image_caption_text_blocks(
        text_blocks,
        [
            block["bbox"]
            for block in prepared.fixed_blocks
            if block.get("type") == "image"
        ],
    )
    text_blocks = _merge_fragmented_header_blocks(text_blocks)
    text_blocks = _merge_repeated_compact_title_continuations(
        text_blocks,
        prepared.page_size,
    )
    absolute_blocks = (
        prepared.fixed_blocks + formula_blocks + index_blocks + text_blocks
    )
    visual_annotation_regions = _classify_and_bind_visual_annotations(
        absolute_blocks,
        prepared.page_size,
        merge_text_block_group=_merge_internal_text_block_group,
    )
    sorted_blocks = _sort_blocks_with_visual_row_groups(
        absolute_blocks,
        prepared.page_size,
        visual_annotation_regions=visual_annotation_regions,
    )
    return [
        normalized
        for block in sorted_blocks
        if (normalized := _normalize_output_block(block, prepared.page_size)) is not None
    ]


def _analyze_page_source(source: _PageSource) -> list[dict[str, Any]]:
    """兼容单页测试入口；单页不凭边缘位置猜测页眉、页脚或页码。"""

    if not source.lines and not source.image_bboxes and not source.signature_bboxes:
        return []
    return _finalize_prepared_page(_prepare_page_source(source), page_index=0)


def _sort_blocks_with_visual_row_groups(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    visual_annotation_regions: list[list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """把视觉注释区域和拆分粗行包装成虚拟项排序，再按各自局部顺序展开。"""

    top_marginals: list[dict[str, Any]] = []
    bottom_marginals: list[dict[str, Any]] = []
    body_blocks: list[dict[str, Any]] = []
    local_page_height = page_size[1]
    for block in blocks:
        block_type = block.get("type")
        bbox = block.get("bbox")
        if block_type == "header" or (
            block_type == "page_number"
            and isinstance(bbox, (list, tuple))
            and _bbox_center_y(bbox) <= 0.5 * local_page_height
        ):
            top_marginals.append(block)
        elif block_type in {"footer", "page_footnote"} or block_type == "page_number":
            bottom_marginals.append(block)
        else:
            body_blocks.append(block)

    body_index_by_identity = {
        id(block): index for index, block in enumerate(body_blocks)
    }
    region_groups: list[list[dict[str, Any]]] = []
    region_consumed_indices: set[int] = set()
    for region in visual_annotation_regions or []:
        indices = [
            body_index_by_identity[id(member)]
            for member in region
            if id(member) in body_index_by_identity
        ]
        if len(indices) < 2 or any(index in region_consumed_indices for index in indices):
            continue
        members = [body_blocks[index] for index in indices]
        for member in members:
            member["_visual_annotation_region_member"] = True
        region_groups.append(members)
        region_consumed_indices.update(indices)

    inline_grouped_indices = _collect_inline_image_text_groups(
        body_blocks,
        excluded_indices=region_consumed_indices,
    )
    inline_consumed_indices = {
        index
        for indices in inline_grouped_indices.values()
        for index in indices
    }
    grouped_indices: dict[int, list[int]] = {}
    for index, block in enumerate(body_blocks):
        if index in region_consumed_indices or index in inline_consumed_indices:
            continue
        row_id = block.get("_single_run_row_id")
        if isinstance(row_id, int):
            grouped_indices.setdefault(row_id, []).append(index)

    virtual_groups: list[dict[str, Any]] = []
    consumed_indices: set[int] = set(region_consumed_indices | inline_consumed_indices)
    for members in region_groups:
        virtual_groups.append(
            {
                "type": "_xycut_visual_annotation_region",
                "bbox": _bbox_union_many([member["bbox"] for member in members]),
                "angle": members[0].get("angle", 0),
                "content": "",
                "_members": members,
                "_visual_annotation_region": True,
            }
        )
    for row_id, indices in inline_grouped_indices.items():
        members = [body_blocks[index] for index in indices]
        virtual_groups.append(
            {
                "type": "_xycut_visual_row_group",
                "bbox": _bbox_union_many([member["bbox"] for member in members]),
                "angle": members[0].get("angle", 0),
                "content": "",
                "_members": members,
                "_inline_visual_row_id": row_id,
            }
        )
    for row_id, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        members = [body_blocks[index] for index in indices]
        virtual_group = {
            "type": "_xycut_visual_row_group",
            "bbox": _bbox_union_many([member["bbox"] for member in members]),
            "angle": members[0].get("angle", 0),
            "content": "",
            "_members": members,
        }
        virtual_groups.append(virtual_group)
        consumed_indices.update(indices)

    sortable_blocks = [
        block
        for index, block in enumerate(body_blocks)
        if index not in consumed_indices
    ]
    sortable_blocks.extend(virtual_groups)
    sorted_payloads = sort_entries(sortable_blocks)
    output: list[dict[str, Any]] = []
    for payload in sorted_payloads:
        members = payload.get("_members")
        if not isinstance(members, list):
            output.append(payload)
            continue
        if payload.get("_visual_annotation_region") is True:
            output.extend(members)
            continue
        if isinstance(payload.get("_inline_visual_row_id"), int):
            members.sort(
                key=lambda member: _inline_visual_group_member_sort_key(
                    member,
                    page_size,
                )
            )
            output.extend(members)
            continue
        angle = int(payload.get("angle", 0) or 0) % 360
        members.sort(
            key=lambda member: (
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[0],
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[1],
            )
        )
        output.extend(members)
    output = _stabilize_overlapping_lane_order(output, page_size)
    return [
        *_sort_marginal_blocks(top_marginals, page_size),
        *output,
        *_sort_marginal_blocks(bottom_marginals, page_size),
    ]


def _collect_inline_image_text_groups(
    body_blocks: list[dict[str, Any]],
    *,
    excluded_indices: set[int] | None = None,
) -> dict[int, list[int]]:
    """把复合图片与包含同一首行的正文块组成专用排序组。"""

    excluded_indices = excluded_indices or set()
    image_indices_by_row: dict[int, list[int]] = {}
    for index, block in enumerate(body_blocks):
        if index in excluded_indices:
            continue
        row_id = block.get("_inline_visual_row_id")
        if block.get("type") == "image" and isinstance(row_id, int):
            image_indices_by_row.setdefault(row_id, []).append(index)

    output: dict[int, list[int]] = {}
    consumed_text_indices: set[int] = set()
    for row_id, image_indices in sorted(image_indices_by_row.items()):
        text_indices = [
            index
            for index, block in enumerate(body_blocks)
            if index not in excluded_indices
            and index not in consumed_text_indices
            and block.get("type") == "text"
            and isinstance(block.get("_visual_row_ids"), set)
            and row_id in block["_visual_row_ids"]
        ]
        if not text_indices:
            continue
        output[row_id] = [*image_indices, *text_indices]
        consumed_text_indices.update(text_indices)
    return output


def _inline_visual_group_member_sort_key(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> tuple[float, float, int]:
    """按图片位置或正文首个局部行位置确定复合视觉行的组内顺序。"""

    local_line_bboxes = block.get("_local_line_bboxes")
    if (
        block.get("type") == "text"
        and isinstance(local_line_bboxes, list)
        and local_line_bboxes
    ):
        local_bbox = local_line_bboxes[0]
    else:
        angle = int(block.get("angle", 0) or 0) % 360
        local_bbox = _rotate_bbox_to_upright(
            block["bbox"],
            page_size,
            angle,
        )
    return (
        float(local_bbox[0]),
        float(local_bbox[1]),
        0 if block.get("type") == "image" else 1,
    )


def _sort_marginal_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """先按视觉中心聚合边缘同排块，再按行内 x 排序以消除字体框顶边抖动。"""

    if len(blocks) < 2:
        return list(blocks)
    geometry = [
        (
            block,
            _rotate_bbox_to_upright(
                block["bbox"],
                page_size,
                int(block.get("angle", 0) or 0) % 360,
            ),
        )
        for block in blocks
    ]
    heights = [
        float(height)
        for block in blocks
        for height in block.get("_line_heights", [])
        if isinstance(height, (int, float)) and height > 0
    ]
    median_height = sorted(heights)[len(heights) // 2] if heights else 1.0
    rows: list[list[tuple[dict[str, Any], tuple[float, float, float, float]]]] = []
    for item in sorted(geometry, key=lambda value: _bbox_center_y(value[1])):
        target = next(
            (
                row
                for row in rows
                if abs(
                    _bbox_center_y(item[1])
                    - sum(_bbox_center_y(member[1]) for member in row) / len(row)
                )
                <= 0.75 * median_height
            ),
            None,
        )
        if target is None:
            rows.append([item])
        else:
            target.append(item)
    rows.sort(
        key=lambda row: sum(_bbox_center_y(member[1]) for member in row) / len(row)
    )
    return [
        block
        for row in rows
        for block, _bbox in sorted(row, key=lambda member: member[1][0])
    ]


def _stabilize_overlapping_lane_order(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """对同栏轻微重叠块按视觉中心纠正局部逆序，不改变跨栏主阅读顺序。"""

    output = list(blocks)
    for _pass_index in range(len(output)):
        changed = False
        for index in range(len(output) - 1):
            first = output[index]
            second = output[index + 1]
            if not _overlapping_lane_pair_is_inverted(first, second, page_size):
                continue
            output[index], output[index + 1] = second, first
            changed = True
        if not changed:
            break
    return output


def _overlapping_lane_pair_is_inverted(
    first: dict[str, Any],
    second: dict[str, Any],
    page_size: tuple[float, float],
) -> bool:
    """判断相邻块是否属于同一内部栏带且视觉中心顺序与当前结果相反。"""

    if first.get("_visual_annotation_region_member") or second.get(
        "_visual_annotation_region_member"
    ):
        return False
    first_interval = first.get("_lane_interval")
    second_interval = second.get("_lane_interval")
    if (
        not isinstance(first_interval, (list, tuple))
        or not isinstance(second_interval, (list, tuple))
        or len(first_interval) != 2
        or len(second_interval) != 2
        or first.get("_lane_is_span") != second.get("_lane_is_span")
        or int(first.get("angle", 0) or 0) % 360
        != int(second.get("angle", 0) or 0) % 360
    ):
        return False
    first_bbox = _rotate_bbox_to_upright(
        first["bbox"],
        page_size,
        int(first.get("angle", 0) or 0) % 360,
    )
    second_bbox = _rotate_bbox_to_upright(
        second["bbox"],
        page_size,
        int(second.get("angle", 0) or 0) % 360,
    )
    line_heights = [
        float(height)
        for block in (first, second)
        for height in block.get("_line_heights", [])
        if isinstance(height, (int, float)) and height > 0
    ]
    tolerance = 0.75 * (min(line_heights) if line_heights else 1.0)
    same_lane = (
        abs(float(first_interval[0]) - float(second_interval[0])) <= tolerance
        and abs(float(first_interval[1]) - float(second_interval[1])) <= tolerance
    )
    vertical_overlap = min(first_bbox[3], second_bbox[3]) - max(
        first_bbox[1],
        second_bbox[1],
    )
    return (
        same_lane
        and _bbox_center_y(first_bbox) > _bbox_center_y(second_bbox) + 0.1 * tolerance
        and _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x") >= 0.35
        and vertical_overlap >= 0.0
    )


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
    if normalized_type not in {"image", "equation", "header", "footer"} and not content.strip():
        return None
    normalized_bbox = _normalize_bbox_to_thousandths(bbox, page_size)
    output_block = {
        "type": normalized_type,
        "bbox": normalized_bbox,
        "angle": 0 if normalized_type == "image" else int(block.get("angle", 0) or 0) % 360,
        "content": content,
    }
    if normalized_type in _LINE_METADATA_OUTPUT_TYPES:
        output_block["lines"] = _normalize_output_line_items(block, page_size)
    return output_block


def _normalize_output_line_items(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> list[dict[str, list[float]]]:
    """将 Flash 正向局部行框逆变换为页面坐标并归一化输出。"""

    local_line_bboxes = block.get("_local_line_bboxes")
    if not isinstance(local_line_bboxes, list):
        return []

    angle = int(block.get("angle", 0) or 0) % 360
    line_items: list[dict[str, list[float]]] = []
    for local_line_bbox in local_line_bboxes:
        try:
            raw_bbox = tuple(float(value) for value in local_line_bbox)
        except (TypeError, ValueError):
            return []
        coerced_bbox = _coerce_bbox(raw_bbox)
        if (
            len(raw_bbox) != 4
            or coerced_bbox is None
            or raw_bbox[2] <= raw_bbox[0]
            or raw_bbox[3] <= raw_bbox[1]
        ):
            return []
        page_bbox = _clip_bbox(
            _rotate_bbox_from_upright(coerced_bbox, page_size, angle),
            page_size,
        )
        if page_bbox is None:
            return []
        line_items.append(
            {"bbox": _normalize_bbox_to_thousandths(page_bbox, page_size)}
        )
    return line_items
