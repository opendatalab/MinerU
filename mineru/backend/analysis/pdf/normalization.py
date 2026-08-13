# Copyright (c) Opendatalab. All rights reserved.
"""标题拆分与 PDF model-list 的最终规范化。"""

from __future__ import annotations

import math
from typing import Any

from mineru.types import BBox, BlockType, RAW_PHONETIC
from mineru.utils.bbox_utils import calculate_overlap_area_2_minbox_area_ratio
from mineru.utils.text_utils import full_to_half_exclude_marks

from .constants import (
    LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD,
    LINE_METADATA_BLOCK_TYPES,
    NATURAL_LANGUAGE_CONTENT_BLOCK_TYPES,
    _INLINE_FORMULA_PATTERN,
    _VLM_UNCLASSIFIED_TITLE_TYPE,
)
from .geometry import _bbox_to_pixel_bbox


def _collect_layout_doc_title_bboxes(layout_res: list[dict[str, Any]], page_size: tuple[int, int]) -> list[BBox]:
    """只收集layout小模型输出的doc_title框，忽略paragraph_title等其他类型。"""
    doc_title_bboxes: list[BBox] = []
    for layout_item in layout_res or []:
        if layout_item.get("label") != BlockType.DOC_TITLE:
            continue
        bbox = _bbox_to_pixel_bbox(layout_item.get("bbox"), page_size)
        if bbox is not None:
            doc_title_bboxes.append(bbox)
    return doc_title_bboxes


def _has_doc_title_overlap(title_bbox: BBox, doc_title_bboxes: list[BBox], overlap_threshold: float) -> bool:
    """判断VLM标题框是否与任一layout doc_title框达到最小框重叠阈值。"""
    return any(
        calculate_overlap_area_2_minbox_area_ratio(title_bbox, doc_title_bbox) >= overlap_threshold
        for doc_title_bbox in doc_title_bboxes
    )


def _apply_layout_title_split(
    model_list: list[list[dict[str, Any]]],
    images_layout_res: list[list[dict[str, Any]]],
    page_sizes: list[tuple[int, int]],
    overlap_threshold: float = LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD,
) -> None:
    """用layout doc_title框将VLM title拆分为doc_title和paragraph_title。"""
    for page_model_list, layout_res, page_size in zip(model_list, images_layout_res, page_sizes):
        doc_title_bboxes = _collect_layout_doc_title_bboxes(layout_res, page_size)
        for block in page_model_list:
            if block.get("type") != _VLM_UNCLASSIFIED_TITLE_TYPE:
                continue
            title_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if title_bbox is None:
                continue
            if _has_doc_title_overlap(title_bbox, doc_title_bboxes, overlap_threshold):
                block["type"] = BlockType.DOC_TITLE
            else:
                block["type"] = BlockType.PARAGRAPH_TITLE


def _is_valid_pdf_text_block(block: dict[str, Any]) -> bool:
    """检查 PDF 文本块是否同时具有非空正文和完整合法的归一化行框。"""

    content = block.get("content")
    if not isinstance(content, str) or not content.strip():
        return False

    lines = block.get("lines")
    if not isinstance(lines, list) or not lines:
        return False
    for line in lines:
        if not isinstance(line, dict):
            return False
        bbox = line.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in bbox
        ):
            return False
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if not all(0.0 <= value <= 1.0 for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            return False
    return True


def _normalize_natural_language_content(content: str) -> str:
    """将自然语言中的全角字母和数字转为半角，同时原样保留行内公式片段。"""
    normalized_parts: list[str] = []
    cursor = 0
    formula_markers = (("\\(", "\\)"), ("<eq>", "</eq>"))

    while cursor < len(content):
        formula_starts = [
            (start, opening, closing) for opening, closing in formula_markers if (start := content.find(opening, cursor)) >= 0
        ]
        if not formula_starts:
            normalized_parts.append(full_to_half_exclude_marks(content[cursor:]))
            break

        formula_start, opening, closing = min(formula_starts, key=lambda item: item[0])
        normalized_parts.append(full_to_half_exclude_marks(content[cursor:formula_start]))
        formula_end = content.find(closing, formula_start + len(opening))
        if formula_end < 0:
            normalized_parts.append(content[formula_start:])
            break

        formula_end += len(closing)
        normalized_parts.append(content[formula_start:formula_end])
        cursor = formula_end

    return "".join(normalized_parts)


def _normalize_pdf_model_list(model_list: list[list[dict[str, Any]]]) -> None:
    """清理 PDF block 元数据、规范公式，并过滤正文或行框无效的文本块。"""
    for page_idx, page_model_list in enumerate(model_list):
        for block_idx, block in enumerate(page_model_list):
            raw_type = block.get("type")
            if raw_type == RAW_PHONETIC:
                block["type"] = BlockType.TEXT
            elif raw_type == BlockType.EQUATION:
                equation_content = block.get("content")
                if isinstance(equation_content, str):
                    if equation_content.startswith("\\["):
                        equation_content = equation_content[2:]
                    if equation_content.endswith("\\]"):
                        equation_content = equation_content[:-2]
                    block["content"] = equation_content.strip()
            elif raw_type == _VLM_UNCLASSIFIED_TITLE_TYPE:
                raise ValueError(f"Unclassified PDF title block: page_idx={page_idx}, block_idx={block_idx}")
            block.pop("angle", None)
            block.pop("score", None)
            block.pop("merge_prev", None)
            content = block.get("content")
            if not isinstance(content, str):
                continue
            if block.get("type") in NATURAL_LANGUAGE_CONTENT_BLOCK_TYPES:
                content = _normalize_natural_language_content(content)
            block["content"] = _INLINE_FORMULA_PATTERN.sub(
                lambda match: f"<eq>{match.group(1)}</eq>",
                content,
            )
        page_model_list[:] = [
            block
            for block in page_model_list
            if block.get("type") not in LINE_METADATA_BLOCK_TYPES or _is_valid_pdf_text_block(block)
        ]
