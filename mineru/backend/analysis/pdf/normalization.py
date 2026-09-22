# Copyright (c) Opendatalab. All rights reserved.
"""标题拆分与 PDF model-list 的最终规范化。"""

from __future__ import annotations

import re
from typing import Any

from docvortex.content import normalize_pdf_model_text
from docvortex.content.spans import append_equation_span, append_text_span, inline_span_plain_text, strip_span_dicts
from docvortex.geometry import calculate_overlap_area_2_minbox_area_ratio
from loguru import logger

from ....types import RAW_ALGORITHM, RAW_PHONETIC, BBox, BlockType
from .constants import (
    _VLM_UNCLASSIFIED_TITLE_TYPE,
    LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD,
    LINE_METADATA_BLOCK_TYPES,
    NATURAL_LANGUAGE_CONTENT_BLOCK_TYPES,
)
from .model_inputs import _bbox_to_pixel_bbox


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
                block["level"] = 1
            else:
                block["type"] = BlockType.PARAGRAPH_TITLE
                block["level"] = 2


def _is_valid_pdf_text_bbox(bbox: object) -> bool:
    """校验文本几何的归一化矩形，拒绝布尔值、非有限坐标和退化框。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    # 区间判断同时排除 NaN/Inf，且避免将超大整数转成 float 时溢出。
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0 for value in bbox):
        return False
    x0, y0, x1, y1 = bbox
    return x1 > x0 and y1 > y0


def _has_valid_pdf_text_lines(lines: object) -> bool:
    """检查非空行列表中的每一条行框是否完整合法。"""
    return (
        isinstance(lines, list)
        and bool(lines)
        and all(isinstance(line, dict) and _is_valid_pdf_text_bbox(line.get("bbox")) for line in lines)
    )


def _is_valid_pdf_text_block(block: dict[str, Any]) -> bool:
    """检查文本块是否具有非空正文，以及合法行框或可用于兜底的块框。"""
    content = block.get("content")
    if isinstance(content, list):
        visible_content = inline_span_plain_text(item for item in content if isinstance(item, dict))
    else:
        visible_content = content if isinstance(content, str) else ""
    if not visible_content.strip():
        return False
    return _has_valid_pdf_text_lines(block.get("lines")) or _is_valid_pdf_text_bbox(block.get("bbox"))


def _natural_language_spans(content: str) -> list[dict[str, Any]]:
    """把 PDF 自然语言字符串及圆括号公式定界符直接转换为 Span。"""
    pattern = re.compile(r"\\\((?P<round>.*?)\\\)", re.DOTALL)
    spans: list[dict[str, Any]] = []
    cursor = 0
    for match in pattern.finditer(content):
        append_text_span(spans, content[cursor : match.start()])
        append_equation_span(spans, match.group("round"))
        cursor = match.end()
    append_text_span(spans, content[cursor:])
    return strip_span_dicts(spans)


def _normalize_pdf_model_list(model_list: list[list[dict[str, Any]]]) -> None:
    """清理 PDF 元数据和公式，修复缺失行框，仅过滤空正文或完全缺少可用几何的文本块。"""
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
            if isinstance(content, list):
                continue
            if not isinstance(content, str):
                continue
            if block.get("type") in NATURAL_LANGUAGE_CONTENT_BLOCK_TYPES:
                block["content"] = _natural_language_spans(content)
            elif block.get("type") == BlockType.CODE:
                spans = _natural_language_spans(content)
                if any(span.get("type") == "equation_inline" for span in spans):
                    block["type"] = RAW_ALGORITHM
                    block["content"] = spans
                else:
                    block["content"] = content
        normalized_blocks: list[dict[str, Any]] = []
        bbox_fallback_count = 0
        dropped_count = 0
        for block in page_model_list:
            if block.get("type") in LINE_METADATA_BLOCK_TYPES:
                if not _is_valid_pdf_text_block(block):
                    dropped_count += 1
                    continue
                if not _has_valid_pdf_text_lines(block.get("lines")):
                    # 公式遮罩可能使 OCR 检测不到行；保留 VLM 正文，并以块框提供粗粒度几何。
                    block["lines"] = [{"bbox": list(block["bbox"])}]
                    bbox_fallback_count += 1
            normalized_blocks.append(block)
        page_model_list[:] = normalized_blocks
        if bbox_fallback_count or dropped_count:
            logger.warning(
                "PDF text block normalization: page_idx={}, bbox_fallback={}, dropped={}",
                page_idx,
                bbox_fallback_count,
                dropped_count,
            )
    # 统一在回填、Span 构造及宿主元数据清理完成后调用共享实现。
    normalize_pdf_model_text(model_list)
