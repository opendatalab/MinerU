# Copyright (c) Opendatalab. All rights reserved.
import re
from typing import Any

from mineru.utils.boxbase import calculate_iou

_VLM_TEXT_BLOCK_TYPES = {
    "text",
    "title",
    "doc_title",
    "paragraph_title",
    "ref_text",
}
_VLM_STRUCTURAL_BLOCK_TYPES = {"list"}
_VLM_DEDUP_BLOCK_TYPES = _VLM_TEXT_BLOCK_TYPES | _VLM_STRUCTURAL_BLOCK_TYPES
_NEAR_IDENTICAL_IOU_THRESHOLD = 0.95


def _normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _block_text(block: dict[str, Any]) -> str:
    if block.get("content") is not None:
        return _normalize_text(block["content"])

    span_contents = (str(span.get("content") or "") for line in block.get("lines", []) for span in line.get("spans", []))
    return _normalize_text("".join(span_contents))


def _has_near_identical_bbox(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_bbox = left.get("bbox")
    right_bbox = right.get("bbox")
    if not left_bbox or not right_bbox:
        return False
    return calculate_iou(left_bbox, right_bbox) >= _NEAR_IDENTICAL_IOU_THRESHOLD


def deduplicate_vlm_model_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """删除同一页面中语义和位置均近似相同的VLM块。"""
    unique_blocks = []

    for block in blocks:
        block_type = block.get("type") or block.get("label")
        if block_type not in _VLM_DEDUP_BLOCK_TYPES:
            unique_blocks.append(block)
            continue

        block_text = _block_text(block)
        duplicate = False
        for kept in unique_blocks:
            kept_type = kept.get("type") or kept.get("label")
            if block_type != kept_type or not _has_near_identical_bbox(block, kept):
                continue
            if block_type in _VLM_STRUCTURAL_BLOCK_TYPES:
                kept_text = _block_text(kept)
                if (not block_text and not kept_text) or block_text == kept_text:
                    duplicate = True
                    break
            if block_text and block_text == _block_text(kept):
                duplicate = True
                break

        if not duplicate:
            unique_blocks.append(block)

    return unique_blocks


def deduplicate_list_sub_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """删除同一列表中语义和位置均近似相同的文本子块。"""
    unique_blocks = []

    for block in blocks:
        block_text = _block_text(block)
        duplicate = any(
            block.get("type") == kept.get("type")
            and block_text
            and block_text == _block_text(kept)
            and _has_near_identical_bbox(block, kept)
            for kept in unique_blocks
        )
        if not duplicate:
            unique_blocks.append(block)

    return unique_blocks
