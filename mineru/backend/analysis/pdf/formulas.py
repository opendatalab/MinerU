# Copyright (c) Opendatalab. All rights reserved.
"""PDF 公式识别输入、结果回填和公式编号规范化。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image

from mineru.backend.local_model_runtime import HybridLocalModelContext
from mineru.utils.bbox_utils import normalize_to_int_bbox
from mineru.utils.ocr_utils import get_rotate_crop_image_for_text_rec

from ....types import RAW_FORMULA_NUMBER, BlockType
from mineru.utils.text_utils import full_to_half
from mineru.utils.text_utils import clean_isolated_formula

from .geometry import (
    _bbox_to_pixel_bbox,
    _medium_bbox_to_quad,
    _normalize_layout_bbox_to_unit,
    _normalize_medium_content,
    _normalize_page_size,
)


def normalize_formula_tag_content(tag_content: str) -> str:
    """归一化公式编号文本，去掉全角字符和包裹括号后用于 \\tag{}。"""
    tag_content = full_to_half(str(tag_content or "").strip())
    if tag_content.startswith("("):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith(")"):
        tag_content = tag_content[:-1].strip()
    return tag_content


def normalize_formula_content_for_tag(formula_content: str) -> str:
    """归一化待合并编号的公式正文，去掉 VLM/Hybrid 可能携带的展示公式分隔符。"""
    return clean_isolated_formula(str(formula_content or ""))


def build_tagged_formula_content(formula_content: str, tag_content: str) -> str | None:
    """将公式正文和编号文本合成为带 LaTeX tag 的纯公式内容。"""
    formula_content = normalize_formula_content_for_tag(formula_content)
    tag_content = normalize_formula_tag_content(tag_content)
    if not formula_content or not tag_content:
        return None
    return f"{formula_content}\\tag{{{tag_content}}}"


def _is_hybrid_equation_block(block: dict[str, Any]) -> bool:
    """判断 raw Hybrid/VLM block 是否表示可合并编号的行间公式。"""
    return str(block.get("type") or "").lower() in {
        BlockType.EQUATION,
        "display_formula",
    }


def _is_hybrid_formula_number_block(block: dict[str, Any]) -> bool:
    """判断 raw Hybrid/VLM block 是否表示公式编号。"""
    return str(block.get("type") or "").lower() == RAW_FORMULA_NUMBER


def _normalize_hybrid_formula_bbox(bbox: Any) -> list[float] | None:
    """将 Hybrid/VLM 公式框规范为合法浮点四元组，非法或退化框返回 None。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) for value in bbox):
        return None
    try:
        normalized_bbox = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) for value in normalized_bbox):
        return None
    x0, y0, x1, y1 = normalized_bbox
    if x1 <= x0 or y1 <= y0:
        return None
    return normalized_bbox


def _merge_hybrid_formula_number_block(equation_block: dict[str, Any], number_block: dict[str, Any]) -> None:
    """把公式编号的 bbox 和非空内容合并到相邻 Hybrid/VLM 公式 block。"""
    equation_bbox = _normalize_hybrid_formula_bbox(equation_block.get("bbox"))
    number_bbox = _normalize_hybrid_formula_bbox(number_block.get("bbox"))
    if equation_bbox is not None and number_bbox is not None:
        equation_block["bbox"] = [
            min(equation_bbox[0], number_bbox[0]),
            min(equation_bbox[1], number_bbox[1]),
            max(equation_bbox[2], number_bbox[2]),
            max(equation_bbox[3], number_bbox[3]),
        ]

    target_key = "latex" if equation_block.get("latex") else "content"
    tagged_content = build_tagged_formula_content(
        str(equation_block.get(target_key) or ""),
        str(number_block.get("content") or number_block.get("text") or ""),
    )
    if tagged_content:
        equation_block[target_key] = tagged_content


def optimize_hybrid_formula_number_blocks(page_model_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并 Hybrid raw model list 中的 formula_number，并返回新的 block 列表。"""
    optimized_blocks: list[dict[str, Any]] = []
    blocks = list(page_model_list)
    for index, block in enumerate(blocks):
        if not _is_hybrid_formula_number_block(block):
            optimized_blocks.append(block)
            continue

        prev_block = blocks[index - 1] if index > 0 else None
        if prev_block and _is_hybrid_equation_block(prev_block):
            _merge_hybrid_formula_number_block(prev_block, block)
            continue

        next_block = blocks[index + 1] if index + 1 < len(blocks) else None
        next_next_block = blocks[index + 2] if index + 2 < len(blocks) else None
        if (
            next_block
            and _is_hybrid_equation_block(next_block)
            and (next_next_block is None or not _is_hybrid_formula_number_block(next_next_block))
        ):
            _merge_hybrid_formula_number_block(next_block, block)
            continue

        fallback_block = dict(block)
        fallback_block["type"] = BlockType.TEXT
        optimized_blocks.append(fallback_block)
    return optimized_blocks


def _build_formula_inputs(images_layout_res: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """构造完整 MFD/MFR 输入，保留全部行内和行间公式框。"""
    formula_inputs = []
    for layout_res in images_layout_res:
        page_formula_inputs = []
        for res in layout_res:
            label = res.get("label")
            if label not in ["inline_formula", "display_formula"]:
                continue
            bbox = res.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            page_formula_inputs.append(
                {
                    "label": label,
                    "bbox": list(bbox),
                    "score": float(res.get("score", 0.0)),
                    # layout 只提供公式位置；未运行 MFR 的 high/xhigh OCR 必须保留空 LaTeX。
                    "latex": "",
                }
            )
        formula_inputs.append(page_formula_inputs)
    return formula_inputs


def _split_formula_results(
    images_formula_list: list[list[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """按原始标签拆分 MFR 结果，避免行间公式进入 inline sidecar。"""
    inline_formula_list = []
    display_formula_list = []
    for page_formula_list in images_formula_list:
        inline_formula_list.append([formula for formula in page_formula_list if formula.get("label") == "inline_formula"])
        display_formula_list.append([formula for formula in page_formula_list if formula.get("label") == "display_formula"])
    return inline_formula_list, display_formula_list


def _apply_medium_display_formula_results(
    model_list: list[list[dict[str, Any]]],
    display_formula_list: list[list[dict[str, Any]]],
    images_pil_list: list[Image.Image],
) -> None:
    """将 medium 行间公式 LaTeX 按页和 bbox 回填到对应 equation 块。"""
    for page_idx, (page_model_list, page_display_formula_list, page_image) in enumerate(
        zip(model_list, display_formula_list, images_pil_list)
    ):
        page_size = _normalize_page_size(page_image)
        equation_blocks_by_bbox: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        for block in page_model_list:
            if block.get("type") != BlockType.EQUATION:
                continue
            block_bbox = block.get("bbox")
            if block_bbox is None or len(block_bbox) != 4:
                continue
            equation_blocks_by_bbox.setdefault(tuple(float(value) for value in block_bbox), []).append(block)

        for formula in page_display_formula_list:
            normalized_bbox = _normalize_layout_bbox_to_unit(formula.get("bbox"), page_size)
            if normalized_bbox is None:
                continue
            matched_blocks = equation_blocks_by_bbox.get(tuple(normalized_bbox), [])
            if len(matched_blocks) != 1:
                raise ValueError(
                    "Hybrid medium display formula must match exactly one equation block: "
                    f"page_idx={page_idx}, bbox={normalized_bbox}, matches={len(matched_blocks)}"
                )
            matched_blocks[0]["content"] = formula.get("latex", "")


def _formula_item_to_pixel_bbox(item: dict[str, Any]) -> list[int] | None:
    """读取公式项的四点 bbox，并转换为后续裁图使用的整数坐标。"""
    bbox = item.get("bbox")
    if bbox is not None and len(bbox) == 4:
        return [int(float(v)) for v in bbox]
    return None


def _apply_medium_formula_number_ocr(
    local_context: HybridLocalModelContext,
    model_list: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> None:
    """对 medium formula_number 裁剪图执行 OCR-rec，并把编号文本回填到原始 layout 项。"""
    need_rec_items: list[dict[str, Any]] = []
    formula_number_crops: list[np.ndarray] = []
    for block_list, np_img in zip(model_list, np_images):
        image_h, image_w = np_img.shape[:2]
        bgr_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        for block_item in block_list:
            if block_item.get("type") != RAW_FORMULA_NUMBER:
                continue

            formula_number_bbox = normalize_to_int_bbox(
                _bbox_to_pixel_bbox(block_item.get("bbox"), (image_w, image_h)),
                image_size=(image_h, image_w),
            )
            if formula_number_bbox is None:
                continue

            # 使用 OCR rec 的标准旋转裁剪逻辑，保证 medium 编号裁图与正文 OCR-rec 输入一致。
            formula_number_crops.append(
                get_rotate_crop_image_for_text_rec(
                    bgr_image,
                    _medium_bbox_to_quad(formula_number_bbox).copy(),
                )
            )
            need_rec_items.append(block_item)

    if not formula_number_crops:
        return

    ocr_result_list = local_context.ocr_model.ocr(
        formula_number_crops,
        det=False,
        tqdm_enable=True,
        tqdm_desc="OCR-rec",
    )[0]
    if len(ocr_result_list) != len(need_rec_items):
        raise ValueError(
            "Hybrid medium formula number OCR rec result count mismatch: "
            f"ocr_result_list={len(ocr_result_list)}, need_rec_items={len(need_rec_items)}"
        )

    for block_item, ocr_result in zip(need_rec_items, ocr_result_list):
        if not ocr_result or len(ocr_result) < 2:
            continue
        ocr_text, _ = ocr_result
        normalized_text = _normalize_medium_content(ocr_text)
        if normalized_text:
            block_item["content"] = normalized_text
