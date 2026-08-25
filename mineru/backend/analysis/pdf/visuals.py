# Copyright (c) Opendatalab. All rights reserved.
"""视觉块容器补全、方向归一化与页面裁图。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger

from ....types import BBox, BlockType
from ....utils.geometry import (
    calculate_overlap_area_2_minbox_area_ratio,
    calculate_overlap_area_in_bbox1_area_ratio,
)

from .constants import (
    IMAGE_BLOCK_CONTAINMENT_THRESHOLD,
    IMAGE_BLOCK_LAYOUT_COVERAGE_THRESHOLD,
    IMAGE_BLOCK_LAYOUT_MIN_VISUAL_COUNT,
    LOCAL_LAYOUT_IMAGE_BLOCK_AREA_TYPES,
    LOCAL_LAYOUT_IMAGE_BLOCK_BODY_TYPES,
    MODEL_JSON_VISUAL_BLOCK_TYPES,
)
from .geometry import (
    _bbox_to_pixel_bbox,
    _encode_page_crop_as_jpeg_data_uri,
    _normalize_page_size,
    _normalize_visual_block_angle,
)


def _normalize_model_bbox_for_containment(raw_bbox: Any) -> BBox | None:
    """校验模型 block 的四点框，返回可用于面积包含判断的浮点坐标。"""
    try:
        if raw_bbox is None or len(raw_bbox) != 4:
            return None
        bbox = tuple(float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _collapse_image_blocks(
    page_model_list: list[dict[str, Any]],
    containment_threshold: float = IMAGE_BLOCK_CONTAINMENT_THRESHOLD,
) -> None:
    """将 image_block 折叠为单个图片，并删除面积上被其包裹的非容器子块。"""
    image_blocks = [block for block in page_model_list if block.get("type") == "image_block"]
    if not image_blocks:
        return

    image_block_ids = {id(block) for block in image_blocks}
    image_block_bboxes = [
        bbox for block in image_blocks if (bbox := _normalize_model_bbox_for_containment(block.get("bbox"))) is not None
    ]

    retained_blocks: list[dict[str, Any]] = []
    for block in page_model_list:
        if id(block) in image_block_ids:
            block["type"] = BlockType.IMAGE
            retained_blocks.append(block)
            continue

        block_bbox = _normalize_model_bbox_for_containment(block.get("bbox"))
        is_contained = block_bbox is not None and any(
            calculate_overlap_area_in_bbox1_area_ratio(block_bbox, image_block_bbox) >= containment_threshold
            for image_block_bbox in image_block_bboxes
        )
        if not is_contained:
            retained_blocks.append(block)

    page_model_list[:] = retained_blocks


def _supplement_missing_image_block_containers(
    model_list: list[list[dict[str, Any]]],
    layout_blocks_list: list[list[dict[str, Any]]],
    containment_threshold: float = IMAGE_BLOCK_CONTAINMENT_THRESHOLD,
    coverage_threshold: float = IMAGE_BLOCK_LAYOUT_COVERAGE_THRESHOLD,
    min_visual_count: int = IMAGE_BLOCK_LAYOUT_MIN_VISUAL_COUNT,
) -> None:
    """用本地 layout 整图框为 xhigh 结果补充缺失的 image_block 容器。"""
    if len(model_list) != len(layout_blocks_list):
        raise ValueError(
            "Hybrid image-block fallback page count mismatch: "
            f"model_list={len(model_list)}, layout_blocks={len(layout_blocks_list)}"
        )

    for page_model_list, page_layout_blocks in zip(model_list, layout_blocks_list):
        existing_image_block_bboxes = [
            bbox
            for block in page_model_list
            if block.get("type") == "image_block"
            if (bbox := _normalize_model_bbox_for_containment(block.get("bbox"))) is not None
        ]

        existing_claimed_block_ids: set[int] = set()
        if existing_image_block_bboxes:
            for block in page_model_list:
                if block.get("type") == "image_block":
                    continue
                block_bbox = _normalize_model_bbox_for_containment(block.get("bbox"))
                if block_bbox is not None and any(
                    calculate_overlap_area_in_bbox1_area_ratio(block_bbox, image_block_bbox) >= containment_threshold
                    for image_block_bbox in existing_image_block_bboxes
                ):
                    existing_claimed_block_ids.add(id(block))

        candidates: list[tuple[int, float, int, int, dict[str, Any], set[int]]] = []
        for layout_order, layout_block in enumerate(page_layout_blocks):
            if layout_block.get("type") != BlockType.IMAGE or layout_block.get("sub_type") == "seal":
                continue

            layout_bbox = _normalize_model_bbox_for_containment(layout_block.get("bbox"))
            if layout_bbox is None:
                continue
            if any(
                calculate_overlap_area_2_minbox_area_ratio(layout_bbox, image_block_bbox) >= containment_threshold
                for image_block_bbox in existing_image_block_bboxes
            ):
                continue

            contained_blocks: list[tuple[int, dict[str, Any], BBox]] = []
            for block_index, block in enumerate(page_model_list):
                if block.get("type") == "image_block" or id(block) in existing_claimed_block_ids:
                    continue
                block_bbox = _normalize_model_bbox_for_containment(block.get("bbox"))
                if block_bbox is None:
                    continue
                if calculate_overlap_area_in_bbox1_area_ratio(block_bbox, layout_bbox) >= containment_threshold:
                    contained_blocks.append((block_index, block, block_bbox))

            contained_visuals = [
                (block_index, block)
                for block_index, block, _ in contained_blocks
                if block.get("type") in LOCAL_LAYOUT_IMAGE_BLOCK_BODY_TYPES
            ]
            if len(contained_visuals) < min_visual_count:
                continue

            layout_area = (layout_bbox[2] - layout_bbox[0]) * (layout_bbox[3] - layout_bbox[1])
            contained_area = sum(
                (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])
                for _, block, block_bbox in contained_blocks
                if block.get("type") in LOCAL_LAYOUT_IMAGE_BLOCK_AREA_TYPES
            )
            coverage_ratio = contained_area / layout_area
            if coverage_ratio < coverage_threshold and not math.isclose(
                coverage_ratio,
                coverage_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                continue

            contained_block_ids = {id(block) for _, block, _ in contained_blocks}
            first_block_index = min(block_index for block_index, _, _ in contained_blocks)
            candidates.append(
                (
                    -len(contained_visuals),
                    layout_area,
                    layout_order,
                    first_block_index,
                    layout_block,
                    contained_block_ids,
                )
            )

        claimed_block_ids: set[int] = set()
        selected_containers: list[tuple[int, dict[str, Any]]] = []
        for _, _, _, first_block_index, layout_block, block_ids in sorted(candidates):
            if claimed_block_ids.intersection(block_ids):
                continue
            claimed_block_ids.update(block_ids)
            selected_containers.append(
                (
                    first_block_index,
                    {
                        "type": "image_block",
                        "bbox": list(layout_block["bbox"]),
                        "angle": layout_block.get("angle", 0),
                        "content": None,
                    },
                )
            )

        for insert_index, image_block in sorted(selected_containers, reverse=True):
            page_model_list.insert(insert_index, image_block)


def _attach_visual_block_images(
    model_list: list[list[dict[str, Any]]],
    images_list: list[dict[str, Any]],
    page_start_index: int = 0,
) -> None:
    """在窗口页图释放前，为最终 model_list 视觉块写入回正后的页面裁图。"""
    if len(model_list) != len(images_list):
        raise ValueError(f"Hybrid visual crop page count mismatch: model_list={len(model_list)}, images={len(images_list)}")

    for page_offset, (page_model_list, image_dict) in enumerate(zip(model_list, images_list)):
        _collapse_image_blocks(page_model_list)
        visual_blocks = [
            (block_idx, block)
            for block_idx, block in enumerate(page_model_list)
            if block.get("type") in MODEL_JSON_VISUAL_BLOCK_TYPES
        ]
        if not visual_blocks:
            continue

        page_index = page_start_index + page_offset
        page_pil_image = image_dict.get("img_pil")
        if page_pil_image is None:
            logger.warning(f"Skipping model visual block crops without page image: page={page_index}")
            continue

        converted_page_image = None
        try:
            if getattr(page_pil_image, "mode", None) == "RGB":
                page_rgb_image = page_pil_image
            else:
                converted_page_image = page_pil_image.convert("RGB")
                page_rgb_image = converted_page_image

            page_size = _normalize_page_size(page_rgb_image)
            np_image = np.asarray(page_rgb_image)
            for block_idx, block in visual_blocks:
                try:
                    pixel_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
                    if pixel_bbox is None:
                        raise ValueError("invalid bbox")
                    angle = _normalize_visual_block_angle(block.get("angle", 0))
                    image_base64 = _encode_page_crop_as_jpeg_data_uri(
                        np_image,
                        pixel_bbox,
                        angle,
                    )
                    if not image_base64:
                        raise ValueError("empty crop or JPEG encoding failure")
                    block["image_base64"] = image_base64
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid model visual block crop: "
                        f"page={page_index}, block={block_idx}, type={block.get('type')}, "
                        f"bbox={block.get('bbox')}, error={exc}"
                    )
        finally:
            if converted_page_image is not None:
                converted_page_image.close()
