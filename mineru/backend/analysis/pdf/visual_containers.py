# Copyright (c) Opendatalab. All rights reserved.
"""依据模型 layout 证据补全 xhigh 图片容器。"""

from __future__ import annotations

import math
from typing import Any

from docvortex.geometry import calculate_overlap_area_2_minbox_area_ratio, calculate_overlap_area_in_bbox1_area_ratio
from docvortex.geometry import normalize_bbox as _normalize_model_bbox_for_containment

from ....types import BBox, BlockType
from .constants import (
    IMAGE_BLOCK_CONTAINMENT_THRESHOLD,
    IMAGE_BLOCK_LAYOUT_COVERAGE_THRESHOLD,
    IMAGE_BLOCK_LAYOUT_MIN_VISUAL_COUNT,
    LOCAL_LAYOUT_IMAGE_BLOCK_AREA_TYPES,
    LOCAL_LAYOUT_IMAGE_BLOCK_BODY_TYPES,
)


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


supplement_missing_image_block_containers = _supplement_missing_image_block_containers

__all__ = ["supplement_missing_image_block_containers"]
