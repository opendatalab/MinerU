# Copyright (c) Opendatalab. All rights reserved.
"""Layout、VLM 块转换和页面坐标归一化。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image

from mineru.types import BBox, BlockType
from mineru.utils.bbox_utils import normalize_to_int_bbox
from mineru.utils.pdf_image_tools import get_crop_np_img

from .geometry import _normalize_layout_bbox_to_unit, _normalize_page_size
from .constants import (
    VLM_LAYOUT_LABEL_MAP,
    VLM_MODEL_LIST_FIELDS,
    VLM_VISUAL_ANNOTATION_TYPE_MAP,
)


def _load_vlm_runtime() -> dict[str, Any]:
    """按需加载 VLM runtime 组件，确保只有 high/extra_high 路径触发 VLM 依赖。"""
    from mineru.model.vlm.runtime import (
        ModelSingleton,
        _get_model_async,
        _maybe_enable_serial_execution,
        aio_predictor_execution_guard,
        predictor_execution_guard,
    )

    return {
        "ModelSingleton": ModelSingleton,
        "_get_model_async": _get_model_async,
        "_maybe_enable_serial_execution": _maybe_enable_serial_execution,
        "aio_predictor_execution_guard": aio_predictor_execution_guard,
        "predictor_execution_guard": predictor_execution_guard,
    }


def _layout_item_to_content_block(layout_item: dict[str, Any], page_size: tuple[int, int]) -> dict | None:
    """将本地 layout 小模型检测项转换为 mineru-vl-utils 的 ContentBlock。"""
    label = layout_item.get("label") or layout_item.get("type")

    block_type = VLM_LAYOUT_LABEL_MAP.get(str(label))
    if block_type is None:
        return None

    bbox = _normalize_layout_bbox_to_unit(layout_item.get("bbox"), page_size)
    if bbox is None:
        return None

    content_block = {
        "type": block_type,
        "bbox": bbox,
        "angle": layout_item.get("angle", 0),
    }

    if block_type == BlockType.IMAGE and label == "seal":
        content_block["sub_type"] = "seal"

    return content_block


def _get_crop_table_img(
    np_img: np.ndarray,
    table_res_bbox: BBox,
    scale: float = 1,
) -> np.ndarray:
    """按指定缩放裁剪表格图，保持 medium 表格处理只使用当前文件窗口图像。"""
    bbox = normalize_to_int_bbox([float(v) / float(scale) for v in table_res_bbox])
    if bbox is None:
        return np_img[0:0, 0:0]
    return get_crop_np_img(bbox, np_img, scale=scale)


def _collect_table_items(
    images_layout_res: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> list[dict[str, Any]]:
    """收集各页有效表格检测项及其裁图，供方向分类阶段批量处理。"""
    table_items = []
    for page_idx, (layout_res, np_img) in enumerate(zip(images_layout_res, np_images)):
        for table_res in layout_res:
            if table_res.get("label") != "table":
                continue
            table_img = _get_crop_table_img(np_img=np_img, table_res_bbox=table_res["bbox"])
            if table_img.size == 0:
                continue
            table_items.append(
                {
                    "table_img": table_img,
                    "layout_item": table_res,
                    "page_idx": page_idx,
                }
            )
    return table_items


def _build_vl_style_layout_blocks(
    images_layout_res: list[list[dict[str, Any]]],
    images_pil_list: list[Image.Image],
) -> list[list[Any]]:
    """按页构造 Hybrid high 模式传给 VLM 的外部 layout blocks。"""
    blocks_list: list[list[Any]] = []
    for layout_res, image in zip(images_layout_res, images_pil_list):
        page_size = _normalize_page_size(image)
        page_blocks = []
        for layout_item in layout_res:
            content_block = _layout_item_to_content_block(layout_item, page_size)
            if content_block is not None:
                page_blocks.append(content_block)
        blocks_list.append(page_blocks)
    return blocks_list


def _convert_vlm_results_to_model_list(
    vlm_results: Iterable[Iterable[Mapping[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """将 VLM 结果投影为允许进入 Analyze 的字段，丢弃未采用的外部属性。"""
    return [
        [
            {field_name: deepcopy(value) for field_name, value in dict(block).items() if field_name in VLM_MODEL_LIST_FIELDS}
            for block in page_blocks
        ]
        for page_blocks in vlm_results
    ]


def _normalize_xhigh_vlm_blocks(model_list: list[list[dict[str, Any]]]) -> None:
    """归一化 xhigh VLM 的视觉注释类型。"""
    for page_model_list in model_list:
        for block in page_model_list:
            normalized_type = VLM_VISUAL_ANNOTATION_TYPE_MAP.get(block.get("type"))
            if normalized_type is not None:
                block["type"] = normalized_type
