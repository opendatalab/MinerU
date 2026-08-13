# Copyright (c) Opendatalab. All rights reserved.
"""OCR 检测、识别、置信度过滤与印章 OCR。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from tqdm import tqdm

from mineru.backend.local_model_runtime import HybridLocalModelContext
from mineru.types import BlockType
from mineru.utils.bbox_utils import normalize_to_int_bbox
from mineru.utils.model_utils import crop_img
from mineru.utils.ocr_utils import (
    OcrConfidence,
    get_adjusted_mfdetrec_res,
    get_ocr_result_list,
    mask_formula_regions_for_ocr_det,
    merge_det_boxes,
    sorted_boxes,
    update_det_boxes,
)

from .constants import (
    BATCH_RATIO,
    OCR_DET_BASE_BATCH_SIZE,
    PIPELINE_DET_TYPE,
    VLM_OCR_DET_TYPE,
    VLM_TXT_DET_TYPE,
)
from .geometry import _bbox_to_pixel_bbox, _normalize_medium_content


@dataclass
class _OcrDetCrop:
    """保存一次 OCR det 裁剪的中间数据。"""

    bgr_image: Any
    useful_list: list[Any]
    adjusted_mfdetrec_res: list[Any]
    page_ocr_res_list: list[dict[str, Any]]


def _build_ocr_det_type_and_mfr_enable(
    parse_mode: Literal["txt", "ocr"],
    effort: Literal["medium", "high", "xhigh"],
) -> tuple[set[str], bool]:
    """返回 OCR 检测块类型，以及是否需要执行小模型公式识别。"""
    if parse_mode not in ("txt", "ocr"):
        raise ValueError(f"Unsupported parse mode: {parse_mode}")
    if effort not in ("medium", "high", "xhigh"):
        raise ValueError(f"Unsupported analyze effort: {effort}")

    if effort == "medium":
        return PIPELINE_DET_TYPE, True
    if parse_mode == "txt":
        return VLM_TXT_DET_TYPE, True
    return VLM_OCR_DET_TYPE, False


def _set_temp_pixel_bbox(res: dict[str, Any], pixel_bbox: list[int]) -> None:
    """临时切换为像素 bbox，便于复用已有裁剪逻辑。"""
    res["_normalized_bbox"] = list(res["bbox"])
    res["bbox"] = pixel_bbox


def _restore_normalized_bbox(res: dict[str, Any]) -> None:
    """恢复归一化 bbox，避免 OCR det 过程污染 Hybrid 输出。"""
    normalized_bbox = res.pop("_normalized_bbox", None)
    if normalized_bbox is not None:
        res["bbox"] = normalized_bbox


def _collect_ocr_det_crops(
    np_images: list[Any],
    model_list: list[list[dict[str, Any]]],
    mfd_res: list[Any],
    ocr_det_type: set[str],
) -> tuple[list[list[dict[str, Any]]], list[_OcrDetCrop]]:
    """收集 OCR det 需要处理的裁剪图，并为每页预建 sidecar 结果列表。"""
    ocr_res_list: list[list[dict[str, Any]]] = []
    crops: list[_OcrDetCrop] = []

    for np_image, page_mfd_res, page_results in zip(np_images, mfd_res, model_list):
        page_ocr_res_list: list[dict[str, Any]] = []
        ocr_res_list.append(page_ocr_res_list)
        img_height, img_width = np_image.shape[:2]
        for res in page_results:
            if res["type"] not in ocr_det_type:
                continue
            x0 = max(0, int(res["bbox"][0] * img_width))
            y0 = max(0, int(res["bbox"][1] * img_height))
            x1 = min(img_width, int(res["bbox"][2] * img_width))
            y1 = min(img_height, int(res["bbox"][3] * img_height))
            if x1 <= x0 or y1 <= y0:
                continue
            _set_temp_pixel_bbox(res, [x0, y0, x1, y1])
            try:
                new_image, useful_list = crop_img(res, np_image, crop_paste_x=50, crop_paste_y=50)
            finally:
                _restore_normalized_bbox(res)
            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(page_mfd_res, useful_list)
            bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)  # type: ignore
            bgr_image = mask_formula_regions_for_ocr_det(bgr_image, adjusted_mfdetrec_res)
            crops.append(
                _OcrDetCrop(
                    bgr_image=bgr_image,
                    useful_list=useful_list,
                    adjusted_mfdetrec_res=adjusted_mfdetrec_res,
                    page_ocr_res_list=page_ocr_res_list,
                )
            )

    return ocr_res_list, crops


def _normalize_batch_ocr_det_boxes(dt_boxes: Any, adjusted_mfdetrec_res: list[Any]) -> list[Any]:
    """对 batch OCR det 的检测框排序、合并，并按公式位置修正。"""
    if dt_boxes is None or len(dt_boxes) == 0:
        return []

    dt_boxes_sorted = sorted_boxes(dt_boxes)
    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []
    if dt_boxes_merged and adjusted_mfdetrec_res:
        return update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
    return dt_boxes_merged


def _append_ocr_det_result(
    crop: _OcrDetCrop,
    ocr_res: Any,
    need_rec_img: bool,
) -> None:
    """将 OCR det 原始框转换为 Hybrid ocr_text sidecar 并写回对应页。"""
    if not ocr_res:
        return
    ocr_result_list = get_ocr_result_list(
        ocr_res,
        crop.useful_list,
        need_rec_img,
        crop.bgr_image,
    )
    crop.page_ocr_res_list.extend(ocr_result_list)


def _ocr_det(
    local_model_context: HybridLocalModelContext,
    np_images: list[np.ndarray],
    model_list: list[list[dict[str, Any]]],
    mfd_res: list[Any],
    need_rec_img: bool,
    ocr_det_type: set[str],
) -> list[list[dict[str, Any]]]:
    """执行 Hybrid OCR det sidecar 生成，按运行时配置选择单图或 batch 模式。"""
    ocr_res_list, crops = _collect_ocr_det_crops(np_images, model_list, mfd_res, ocr_det_type)

    if crops:
        batch_images = [crop.bgr_image for crop in crops]
        det_batch_size = min(len(batch_images), BATCH_RATIO * OCR_DET_BASE_BATCH_SIZE)
        batch_results = local_model_context.ocr_model.text_detector.batch_predict(
            batch_images,
            det_batch_size,
            tqdm_enable=True,
            tqdm_desc="OCR-det",
        )

        for crop, (dt_boxes, _) in zip(crops, batch_results):
            dt_boxes_final = _normalize_batch_ocr_det_boxes(dt_boxes, crop.adjusted_mfdetrec_res)
            if dt_boxes_final:
                ocr_res = [box.tolist() if hasattr(box, "tolist") else box for box in dt_boxes_final]
                _append_ocr_det_result(crop, ocr_res, need_rec_img)
    return ocr_res_list


def _collect_ocr_rec_inputs(
    ocr_res_list: list[list[dict[str, Any]]],
) -> tuple[list[tuple[list[dict[str, Any]], dict[str, Any]]], list[Any]]:
    """收集需要 OCR rec 的裁剪图，同时从 sidecar 中移除临时图像对象。"""
    need_ocr_list = []
    img_crop_list = []
    for page_ocr_res_list in ocr_res_list:
        for ocr_res in page_ocr_res_list:
            if "np_img" in ocr_res:
                need_ocr_list.append((page_ocr_res_list, ocr_res))
                img_crop_list.append(ocr_res.pop("np_img"))
    return need_ocr_list, img_crop_list


def _should_remove_low_confidence_ocr_text(ocr_text: str, ocr_score: float, ocr_res: dict[str, Any]) -> bool:
    """判断 OCR rec 结果是否应因低置信或竖排噪声被丢弃。"""
    if ocr_score < OcrConfidence.min_confidence:
        return True

    layout_res_bbox = ocr_res.get("bbox")
    if layout_res_bbox is None and ocr_res.get("poly") is not None:
        layout_res_bbox = [
            ocr_res["poly"][0],
            ocr_res["poly"][1],
            ocr_res["poly"][4],
            ocr_res["poly"][5],
        ]
    if layout_res_bbox is None:
        return True

    layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
    layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
    return (
        ocr_text
        in [
            "（204号",
            "（20",
            "（2",
            "（2号",
            "（20号",
            "号",
            "（204",
            "(cid:)",
            "(ci:)",
            "(cd:1)",
            "cd:)",
            "c)",
            "(cd:)",
            "c",
            "id:)",
            ":)",
            "√:)",
            "√i:)",
            "−i:)",
            "−:",
            "i:)",
        ]
        and ocr_score < 0.8
        and layout_res_width < layout_res_height
    )


def _apply_ocr_rec_results(
    local_model_context: HybridLocalModelContext,
    ocr_res_list: list[list[dict[str, Any]]],
) -> None:
    """执行 OCR rec 并把文本写回 sidecar，结果数量异常时显式报错。"""
    need_ocr_list, img_crop_list = _collect_ocr_rec_inputs(ocr_res_list)
    if not img_crop_list:
        return

    ocr_result_list = local_model_context.ocr_model.ocr(
        img_crop_list,
        det=False,
        tqdm_enable=True,
    )[0]

    if len(ocr_result_list) != len(need_ocr_list):
        raise ValueError(
            f"Hybrid OCR rec result count mismatch: ocr_result_list={len(ocr_result_list)}, need_ocr_list={len(need_ocr_list)}"
        )

    items_to_remove = []
    for index, (page_ocr_res_list, need_ocr_res) in enumerate(need_ocr_list):
        ocr_text, ocr_score = ocr_result_list[index]
        need_ocr_res["text"] = ocr_text
        need_ocr_res["score"] = float(f"{ocr_score:.3f}")
        if _should_remove_low_confidence_ocr_text(ocr_text, ocr_score, need_ocr_res):
            items_to_remove.append((page_ocr_res_list, need_ocr_res))

    for page_ocr_res_list, need_ocr_res in items_to_remove:
        if need_ocr_res in page_ocr_res_list:
            page_ocr_res_list.remove(need_ocr_res)


def _apply_seal_ocr(
    local_model_context: HybridLocalModelContext,
    model_list: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> None:
    """对 medium/high 最终 seal block 逐张执行专用 OCR，并将多段文本按行写回 content。"""
    if len(model_list) != len(np_images):
        raise ValueError(f"Hybrid seal OCR page count mismatch: model_list={len(model_list)}, images={len(np_images)}")

    seal_tasks: list[tuple[dict[str, Any], np.ndarray]] = []
    for page_model_list, np_img in zip(model_list, np_images):
        image_h, image_w = np_img.shape[:2]
        for block_item in page_model_list:
            if block_item.get("type") != BlockType.IMAGE or block_item.get("sub_type") != "seal":
                continue

            seal_bbox = normalize_to_int_bbox(
                _bbox_to_pixel_bbox(block_item.get("bbox"), (image_w, image_h)),
                image_size=(image_h, image_w),
            )
            if seal_bbox is None:
                continue

            x0, y0, x1, y1 = seal_bbox
            seal_crop_rgb = np_img[y0:y1, x0:x1]
            if seal_crop_rgb.size == 0:
                continue

            seal_crop_bgr = cv2.cvtColor(seal_crop_rgb, cv2.COLOR_RGB2BGR)
            seal_tasks.append((block_item, seal_crop_bgr))

    if not seal_tasks:
        return

    seal_model = local_model_context.seal_model
    for block_item, seal_crop_bgr in tqdm(
        seal_tasks,
        total=len(seal_tasks),
        desc="OCR-seal",
    ):
        seal_ocr_results = seal_model.ocr(seal_crop_bgr)
        seal_ocr_result = seal_ocr_results[0] if seal_ocr_results else []

        seal_texts = []
        for seal_item in seal_ocr_result or []:
            if not isinstance(seal_item, (list, tuple)) or len(seal_item) != 2:
                continue
            rec_result = seal_item[1]
            if not isinstance(rec_result, (list, tuple)) or not rec_result:
                continue
            seal_text = _normalize_medium_content(rec_result[0])
            if seal_text:
                seal_texts.append(seal_text)

        seal_content = "\n".join(seal_texts)
        if seal_content:
            block_item["content"] = seal_content
