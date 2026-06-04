# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.structs import BlockType, ExtractResult
from PIL import Image
from tqdm import tqdm

from ...data.data_reader_writer import DataWriter
from ...utils.config_reader import get_device, get_processing_window_size
from ...utils.enum_class import BlockType as MineruBlockType
from ...utils.enum_class import ImageType, NotExtractType
from ...utils.model_utils import clean_memory, crop_img, get_vram
from ...utils.ocr_utils import (
    OcrConfidence,
    get_adjusted_mfdetrec_res,
    get_ocr_result_list,
    merge_det_boxes,
    sorted_boxes,
    update_det_boxes,
)
from ...utils.pdf_classify import classify
from ...utils.pdf_image_tools import aio_load_images_from_pdf_bytes_range, load_images_from_pdf_doc
from ...utils.pdfium_guard import close_pdfium_document, get_pdfium_document_page_count, open_pdfium_document
from ..pipeline.model_init import (
    HybridModelSingleton,
    MineruHybridModel,
    run_layout_inference,
    run_mfr_inference,
    run_ocr_det_inference,
    run_ocr_rec_inference,
)
from ..utils.boxbase import calculate_overlap_area_2_minbox_area_ratio
from ..utils.middle_json_utils import append_pages
from ..utils.runtime_utils import exclude_progress_bar_idle_time
from ..vlm.vlm_analyze import (
    ModelSingleton,
    _get_model_async,
    _maybe_enable_serial_execution,
    aio_predictor_execution_guard,
    predictor_execution_guard,
)
from .model_output_to_middle_json import (
    apply_server_side_postprocess,
    blocks_to_page_info,
    finalize_middle_json,
    init_middle_json,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 让mps可以fallback

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8

not_extract_list = [item.value for item in NotExtractType]
HYBRID_OCR_DET_TEXT_TYPES = set(not_extract_list)


def _is_hybrid_ocr_det_candidate(block: dict[str, Any]) -> bool:
    """判断 Hybrid 文本类块是否需要 OCR det 生成行级视觉信息。"""
    return (block.get("type") or block.get("label")) in HYBRID_OCR_DET_TEXT_TYPES


def ocr_classify(
    pdf_bytes: bytes,
    parse_method: str = "auto",
) -> bool:
    # 确定OCR设置
    _ocr_enable = False
    if parse_method == "auto":
        if classify(pdf_bytes) == "ocr":
            _ocr_enable = True
    elif parse_method == "ocr":
        _ocr_enable = True
    return _ocr_enable


def ocr_det(
    hybrid_pipeline_model: Any,
    np_images: list[Any],
    model_list: list[list[dict[str, Any]]],
    mfd_res: list[Any],
    _ocr_enable: bool,
    batch_ratio: int = 1,
    *,
    fill_text: bool = True,
) -> None:
    def _set_temp_pixel_bbox(res: dict[str, Any], pixel_bbox: list[int]) -> None:
        res["_normalized_bbox"] = list(res["bbox"])
        res["bbox"] = pixel_bbox

    def _restore_normalized_bbox(res: dict[str, Any]) -> None:
        normalized_bbox = res.pop("_normalized_bbox", None)
        if normalized_bbox is not None:
            res["bbox"] = normalized_bbox

    ocr_res_list = []
    if not hybrid_pipeline_model.enable_ocr_det_batch:
        # 非批处理模式 - 逐页处理
        for np_image, page_mfd_res, page_results in tqdm(
            zip(np_images, mfd_res, model_list), total=len(np_images), desc="OCR-det"
        ):
            ocr_res_list.append([])
            img_height, img_width = np_image.shape[:2]
            for res in page_results:
                if not _is_hybrid_ocr_det_candidate(res):
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
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                ocr_res = run_ocr_det_inference(
                    hybrid_pipeline_model.ocr_model.ocr,
                    bgr_image,
                    mfd_res=adjusted_mfdetrec_res,
                    rec=False,
                )[0]
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(
                        ocr_res,
                        useful_list,
                        _ocr_enable if fill_text else False,
                        bgr_image,
                        hybrid_pipeline_model.lang,
                    )

                    ocr_res_list[-1].extend(ocr_result_list)
    else:
        # 批处理模式 - 按语言和分辨率分组
        # 收集所有需要OCR检测的裁剪图像
        all_cropped_images_info = []

        for np_image, page_mfd_res, page_results in zip(np_images, mfd_res, model_list):
            ocr_res_list.append([])
            img_height, img_width = np_image.shape[:2]
            for res in page_results:
                if not _is_hybrid_ocr_det_candidate(res):
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
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                all_cropped_images_info.append((bgr_image, useful_list, adjusted_mfdetrec_res, ocr_res_list[-1]))

        # 按分辨率分组并同时完成padding
        RESOLUTION_GROUP_STRIDE = 64  # 32

        resolution_groups = defaultdict(list)
        for crop_info in all_cropped_images_info:
            cropped_img = crop_info[0]
            h, w = cropped_img.shape[:2]
            # 直接计算目标尺寸并用作分组键
            target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            group_key = (target_h, target_w)
            resolution_groups[group_key].append(crop_info)

        # 对每个分辨率组进行批处理
        for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det"):
            # 对所有图像进行padding到统一尺寸
            batch_images = []
            for crop_info in group_crops:
                img = crop_info[0]
                h, w = img.shape[:2]
                # 创建目标尺寸的白色背景
                padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                padded_img[:h, :w] = img
                batch_images.append(padded_img)

            # 批处理检测
            det_batch_size = min(len(batch_images), batch_ratio * OCR_DET_BASE_BATCH_SIZE)
            batch_results = run_ocr_det_inference(
                hybrid_pipeline_model.ocr_model.text_detector.batch_predict,
                batch_images,
                det_batch_size,
            )

            # 处理批处理结果
            for crop_info, (dt_boxes, _) in zip(group_crops, batch_results):
                bgr_image, useful_list, adjusted_mfdetrec_res, ocr_page_res_list = crop_info

                if dt_boxes is not None and len(dt_boxes) > 0:
                    # 处理检测框
                    dt_boxes_sorted = sorted_boxes(dt_boxes)
                    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                    # 根据公式位置更新检测框
                    dt_boxes_final = (
                        update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                        if dt_boxes_merged and adjusted_mfdetrec_res
                        else dt_boxes_merged
                    )

                    if dt_boxes_final:
                        ocr_res = [box.tolist() if hasattr(box, "tolist") else box for box in dt_boxes_final]
                        ocr_result_list = get_ocr_result_list(
                            ocr_res,
                            useful_list,
                            _ocr_enable if fill_text else False,
                            bgr_image,
                            hybrid_pipeline_model.lang,
                        )
                        ocr_page_res_list.extend(ocr_result_list)
    return ocr_res_list


def mask_image_regions(np_images: list[Any], model_list: list[list[dict[str, Any]]]) -> list[Any]:
    # 根据vlm返回的结果，在每一页中将image、table、equation块mask成白色背景图像
    for np_image, vlm_page_results in zip(np_images, model_list):
        img_height, img_width = np_image.shape[:2]
        # 收集需要mask的区域
        mask_regions = []
        for block in vlm_page_results:
            if block["type"] in [BlockType.IMAGE, BlockType.TABLE, BlockType.EQUATION]:
                bbox = block["bbox"]
                # 批量转换归一化坐标到像素坐标,并进行边界检查
                x0 = max(0, int(bbox[0] * img_width))
                y0 = max(0, int(bbox[1] * img_height))
                x1 = min(img_width, int(bbox[2] * img_width))
                y1 = min(img_height, int(bbox[3] * img_height))
                # 只添加有效区域
                if x1 > x0 and y1 > y0:
                    mask_regions.append((y0, y1, x0, x1))
        # 批量应用mask
        for y0, y1, x0, x1 in mask_regions:
            np_image[y0:y1, x0:x1, :] = 255
    return np_images


def normalize_bbox_to_unit(item: dict[str, Any], page_width: int, page_height: int) -> bool:
    """将像素级bbox归一化为[0, 1]区间"""
    bbox = item.get("bbox")
    if bbox is None or len(bbox) != 4:
        return False

    x0, y0, x1, y1 = [float(v) for v in bbox]
    if 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0:
        normalized_bbox = [x0, y0, x1, y1]
    else:
        normalized_bbox = [
            x0 / page_width,
            y0 / page_height,
            x1 / page_width,
            y1 / page_height,
        ]
    item["bbox"] = [round(min(max(v, 0), 1), 3) for v in normalized_bbox]
    return True


def _formula_item_to_pixel_bbox(item: dict[str, Any]) -> list[int] | None:
    bbox = item.get("bbox")
    if bbox is not None and len(bbox) == 4:
        return [int(float(v)) for v in bbox]
    return None


def _build_inline_formula_inputs(images_layout_res: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    inline_formula_inputs = []
    for layout_res in images_layout_res:
        page_inline_formula_inputs = []
        for res in layout_res:
            if res.get("label") not in ["inline_formula", "display_formula"]:
                continue
            bbox = res.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            page_inline_formula_inputs.append(
                {
                    "label": "inline_formula",
                    "bbox": list(bbox),
                    "score": float(res.get("score", 0.0)),
                    "latex": res.get("latex", ""),
                }
            )
        inline_formula_inputs.append(page_inline_formula_inputs)
    return inline_formula_inputs


def _build_formula_mask_inputs(images_layout_res: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """从 layout 检测结果提取公式框，供 OCR det 规避行内/行间公式区域。"""
    page_formula_masks = []
    for layout_res in images_layout_res:
        page_masks = []
        for res in layout_res:
            if res.get("label") not in ["inline_formula", "display_formula"]:
                continue
            bbox = _formula_item_to_pixel_bbox(res)
            if bbox is not None:
                page_masks.append({"bbox": bbox})
        page_formula_masks.append(page_masks)
    return page_formula_masks


def _normalize_page_size(page_image: Any) -> list[int]:
    """从PIL或numpy图像中读取页面宽高，供归一化bbox还原为像素bbox。"""
    if hasattr(page_image, "size"):
        return page_image.size

    height, width = page_image.shape[:2]
    return width, height


def _bbox_to_pixel_bbox(bbox: list[float], page_size: list[int]) -> list[int]:
    """将归一化或像素bbox统一成像素bbox，异常bbox返回None。"""
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    width, height = page_size
    if all(0.0 <= value <= 1.0 for value in [x0, y0, x1, y1]):
        x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height

    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _collect_layout_doc_title_bboxes(layout_res: list[dict[str, Any]], page_size: list[int]) -> list[list[float]]:
    """只收集layout小模型输出的doc_title框，忽略paragraph_title等其他类型。"""
    doc_title_bboxes = []
    for layout_item in layout_res or []:
        if layout_item.get("label") != MineruBlockType.DOC_TITLE:
            continue
        bbox = _bbox_to_pixel_bbox(layout_item.get("bbox"), page_size)
        if bbox is not None:
            doc_title_bboxes.append(bbox)
    return doc_title_bboxes


def _has_doc_title_overlap(title_bbox: list[float], doc_title_bboxes: list[list[float]], overlap_threshold: float) -> bool:
    """判断VLM标题框是否与任一layout doc_title框达到最小框重叠阈值。"""
    return any(
        calculate_overlap_area_2_minbox_area_ratio(title_bbox, doc_title_bbox) >= overlap_threshold
        for doc_title_bbox in doc_title_bboxes
    )


def _apply_layout_title_split(
    model_list: list[list[dict[str, Any]]],
    images_layout_res: list[list[dict[str, Any]]],
    page_sizes: list[list[int]],
    overlap_threshold: float = LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD,
) -> None:
    """用layout doc_title框将VLM title拆分为doc_title和paragraph_title。"""
    for page_model_list, layout_res, page_size in zip(model_list, images_layout_res, page_sizes):
        doc_title_bboxes = _collect_layout_doc_title_bboxes(layout_res, page_size)
        for block in page_model_list:
            if block.get("type") != MineruBlockType.TITLE:
                continue
            title_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if title_bbox is None:
                continue
            if _has_doc_title_overlap(title_bbox, doc_title_bboxes, overlap_threshold):
                block["type"] = MineruBlockType.DOC_TITLE
            else:
                block["type"] = MineruBlockType.PARAGRAPH_TITLE


def _predict_layout_for_title_split(
    hybrid_pipeline_model: MineruHybridModel,
    images: list[Image.Image],
    batch_ratio: int,
) -> list[dict[str, Any]]:
    """执行layout小模型检测，专门为Hybrid标题拆分提供页面layout结果。"""
    return run_layout_inference(
        hybrid_pipeline_model.layout_model.batch_predict,
        images,
        batch_size=min(8, batch_ratio * LAYOUT_BASE_BATCH_SIZE),
    )


def _process_ocr_and_formulas(
    images_pil_list: list[Image.Image],
    model_list: list[ExtractResult],
    language: str | None,
    inline_formula_enable: bool,
    _ocr_enable: bool,
    batch_ratio: int = 1,
) -> tuple[list[list[dict[str, Any]]], MineruHybridModel]:
    """处理OCR和公式识别"""

    # 遍历model_list,对文本块截图交由OCR识别
    # 根据_ocr_enable决定ocr只开det还是det+rec
    # 根据inline_formula_enable决定是使用mfd和ocr结合的方式,还是纯ocr方式

    # 将PIL图片转换为numpy数组
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]

    # 获取混合模型实例
    hybrid_model_singleton = HybridModelSingleton()
    hybrid_pipeline_model = hybrid_model_singleton.get_model(
        lang=language,
        formula_enable=inline_formula_enable,
    )

    # 在进行`行内`公式检测和识别前，先将图像中的图片、表格、`行间`公式区域mask掉
    layout_images = mask_image_regions(np_images, model_list) if inline_formula_enable else np_images
    images_layout_res = _predict_layout_for_title_split(
        hybrid_pipeline_model,
        layout_images,
        batch_ratio,
    )

    if inline_formula_enable:
        images_mfd_res = _build_inline_formula_inputs(images_layout_res)
        # 公式识别
        inline_formula_list = run_mfr_inference(
            hybrid_pipeline_model.mfr_model.batch_predict,
            images_mfd_res,
            np_images,
            batch_size=batch_ratio * MFR_BASE_BATCH_SIZE,
            interline_enable=True,
        )
    else:
        inline_formula_list = [[] for _ in range(len(images_pil_list))]

    mfd_res = []
    for page_inline_formula_list in inline_formula_list:
        page_mfd_res = []
        for formula in page_inline_formula_list:
            bbox = _formula_item_to_pixel_bbox(formula)
            if bbox is None:
                continue
            page_mfd_res.append({"bbox": bbox})
        mfd_res.append(page_mfd_res)

    # vlm没有执行ocr，需要ocr_det
    ocr_res_list = ocr_det(
        hybrid_pipeline_model,
        np_images,
        model_list,
        mfd_res,
        _ocr_enable,
        batch_ratio=batch_ratio,
    )

    # 如果需要ocr则做ocr_rec
    if _ocr_enable:
        need_ocr_list = []
        img_crop_list = []
        for page_ocr_res_list in ocr_res_list:
            for ocr_res in page_ocr_res_list:
                if "np_img" in ocr_res:
                    need_ocr_list.append((page_ocr_res_list, ocr_res))
                    img_crop_list.append(ocr_res.pop("np_img"))
        if len(img_crop_list) > 0:
            # Process OCR
            ocr_result_list = run_ocr_rec_inference(
                hybrid_pipeline_model.ocr_model.ocr,
                img_crop_list,
                det=False,
                tqdm_enable=True,
            )[0]

            # Verify we have matching counts
            assert len(ocr_result_list) == len(need_ocr_list), (
                f"ocr_result_list: {len(ocr_result_list)}, need_ocr_list: {len(need_ocr_list)}"
            )

            items_to_remove = []
            # Process OCR results for this language
            for index, (page_ocr_res_list, need_ocr_res) in enumerate(need_ocr_list):
                ocr_text, ocr_score = ocr_result_list[index]
                need_ocr_res["text"] = ocr_text
                need_ocr_res["score"] = float(f"{ocr_score:.3f}")
                should_remove = False
                if ocr_score < OcrConfidence.min_confidence:
                    should_remove = True
                else:
                    layout_res_bbox = need_ocr_res.get("bbox")
                    if layout_res_bbox is None and need_ocr_res.get("poly") is not None:
                        layout_res_bbox = [
                            need_ocr_res["poly"][0],
                            need_ocr_res["poly"][1],
                            need_ocr_res["poly"][4],
                            need_ocr_res["poly"][5],
                        ]
                    if layout_res_bbox is None:
                        should_remove = True
                        continue
                    layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                    layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                    if (
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
                    ):
                        should_remove = True

                if should_remove:
                    items_to_remove.append((page_ocr_res_list, need_ocr_res))

            for page_ocr_res_list, need_ocr_res in items_to_remove:
                if need_ocr_res in page_ocr_res_list:
                    page_ocr_res_list.remove(need_ocr_res)

    _apply_layout_title_split(
        model_list,
        images_layout_res,
        [_normalize_page_size(image) for image in images_pil_list],
    )

    _normalize_bbox(inline_formula_list, ocr_res_list, images_pil_list)
    merged_model_list = _merge_page_sidecar_items(
        model_list,
        inline_formula_list,
        ocr_res_list,
    )
    return merged_model_list, hybrid_pipeline_model


def _apply_layout_title_split_for_window(
    images_pil_list: list[Image.Image],
    model_list: list[ExtractResult],
    language: str | None,
    batch_ratio: int,
) -> MineruHybridModel:
    """为VLM-OCR路径补跑layout小模型，先基于VLM原始title做OCR det，再拆分标题。"""
    hybrid_model_singleton = HybridModelSingleton()
    hybrid_pipeline_model = hybrid_model_singleton.get_model(
        lang=language,
        formula_enable=False,
    )
    images_layout_res = _predict_layout_for_title_split(
        hybrid_pipeline_model,
        images_pil_list,
        batch_ratio,
    )
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    ocr_res_list = ocr_det(
        hybrid_pipeline_model,
        np_images,
        model_list,
        _build_formula_mask_inputs(images_layout_res),
        False,
        batch_ratio=batch_ratio,
        fill_text=False,
    )
    _normalize_bbox([[] for _ in images_pil_list], ocr_res_list, images_pil_list)
    model_list[:] = _merge_page_sidecar_items(
        model_list,
        [[] for _ in images_pil_list],
        ocr_res_list,
        keep_ocr_text=False,
    )
    _apply_layout_title_split(
        model_list,
        images_layout_res,
        [_normalize_page_size(image) for image in images_pil_list],
    )
    return hybrid_pipeline_model


def _normalize_bbox(
    inline_formula_list: list[list[dict[str, Any]]],
    ocr_res_list: list[list[dict[str, Any]]],
    images_pil_list: list[Image.Image],
) -> None:
    """归一化坐标并生成最终结果"""
    for page_inline_formula_list, page_ocr_res_list, page_pil_image in zip(inline_formula_list, ocr_res_list, images_pil_list):
        if page_inline_formula_list or page_ocr_res_list:
            page_width, page_height = page_pil_image.size
            # 处理公式列表
            for formula in page_inline_formula_list:
                normalize_bbox_to_unit(formula, page_width, page_height)
            # 处理OCR结果列表
            for ocr_res in page_ocr_res_list:
                normalize_bbox_to_unit(ocr_res, page_width, page_height)


def _build_inline_formula_model_item(formula: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "inline_formula",
        "bbox": list(formula["bbox"]),
        "latex": formula.get("latex", ""),
        "score": float(formula.get("score", 0.0)),
    }


def _build_ocr_text_model_item(ocr_res: dict[str, Any], keep_text: bool = True) -> dict[str, Any]:
    """构造 OCR det sidecar；VLM-OCR 路径可只保留空文本行提示。"""
    return {
        "type": "ocr_text",
        "bbox": list(ocr_res["bbox"]),
        "text": ocr_res.get("text", "") if keep_text else "",
        "score": float(ocr_res.get("score", 0.0)),
    }


def _merge_page_sidecar_items(
    model_list: list[list[dict[str, Any]]],
    inline_formula_list: list[list[dict[str, Any]]],
    ocr_res_list: list[list[dict[str, Any]]],
    keep_ocr_text: bool = True,
) -> list[list[dict[str, Any]]]:
    merged_model_list = []
    for page_model_list, page_inline_formula_list, page_ocr_res_list in zip(model_list, inline_formula_list, ocr_res_list):
        merged_page_model_list = list(page_model_list)
        merged_page_model_list.extend(
            _build_inline_formula_model_item(formula) for formula in page_inline_formula_list if formula.get("bbox") is not None
        )
        merged_page_model_list.extend(
            _build_ocr_text_model_item(ocr_res, keep_text=keep_ocr_text)
            for ocr_res in page_ocr_res_list
            if ocr_res.get("bbox") is not None
        )
        merged_model_list.append(merged_page_model_list)
    return merged_model_list


def get_batch_ratio(device: str) -> int:
    """
    根据显存大小或环境变量获取 batch ratio
    """
    # 1. 优先尝试从环境变量获取
    """
    c/s架构分离部署时，建议通过设置环境变量 MINERU_HYBRID_BATCH_RATIO 来指定 batch ratio
    建议的设置值如如下，以下配置值已考虑一定的冗余，单卡多终端部署时为了保证稳定性，可以额外保留一个client端的显存作为整体冗余
    单个client端显存大小 | MINERU_HYBRID_BATCH_RATIO
    ------------------|------------------------
    <= 6   GB         | 8
    <= 4   GB         | 4
    <= 3   GB         | 2
    <= 2   GB         | 1
    例如：
    export MINERU_HYBRID_BATCH_RATIO=4
    """
    env_val = os.getenv("MINERU_HYBRID_BATCH_RATIO")
    if env_val:
        try:
            batch_ratio = int(env_val)
            logger.info(f"hybrid batch ratio (from env): {batch_ratio}")
            return batch_ratio
        except ValueError as e:
            logger.warning(f"Invalid MINERU_HYBRID_BATCH_RATIO value: {env_val}, switching to auto mode. Error: {e}")

    # 2. 根据显存自动推断
    """
    根据总显存大小粗略估计 batch ratio，需要排除掉vllm等推理框架占用的显存开销
    """
    gpu_memory = get_vram(device)
    if gpu_memory >= 32:
        batch_ratio = 16
    elif gpu_memory >= 16:
        batch_ratio = 8
    elif gpu_memory >= 12:
        batch_ratio = 4
    elif gpu_memory >= 8:
        batch_ratio = 2
    else:
        batch_ratio = 1

    logger.info(f"hybrid batch ratio (auto, vram={gpu_memory}GB): {batch_ratio}")
    return batch_ratio


def _should_enable_vlm_ocr(ocr_enable: bool, language: str, inline_formula_enable: bool) -> bool:
    """判断是否启用VLM OCR"""
    force_enable = os.getenv("MINERU_FORCE_VLM_OCR_ENABLE", "0").lower() in ("1", "true", "yes")
    if force_enable:
        return True

    force_pipeline = os.getenv("MINERU_HYBRID_FORCE_PIPELINE_ENABLE", "0").lower() in ("1", "true", "yes")
    return ocr_enable and language in ["ch", "en"] and inline_formula_enable and not force_pipeline


def _close_images(images_list: list[dict[str, Any]]) -> None:
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def doc_analyze(
    pdf_bytes: bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend: str = "transformers",
    parse_method: str = "auto",
    language: str = "ch",
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs: object,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]], bool]:
    client_side_output_generation = bool(kwargs.pop("client_side_output_generation", False))
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    device = get_device()
    _ocr_enable = ocr_classify(pdf_bytes, parse_method=parse_method)
    _vlm_ocr_enable = _should_enable_vlm_ocr(_ocr_enable, language, inline_formula_enable)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json(_ocr_enable, _vlm_ocr_enable)
    model_list: list[ExtractResult] = []
    doc_closed = False
    hybrid_pipeline_model = None
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        total_windows = (page_count + effective_window_size - 1) // effective_window_size if effective_window_size else 0
        logger.info(
            f"Hybrid processing-window run. page_count={page_count}, "
            f"window_size={configured_window_size}, total_windows={total_windows}"
        )

        batch_ratio = get_batch_ratio(device) if not _vlm_ocr_enable else 1

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = load_images_from_pdf_doc(
                    pdf_doc,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                    pdf_bytes=pdf_bytes,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    logger.info(
                        f"Hybrid processing window {window_index + 1}/{total_windows}: "
                        f"pages {window_start + 1}-{window_end + 1}/{page_count} "
                        f"({len(images_pil_list)} pages)"
                    )
                    if _vlm_ocr_enable:
                        with predictor_execution_guard(predictor):
                            window_model_list = predictor.batch_two_step_extract(
                                images=images_pil_list,
                                image_analysis=image_analysis,
                            )
                        hybrid_pipeline_model = _apply_layout_title_split_for_window(
                            images_pil_list,
                            window_model_list,
                            language,
                            batch_ratio,
                        )
                    else:
                        with predictor_execution_guard(predictor):
                            window_model_list = predictor.batch_two_step_extract(
                                images=images_pil_list,
                                not_extract_list=not_extract_list,
                                image_analysis=image_analysis,
                            )
                        window_model_list, hybrid_pipeline_model = _process_ocr_and_formulas(
                            images_pil_list,
                            window_model_list,
                            language,
                            inline_formula_enable,
                            _ocr_enable,
                            batch_ratio=batch_ratio,
                        )

                    model_list.extend(window_model_list)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_pages(
                        middle_json,
                        window_model_list,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_cvt_fn=blocks_to_page_info,
                        page_start_index=window_start,
                        _ocr_enable=_ocr_enable,
                        _vlm_ocr_enable=_vlm_ocr_enable,
                        progress_bar=progress_bar,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"processing-window infer finished, cost: {infer_time}, speed: {round(len(model_list) / infer_time, 3)} page/s"
            )

        if client_side_output_generation:
            apply_server_side_postprocess(
                middle_json["pdf_info"],
                hybrid_pipeline_model,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        else:
            finalize_middle_json(
                middle_json["pdf_info"],
                hybrid_pipeline_model,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        close_pdfium_document(pdf_doc)
        doc_closed = True
        clean_memory(device)
        return middle_json, model_list, _vlm_ocr_enable
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)


async def aio_doc_analyze(
    pdf_bytes: bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend: str = "transformers",
    parse_method: str = "auto",
    language: str = "ch",
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs: object,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]], bool]:
    client_side_output_generation = bool(kwargs.pop("client_side_output_generation", False))
    if predictor is None:
        predictor = await _get_model_async(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    device = get_device()
    _ocr_enable = ocr_classify(pdf_bytes, parse_method=parse_method)
    _vlm_ocr_enable = _should_enable_vlm_ocr(_ocr_enable, language, inline_formula_enable)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json(_ocr_enable, _vlm_ocr_enable)
    model_list = []
    doc_closed = False
    hybrid_pipeline_model = None
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        total_windows = (page_count + effective_window_size - 1) // effective_window_size if effective_window_size else 0
        logger.info(
            f"Hybrid processing-window run. page_count={page_count}, "
            f"window_size={configured_window_size}, total_windows={total_windows}"
        )

        batch_ratio = get_batch_ratio(device) if not _vlm_ocr_enable else 1

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = await aio_load_images_from_pdf_bytes_range(
                    pdf_bytes,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    logger.info(
                        f"Hybrid processing window {window_index + 1}/{total_windows}: "
                        f"pages {window_start + 1}-{window_end + 1}/{page_count} "
                        f"({len(images_pil_list)} pages)"
                    )
                    if _vlm_ocr_enable:
                        async with aio_predictor_execution_guard(predictor):
                            window_model_list = await predictor.aio_batch_two_step_extract(
                                images=images_pil_list,
                                image_analysis=image_analysis,
                            )
                        hybrid_pipeline_model = await asyncio.to_thread(
                            _apply_layout_title_split_for_window,
                            images_pil_list,
                            window_model_list,
                            language,
                            batch_ratio,
                        )
                    else:
                        async with aio_predictor_execution_guard(predictor):
                            window_model_list = await predictor.aio_batch_two_step_extract(
                                images=images_pil_list,
                                not_extract_list=not_extract_list,
                                image_analysis=image_analysis,
                            )
                        window_model_list, hybrid_pipeline_model = await asyncio.to_thread(
                            _process_ocr_and_formulas,
                            images_pil_list,
                            window_model_list,
                            language,
                            inline_formula_enable,
                            _ocr_enable,
                            batch_ratio=batch_ratio,
                        )

                    model_list.extend(window_model_list)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_pages(
                        middle_json,
                        window_model_list,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_cvt_fn=blocks_to_page_info,
                        page_start_index=window_start,
                        _ocr_enable=_ocr_enable,
                        _vlm_ocr_enable=_vlm_ocr_enable,
                        progress_bar=progress_bar,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"processing-window infer finished, cost: {infer_time}, speed: {round(len(model_list) / infer_time, 3)} page/s"
            )

        if client_side_output_generation:
            await asyncio.to_thread(
                apply_server_side_postprocess,
                middle_json["pdf_info"],
                hybrid_pipeline_model,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        else:
            await asyncio.to_thread(
                finalize_middle_json,
                middle_json["pdf_info"],
                hybrid_pipeline_model,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        close_pdfium_document(pdf_doc)
        doc_closed = True
        clean_memory(device)
        return middle_json, model_list, _vlm_ocr_enable
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)
