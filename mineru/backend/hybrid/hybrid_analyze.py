# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import base64
import html
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from ...types import NOT_EXTRACT_TYPES, BBox, PageInfo
from ...types import BlockType as MineruBlockType
from ...utils.backend_options import (
    DEFAULT_HYBRID_EFFORT,
    LAYOUT_HYBRID_EFFORT,
    LOCAL_HYBRID_EFFORT,
    MAX_HYBRID_EFFORT,
    validate_effort,
)
from ...utils.config_reader import get_device, get_processing_window_size
from ...utils.enum_class import ImageType
from ...utils.image_payload import ImagePayloadCache
from ...utils.model_utils import clean_memory, clean_vram, crop_img, get_vram
from ...utils.ocr_utils import (
    OcrConfidence,
    get_adjusted_mfdetrec_res,
    get_ocr_result_list,
    get_rotate_crop_image_for_text_rec,
    mask_formula_regions_for_ocr_det,
    merge_det_boxes,
    sorted_boxes,
    update_det_boxes,
)
from ...utils.pdf_document import PDFDocument
from ...utils.pdf_image_tools import aio_load_images_from_pdf_bytes_range, load_images_from_pdf_bytes_range
from ...utils.bbox_utils import normalize_to_int_bbox
from ...utils.pdf_image_tools import get_crop_np_img
from ..local_model_runtime import (
    AtomicModel,
    HybridLocalModelContextSingleton,
    HybridLocalModelContext,
    run_layout_inference,
    run_mfr_inference,
    run_ocr_inference,
)
from ..utils.boxbase import calculate_overlap_area_2_minbox_area_ratio
from ..utils.middle_json_utils import append_pages
from ..utils.runtime_utils import exclude_progress_bar_idle_time
from .model_output_to_middle_json import (
    apply_server_side_postprocess,
    blocks_to_page_info,
    finalize_middle_json,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 让mps可以fallback

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8
TABLE_OCR_REC_SINGLE_CHAR_REPLACEMENTS = {
    "香": "否",
    "哦樂": "哦",
}
TABLE_OCR_REC_REGEX_REPLACEMENTS = (
    # 仅规范化完整的“单个数字 + 號”，避免影响“10號”“第6號”等普通文本。
    (re.compile(r"^([0-9])號$"), r"\1"),
)


@dataclass(frozen=True)
class _ProcessingWindow:
    """记录单个 Hybrid 处理窗口的页码范围，统一同步和异步入口的窗口计算。"""

    index: int
    total: int
    start: int
    end: int


@dataclass
class _OcrDetCrop:
    """保存一次 OCR det 裁剪的中间数据，避免用裸 tuple 传递阶段状态。"""

    bgr_image: Any
    useful_list: list[Any]
    adjusted_mfdetrec_res: list[Any]
    page_ocr_res_list: list[dict[str, Any]]
HYBRID_VLM_LAYOUT_LABEL_MAP = {
    "abstract": MineruBlockType.TEXT,
    "algorithm": MineruBlockType.ALGORITHM,
    "aside_text": MineruBlockType.ASIDE_TEXT,
    "chart": MineruBlockType.CHART,
    "content": MineruBlockType.INDEX,
    "display_formula": MineruBlockType.EQUATION,
    "doc_title": MineruBlockType.TITLE,
    "figure_title": MineruBlockType.IMAGE_CAPTION,
    "footer": MineruBlockType.FOOTER,
    "footer_image": MineruBlockType.FOOTER,
    "footnote": MineruBlockType.PAGE_FOOTNOTE,
    "formula_number": MineruBlockType.FORMULA_NUMBER,
    "header": MineruBlockType.HEADER,
    "header_image": MineruBlockType.HEADER,
    "image": MineruBlockType.IMAGE,
    "number": MineruBlockType.PAGE_NUMBER,
    "paragraph_title": MineruBlockType.TITLE,
    "reference_content": MineruBlockType.REF_TEXT,
    "seal": MineruBlockType.IMAGE,
    "table": MineruBlockType.TABLE,
    "text": MineruBlockType.TEXT,
    "vertical_text": MineruBlockType.TEXT,
}

HYBRID_MEDIUM_LAYOUT_LABEL_MAP = {
    **HYBRID_VLM_LAYOUT_LABEL_MAP,
    "doc_title": MineruBlockType.DOC_TITLE,
    "paragraph_title": MineruBlockType.PARAGRAPH_TITLE,
    "figure_title": MineruBlockType.IMAGE_CAPTION,
    "vision_footnote": MineruBlockType.PAGE_FOOTNOTE,
}

HYBRID_VLM_OCR_DET_TEXT_TYPES = {
    MineruBlockType.TEXT,
    MineruBlockType.TITLE,
    MineruBlockType.DOC_TITLE,
    MineruBlockType.PARAGRAPH_TITLE,
}


def _load_vlm_content_block() -> Any:
    """按需加载 VLM ContentBlock，避免 Hybrid medium 导入阶段依赖 mineru_vl_utils。"""
    from mineru_vl_utils.structs import ContentBlock

    return ContentBlock


def _load_vlm_runtime() -> dict[str, Any]:
    """按需加载 VLM runtime 组件，确保只有 high/extra_high 路径触发 VLM 依赖。"""
    from ...model.vlm.runtime import (
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


def _discard_legacy_formula_table_kwargs(kwargs: dict[str, object]) -> None:
    """丢弃旧公式/表格开关，避免兼容参数继续影响 Hybrid analyzer 行为。"""
    for key in ("inline_formula_enable", "formula_enable", "table_enable"):
        kwargs.pop(key, None)


def _is_hybrid_ocr_det_candidate(block: dict[str, Any]) -> bool:
    """判断 Hybrid 文本类块是否需要 OCR det 生成行级视觉信息。"""
    return (block.get("type") or block.get("label")) in NOT_EXTRACT_TYPES


def _is_hybrid_medium_ocr_det_candidate(block: dict[str, Any]) -> bool:
    """判断 Hybrid medium 本地 layout 块是否需要 OCR det，覆盖 doc/index/list 等 layout 标签。"""
    item_type = block.get("type") or block.get("label")
    return item_type in {
        *NOT_EXTRACT_TYPES,
        MineruBlockType.DOC_TITLE,
        MineruBlockType.PARAGRAPH_TITLE,
        MineruBlockType.INDEX,
        MineruBlockType.ABSTRACT,
        MineruBlockType.ASIDE_TEXT,
        MineruBlockType.PHONETIC,
        MineruBlockType.LIST,
        MineruBlockType.CHART_CAPTION,
        MineruBlockType.CHART_FOOTNOTE,
        MineruBlockType.CODE_FOOTNOTE,
    }


def _is_hybrid_vlm_text_ocr_det_candidate(block: dict[str, Any]) -> bool:
    """判断 VLM 文本内容路径中哪些块只需要小模型 OCR det 行提示。"""
    return (block.get("type") or block.get("label")) in HYBRID_VLM_OCR_DET_TEXT_TYPES


def ocr_classify(pdf_doc: PDFDocument, parse_method: str = "auto") -> bool:
    # 确定OCR设置
    _ocr_enable = False
    if parse_method == "auto":
        if pdf_doc.classify() == "ocr":
            _ocr_enable = True
    elif parse_method == "ocr":
        _ocr_enable = True
    return _ocr_enable


def _build_processing_windows(page_count: int, configured_window_size: int) -> list[_ProcessingWindow]:
    """根据页数和配置窗口大小生成稳定的 Hybrid 分段处理计划。"""
    effective_window_size = min(page_count, configured_window_size) if page_count else 0
    if effective_window_size <= 0:
        return []

    total_windows = (page_count + effective_window_size - 1) // effective_window_size
    return [
        _ProcessingWindow(
            index=window_index,
            total=total_windows,
            start=window_start,
            end=min(page_count - 1, window_start + effective_window_size - 1),
        )
        for window_index, window_start in enumerate(range(0, page_count, effective_window_size))
    ]


def _log_processing_window_plan(page_count: int, configured_window_size: int, total_windows: int) -> None:
    """输出 Hybrid 分段处理计划日志，避免同步和异步入口文案漂移。"""
    logger.info(
        f"Hybrid processing-window run. page_count={page_count}, "
        f"window_size={configured_window_size}, total_windows={total_windows}"
    )


def _log_processing_window(window: _ProcessingWindow, page_count: int, image_count: int) -> None:
    """输出单个 Hybrid 处理窗口的页码范围日志。"""
    logger.info(
        f"Hybrid processing window {window.index + 1}/{window.total}: "
        f"pages {window.start + 1}-{window.end + 1}/{page_count} "
        f"({image_count} pages)"
    )


def _finalize_hybrid_middle_json(
    middle_json: list[PageInfo],
    local_context: HybridLocalModelContext | None,
    _ocr_enable: bool,
    use_vlm_text_content: bool,
    *,
    effort: str,
    client_side_output_generation: bool,
) -> None:
    """统一 Hybrid finalize 入口，并在缺少本地上下文时给出明确错误。"""
    if local_context is None:
        if middle_json:
            raise ValueError("Hybrid local context was not initialized")
        return

    if client_side_output_generation:
        apply_server_side_postprocess(
            middle_json,
            local_context,
            _ocr_enable,
            use_vlm_text_content,
        )
    else:
        finalize_middle_json(
            middle_json,
            local_context,
            _ocr_enable,
            use_vlm_text_content,
            effort=effort,
        )


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
    candidate_fn: Any,
) -> tuple[list[list[dict[str, Any]]], list[_OcrDetCrop]]:
    """收集 OCR det 需要处理的裁剪图，并为每页预建 sidecar 结果列表。"""
    ocr_res_list: list[list[dict[str, Any]]] = []
    crops: list[_OcrDetCrop] = []

    for np_image, page_mfd_res, page_results in zip(np_images, mfd_res, model_list):
        page_ocr_res_list: list[dict[str, Any]] = []
        ocr_res_list.append(page_ocr_res_list)
        img_height, img_width = np_image.shape[:2]
        for res in page_results:
            if not candidate_fn(res):
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


def _append_ocr_det_result(
    local_context: Any,
    crop: _OcrDetCrop,
    ocr_res: Any,
    _ocr_enable: bool,
    fill_text: bool,
) -> None:
    """将 OCR det 原始框转换为 Hybrid ocr_text sidecar 并写回对应页。"""
    if not ocr_res:
        return
    ocr_result_list = get_ocr_result_list(
        ocr_res,
        crop.useful_list,
        _ocr_enable if fill_text else False,
        crop.bgr_image,
        local_context.lang,
    )
    crop.page_ocr_res_list.extend(ocr_result_list)


def _normalize_batch_ocr_det_boxes(dt_boxes: Any, adjusted_mfdetrec_res: list[Any]) -> list[Any]:
    """对 batch OCR det 的检测框排序、合并，并按公式位置修正。"""
    if dt_boxes is None or len(dt_boxes) == 0:
        return []

    dt_boxes_sorted = sorted_boxes(dt_boxes)
    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []
    if dt_boxes_merged and adjusted_mfdetrec_res:
        return update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
    return dt_boxes_merged


def _ocr_det(
    local_context: Any,
    np_images: list[Any],
    model_list: list[list[dict[str, Any]]],
    mfd_res: list[Any],
    _ocr_enable: bool,
    batch_ratio: int = 1,
    *,
    fill_text: bool = True,
    candidate_fn: Any = _is_hybrid_ocr_det_candidate,
) -> list[list[dict[str, Any]]]:
    """执行 Hybrid OCR det sidecar 生成，按运行时配置选择单图或 batch 模式。"""
    ocr_res_list, crops = _collect_ocr_det_crops(np_images, model_list, mfd_res, candidate_fn)

    if not local_context.enable_ocr_det_batch:
        for crop in tqdm(crops, total=len(crops), desc="OCR-det"):
            ocr_res = run_ocr_inference(
                local_context.ocr_model.ocr,
                crop.bgr_image,
                mfd_res=crop.adjusted_mfdetrec_res,
                rec=False,
            )[0]
            _append_ocr_det_result(local_context, crop, ocr_res, _ocr_enable, fill_text)
        return ocr_res_list

    if crops:
        batch_images = [crop.bgr_image for crop in crops]
        det_batch_size = min(len(batch_images), batch_ratio * OCR_DET_BASE_BATCH_SIZE)
        batch_results = run_ocr_inference(
            local_context.ocr_model.text_detector.batch_predict,
            batch_images,
            det_batch_size,
            tqdm_enable=True,
            tqdm_desc="OCR-det",
        )

        for crop, (dt_boxes, _) in zip(crops, batch_results):
            dt_boxes_final = _normalize_batch_ocr_det_boxes(dt_boxes, crop.adjusted_mfdetrec_res)
            if dt_boxes_final:
                ocr_res = [box.tolist() if hasattr(box, "tolist") else box for box in dt_boxes_final]
                _append_ocr_det_result(local_context, crop, ocr_res, _ocr_enable, fill_text)
    return ocr_res_list


def _mask_image_regions(np_images: list[Any], model_list: list[list[dict[str, Any]]]) -> list[Any]:
    # 根据vlm返回的结果，在每一页中将image、table、equation块mask成白色背景图像
    for np_image, vlm_page_results in zip(np_images, model_list):
        img_height, img_width = np_image.shape[:2]
        # 收集需要mask的区域
        mask_regions = []
        for block in vlm_page_results:
            if block["type"] in [MineruBlockType.IMAGE, MineruBlockType.TABLE, MineruBlockType.EQUATION]:
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


def _build_inline_formula_inputs(images_layout_res: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
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


def _build_formula_mask_inputs(images_layout_res: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
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


def _build_inline_formula_det_inputs(images_layout_res: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """从 layout 检测结果提取行内公式框，供 VLM 文本路径作为空内容公式行提示。"""
    inline_formula_inputs = []
    for layout_res in images_layout_res:
        page_inline_formula_inputs = []
        for res in layout_res:
            if res.get("label") != "inline_formula":
                continue
            bbox = _formula_item_to_pixel_bbox(res)
            if bbox is None:
                continue
            page_inline_formula_inputs.append(
                {
                    "bbox": bbox,
                    "score": float(res.get("score", 0.0)),
                    "latex": "",
                }
            )
        inline_formula_inputs.append(page_inline_formula_inputs)
    return inline_formula_inputs


def _normalize_page_size(page_image: Any) -> tuple[int, int]:
    """从PIL或numpy图像中读取页面宽高，供归一化bbox还原为像素bbox。"""
    if hasattr(page_image, "size"):
        return page_image.size

    height, width = page_image.shape[:2]
    return width, height


def _bbox_to_pixel_bbox(bbox: BBox | None, page_size: tuple[int, int]) -> BBox | None:
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
    return (left, top, right, bottom)


def _collect_layout_doc_title_bboxes(layout_res: list[dict[str, Any]], page_size: tuple[int, int]) -> list[BBox]:
    """只收集layout小模型输出的doc_title框，忽略paragraph_title等其他类型。"""
    doc_title_bboxes: list[BBox] = []
    for layout_item in layout_res or []:
        if layout_item.get("label") != MineruBlockType.DOC_TITLE:
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
    local_context: HybridLocalModelContext,
    np_images: list[np.ndarray] | list[Image.Image],
    batch_ratio: int,
) -> list[list[dict[str, Any]]]:
    """执行layout小模型检测，专门为Hybrid标题拆分提供页面layout结果。"""
    return run_layout_inference(
        local_context.layout_model.batch_predict,
        np_images,
        batch_size=min(8, batch_ratio * LAYOUT_BASE_BATCH_SIZE),
    )


def _normalize_layout_bbox_to_unit(bbox: BBox | None, page_size: tuple[int, int]) -> list[float] | None:
    """将 layout 像素 bbox 归一化为 VLM ContentBlock 需要的 0-1 坐标。"""
    pixel_bbox = _bbox_to_pixel_bbox(bbox, page_size)
    if pixel_bbox is None:
        return None

    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return None

    x0, y0, x1, y1 = pixel_bbox
    unit_bbox = [
        max(0.0, min(1.0, float(x0) / page_width)),
        max(0.0, min(1.0, float(y0) / page_height)),
        max(0.0, min(1.0, float(x1) / page_width)),
        max(0.0, min(1.0, float(y1) / page_height)),
    ]
    if unit_bbox[2] <= unit_bbox[0] or unit_bbox[3] <= unit_bbox[1]:
        return None
    return unit_bbox


def _layout_item_to_content_block(layout_item: dict[str, Any], page_size: tuple[int, int]) -> Any | None:
    """将本地 layout 小模型检测项转换为 mineru-vl-utils 的 ContentBlock。"""
    label = layout_item.get("label") or layout_item.get("type")
    block_type = HYBRID_VLM_LAYOUT_LABEL_MAP.get(str(label))
    if block_type is None:
        return None

    bbox = _normalize_layout_bbox_to_unit(layout_item.get("bbox"), page_size)
    if bbox is None:
        return None

    ContentBlock = _load_vlm_content_block()
    try:
        return ContentBlock(type=block_type, bbox=bbox, angle=layout_item.get("angle", 0) or 0)
    except AssertionError:
        logger.debug(f"Skip invalid Hybrid high layout block: {layout_item}")
        return None


def _build_high_layout_blocks(
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


def _normalize_medium_content(value: Any) -> str:
    """将 medium 本地模型输出的文本字段规范成 Hybrid block 可消费的字符串。"""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return value.strip()
    return ""


def _medium_layout_item_to_hybrid_item(
    layout_item: dict[str, Any],
    page_size: tuple[int, int],
) -> dict[str, Any] | None:
    """将本地 layout/OCR 结果转换为 Hybrid MagicModel 可消费的 block 或 sidecar。"""
    label = str(layout_item.get("label") or layout_item.get("type") or "")
    bbox = _normalize_layout_bbox_to_unit(layout_item.get("bbox"), page_size)
    if bbox is None:
        return None

    score = float(layout_item.get("score", 0.0) or 0.0)
    if label == "inline_formula":
        latex = _normalize_medium_content(layout_item.get("latex", ""))
        if not latex:
            return None
        return {"type": "inline_formula", "bbox": bbox, "latex": latex, "score": score}
    if label == "ocr_text":
        return {
            "type": "ocr_text",
            "bbox": bbox,
            "text": _normalize_medium_content(layout_item.get("text", "")),
            "score": score,
        }

    block_type = HYBRID_MEDIUM_LAYOUT_LABEL_MAP.get(label)
    if block_type is None:
        return None

    hybrid_item: dict[str, Any] = {
        "type": block_type,
        "bbox": bbox,
        "angle": layout_item.get("angle", 0) or 0,
    }
    if label == "seal":
        hybrid_item["sub_type"] = "seal"

    if block_type == MineruBlockType.TABLE:
        hybrid_item["content"] = _normalize_medium_content(layout_item.get("html", ""))
    elif block_type == MineruBlockType.EQUATION:
        hybrid_item["content"] = _normalize_medium_content(layout_item.get("latex", ""))
    else:
        content = _normalize_medium_content(layout_item.get("text", ""))
        if content:
            hybrid_item["content"] = content

    return hybrid_item


def _build_medium_hybrid_model_list(
    images_layout_res: list[list[dict[str, Any]]],
    images_pil_list: list[Image.Image],
) -> list[list[dict[str, Any]]]:
    """按页构造 Hybrid medium 的模型输出，保持最终 middle-json 为 Hybrid 形态。"""
    model_list: list[list[dict[str, Any]]] = []
    for layout_res, image in zip(images_layout_res, images_pil_list):
        page_size = _normalize_page_size(image)
        page_items = []
        for layout_item in layout_res:
            hybrid_item = _medium_layout_item_to_hybrid_item(layout_item, page_size)
            if hybrid_item is not None:
                page_items.append(hybrid_item)
        model_list.append(page_items)
    return model_list


def _run_medium_formula_recognition(
    local_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
    batch_ratio: int,
) -> list[list[dict[str, Any]]]:
    """执行 Hybrid medium 的本地公式识别，并返回供 OCR det 遮罩使用的公式框。"""
    images_mfd_res = []
    for layout_res in images_layout_res:
        page_formula_res = []
        for res in layout_res:
            if res.get("label") in ["display_formula", "inline_formula"]:
                res.setdefault("latex", "")
                page_formula_res.append(res)
        images_mfd_res.append(page_formula_res)

    if any(images_mfd_res):
        images_formula_list = run_mfr_inference(
            local_context.mfr_model.batch_predict,
            images_mfd_res,
            np_images,
            batch_size=batch_ratio * MFR_BASE_BATCH_SIZE,
        )
        for page_formula_res, page_formula_with_latex in zip(images_mfd_res, images_formula_list):
            for formula_res, formula_with_latex in zip(page_formula_res, page_formula_with_latex):
                formula_res["latex"] = formula_with_latex.get("latex", "")

    mfd_res = []
    for layout_res in images_layout_res:
        page_mfd_res = []
        for formula in layout_res:
            if formula.get("label") not in ["display_formula", "inline_formula"]:
                continue
            bbox = _formula_item_to_pixel_bbox(formula)
            if bbox is not None:
                page_mfd_res.append({"bbox": bbox})
        mfd_res.append(page_mfd_res)
    return mfd_res


def _medium_bbox_center(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """计算 bbox 中心点，供表格内联对象匹配所属表格。"""
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def _medium_is_point_in_bbox(point: tuple[float, float], bbox: list[float] | tuple[float, ...]) -> bool:
    """判断点是否落在 bbox 内部，使用闭区间兼容边界贴合的 layout 框。"""
    x, y = point
    return float(bbox[0]) <= x <= float(bbox[2]) and float(bbox[1]) <= y <= float(bbox[3])


def _medium_bbox_intersection(
    bbox1: list[float] | tuple[float, ...],
    bbox2: list[float] | tuple[float, ...],
) -> list[float] | None:
    """计算两个 bbox 的交集；无有效重叠时返回 None。"""
    x0 = max(float(bbox1[0]), float(bbox2[0]))
    y0 = max(float(bbox1[1]), float(bbox2[1]))
    x1 = min(float(bbox1[2]), float(bbox2[2]))
    y1 = min(float(bbox1[3]), float(bbox2[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _medium_bbox_intersection_area(
    bbox1: list[float] | tuple[float, ...],
    bbox2: list[float] | tuple[float, ...],
) -> float:
    """计算 bbox 交集面积，用于多个候选表格时选重叠最大的表格。"""
    overlap_bbox = _medium_bbox_intersection(bbox1, bbox2)
    if overlap_bbox is None:
        return 0.0
    return float(overlap_bbox[2] - overlap_bbox[0]) * float(overlap_bbox[3] - overlap_bbox[1])


def _medium_bbox_to_relative_bbox(
    bbox: list[float] | tuple[float, ...],
    base_bbox: list[float] | tuple[float, ...],
) -> list[float]:
    """将页面坐标 bbox 转成表格裁剪图内的相对坐标。"""
    return [
        float(bbox[0]) - float(base_bbox[0]),
        float(bbox[1]) - float(base_bbox[1]),
        float(bbox[2]) - float(base_bbox[0]),
        float(bbox[3]) - float(base_bbox[1]),
    ]


def _medium_bbox_to_quad(bbox: list[float] | tuple[float, ...]) -> np.ndarray:
    """将普通 bbox 转为表格模型 OCR token 使用的四点框。"""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _encode_medium_table_inline_image(np_img: np.ndarray, bbox: list[float] | tuple[float, ...]) -> str:
    """裁剪并编码表格内联图片，返回供后续 image_cache 外置化的 data URI。"""
    image_h, image_w = np_img.shape[:2]
    image_bbox = normalize_to_int_bbox(bbox, image_size=(image_h, image_w))
    if image_bbox is None:
        return ""

    x0, y0, x1, y1 = image_bbox
    if x1 <= x0 or y1 <= y0:
        return ""

    crop_rgb = np_img[y0:y1, x0:x1]
    if crop_rgb.size == 0:
        return ""

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", crop_bgr)
    if not success:
        return ""

    b64_str = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def _get_medium_virtual_image_bbox(bbox: list[float] | tuple[float, ...], box_size: float = 10.0) -> list[float]:
    """为表格内图片构造虚拟 OCR token 小框，避免图片占据整块区域干扰表格结构。"""
    center_x, center_y = _medium_bbox_center(bbox)
    half_size = box_size / 2.0
    return [
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size,
    ]


def _medium_table_supports_inline_objects(table_item: dict[str, Any]) -> bool:
    """判断当前表格方向是否支持直接注入内联对象；旋转表格仅清理不注入。"""
    return str(table_item.get("rotate_label", "0")) == "0"


def _normalize_medium_table_ocr_rec_text(text: Any) -> Any:
    """规范化表格 OCR rec 的已知误识别，保持与 dev pipeline 表格输入一致。"""
    if not isinstance(text, str):
        return text
    if text in TABLE_OCR_REC_SINGLE_CHAR_REPLACEMENTS:
        return TABLE_OCR_REC_SINGLE_CHAR_REPLACEMENTS[text]
    for pattern, replacement in TABLE_OCR_REC_REGEX_REPLACEMENTS:
        match = pattern.fullmatch(text)
        if match:
            return match.expand(replacement)
    return text


def _sort_medium_table_ocr_result(ocr_result: list[list[Any]]) -> None:
    """按表格模型期望的从上到下、同行从左到右顺序整理 OCR token。"""
    if not ocr_result:
        return

    sorted_result = sorted(
        ocr_result,
        key=lambda item: (float(np.asarray(item[0])[0][1]), float(np.asarray(item[0])[0][0])),
    )

    for i in range(len(sorted_result) - 1):
        for j in range(i, -1, -1):
            cur_box = np.asarray(sorted_result[j][0], dtype=np.float32)
            next_box = np.asarray(sorted_result[j + 1][0], dtype=np.float32)
            if (
                abs(float(next_box[0][1]) - float(cur_box[0][1])) < 10
                and float(next_box[0][0]) < float(cur_box[0][0])
            ):
                sorted_result[j], sorted_result[j + 1] = sorted_result[j + 1], sorted_result[j]
            else:
                break

    ocr_result[:] = sorted_result


def _extract_medium_table_inline_objects(
    layout_res: list[dict[str, Any]],
    np_img: np.ndarray,
    *,
    formula_enable: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    """从同页 layout 结果中收集落在表格内部的图片和公式候选对象。"""
    image_h, image_w = np_img.shape[:2]
    image_size = (image_h, image_w)

    tables: list[tuple[dict[str, Any], list[int]]] = []
    for res in layout_res:
        if res.get("label") != "table":
            continue
        table_bbox = normalize_to_int_bbox(res.get("bbox"), image_size=image_size)
        if table_bbox is not None:
            tables.append((res, table_bbox))

    if not tables:
        return {}

    table_inline_objects: dict[int, list[dict[str, Any]]] = {id(table_res): [] for table_res, _ in tables}
    candidate_labels = {"image"}
    if formula_enable:
        candidate_labels.update({"inline_formula", "display_formula"})

    for layout_item in layout_res:
        label = layout_item.get("label")
        if label not in candidate_labels:
            continue

        item_bbox = normalize_to_int_bbox(layout_item.get("bbox"), image_size=image_size)
        if item_bbox is None:
            continue

        item_center = _medium_bbox_center(item_bbox)
        matched_tables = []
        for table_res, table_bbox in tables:
            if not _medium_is_point_in_bbox(item_center, table_bbox):
                continue
            overlap_area = _medium_bbox_intersection_area(item_bbox, table_bbox)
            matched_tables.append((overlap_area, table_res, table_bbox))

        if not matched_tables:
            continue

        matched_tables.sort(key=lambda item: item[0], reverse=True)
        _, table_res, table_bbox = matched_tables[0]
        overlap_bbox = _medium_bbox_intersection(item_bbox, table_bbox)
        if overlap_bbox is None:
            continue

        rel_overlap_bbox = _medium_bbox_to_relative_bbox(overlap_bbox, table_bbox)
        score = float(layout_item.get("score", 1.0) or 0.0)
        if label == "image":
            image_src = _encode_medium_table_inline_image(np_img, item_bbox)
            if not image_src:
                continue
            content = f'<img src="{image_src}"/>'
            token_bbox = _get_medium_virtual_image_bbox(rel_overlap_bbox)
            kind = "image"
        else:
            latex = _normalize_medium_content(layout_item.get("latex", ""))
            if not latex:
                continue
            content = f"<eq>{html.escape(latex)}</eq>"
            token_bbox = rel_overlap_bbox
            kind = "formula"

        table_inline_objects[id(table_res)].append(
            {
                "kind": kind,
                "source_layout_id": id(layout_item),
                "page_bbox": item_bbox,
                "table_rel_mask_bbox": rel_overlap_bbox,
                "table_token_bbox": token_bbox,
                "content": content,
                "score": score,
            }
        )

    return table_inline_objects


def _apply_medium_table_rotate_label(table_res_dict: dict[str, Any], rotate_label: str) -> None:
    """写回表格方向预测结果，并同步旋转无线和有线表格裁剪图。"""
    rotate_label = str(rotate_label or "0")
    table_res_dict["rotate_label"] = rotate_label
    if rotate_label == "270":
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotate_label == "90":
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return
    table_res_dict["table_img"] = cv2.rotate(np.asarray(table_res_dict["table_img"]), rotate_code)
    table_res_dict["wired_table_img"] = cv2.rotate(np.asarray(table_res_dict["wired_table_img"]), rotate_code)


def _collect_medium_table_items(
    images_layout_res: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
    lang_list: list[str | None],
) -> list[dict[str, Any]]:
    """收集 Hybrid medium 单文件窗口中的表格裁剪项，不跨文件聚合。"""
    table_items = []
    for layout_res, np_img, lang in zip(images_layout_res, np_images, lang_list):
        table_inline_objects = _extract_medium_table_inline_objects(
            layout_res,
            np_img,
            formula_enable=True,
        )
        for table_res in layout_res:
            if table_res.get("label") != "table":
                continue

            def get_crop_table_img(scale: float) -> np.ndarray:
                """按指定缩放裁剪表格图，保持 medium 表格处理只使用当前文件窗口图像。"""
                bbox = normalize_to_int_bbox([float(v) / float(scale) for v in table_res["bbox"]])
                if bbox is None:
                    return np_img[0:0, 0:0]
                return get_crop_np_img(bbox, np_img, scale=scale)

            table_img = get_crop_table_img(scale=1)
            wired_table_img = get_crop_table_img(scale=10 / 3)
            if table_img.size == 0 or wired_table_img.size == 0:
                continue
            table_items.append(
                {
                    "table_res": table_res,
                    "layout_res": layout_res,
                    "lang": lang,
                    "table_img": table_img,
                    "wired_table_img": wired_table_img,
                    "table_inline_objects": table_inline_objects.get(id(table_res), []),
                }
            )
    return table_items


def _collect_medium_table_ocr_rec_inputs(
    atom_model_manager: Any,
    table_items: list[dict[str, Any]],
    batch_ratio: int,
) -> dict[str | None, list[dict[str, Any]]]:
    """对表格裁剪图执行 OCR det，并按语言收集 OCR rec 输入。"""
    rec_img_lang_group: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    det_ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        enable_merge_det_boxes=False,
    )
    table_det_items = _build_medium_table_ocr_det_items(table_items)
    det_images = [item["det_image"] for item in table_det_items]
    if not det_images:
        return rec_img_lang_group

    det_batch_size = max(1, min(len(det_images), batch_ratio * OCR_DET_BASE_BATCH_SIZE))
    batch_results = run_ocr_inference(
        det_ocr_engine.text_detector.batch_predict,
        det_images,
        det_batch_size,
        tqdm_enable=True,
        tqdm_desc="Hybrid-medium table OCR-det",
    )
    if len(batch_results) != len(table_det_items):
        raise ValueError("Hybrid medium table OCR det batch result count mismatch")

    for table_det_item, (dt_boxes, _) in zip(table_det_items, batch_results):
        _append_medium_table_ocr_det_result(
            table_det_item,
            dt_boxes,
            rec_img_lang_group,
        )
    return rec_img_lang_group


def _build_medium_table_ocr_det_items(table_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造表格 OCR-det 输入项，保留原图、遮罩图和公式修正框。"""
    table_det_items = []
    for table_id, table_item in enumerate(table_items):
        bgr_image = cv2.cvtColor(table_item["table_img"], cv2.COLOR_RGB2BGR)
        inline_objects = (
            table_item.get("table_inline_objects", [])
            if _medium_table_supports_inline_objects(table_item)
            else []
        )
        inline_mask_boxes = [{"bbox": inline_object["table_rel_mask_bbox"]} for inline_object in inline_objects]
        formula_mask_boxes = [
            {"bbox": inline_object["table_rel_mask_bbox"]}
            for inline_object in inline_objects
            if inline_object["kind"] == "formula"
        ]
        det_image = mask_formula_regions_for_ocr_det(bgr_image.copy(), inline_mask_boxes) if inline_mask_boxes else bgr_image
        table_det_items.append(
            {
                "bgr_image": bgr_image,
                "det_image": det_image,
                "formula_mask_boxes": formula_mask_boxes,
                "lang": table_item["lang"],
                "table_id": table_id,
            }
        )
    return table_det_items


def _append_medium_table_ocr_det_result(
    table_det_item: dict[str, Any],
    dt_boxes: Any,
    rec_img_lang_group: dict[str | None, list[dict[str, Any]]],
) -> None:
    """整理单表 OCR-det 结果，跳过公式框并按语言归集 OCR-rec 输入。"""
    if dt_boxes is None or len(dt_boxes) == 0:
        return

    ocr_result = dt_boxes
    formula_mask_boxes = table_det_item["formula_mask_boxes"]
    if formula_mask_boxes:
        ocr_result = update_det_boxes(ocr_result, formula_mask_boxes)
        if ocr_result is None or len(ocr_result) == 0:
            return

    for dt_box in sorted_boxes(ocr_result):
        dt_box_array = np.asarray(dt_box, dtype=np.float32)
        rec_img_lang_group[table_det_item["lang"]].append(
            {
                "cropped_img": get_rotate_crop_image_for_text_rec(
                    table_det_item["bgr_image"],
                    dt_box_array.copy(),
                ),
                "dt_box": dt_box_array.copy(),
                "table_id": table_det_item["table_id"],
            }
        )


def _apply_medium_table_orientation(atom_model_manager: Any, table_items: list[dict[str, Any]], batch_ratio: int) -> None:
    """执行 medium 表格方向分类，失败时保留原始裁剪图继续后续识别。"""
    table_orientation_cls_model = atom_model_manager.get_atom_model(atom_model_name=AtomicModel.TableOrientationCls)
    try:
        rotate_labels = table_orientation_cls_model.batch_predict(
            table_items,
            det_batch_size=batch_ratio * OCR_DET_BASE_BATCH_SIZE,
        )
        if len(rotate_labels) != len(table_items):
            raise ValueError("Hybrid medium table orientation result count mismatch")
        for table_item, rotate_label in zip(table_items, rotate_labels):
            _apply_medium_table_rotate_label(table_item, rotate_label)
    except Exception as exc:
        logger.warning(f"Hybrid medium table orientation classification failed: {exc}, using original image")


def _apply_medium_table_classification(atom_model_manager: Any, table_items: list[dict[str, Any]]) -> None:
    """执行 medium 表格有线/无线分类，失败时走默认模型选择。"""
    table_cls_model = atom_model_manager.get_atom_model(atom_model_name=AtomicModel.TableCls)
    try:
        table_cls_model.batch_predict(table_items)
    except Exception as exc:
        logger.warning(f"Hybrid medium table classification failed: {exc}, using default model")


def _apply_medium_table_ocr_rec(
    atom_model_manager: Any,
    table_items: list[dict[str, Any]],
    batch_ratio: int,
) -> None:
    """执行 medium 表格 OCR det/rec，并把识别结果按 table_id 写回 table_items。"""
    rec_img_lang_group = _collect_medium_table_ocr_rec_inputs(atom_model_manager, table_items, batch_ratio)
    for lang, rec_img_list in rec_img_lang_group.items():
        if not rec_img_list:
            continue
        ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            lang=lang,
            enable_merge_det_boxes=False,
        )
        ocr_res_list = run_ocr_inference(
            ocr_engine.ocr,
            [item["cropped_img"] for item in rec_img_list],
            det=False,
            tqdm_enable=True,
            tqdm_desc=f"Hybrid-medium table OCR-rec {lang}",
        )[0]
        for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
            table_item = table_items[img_dict["table_id"]]
            ocr_text = _normalize_medium_table_ocr_rec_text(ocr_res[0])
            ocr_result_item = [img_dict["dt_box"], html.escape(str(ocr_text)), ocr_res[1]]
            table_item.setdefault("ocr_result", []).append(ocr_result_item)
    for table_item in table_items:
        _sort_medium_table_ocr_result(table_item.get("ocr_result", []))


def _append_medium_table_inline_objects_to_ocr_result(table_items: list[dict[str, Any]]) -> None:
    """删除表格内图片/公式原始 layout 项，非旋转表额外注入虚拟 OCR token。"""
    remove_ids_by_layout: dict[int, tuple[list[dict[str, Any]], set[int]]] = {}
    for table_item in table_items:
        table_inline_objects = table_item.get("table_inline_objects", [])
        if not table_inline_objects:
            continue

        layout_res = table_item.get("layout_res")
        consumed_layout_ids = None
        if layout_res is not None:
            consumed_layout_ids = remove_ids_by_layout.setdefault(
                id(layout_res),
                (layout_res, set()),
            )[1]

        supports_inline_injection = _medium_table_supports_inline_objects(table_item)
        table_ocr_result = table_item.setdefault("ocr_result", []) if supports_inline_injection else None
        for inline_object in table_inline_objects:
            if consumed_layout_ids is not None:
                consumed_layout_ids.add(inline_object["source_layout_id"])
            if table_ocr_result is not None:
                table_ocr_result.append(
                    [
                        _medium_bbox_to_quad(inline_object["table_token_bbox"]),
                        inline_object["content"],
                        inline_object["score"],
                    ]
                )

        if table_ocr_result is not None:
            _sort_medium_table_ocr_result(table_ocr_result)

    for layout_res, remove_ids in remove_ids_by_layout.values():
        layout_res[:] = [item for item in layout_res if id(item) not in remove_ids]


def _apply_medium_wireless_table_recognition(atom_model_manager: Any, table_items: list[dict[str, Any]]) -> None:
    """先执行无线表格识别，为后续有线表格回退提供初始 html。"""
    wireless_table_model = atom_model_manager.get_atom_model(atom_model_name=AtomicModel.WirelessTable)
    wireless_table_model.batch_predict(table_items)


def _select_medium_wired_table_items(table_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """根据表格分类结果挑选需要有线模型二次识别的表格项。"""
    wired_table_items = []
    for table_item in table_items:
        cls_label = table_item["table_res"].get("cls_label")
        cls_score = table_item["table_res"].get("cls_score", 0.0)
        if (cls_label == AtomicModel.WirelessTable and cls_score < 0.9) or cls_label == AtomicModel.WiredTable:
            wired_table_items.append(table_item)
        table_item["table_res"].pop("cls_label", None)
        table_item["table_res"].pop("cls_score", None)
    return wired_table_items


def _apply_medium_wired_table_recognition(atom_model_manager: Any, wired_table_items: list[dict[str, Any]]) -> None:
    """对筛选出的有线表格执行二次识别，并覆盖对应 table html。"""
    for table_item in tqdm(wired_table_items, desc="Hybrid-medium table-wired predict"):
        if not table_item.get("ocr_result"):
            continue
        wired_table_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.WiredTable,
            lang=table_item["lang"],
        )
        table_item["table_res"]["html"] = wired_table_model.predict(
            table_item["wired_table_img"],
            table_item["ocr_result"],
            table_item["table_res"].get("html", None),
        )


def _trim_medium_table_html(table_items: list[dict[str, Any]]) -> None:
    """保留表格识别结果中的核心 table 片段，去掉模型可能返回的外围文本。"""
    for table_item in table_items:
        html_code = table_item["table_res"].get("html", "") or ""
        if "<table>" in html_code and "</table>" in html_code:
            start_index = html_code.find("<table>")
            end_index = html_code.rfind("</table>") + len("</table>")
            table_item["table_res"]["html"] = html_code[start_index:end_index]


def _apply_medium_table_recognition(
    local_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
    lang_list: list[str | None],
    batch_ratio: int,
) -> None:
    """执行 Hybrid medium 的本地表格识别，结果写回 layout table 项的 html 字段。"""
    table_items = _collect_medium_table_items(images_layout_res, np_images, lang_list)
    if not table_items:
        return

    atom_model_manager = local_context.atom_model_manager
    _apply_medium_table_orientation(atom_model_manager, table_items, batch_ratio)
    _apply_medium_table_classification(atom_model_manager, table_items)
    _apply_medium_table_ocr_rec(atom_model_manager, table_items, batch_ratio)
    _append_medium_table_inline_objects_to_ocr_result(table_items)
    _apply_medium_wireless_table_recognition(atom_model_manager, table_items)
    wired_table_items = _select_medium_wired_table_items(table_items)
    _apply_medium_wired_table_recognition(atom_model_manager, wired_table_items)
    _trim_medium_table_html(table_items)


def _apply_medium_seal_ocr(
    local_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> None:
    """执行 Hybrid medium 的印章 OCR，并将识别文本写回 seal layout 项。"""
    seal_ocr_model = None
    for layout_res, np_img in zip(images_layout_res, np_images):
        for layout_item in layout_res:
            if layout_item.get("label") != "seal":
                continue
            layout_item["text"] = ""
            seal_bbox = normalize_to_int_bbox(layout_item.get("bbox"), image_size=np_img.shape[:2])
            if seal_bbox is None:
                continue
            x0, y0, x1, y1 = seal_bbox
            seal_crop_rgb = np_img[y0:y1, x0:x1]
            if seal_crop_rgb.size == 0:
                continue
            if seal_ocr_model is None:
                seal_ocr_model = local_context.atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    lang="seal",
                )
            seal_crop_bgr = cv2.cvtColor(seal_crop_rgb, cv2.COLOR_RGB2BGR)
            seal_ocr_res = run_ocr_inference(seal_ocr_model.ocr, seal_crop_bgr, det=True, rec=True)[0]
            seal_texts = []
            for seal_item in seal_ocr_res or []:
                if not seal_item or len(seal_item) != 2:
                    continue
                rec_result = seal_item[1]
                if rec_result and rec_result[0]:
                    seal_texts.append(rec_result[0])
            layout_item["text"] = seal_texts


def _prune_medium_empty_ocr_text_blocks(images_layout_res: list[list[dict[str, Any]]], ocr_enable: bool) -> None:
    """清理低置信 OCR 后留下的空 ocr_text，避免空块进入 Hybrid MagicModel。"""
    if not ocr_enable:
        return
    for layout_res in images_layout_res:
        layout_res[:] = [
            item
            for item in layout_res
            if item.get("label") != "ocr_text" or bool(_normalize_medium_content(item.get("text", "")))
        ]


def _extract_with_local_layout(
    images_pil_list: list[Image.Image],
    language: str | None,
    _ocr_enable: bool,
    batch_ratio: int,
) -> tuple[list[list[dict[str, Any]]], HybridLocalModelContext]:
    """Hybrid medium 路径：单文件窗口内执行本地 layout/OCR/table/formula，不跨文件 batch。"""
    local_context_singleton = HybridLocalModelContextSingleton()
    local_context = local_context_singleton.get_model(
        lang=language,
        # medium 本地路径会直接执行公式识别；OCR 扫描件也需要加载 MFR。
        formula_enable=True,
    )
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    images_layout_res = run_layout_inference(
        local_context.layout_model.batch_predict,
        images_pil_list,
        batch_size=min(8, batch_ratio * LAYOUT_BASE_BATCH_SIZE),
    )
    clean_vram(local_context.device, vram_threshold=8)
    mfd_res = _run_medium_formula_recognition(
        local_context,
        images_layout_res,
        np_images,
        batch_ratio,
    )
    clean_vram(local_context.device, vram_threshold=8)
    _apply_medium_table_recognition(
        local_context,
        images_layout_res,
        np_images,
        [language for _ in images_pil_list],
        batch_ratio,
    )
    _apply_medium_seal_ocr(local_context, images_layout_res, np_images)
    _prune_medium_empty_ocr_text_blocks(images_layout_res, _ocr_enable)
    medium_model_list = _build_medium_hybrid_model_list(images_layout_res, images_pil_list)
    ocr_res_list = _ocr_det(
        local_context,
        np_images,
        medium_model_list,
        mfd_res,
        _ocr_enable,
        batch_ratio=batch_ratio,
        candidate_fn=_is_hybrid_medium_ocr_det_candidate,
    )
    if _ocr_enable:
        # medium 路径也需要在合并 sidecar 前完成 OCR rec，否则 ocr_text 只保留空文本裁剪。
        _apply_ocr_rec_results(local_context, ocr_res_list)
    _normalize_bbox([[] for _ in images_pil_list], ocr_res_list, images_pil_list)
    merged_model_list = _merge_page_sidecar_items(
        medium_model_list,
        [[] for _ in images_pil_list],
        ocr_res_list,
    )
    return merged_model_list, local_context


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
    local_context: HybridLocalModelContext,
    ocr_res_list: list[list[dict[str, Any]]],
) -> None:
    """执行 OCR rec 并把文本写回 sidecar，结果数量异常时显式报错。"""
    need_ocr_list, img_crop_list = _collect_ocr_rec_inputs(ocr_res_list)
    if not img_crop_list:
        return

    ocr_result_list = run_ocr_inference(
        local_context.ocr_model.ocr,
        img_crop_list,
        det=False,
        tqdm_enable=True,
    )[0]

    if len(ocr_result_list) != len(need_ocr_list):
        raise ValueError(
            "Hybrid OCR rec result count mismatch: "
            f"ocr_result_list={len(ocr_result_list)}, need_ocr_list={len(need_ocr_list)}"
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


def _process_ocr_and_formulas(
    images_pil_list: list[Image.Image],
    model_list: list[list[dict[str, Any]]],
    language: str | None,
    _ocr_enable: bool,
    batch_ratio: int = 1,
    *,
    local_context: HybridLocalModelContext | None = None,
    images_layout_res: list[list[dict[str, Any]]] | None = None,
) -> tuple[list[list[dict[str, Any]]], HybridLocalModelContext]:
    """处理OCR和公式识别"""

    # 遍历model_list,对文本块截图交由OCR识别
    # 根据_ocr_enable决定ocr只开det还是det+rec

    # 将PIL图片转换为numpy数组
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]

    if local_context is None:
        # 允许 high 路径复用已经初始化的本地上下文，避免同一窗口重复取模型。
        local_context_singleton = HybridLocalModelContextSingleton()
        local_context = local_context_singleton.get_model(
            lang=language,
            formula_enable=True,
        )

    if images_layout_res is None:
        # 没有外部复用的 layout 时，按旧逻辑为公式和标题拆分补跑一次本地 layout。
        layout_images = _mask_image_regions(np_images, model_list)
        images_layout_res = _predict_layout_for_title_split(local_context, layout_images, batch_ratio)

    images_mfd_res = _build_inline_formula_inputs(images_layout_res)
    # 公式识别
    inline_formula_list = run_mfr_inference(
        local_context.mfr_model.batch_predict,
        images_mfd_res,
        np_images,
        batch_size=batch_ratio * MFR_BASE_BATCH_SIZE,
        interline_enable=True,
    )

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
    ocr_res_list = _ocr_det(
        local_context,
        np_images,
        model_list,
        mfd_res,
        _ocr_enable,
        batch_ratio=batch_ratio,
    )

    # 如果需要ocr则做ocr_rec
    if _ocr_enable:
        _apply_ocr_rec_results(local_context, ocr_res_list)

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
    return merged_model_list, local_context


def _extract_high_with_local_layout(
    predictor: Any,
    images_pil_list: list[Image.Image],
    language: str | None,
    _ocr_enable: bool,
    batch_ratio: int,
) -> tuple[list[list[dict[str, Any]]], HybridLocalModelContext]:
    """Hybrid high 路径：用本地 layout 小模型约束 VLM 抽取，再补 OCR/formula sidecar。"""
    local_context_singleton = HybridLocalModelContextSingleton()
    local_context = local_context_singleton.get_model(
        lang=language,
        # VLM 文本路径只需要本地 layout 与 OCR det，不需要加载 MFR。
        formula_enable=not _ocr_enable,
    )
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    images_layout_res = _predict_layout_for_title_split(local_context, np_images, batch_ratio)
    layout_blocks = _build_high_layout_blocks(images_layout_res, images_pil_list)
    window_model_list = predictor.batch_extract_with_layout(
        images=images_pil_list,
        blocks_list=layout_blocks,
        not_extract_list=None if _ocr_enable else list(NOT_EXTRACT_TYPES),
        image_analysis=False,
    )
    if _ocr_enable:
        local_context = _apply_vlm_text_det_sidecars_for_window(
            images_pil_list,
            window_model_list,
            language,
            batch_ratio,
            images_layout_res=images_layout_res,
            local_context=local_context,
        )
    else:
        window_model_list, local_context = _process_ocr_and_formulas(
            images_pil_list,
            window_model_list,
            language,
            False,
            batch_ratio=batch_ratio,
            local_context=local_context,
            images_layout_res=images_layout_res,
        )
    return window_model_list, local_context


async def _aio_extract_high_with_local_layout(
    predictor: Any,
    images_pil_list: list[Image.Image],
    language: str | None,
    _ocr_enable: bool,
    batch_ratio: int,
) -> tuple[list[list[dict[str, Any]]], HybridLocalModelContext]:
    """Hybrid high 异步路径：异步调用 VLM 的本地 layout 约束抽取，其余本地步骤放线程执行。"""
    local_context_singleton = HybridLocalModelContextSingleton()
    local_context = local_context_singleton.get_model(
        lang=language,
        formula_enable=not _ocr_enable,
    )
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    images_layout_res = await asyncio.to_thread(
        _predict_layout_for_title_split,
        local_context,
        np_images,
        batch_ratio,
    )
    layout_blocks = _build_high_layout_blocks(images_layout_res, images_pil_list)
    window_model_list = await predictor.aio_batch_extract_with_layout(
        images=images_pil_list,
        blocks_list=layout_blocks,
        not_extract_list=None if _ocr_enable else list(NOT_EXTRACT_TYPES),
        image_analysis=False,
    )
    if _ocr_enable:
        local_context = await asyncio.to_thread(
            _apply_vlm_text_det_sidecars_for_window,
            images_pil_list,
            window_model_list,
            language,
            batch_ratio,
            images_layout_res=images_layout_res,
            local_context=local_context,
        )
    else:
        window_model_list, local_context = await asyncio.to_thread(
            _process_ocr_and_formulas,
            images_pil_list,
            window_model_list,
            language,
            False,
            batch_ratio,
            local_context=local_context,
            images_layout_res=images_layout_res,
        )
    return window_model_list, local_context


def _apply_vlm_text_det_sidecars_for_window(
    images_pil_list: list[Image.Image],
    model_list: list[list[dict[str, Any]]],
    language: str | None,
    batch_ratio: int,
    *,
    images_layout_res: list[list[dict[str, Any]]] | None = None,
    local_context: HybridLocalModelContext | None = None,
) -> HybridLocalModelContext:
    """为使用 VLM 文本内容的路径追加小模型 OCR det 空文本行提示。"""
    if local_context is None:
        local_context_singleton = HybridLocalModelContextSingleton()
        local_context = local_context_singleton.get_model(
            lang=language,
            formula_enable=False,
        )
    if images_layout_res is None:
        images_layout_res = _predict_layout_for_title_split(
            local_context,
            images_pil_list,
            batch_ratio,
        )
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    inline_formula_list = _build_inline_formula_det_inputs(images_layout_res)
    ocr_res_list = _ocr_det(
        local_context,
        np_images,
        model_list,
        _build_formula_mask_inputs(images_layout_res),
        False,
        batch_ratio=batch_ratio,
        fill_text=False,
        candidate_fn=_is_hybrid_vlm_text_ocr_det_candidate,
    )
    _normalize_bbox(inline_formula_list, ocr_res_list, images_pil_list)
    model_list[:] = _merge_page_sidecar_items(
        model_list,
        inline_formula_list,
        ocr_res_list,
        keep_ocr_text=False,
    )
    _apply_layout_title_split(
        model_list,
        images_layout_res,
        [_normalize_page_size(image) for image in images_pil_list],
    )
    return local_context


def _apply_layout_title_split_for_window(
    images_pil_list: list[Image.Image],
    model_list: list[list[dict[str, Any]]],
    language: str | None,
    batch_ratio: int,
) -> HybridLocalModelContext:
    """兼容旧内部入口，实际委托 VLM 文本路径的 det sidecar 与标题拆分流程。"""
    return _apply_vlm_text_det_sidecars_for_window(
        images_pil_list,
        model_list,
        language,
        batch_ratio,
    )


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
    merged_model_list: list[list[dict[str, Any]]] = []
    for page_model_list, page_inline_formula_list, page_ocr_res_list in zip(model_list, inline_formula_list, ocr_res_list):
        merged_page_model_list: list[dict[str, Any]] = list(page_model_list)
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
    predictor: Any | None = None,
    backend: Literal[
        "http-client",
        "transformers",
        "mlx-engine",
        "lmdeploy-engine",
        "vllm-engine",
        "vllm-async-engine",
    ] = "transformers",
    parse_method: str = "auto",
    language: str = "ch",
    model_path: str | None = None,
    server_url: str | None = None,
    effort: Literal["medium", "high", "extra_high"] = DEFAULT_HYBRID_EFFORT,
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    image_cache: ImagePayloadCache | None = None,
    **kwargs: object,
) -> tuple[list[PageInfo], list[list[dict[str, Any]]], bool]:
    client_side_output_generation = bool(kwargs.pop("client_side_output_generation", False))
    _discard_legacy_formula_table_kwargs(kwargs)
    effort = validate_effort(effort)
    image_analysis = image_analysis if effort == MAX_HYBRID_EFFORT else False
    if effort == LOCAL_HYBRID_EFFORT:
        predictor = None
    else:
        vlm_runtime = _load_vlm_runtime()
        if predictor is None:
            predictor = vlm_runtime["ModelSingleton"]().get_model(backend, model_path, server_url, **kwargs)
        predictor = vlm_runtime["_maybe_enable_serial_execution"](predictor, backend)

    device = get_device()

    pdf_doc = PDFDocument(pdf_bytes)
    _ocr_enable = ocr_classify(pdf_doc, parse_method=parse_method)
    use_vlm_text_content = effort in {LAYOUT_HYBRID_EFFORT, MAX_HYBRID_EFFORT} and _ocr_enable

    middle_json: list[PageInfo] = []
    model_list: list[list[dict[str, Any]]] = []
    doc_closed = False
    local_context = None
    try:
        page_count = pdf_doc.page_count
        configured_window_size = get_processing_window_size(default=64)
        windows = _build_processing_windows(page_count, configured_window_size)
        _log_processing_window_plan(page_count, configured_window_size, len(windows))

        batch_ratio = 1 if use_vlm_text_content else get_batch_ratio(device)

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window in windows:
                images_list = load_images_from_pdf_bytes_range(
                    pdf_bytes=pdf_bytes,
                    start_page_id=window.start,
                    end_page_id=window.end,
                    image_type=ImageType.PIL,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    _log_processing_window(window, page_count, len(images_pil_list))
                    if effort == LOCAL_HYBRID_EFFORT:
                        window_model_list, local_context = _extract_with_local_layout(
                            images_pil_list,
                            language,
                            _ocr_enable,
                            batch_ratio,
                        )
                    elif effort == LAYOUT_HYBRID_EFFORT:
                        with vlm_runtime["predictor_execution_guard"](predictor):
                            window_model_list, local_context = _extract_high_with_local_layout(
                                predictor,
                                images_pil_list,
                                language,
                                _ocr_enable,
                                batch_ratio,
                            )
                    elif effort == MAX_HYBRID_EFFORT:
                        if _ocr_enable:
                            with vlm_runtime["predictor_execution_guard"](predictor):
                                window_model_list = predictor.batch_two_step_extract(
                                    images=images_pil_list,
                                    image_analysis=image_analysis,
                                )
                            local_context = _apply_vlm_text_det_sidecars_for_window(
                                images_pil_list,
                                window_model_list,
                                language,
                                batch_ratio,
                            )
                        else:
                            with vlm_runtime["predictor_execution_guard"](predictor):
                                window_model_list = predictor.batch_two_step_extract(
                                    images=images_pil_list,
                                    not_extract_list=list(NOT_EXTRACT_TYPES),
                                    image_analysis=image_analysis,
                                )
                            window_model_list, local_context = _process_ocr_and_formulas(
                                images_pil_list,
                                window_model_list,
                                language,
                                False,
                                batch_ratio=batch_ratio,
                            )
                    else:
                        raise ValueError(f"Unsupported hybrid effort: {effort}")

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
                        page_cvt_fn=blocks_to_page_info,
                        page_start_index=window.start,
                        page_index_map=page_index_map,
                        _ocr_enable=_ocr_enable,
                        use_vlm_text_content=use_vlm_text_content,
                        progress_bar=progress_bar,
                        image_cache=image_cache,
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

        _finalize_hybrid_middle_json(
            middle_json,
            local_context,
            _ocr_enable,
            use_vlm_text_content,
            effort=effort,
            client_side_output_generation=client_side_output_generation,
        )
        pdf_doc.close()
        doc_closed = True
        clean_memory(device)
        return middle_json, model_list, use_vlm_text_content
    finally:
        if not doc_closed:
            pdf_doc.close()


async def aio_doc_analyze(
    pdf_bytes: bytes,
    predictor: Any | None = None,
    backend: Literal[
        "http-client",
        "transformers",
        "mlx-engine",
        "lmdeploy-engine",
        "vllm-engine",
        "vllm-async-engine",
    ] = "transformers",
    parse_method: str = "auto",
    language: str = "ch",
    model_path: str | None = None,
    server_url: str | None = None,
    effort: Literal["medium", "high", "extra_high"] = DEFAULT_HYBRID_EFFORT,
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    image_cache: ImagePayloadCache | None = None,
    **kwargs: object,
) -> tuple[list[PageInfo], list[list[dict[str, Any]]], bool]:
    client_side_output_generation = bool(kwargs.pop("client_side_output_generation", False))
    _discard_legacy_formula_table_kwargs(kwargs)
    effort = validate_effort(effort)
    image_analysis = image_analysis if effort == MAX_HYBRID_EFFORT else False
    if effort == LOCAL_HYBRID_EFFORT:
        predictor = None
    else:
        vlm_runtime = _load_vlm_runtime()
        if predictor is None:
            predictor = await vlm_runtime["_get_model_async"](backend, model_path, server_url, **kwargs)
        predictor = vlm_runtime["_maybe_enable_serial_execution"](predictor, backend)

    device = get_device()

    pdf_doc = PDFDocument(pdf_bytes)
    _ocr_enable = ocr_classify(pdf_doc, parse_method=parse_method)
    use_vlm_text_content = effort in {LAYOUT_HYBRID_EFFORT, MAX_HYBRID_EFFORT} and _ocr_enable

    middle_json: list[PageInfo] = []
    model_list = []
    doc_closed = False
    local_context = None
    try:
        page_count = pdf_doc.page_count
        configured_window_size = get_processing_window_size(default=64)
        windows = _build_processing_windows(page_count, configured_window_size)
        _log_processing_window_plan(page_count, configured_window_size, len(windows))

        batch_ratio = 1 if use_vlm_text_content else get_batch_ratio(device)

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window in windows:
                images_list = await aio_load_images_from_pdf_bytes_range(
                    pdf_bytes,
                    start_page_id=window.start,
                    end_page_id=window.end,
                    image_type=ImageType.PIL,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    _log_processing_window(window, page_count, len(images_pil_list))
                    if effort == LOCAL_HYBRID_EFFORT:
                        window_model_list, local_context = await asyncio.to_thread(
                            _extract_with_local_layout,
                            images_pil_list,
                            language,
                            _ocr_enable,
                            batch_ratio,
                        )
                    elif effort == LAYOUT_HYBRID_EFFORT:
                        async with vlm_runtime["aio_predictor_execution_guard"](predictor):
                            window_model_list, local_context = await _aio_extract_high_with_local_layout(
                                predictor,
                                images_pil_list,
                                language,
                                _ocr_enable,
                                batch_ratio,
                            )
                    elif effort == MAX_HYBRID_EFFORT:
                        if _ocr_enable:
                            async with vlm_runtime["aio_predictor_execution_guard"](predictor):
                                window_model_list = await predictor.aio_batch_two_step_extract(
                                    images=images_pil_list,
                                    image_analysis=image_analysis,
                                )
                            local_context = await asyncio.to_thread(
                                _apply_vlm_text_det_sidecars_for_window,
                                images_pil_list,
                                window_model_list,
                                language,
                                batch_ratio,
                            )
                        else:
                            async with vlm_runtime["aio_predictor_execution_guard"](predictor):
                                window_model_list = await predictor.aio_batch_two_step_extract(
                                    images=images_pil_list,
                                    not_extract_list=list(NOT_EXTRACT_TYPES),
                                    image_analysis=image_analysis,
                                )
                            window_model_list, local_context = await asyncio.to_thread(
                                _process_ocr_and_formulas,
                                images_pil_list,
                                window_model_list,
                                language,
                                False,
                                batch_ratio=batch_ratio,
                            )
                    else:
                        raise ValueError(f"Unsupported hybrid effort: {effort}")

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
                        page_cvt_fn=blocks_to_page_info,
                        page_start_index=window.start,
                        page_index_map=page_index_map,
                        _ocr_enable=_ocr_enable,
                        use_vlm_text_content=use_vlm_text_content,
                        progress_bar=progress_bar,
                        image_cache=image_cache,
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

        await asyncio.to_thread(
            _finalize_hybrid_middle_json,
            middle_json,
            local_context,
            _ocr_enable,
            use_vlm_text_content,
            effort=effort,
            client_side_output_generation=client_side_output_generation,
        )
        pdf_doc.close()
        doc_closed = True
        clean_memory(device)
        return middle_json, model_list, use_vlm_text_content
    finally:
        if not doc_closed:
            pdf_doc.close()
