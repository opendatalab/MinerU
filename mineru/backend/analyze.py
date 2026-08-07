# Copyright (c) Opendatalab. All rights reserved.
import base64
import html
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from mineru.backend.utils.table_text import project_ocr_table_text, project_pdf_table_text
from mineru.backend.local_model_runtime import (
    AtomicModel,
    HybridLocalModelContext,
    HybridLocalModelContextSingleton,
    run_ocr_inference,
)
from mineru.backend.utils.boxbase import (
    calculate_overlap_area_2_minbox_area_ratio,
    calculate_overlap_area_in_bbox1_area_ratio,
)
from mineru.backend.utils.char_utils import resolve_text_line_boundary
from mineru.backend.utils.formula_number import optimize_hybrid_formula_number_blocks
from mineru.backend.utils.span_block_fix import fix_text_block
from mineru.backend.utils.span_pre_proc import (
    SpanBlockMatcher,
    __replace_ligatures,
    __replace_unicode,
    _clear_post_ocr_fallback,
    _is_supported_rotation,
    _restore_post_ocr_fallback,
    txt_spans_extract,
)
from mineru.model.flash import DocxModel, PptxModel, XlsxModel
from mineru.utils.bbox_utils import normalize_to_int_bbox
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.language import detect_lang
from mineru.utils.model_utils import clean_memory, crop_img
from mineru.utils.ocr_utils import (
    OcrConfidence,
    get_adjusted_mfdetrec_res,
    get_ocr_result_list,
    get_rotate_crop_image_for_text_rec,
    mask_formula_regions_for_ocr_det,
    merge_det_boxes,
    rotate_vertical_crop_if_needed,
    sorted_boxes,
    update_det_boxes,
)
from mineru.utils.pdf_image_tools import load_images_from_pdf_bytes_range, get_crop_np_img
from tqdm import tqdm

from ..types import BBox, Block, BlockType, ContentType, Line, NOT_EXTRACT_TYPES, PageInfo, Span
from ..utils.config_reader import get_processing_window_size
from ..utils.pdf_document import PDFDocument, PDFPage, get_lines_from_chars


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 让mps可以fallback

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8
BATCH_RATIO = 2
TABLE_TEXT_LINE_OVERLAP_THRESHOLD = 0.5
TABLE_TEXT_ORIENTATION_MIN_VALID_LINES = 3
TABLE_TEXT_ORIENTATION_MIN_DOMINANCE_RATIO = 0.6
TABLE_TEXT_ORIENTATION_ANGLES = frozenset({0, 90, 180, 270})
_OFFICE_MODEL_MAP = {
    "docx": DocxModel,
    "pptx": PptxModel,
    "xlsx": XlsxModel,
}
_SUPPORTED_FILE_SUFFIXES = {"pdf", *_OFFICE_MODEL_MAP}
TITLE_BLOCK_TYPES = {
    BlockType.TITLE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}
CODE_CONTENT_BLOCK_TYPES = {
    BlockType.CODE,
    BlockType.ALGORITHM,
}
MODEL_JSON_VISUAL_BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.CHART,
    BlockType.TABLE,
    BlockType.EQUATION,
}
_INLINE_FORMULA_PATTERN = re.compile(r"\\\((.*?)\\\)")

VLM_LAYOUT_LABEL_MAP = {
    "abstract": BlockType.TEXT,
    "algorithm": BlockType.CODE,
    "aside_text": BlockType.ASIDE_TEXT,
    "chart": BlockType.CHART,
    "content": BlockType.INDEX,
    "display_formula": BlockType.EQUATION,
    "doc_title": BlockType.DOC_TITLE,
    "figure_title": BlockType.CAPTION,
    "footer": BlockType.FOOTER,
    "footer_image": BlockType.FOOTER,
    "footnote": BlockType.PAGE_FOOTNOTE,
    "formula_number": BlockType.FORMULA_NUMBER,
    "header": BlockType.HEADER,
    "header_image": BlockType.HEADER,
    "image": BlockType.IMAGE,
    "number": BlockType.PAGE_NUMBER,
    "paragraph_title": BlockType.PARAGRAPH_TITLE,
    "reference_content": BlockType.REF_TEXT,
    "seal": BlockType.IMAGE,
    "table": BlockType.TABLE,
    "text": BlockType.TEXT,
    "vertical_text": BlockType.TEXT,
    "vision_footnote": BlockType.FOOTNOTE,
}

PIPELINE_DET_TYPE = {
    BlockType.TEXT,
    BlockType.CODE,
    BlockType.ASIDE_TEXT,
    BlockType.INDEX,
    BlockType.DOC_TITLE,
    BlockType.CAPTION,
    BlockType.FOOTER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.HEADER,
    BlockType.PAGE_NUMBER,
    BlockType.PARAGRAPH_TITLE,
    BlockType.REF_TEXT,
    BlockType.FOOTNOTE,
}
VLM_TXT_DET_TYPE = NOT_EXTRACT_TYPES
VLM_OCR_DET_TYPE = {
    BlockType.TEXT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}


@dataclass
class _OcrDetCrop:
    """保存一次 OCR det 裁剪的中间数据，避免用裸 tuple 传递阶段状态。"""

    bgr_image: Any
    useful_list: list[Any]
    adjusted_mfdetrec_res: list[Any]
    page_ocr_res_list: list[dict[str, Any]]


def _load_vlm_runtime() -> dict[str, Any]:
    """按需加载 VLM runtime 组件，确保只有 high/extra_high 路径触发 VLM 依赖。"""
    from ..model.vlm.runtime import (
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


@dataclass(frozen=True)
class _ProcessingWindow:
    """记录单个 Hybrid 处理窗口的页码范围，统一同步和异步入口的窗口计算。"""

    index: int
    total: int
    start: int
    end: int


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


def _get_window_pdf_pages(pdf_doc: PDFDocument, window: _ProcessingWindow) -> list[PDFPage]:
    """按窗口闭区间获取对应的 PDFPage 对象，供窗口内后续处理复用。"""
    return [pdf_doc[page_idx] for page_idx in range(window.start, window.end + 1)]


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


def _close_images(images_list: list[dict[str, Any]]) -> None:
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


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
        round(max(0.0, min(1.0, float(x0) / page_width)), 3),
        round(max(0.0, min(1.0, float(y0) / page_height)), 3),
        round(max(0.0, min(1.0, float(x1) / page_width)), 3),
        round(max(0.0, min(1.0, float(y1) / page_height)), 3),
    ]
    if unit_bbox[2] <= unit_bbox[0] or unit_bbox[3] <= unit_bbox[1]:
        return None
    return unit_bbox


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


def _apply_table_rotate_labels(
    table_items: list[dict[str, Any]],
    rotate_labels: list[str],
) -> None:
    """按分类输入顺序将表格旋转角写回原始 layout 检测项。"""
    if len(rotate_labels) != len(table_items):
        raise ValueError("Hybrid table orientation result count mismatch")

    for table_item, rotate_label in zip(table_items, rotate_labels):
        table_item["layout_item"]["angle"] = int(rotate_label or "0")


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


def _formula_item_to_pixel_bbox(item: dict[str, Any]) -> list[int] | None:
    bbox = item.get("bbox")
    if bbox is not None and len(bbox) == 4:
        return [int(float(v)) for v in bbox]
    return None


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


def _validate_text_formula_window_inputs(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    images_layout_res: list[list[dict[str, Any]]],
) -> None:
    """校验文本公式处理所需的窗口分页数据，避免 zip 静默截断。"""
    page_counts = {
        "images": len(images_list),
        "pdf_pages": len(pdf_pages),
        "model_list": len(model_list),
        "layout": len(images_layout_res),
    }
    if len(set(page_counts.values())) != 1:
        raise ValueError(f"Hybrid text/formula window page count mismatch: {page_counts}")

    for page_idx, image_dict in enumerate(images_list):
        if image_dict.get("img_pil") is None:
            raise ValueError(f"Hybrid text/formula window image is missing img_pil: page_idx={page_idx}")
        scale = float(image_dict.get("scale", 0) or 0)
        if scale <= 0:
            raise ValueError(f"Hybrid text/formula window image scale must be positive: page_idx={page_idx}")


def _build_pdf_text_line_spans(pdf_page: PDFPage) -> list[Span]:
    """将标准方向的 pdftext line 转为文本 Span，并复用现有字符清洗规则。"""
    page_spans: list[Span] = []
    for pdf_line in get_lines_from_chars(pdf_page.get_chars()):
        try:
            rotation = float(pdf_line.get("rotation", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not _is_supported_rotation(rotation):
            continue

        raw_bbox = pdf_line.get("bbox")
        bbox = getattr(raw_bbox, "bbox", raw_bbox)
        try:
            if bbox is None or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue

        content = "".join(str(pdf_span.get("text", "") or "") for pdf_span in pdf_line.get("spans", []))
        content = __replace_unicode(content)
        content = __replace_ligatures(content).strip()
        if not content:
            continue

        page_spans.append(
            Span(
                type=ContentType.TEXT,
                bbox=(x0, y0, x1, y1),
                content=content,
                score=1.0,
            )
        )
    return page_spans


def _sidecar_bbox_to_page_bbox(
    bbox: BBox | None,
    page_size: tuple[float, float],
    render_scale: float,
) -> BBox | None:
    """将公式或 OCR sidecar bbox 转为 PDF point 坐标，供原生字符匹配和组行。"""
    if bbox is None or len(bbox) != 4 or render_scale <= 0:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return None
    if all(0.0 <= value <= 1.0 for value in [x0, y0, x1, y1]):
        x0, y0, x1, y1 = x0 * page_width, y0 * page_height, x1 * page_width, y1 * page_height
    else:
        x0, y0, x1, y1 = (
            x0 / render_scale,
            y0 / render_scale,
            x1 / render_scale,
            y1 / render_scale,
        )

    left, right = sorted([max(0.0, min(page_width, x0)), max(0.0, min(page_width, x1))])
    top, bottom = sorted([max(0.0, min(page_height, y0)), max(0.0, min(page_height, y1))])
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _detect_table_angle_from_pdf_lines(
    pdf_lines: list[dict[str, Any]],
    table_bbox: BBox,
) -> int | None:
    """统计表格框内标准方向文本行，满足强众数门槛时返回表格角度。"""
    angle_counts: Counter[int] = Counter()
    for pdf_line in pdf_lines:
        try:
            raw_bbox = pdf_line.get("bbox")
            line_bbox = getattr(raw_bbox, "bbox", raw_bbox)
            if line_bbox is None or len(line_bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in line_bbox]
            if x1 <= x0 or y1 <= y0:
                continue

            line_text = "".join(
                str(pdf_span.get("text", "") or "")
                for pdf_span in pdf_line.get("spans", [])
            ).strip()
            if not line_text:
                continue

            rotation = float(pdf_line.get("rotation", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            continue

        if (
            calculate_overlap_area_in_bbox1_area_ratio(
                (x0, y0, x1, y1),
                table_bbox,
            )
            <= TABLE_TEXT_LINE_OVERLAP_THRESHOLD
        ):
            continue
        if not _is_supported_rotation(rotation):
            continue

        angle = int(round(math.degrees(rotation))) % 360
        if angle in TABLE_TEXT_ORIENTATION_ANGLES:
            angle_counts[angle] += 1

    valid_line_count = sum(angle_counts.values())
    if valid_line_count < TABLE_TEXT_ORIENTATION_MIN_VALID_LINES:
        return None

    max_count = max(angle_counts.values())
    dominant_angles = [angle for angle, count in angle_counts.items() if count == max_count]
    if len(dominant_angles) != 1:
        return None
    if max_count / valid_line_count < TABLE_TEXT_ORIENTATION_MIN_DOMINANCE_RATIO:
        return None
    return dominant_angles[0]


def _resolve_txt_table_orientations(
    table_items: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    images_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """优先用原生 PDF 文本行写回表格角度，并返回需要视觉兜底的表格。"""
    fallback_table_items: list[dict[str, Any]] = []
    page_lines_cache: dict[int, list[dict[str, Any]] | None] = {}

    for table_item in table_items:
        page_idx = table_item.get("page_idx")
        if (
            not isinstance(page_idx, int)
            or page_idx < 0
            or page_idx >= len(pdf_pages)
            or page_idx >= len(images_list)
        ):
            fallback_table_items.append(table_item)
            continue

        try:
            pdf_page = pdf_pages[page_idx]
            render_scale = float(images_list[page_idx].get("scale", 0) or 0)
            table_bbox = _sidecar_bbox_to_page_bbox(
                table_item["layout_item"].get("bbox"),
                tuple(float(value) for value in pdf_page.size),
                render_scale,
            )
        except Exception as exc:
            logger.warning(
                "Hybrid txt table orientation falls back to visual model: "
                f"page_idx={page_idx}, error={exc}"
            )
            fallback_table_items.append(table_item)
            continue

        if table_bbox is None:
            fallback_table_items.append(table_item)
            continue

        if page_idx not in page_lines_cache:
            try:
                page_lines_cache[page_idx] = get_lines_from_chars(pdf_page.get_chars())
            except Exception as exc:
                logger.warning(
                    "Hybrid txt table orientation falls back to visual model: "
                    f"page_idx={page_idx}, error={exc}"
                )
                page_lines_cache[page_idx] = None

        pdf_lines = page_lines_cache[page_idx]
        if pdf_lines is None:
            fallback_table_items.append(table_item)
            continue

        table_angle = _detect_table_angle_from_pdf_lines(pdf_lines, table_bbox)
        if table_angle is None:
            fallback_table_items.append(table_item)
            continue
        table_item["layout_item"]["angle"] = table_angle

    return fallback_table_items


def _apply_table_orientations(
    table_items: list[dict[str, Any]],
    parse_mode: Literal["txt", "ocr"],
    pdf_pages: list[PDFPage],
    images_list: list[dict[str, Any]],
    hybrid_model: HybridLocalModelContext,
) -> None:
    """按解析模式写回表格角度，文本证据不足时批量调用视觉方向模型。"""
    if parse_mode == "txt":
        fallback_table_items = _resolve_txt_table_orientations(
            table_items,
            pdf_pages,
            images_list,
        )
    elif parse_mode == "ocr":
        fallback_table_items = table_items
    else:
        raise ValueError(f"Unsupported parse mode: {parse_mode}")

    if not fallback_table_items:
        return

    rotate_labels = hybrid_model.table_orientation_cls_model.batch_predict(
        fallback_table_items,
        det_batch_size=BATCH_RATIO * OCR_DET_BASE_BATCH_SIZE,
        tqdm_enable=True,
    )
    _apply_table_rotate_labels(fallback_table_items, rotate_labels)


def _page_bbox_to_unit_bbox(bbox: BBox, page_size: tuple[float, float]) -> list[float] | None:
    """将 PDF point bbox 转为页面级 0-1 坐标，并统一保留三位小数。"""
    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0 or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if x1 <= x0 or y1 <= y0:
        return None
    normalized_bbox = [
        round(max(0.0, min(1.0, x0 / page_width)), 3),
        round(max(0.0, min(1.0, y0 / page_height)), 3),
        round(max(0.0, min(1.0, x1 / page_width)), 3),
        round(max(0.0, min(1.0, y1 / page_height)), 3),
    ]
    if normalized_bbox[2] <= normalized_bbox[0] or normalized_bbox[3] <= normalized_bbox[1]:
        return None
    return normalized_bbox


def _build_page_text_formula_spans(
    page_inline_formula_list: list[dict[str, Any]],
    page_ocr_res_list: list[dict[str, Any]],
    page_size: tuple[float, float],
    render_scale: float,
) -> list[Span]:
    """将当前页行内公式和 OCR 结果转换为统一 Span，正文与公式后续共同组行。"""
    page_spans: list[Span] = []
    for formula in page_inline_formula_list:
        bbox = _sidecar_bbox_to_page_bbox(formula.get("bbox"), page_size, render_scale)
        if bbox is None:
            continue
        page_spans.append(
            Span(
                type=ContentType.INLINE_EQUATION,
                bbox=bbox,
                content=str(formula.get("latex", "") or "").strip(),
                score=float(formula.get("score", 0.0) or 0.0),
            )
        )

    for ocr_res in page_ocr_res_list:
        bbox = _sidecar_bbox_to_page_bbox(ocr_res.get("bbox"), page_size, render_scale)
        if bbox is None:
            continue
        page_spans.append(
            Span(
                type=ContentType.TEXT,
                bbox=bbox,
                content=str(ocr_res.get("text", "") or ""),
                score=float(ocr_res.get("score", 0.0) or 0.0),
            )
        )
    return page_spans


def _fill_native_pdf_text_spans(
    pdf_page: PDFPage,
    page_spans: list[Span],
    page_pil_image: Image.Image,
    render_scale: float,
    page_size: tuple[float, float],
) -> list[Span]:
    """复用原生 PDF 字符回填逻辑，并为内容不足的 span 准备后置 OCR 裁图。"""
    page_width, page_height = page_size
    virtual_block = (0, 0, page_width, page_height, None, None, None, BlockType.TEXT)
    return txt_spans_extract(
        pdf_page,
        page_spans,
        page_pil_image,
        render_scale,
        [virtual_block],
        [],
    )


def _group_page_spans_by_block(
    page_model_list: list[dict[str, Any]],
    page_spans: list[Span],
    page_size: tuple[float, float],
    target_block_types: set[str],
) -> dict[int, list[Line]]:
    """按 block 原始顺序消费 span，并使用现有文本修复逻辑形成真实行。"""
    span_matcher = SpanBlockMatcher(page_spans)
    block_lines: dict[int, list[Line]] = {}
    for block_idx, block_item in enumerate(page_model_list):
        block_type = str(block_item.get("type") or block_item.get("label") or "")
        if block_type not in target_block_types:
            continue
        block_bbox = _bbox_to_pixel_bbox(block_item.get("bbox"), page_size)
        if block_bbox is None:
            block_lines[block_idx] = []
            continue

        fix_block = Block(
            index=block_idx,
            type=block_type,
            bbox=block_bbox,
            _fix_spans=span_matcher.collect_for_block(block_bbox),
        )
        block_lines[block_idx] = fix_text_block(fix_block).lines
    return block_lines


def _apply_window_post_ocr(
    local_model_context: HybridLocalModelContext,
    block_lines: dict[int, list[Line]],
) -> None:
    """在当前窗口内识别原生字符不足的 span，保持 finalize 后置 OCR 的回退语义。"""
    need_ocr_spans: list[Span] = []
    img_crop_list: list[np.ndarray] = []
    for lines in block_lines.values():
        for line in lines:
            for span in line.spans:
                if span._np_img is None:
                    continue
                need_ocr_spans.append(span)
                img_crop_list.append(rotate_vertical_crop_if_needed(span._np_img))
                span._np_img = None

    if not img_crop_list:
        return
    ocr_res_list = run_ocr_inference(
        local_model_context.ocr_model.ocr,
        img_crop_list,
        det=False,
        tqdm_enable=True,
    )[0]
    if len(ocr_res_list) != len(need_ocr_spans):
        raise ValueError(
            f"Hybrid post-OCR result count mismatch: ocr_res_list={len(ocr_res_list)}, need_ocr_spans={len(need_ocr_spans)}"
        )

    for span, ocr_res in zip(need_ocr_spans, ocr_res_list):
        ocr_text, ocr_score = ocr_res
        if ocr_score > OcrConfidence.min_confidence:
            span.content = ocr_text
            span.score = float(f"{ocr_score:.3f}")
            _clear_post_ocr_fallback(span)
        elif _restore_post_ocr_fallback(span):
            continue
        else:
            span.content = ""
            span.score = 0.0


def _line_content_parts(line: Line) -> list[tuple[str, str]]:
    """提取一行内可输出的文本与行内公式，公式统一包装为反斜杠圆括号格式。"""
    parts: list[tuple[str, str]] = []
    for span in line.spans:
        if span.type == ContentType.TEXT:
            content = str(span.content or "").strip()
        elif span.type == ContentType.INLINE_EQUATION:
            latex = str(span.content or "").strip()
            content = f"\\({latex}\\)" if latex else ""
        else:
            continue
        if content:
            parts.append((span.type, content))
    return parts


def _lines_to_block_content(lines: list[Line], block_type: str) -> str:
    """将真实行折叠为统一 block content，保留代码换行并处理自然语言跨行连接。"""
    content_lines = [parts for line in lines if (parts := _line_content_parts(line))]
    if not content_lines:
        return ""

    rendered_lines = [" ".join(content for _, content in parts) for parts in content_lines]
    if block_type in CODE_CONTENT_BLOCK_TYPES:
        return "\n".join(rendered_lines).strip()

    text_for_language = "".join(
        content for parts in content_lines for span_type, content in parts if span_type == ContentType.TEXT
    )
    block_language = detect_lang(text_for_language)
    content_parts = [rendered_lines[0]]
    for line_idx in range(1, len(rendered_lines)):
        current_type, current_content = content_lines[line_idx][0]
        next_starts_with_lowercase = (
            current_type == ContentType.TEXT
            and bool(current_content)
            and current_content[0].islower()
        )
        content_parts[-1], separator = resolve_text_line_boundary(
            content_parts[-1],
            block_language=block_language,
            next_starts_with_lowercase=next_starts_with_lowercase,
        )
        content_parts.extend([separator, rendered_lines[line_idx]])
    return "".join(content_parts).strip()


def _build_ocr_det_line_items(lines: list[Line], page_size: tuple[float, float]) -> list[dict[str, Any]]:
    """将内部 Line 转换为归一化行框"""
    line_items = []
    for line in lines:
        normalized_bbox = _page_bbox_to_unit_bbox(line.bbox, page_size)
        if normalized_bbox is not None:
            line_items.append({"bbox": normalized_bbox})
    return line_items


def _apply_block_content_and_line_metadata(
    page_model_list: list[dict[str, Any]],
    block_lines: dict[int, list[Line]],
    page_size: tuple[float, float],
) -> None:
    """将组行结果回填到 block，并只为 TEXT 保存行框。"""
    for block_item in page_model_list:
        block_type = str(block_item.get("type") or block_item.get("label") or "")
        if block_type == BlockType.TEXT:
            block_item["_lines"] = []
        else:
            block_item.pop("_lines", None)

    for block_idx, lines in block_lines.items():
        block_item = page_model_list[block_idx]
        block_type = str(block_item.get("type") or block_item.get("label") or "")
        block_content = block_item.get("content")
        has_nonempty_content = bool(block_content.strip()) if isinstance(block_content, str) else bool(block_content)
        if not has_nonempty_content:
            block_item["content"] = _lines_to_block_content(lines, block_type)

        if block_type == BlockType.TEXT:
            block_item["_lines"] = _build_ocr_det_line_items(lines, page_size)


def _fill_window_block_content_and_lines(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    inline_formula_list: list[list[dict[str, Any]]],
    ocr_res_list: list[list[dict[str, Any]]],
    parse_mode: Literal["txt", "ocr"],
    ocr_det_type: set[str],
    local_model_context: HybridLocalModelContext,
) -> list[list[dict[str, Any]]]:
    """按页完成 span 回填与行级元数据构造，返回不含页面级 sidecar 的 model list。"""
    page_counts = {
        "images": len(images_list),
        "pdf_pages": len(pdf_pages),
        "model_list": len(model_list),
        "inline_formulas": len(inline_formula_list),
        "ocr_results": len(ocr_res_list),
    }
    if len(set(page_counts.values())) != 1:
        raise ValueError(f"Hybrid block content page count mismatch: {page_counts}")

    target_block_types = set(ocr_det_type) | TITLE_BLOCK_TYPES | {BlockType.TEXT}
    for image_dict, pdf_page, page_model_list, page_inline_formula_list, page_ocr_res_list in zip(
        images_list,
        pdf_pages,
        model_list,
        inline_formula_list,
        ocr_res_list,
    ):
        page_pil_image = image_dict["img_pil"]
        render_scale = float(image_dict["scale"])
        page_size = tuple(float(value) for value in pdf_page.size)
        page_spans = _build_page_text_formula_spans(
            page_inline_formula_list,
            page_ocr_res_list,
            page_size,
            render_scale,
        )
        if parse_mode == "txt":
            page_spans = _fill_native_pdf_text_spans(
                pdf_page,
                page_spans,
                page_pil_image,
                render_scale,
                page_size,
            )

        block_lines = _group_page_spans_by_block(
            page_model_list,
            page_spans,
            page_size,
            target_block_types,
        )
        if parse_mode == "txt":
            _apply_window_post_ocr(local_model_context, block_lines)
        _apply_block_content_and_line_metadata(
            page_model_list,
            block_lines,
            page_size,
        )
    return model_list


def _medium_bbox_to_quad(bbox: list[float] | tuple[float, ...]) -> np.ndarray:
    """将普通 bbox 转为表格模型 OCR token 使用的四点框。"""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _normalize_medium_content(value: Any) -> str:
    """将 medium 本地模型输出的文本字段规范成 Hybrid block 可消费的字符串。"""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return value.strip()
    return ""


def _apply_seal_ocr(
    local_model_context: HybridLocalModelContext,
    model_list: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> None:
    """对 medium/high 最终 seal block 逐张执行专用 OCR，并将多段文本按行写回 content。"""
    if len(model_list) != len(np_images):
        raise ValueError(
            "Hybrid seal OCR page count mismatch: "
            f"model_list={len(model_list)}, images={len(np_images)}"
        )

    seal_tasks: list[tuple[dict[str, Any], np.ndarray]] = []
    for page_model_list, np_img in zip(model_list, np_images):
        image_h, image_w = np_img.shape[:2]
        for block_item in page_model_list:
            if (
                block_item.get("type") != BlockType.IMAGE
                or block_item.get("sub_type") != "seal"
            ):
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


def _table_bbox_center(bbox: BBox) -> tuple[float, float]:
    """计算 bbox 中心点，用于判断图片或公式应归属哪个表格。"""
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def _table_bbox_intersection(bbox1: BBox, bbox2: BBox) -> BBox | None:
    """计算两个 bbox 的有效交集，无重叠时返回 None。"""
    x0 = max(float(bbox1[0]), float(bbox2[0]))
    y0 = max(float(bbox1[1]), float(bbox2[1]))
    x1 = min(float(bbox1[2]), float(bbox2[2]))
    y1 = min(float(bbox1[3]), float(bbox2[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _select_table_owner(
    item_bbox: BBox,
    table_entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """为 low/medium 对象匹配所属表格，多表命中时选择交叠面积最大的表格。"""
    center_x, center_y = _table_bbox_center(item_bbox)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for table_entry in table_entries:
        table_bbox = table_entry["table_bbox"]
        if not (
            float(table_bbox[0]) <= center_x <= float(table_bbox[2])
            and float(table_bbox[1]) <= center_y <= float(table_bbox[3])
        ):
            continue
        overlap_bbox = _table_bbox_intersection(item_bbox, table_bbox)
        if overlap_bbox is None:
            continue
        overlap_area = float(overlap_bbox[2] - overlap_bbox[0]) * float(overlap_bbox[3] - overlap_bbox[1])
        candidates.append((overlap_area, table_entry))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _normalize_visual_block_angle(angle: Any) -> int:
    """规范视觉块角度为 0/90/180/270，无法识别的角度按 0 处理。"""
    try:
        normalized_angle = int(float(angle or 0)) % 360
    except (TypeError, ValueError):
        logger.warning(f"Unsupported visual block angle: {angle}, using 0")
        return 0
    if normalized_angle not in {0, 90, 180, 270}:
        logger.warning(f"Unsupported visual block angle: {angle}, using 0")
        return 0
    return normalized_angle


def _rotate_visual_block_image_to_upright(image: np.ndarray, angle: int) -> np.ndarray:
    """按 layout 视觉块角度把裁图旋转至正向，角度语义与方向分类模型保持一致。"""
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def _rotate_medium_table_bbox(
    bbox: BBox,
    image_width: float,
    image_height: float,
    angle: int,
) -> BBox:
    """把原表格裁图中的 bbox 同步转换到旋转后裁图坐标系。"""
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if angle == 270:
        # 顺时针旋转 90 度后，新 x 轴来自原 y 轴的反方向。
        return (image_height - y1, x0, image_height - y0, x1)
    if angle == 90:
        # 逆时针旋转 90 度后，新 y 轴来自原 x 轴的反方向。
        return (y0, image_width - x1, y1, image_width - x0)
    if angle == 180:
        return (image_width - x1, image_height - y1, image_width - x0, image_height - y0)
    return (x0, y0, x1, y1)


def _get_medium_table_virtual_image_bbox(
    bbox: BBox,
    image_size: tuple[int, int],
    box_size: float = 10.0,
) -> BBox:
    """在图片中心生成小 OCR token 框，避免图片大框干扰单元格匹配。"""
    image_width, image_height = image_size
    center_x, center_y = _table_bbox_center(bbox)
    half_size = box_size / 2.0
    return (
        max(0.0, center_x - half_size),
        max(0.0, center_y - half_size),
        min(float(image_width), center_x + half_size),
        min(float(image_height), center_y + half_size),
    )


def _encode_page_crop_as_jpeg_data_uri(
    np_image: np.ndarray,
    page_bbox: BBox,
    angle: int,
) -> str:
    """从页面原图按像素框裁剪，按视觉块方向回正后编码为 JPEG data URI。"""
    image_h, image_w = np_image.shape[:2]
    image_bbox = normalize_to_int_bbox(page_bbox, image_size=(image_h, image_w))
    if image_bbox is None:
        return ""
    x0, y0, x1, y1 = image_bbox
    crop_rgb = np_image[y0:y1, x0:x1].copy()
    if crop_rgb.size == 0:
        return ""

    crop_rgb = _rotate_visual_block_image_to_upright(crop_rgb, angle)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", crop_bgr)
    if not success:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def _attach_visual_block_images(
    model_list: list[list[dict[str, Any]]],
    images_list: list[dict[str, Any]],
    page_start_index: int = 0,
) -> None:
    """在窗口页图释放前，为最终 model_list 视觉块写入回正后的页面裁图。"""
    if len(model_list) != len(images_list):
        raise ValueError(f"Hybrid visual crop page count mismatch: model_list={len(model_list)}, images={len(images_list)}")

    for page_offset, (page_model_list, image_dict) in enumerate(zip(model_list, images_list)):
        visual_blocks = [
            (block_idx, block)
            for block_idx, block in enumerate(page_model_list)
            if block.get("type") in MODEL_JSON_VISUAL_BLOCK_TYPES
        ]
        if not visual_blocks:
            continue

        # 先清理模型可能携带的同名字段，保证最终载荷只来自当前 PDF 页面裁图。
        for _, block in visual_blocks:
            block.pop("image_base64", None)

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


def _collect_medium_table_tasks(
    model_list: list[list[dict[str, Any]]],
    inline_formula_list: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> list[dict[str, Any]]:
    """从当前窗口 model_list 收集表格任务，并提前消费表内图片和行内公式。"""
    table_tasks: list[dict[str, Any]] = []
    for page_idx, (page_model_list, page_inline_formulas, np_image) in enumerate(
        zip(model_list, inline_formula_list, np_images)
    ):
        image_h, image_w = np_image.shape[:2]
        page_size = (image_w, image_h)
        table_entries: list[dict[str, Any]] = []
        for block in page_model_list:
            if block.get("type") != BlockType.TABLE:
                continue
            pixel_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            table_bbox = normalize_to_int_bbox(pixel_bbox, image_size=(image_h, image_w))
            if table_bbox is None:
                continue
            x0, y0, x1, y1 = table_bbox
            table_crop = np_image[y0:y1, x0:x1].copy()
            if table_crop.size == 0:
                continue
            table_entries.append(
                {
                    "table_block": block,
                    "table_bbox": table_bbox,
                    "table_crop": table_crop,
                    "angle": _normalize_visual_block_angle(block.get("angle", 0)),
                    "inline_objects": [],
                }
            )

        if not table_entries:
            continue

        consumed_image_ids: set[int] = set()
        consumed_formula_ids: set[int] = set()
        for block in page_model_list:
            if block.get("type") != BlockType.IMAGE:
                continue
            image_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if image_bbox is None:
                continue
            owner = _select_table_owner(image_bbox, table_entries)
            if owner is None:
                continue
            owner["inline_objects"].append(
                {
                    "kind": "image",
                    "page_bbox": image_bbox,
                    "score": float(block.get("score", 1.0) or 0.0),
                }
            )
            consumed_image_ids.add(id(block))

        for formula in page_inline_formulas:
            latex = _normalize_medium_content(formula.get("latex", ""))
            formula_bbox = _bbox_to_pixel_bbox(formula.get("bbox"), page_size)
            if formula_bbox is None:
                continue
            owner = _select_table_owner(formula_bbox, table_entries)
            if owner is None:
                continue
            owner["inline_objects"].append(
                {
                    "kind": "formula",
                    "page_bbox": formula_bbox,
                    "latex": latex,
                    "score": float(formula.get("score", 1.0) or 0.0),
                }
            )
            consumed_formula_ids.add(id(formula))

        # 匹配完成后立即删除原始对象；后续 OCR 或结构识别失败均不回滚。
        page_model_list[:] = [block for block in page_model_list if id(block) not in consumed_image_ids]
        page_inline_formulas[:] = [formula for formula in page_inline_formulas if id(formula) not in consumed_formula_ids]

        for table_entry in table_entries:
            table_bbox = table_entry["table_bbox"]
            table_crop = table_entry["table_crop"]
            angle = table_entry["angle"]
            crop_h, crop_w = table_crop.shape[:2]
            rotated_crop = _rotate_visual_block_image_to_upright(table_crop, angle)
            rotated_h, rotated_w = rotated_crop.shape[:2]
            prepared_objects: list[dict[str, Any]] = []
            for inline_object in table_entry["inline_objects"]:
                overlap_bbox = _table_bbox_intersection(inline_object["page_bbox"], table_bbox)
                if overlap_bbox is None:
                    continue
                relative_bbox = (
                    float(overlap_bbox[0]) - float(table_bbox[0]),
                    float(overlap_bbox[1]) - float(table_bbox[1]),
                    float(overlap_bbox[2]) - float(table_bbox[0]),
                    float(overlap_bbox[3]) - float(table_bbox[1]),
                )
                rotated_bbox = _rotate_medium_table_bbox(relative_bbox, crop_w, crop_h, angle)
                if inline_object["kind"] == "formula":
                    latex = inline_object["latex"]
                    content = f"<eq>{html.escape(latex)}</eq>" if latex else ""
                    token_bbox = rotated_bbox
                else:
                    image_src = _encode_page_crop_as_jpeg_data_uri(np_image, inline_object["page_bbox"], angle)
                    content = f'<img src="{image_src}"/>' if image_src else ""
                    token_bbox = _get_medium_table_virtual_image_bbox(
                        rotated_bbox,
                        (rotated_w, rotated_h),
                    )
                prepared_objects.append(
                    {
                        **inline_object,
                        "mask_bbox": rotated_bbox,
                        "token_bbox": token_bbox,
                        "content": content,
                    }
                )

            table_tasks.append(
                {
                    "page_idx": page_idx,
                    "table_block": table_entry["table_block"],
                    "table_bbox": table_bbox,
                    "angle": angle,
                    "table_img": rotated_crop,
                    "wired_table_img": rotated_crop,
                    "ocr_result": [],
                    "table_res": {},
                    "inline_objects": prepared_objects,
                }
            )
    return table_tasks


def _sort_medium_table_ocr_result(ocr_result: list[list[Any]]) -> None:
    """按 token 顶边和左边坐标排序，保证表格 OCR 输入顺序稳定。"""
    def sort_key(item: list[Any]) -> tuple[float, float]:
        """提取四点框的最小 y/x，兼容普通列表和 numpy 数组。"""
        box = np.asarray(item[0], dtype=np.float32).reshape(-1, 2)
        return float(np.min(box[:, 1])), float(np.min(box[:, 0]))

    ocr_result.sort(key=sort_key)


def _prepare_medium_table_ocr_results(
    local_model_context: HybridLocalModelContext,
    table_tasks: list[dict[str, Any]],
) -> None:
    """遮盖表内图片和公式后逐表执行 OCR，并合并内联对象 token。"""
    table_ocr_model = None
    try:
        table_ocr_model = local_model_context.get_ocr_model(
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            enable_merge_det_boxes=False,
        )
    except Exception as exc:
        logger.warning(f"Hybrid medium table OCR model initialization failed: {exc}")

    for table_task in table_tasks:
        ocr_result: list[list[Any]] = []
        bgr_image = cv2.cvtColor(table_task["table_img"], cv2.COLOR_RGB2BGR)
        mask_boxes = [{"bbox": item["mask_bbox"]} for item in table_task["inline_objects"]]
        masked_image = mask_formula_regions_for_ocr_det(bgr_image, mask_boxes)

        if table_ocr_model is not None:
            try:
                page_ocr_results = run_ocr_inference(table_ocr_model.ocr, masked_image)
                raw_ocr_result = page_ocr_results[0] if page_ocr_results else None
                for raw_item in raw_ocr_result or []:
                    if not raw_item or len(raw_item) < 2:
                        continue
                    box, rec_result = raw_item[0], raw_item[1]
                    if not rec_result or len(rec_result) < 2:
                        continue
                    text, score = rec_result[0], rec_result[1]
                    normalized_text = _normalize_medium_content(text)
                    if not normalized_text:
                        continue
                    ocr_result.append(
                        [
                            np.asarray(box, dtype=np.float32),
                            html.escape(normalized_text),
                            float(score or 0.0),
                        ]
                    )
            except Exception as exc:
                logger.warning(
                    f"Hybrid medium table OCR failed: page_idx={table_task['page_idx']}, error={exc}"
                )

        for inline_object in table_task["inline_objects"]:
            if not inline_object["content"]:
                continue
            ocr_result.append(
                [
                    _medium_bbox_to_quad(inline_object["token_bbox"]),
                    inline_object["content"],
                    inline_object["score"],
                ]
            )
        _sort_medium_table_ocr_result(ocr_result)
        table_task["ocr_result"] = ocr_result


def _trim_medium_table_html(html_code: Any) -> str:
    """从模型输出中截取核心 table HTML；未带外围标签时保留原字符串。"""
    if not isinstance(html_code, str):
        return ""
    stripped_html = html_code.strip()
    lower_html = stripped_html.lower()
    start_index = lower_html.find("<table")
    end_index = lower_html.rfind("</table>")
    if start_index >= 0 and end_index >= start_index:
        return stripped_html[start_index : end_index + len("</table>")]
    return stripped_html


def _apply_medium_table_recognition(
    local_model_context: HybridLocalModelContext,
    model_list: list[list[dict[str, Any]]],
    inline_formula_list: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> None:
    """基于 model_list 执行 medium 表格解析，采用分类 batch、无线 batch、有线单表调度。"""
    table_tasks = _collect_medium_table_tasks(model_list, inline_formula_list, np_images)
    if not table_tasks:
        return

    _prepare_medium_table_ocr_results(local_model_context, table_tasks)

    try:
        local_model_context.table_cls_model.batch_predict(table_tasks)
    except Exception as exc:
        # 分类失败不阻断无线模型，未获得分类结果的表格按 wireless-only 处理。
        logger.warning(f"Hybrid medium table classification failed: {exc}")

    try:
        local_model_context.wireless_table_model.batch_predict(table_tasks)
    except Exception as exc:
        logger.warning(f"Hybrid medium wireless table recognition failed: {exc}")

    for table_task in table_tasks:
        cls_label = table_task["table_res"].get("cls_label")
        try:
            cls_score = float(table_task["table_res"].get("cls_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            cls_score = 0.0
        use_wired_model = cls_label == AtomicModel.WiredTable or (
            cls_label == AtomicModel.WirelessTable and cls_score < 0.9
        )
        if use_wired_model:
            try:
                wireless_html = table_task["table_res"].get("html", "") or ""
                wired_html = local_model_context.wired_table_model.predict(
                    table_task["wired_table_img"],
                    table_task["ocr_result"],
                    wireless_html,
                )
                if wired_html is not None:
                    table_task["table_res"]["html"] = wired_html
            except Exception as exc:
                logger.warning(
                    f"Hybrid medium wired table recognition failed: page_idx={table_task['page_idx']}, error={exc}"
                )

        html_code = _trim_medium_table_html(table_task["table_res"].get("html", ""))
        if html_code:
            table_task["table_block"]["content"] = html_code


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
            if block_item.get("type") != BlockType.FORMULA_NUMBER:
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


def _remove_low_table_inner_blocks(
    images_list: list[dict[str, Any]],
    model_list: list[list[dict[str, Any]]],
) -> None:
    """按表格中心点归属规则移除 low 表内图片、公式和公式编号块。"""
    inner_block_types = {
        BlockType.IMAGE,
        BlockType.EQUATION,
        BlockType.FORMULA_NUMBER,
    }
    for image_dict, page_model_list in zip(images_list, model_list):
        page_size = _normalize_page_size(image_dict["img_pil"])
        table_entries = []
        for block in page_model_list:
            if block.get("type") != BlockType.TABLE:
                continue
            table_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if table_bbox is not None:
                table_entries.append({"table_bbox": table_bbox})

        if not table_entries:
            continue

        retained_blocks = []
        for block in page_model_list:
            if block.get("type") not in inner_block_types:
                retained_blocks.append(block)
                continue
            block_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if block_bbox is None or _select_table_owner(block_bbox, table_entries) is None:
                retained_blocks.append(block)
        page_model_list[:] = retained_blocks


def _fill_low_table_contents(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    parse_mode: Literal["txt", "ocr"],
    local_model_context: HybridLocalModelContext,
) -> None:
    """为 low 表格回填空间投影纯文本，单表失败时保留空内容并继续。"""
    table_entries: list[dict[str, Any]] = []
    for page_idx, page_model_list in enumerate(model_list):
        table_idx = 0
        for block in page_model_list:
            if block.get("type") != BlockType.TABLE:
                continue
            block["content"] = ""
            table_entries.append(
                {
                    "page_idx": page_idx,
                    "table_idx": table_idx,
                    "block": block,
                }
            )
            table_idx += 1

    if not table_entries:
        return

    # 表格一旦认领内部对象便立即从 model_list 删除，后续文本回填失败也不恢复。
    _remove_low_table_inner_blocks(images_list, model_list)

    if parse_mode == "txt":
        page_chars_cache: dict[int, list[Any]] = {}
        for table_entry in table_entries:
            page_idx = table_entry["page_idx"]
            table_idx = table_entry["table_idx"]
            table_block = table_entry["block"]
            try:
                pdf_page = pdf_pages[page_idx]
                page_width, page_height = pdf_page.size
                table_bbox = _bbox_to_pixel_bbox(
                    table_block.get("bbox"),
                    (int(page_width), int(page_height)),
                )
                if table_bbox is None:
                    raise ValueError("invalid table bbox")
                if page_idx not in page_chars_cache:
                    page_chars_cache[page_idx] = pdf_page.get_chars()
                table_block["content"] = project_pdf_table_text(
                    page_chars_cache[page_idx],
                    table_bbox,
                    table_block.get("angle", 0),
                )
                if not table_block["content"]:
                    logger.warning(
                        "Hybrid low table text is empty: "
                        f"parse_mode={parse_mode}, page_idx={page_idx}, "
                        f"table_idx={table_idx}, bbox={table_block.get('bbox')}"
                    )
            except Exception as exc:
                logger.warning(
                    "Hybrid low table text failed: "
                    f"parse_mode={parse_mode}, page_idx={page_idx}, "
                    f"table_idx={table_idx}, bbox={table_block.get('bbox')}, error={exc}"
                )
        return

    if parse_mode != "ocr":
        raise ValueError(f"Unsupported parse mode: {parse_mode}")

    try:
        table_ocr_model = local_model_context.get_ocr_model(
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            enable_merge_det_boxes=False,
        )
    except Exception as exc:
        for table_entry in table_entries:
            logger.warning(
                "Hybrid low table text failed: "
                f"parse_mode={parse_mode}, page_idx={table_entry['page_idx']}, "
                f"table_idx={table_entry['table_idx']}, "
                f"bbox={table_entry['block'].get('bbox')}, "
                f"error=OCR model initialization failed: {exc}"
            )
        return

    page_image_cache: dict[int, np.ndarray] = {}
    for table_entry in table_entries:
        page_idx = table_entry["page_idx"]
        table_idx = table_entry["table_idx"]
        table_block = table_entry["block"]
        try:
            if page_idx not in page_image_cache:
                page_image_cache[page_idx] = np.asarray(images_list[page_idx]["img_pil"]).copy()
            np_image = page_image_cache[page_idx]
            image_height, image_width = np_image.shape[:2]
            pixel_bbox = _bbox_to_pixel_bbox(
                table_block.get("bbox"),
                (image_width, image_height),
            )
            table_bbox = normalize_to_int_bbox(
                pixel_bbox,
                image_size=(image_height, image_width),
            )
            if table_bbox is None:
                raise ValueError("invalid table bbox")
            x0, y0, x1, y1 = table_bbox
            table_crop = np_image[y0:y1, x0:x1].copy()
            if table_crop.size == 0:
                raise ValueError("empty table crop")

            angle = _normalize_visual_block_angle(table_block.get("angle", 0))
            rotated_crop = _rotate_visual_block_image_to_upright(table_crop, angle)
            bgr_crop = cv2.cvtColor(rotated_crop, cv2.COLOR_RGB2BGR)
            page_ocr_results = run_ocr_inference(table_ocr_model.ocr, bgr_crop)
            raw_ocr_result = page_ocr_results[0] if page_ocr_results else None
            rotated_height, rotated_width = rotated_crop.shape[:2]
            table_block["content"] = project_ocr_table_text(
                raw_ocr_result,
                (rotated_width, rotated_height),
            )
            if not table_block["content"]:
                logger.warning(
                    "Hybrid low table text is empty: "
                    f"parse_mode={parse_mode}, page_idx={page_idx}, "
                    f"table_idx={table_idx}, bbox={table_block.get('bbox')}"
                )
        except Exception as exc:
            logger.warning(
                "Hybrid low table text failed: "
                f"parse_mode={parse_mode}, page_idx={page_idx}, "
                f"table_idx={table_idx}, bbox={table_block.get('bbox')}, error={exc}"
            )


def _process_low_text(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    parse_mode: Literal["txt", "ocr"],
    local_model_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """使用 pdftext line 或本地 OCR 为 low layout block 填充文本内容。"""
    _validate_text_formula_window_inputs(
        images_list,
        pdf_pages,
        model_list,
        images_layout_res,
    )

    if parse_mode == "txt":
        target_block_types = set(PIPELINE_DET_TYPE) | TITLE_BLOCK_TYPES | {BlockType.TEXT}
        for pdf_page, page_model_list in zip(pdf_pages, model_list):
            page_size = tuple(float(value) for value in pdf_page.size)
            block_lines = _group_page_spans_by_block(
                page_model_list,
                _build_pdf_text_line_spans(pdf_page),
                page_size,
                target_block_types,
            )
            _apply_block_content_and_line_metadata(
                page_model_list,
                block_lines,
                page_size,
            )
    elif parse_mode == "ocr":
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
        np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
        empty_formula_list: list[list[dict[str, Any]]] = [[] for _ in model_list]
        ocr_res_list = _ocr_det(
            local_model_context,
            np_images,
            model_list,
            empty_formula_list,
            True,
            PIPELINE_DET_TYPE,
        )
        _apply_ocr_rec_results(local_model_context, ocr_res_list)
        model_list = _fill_window_block_content_and_lines(
            images_list,
            pdf_pages,
            model_list,
            empty_formula_list,
            ocr_res_list,
            parse_mode,
            PIPELINE_DET_TYPE,
            local_model_context,
        )
    else:
        raise ValueError(f"Unsupported parse mode: {parse_mode}")

    _fill_low_table_contents(
        images_list,
        pdf_pages,
        model_list,
        parse_mode,
        local_model_context,
    )
    return model_list


def _process_text_and_formulas(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    parse_mode: Literal["txt", "ocr"],
    effort: Literal["medium", "high", "xhigh"],
    local_model_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """在当前窗口内完成 OCR、公式、原生文本及 block 行信息回填。"""

    _validate_text_formula_window_inputs(
        images_list,
        pdf_pages,
        model_list,
        images_layout_res,
    )
    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]

    # 遍历model_list,对文本块截图交由OCR识别
    # 根据 parse_mode 和 effort 决定需要ocr的文本块的类型以及只开det还是det+rec
    ocr_det_type, mfr_enable = _build_ocr_det_type_and_mfr_enable(
        parse_mode=parse_mode,
        effort=effort,
    )

    # 将PIL图片转换为numpy数组
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]

    mfd_res = _build_formula_inputs(images_layout_res)
    images_formula_list = mfd_res
    interline_enable = effort == "medium"

    # medium 识别行内和行间公式；high/xhigh 的 txt 路径只识别行内公式。
    if mfr_enable:
        images_formula_list = local_model_context.mfr_model.batch_predict(
            mfd_res,
            np_images,
            batch_size=BATCH_RATIO * MFR_BASE_BATCH_SIZE,
            interline_enable=interline_enable,
        )

    inline_formula_list, display_formula_list = _split_formula_results(images_formula_list)
    if effort == "medium":
        # 表格解析必须早于正文行填充，确保表内图片和行内公式只由表格模型消费一次。
        _apply_medium_table_recognition(
            local_model_context,
            model_list,
            inline_formula_list,
            np_images,
        )
        # 将行间公式span回填入block
        _apply_medium_display_formula_results(
            model_list,
            display_formula_list,
            images_pil_list,
        )
        # 使用ocr识别行间公式标号
        _apply_medium_formula_number_ocr(
            local_model_context,
            model_list,
            np_images,
        )

    # 行间公式标号回填到block
    for page_model_list in model_list:
        page_model_list[:] = optimize_hybrid_formula_number_blocks(page_model_list)

    need_rec_img = parse_mode == "ocr" and effort == "medium"
    # vlm没有执行ocr，需要ocr_det
    ocr_res_list = _ocr_det(
        local_model_context,
        np_images,
        model_list,
        mfd_res,
        need_rec_img,
        ocr_det_type,
    )

    # 如果有rec_img则做ocr_rec
    if need_rec_img:
        _apply_ocr_rec_results(local_model_context, ocr_res_list)

    return _fill_window_block_content_and_lines(
        images_list,
        pdf_pages,
        model_list,
        inline_formula_list,
        ocr_res_list,
        parse_mode,
        ocr_det_type,
        local_model_context,
    )


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
            if block.get("type") != BlockType.TITLE:
                continue
            title_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if title_bbox is None:
                continue
            if _has_doc_title_overlap(title_bbox, doc_title_bboxes, overlap_threshold):
                block["type"] = BlockType.DOC_TITLE
            else:
                block["type"] = BlockType.PARAGRAPH_TITLE


def _replace_inline_formula_delimiters(model_list: list[list[dict[str, Any]]]) -> None:
    """将 model JSON content 中的行内公式定界符原地替换为 eq 标签。"""
    for page_model_list in model_list:
        for block in page_model_list:
            content = block.get("content")
            if not isinstance(content, str):
                continue
            block["content"] = _INLINE_FORMULA_PATTERN.sub(
                lambda match: f"<eq>{match.group(1)}</eq>",
                content,
            )


def _log_infer_performance(file_suffix: str, page_count: int, elapsed: float) -> None:
    """使用未舍入耗时统一记录 model_list 生产阶段的处理速度。"""
    speed = page_count / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"model_list infer finished, file_suffix={file_suffix}, pages={page_count}, "
        f"cost={elapsed:.6f}s, speed={speed:.3f} page/s"
    )


def doc_analyze(
    file_bytes: bytes,
    effort: Literal["flash", "low", "medium", "high", "xhigh"] = "high",
    parse_mode: Literal["auto", "txt", "ocr"] = "auto",
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    file_suffix: Literal["pdf", "docx", "pptx", "xlsx"] = "pdf",
) -> tuple[list[PageInfo], list[list[dict[str, Any]]]]:
    if file_suffix not in _SUPPORTED_FILE_SUFFIXES:
        raise ValueError(f"Unsupported file suffix: {file_suffix!r}")

    if file_suffix in _OFFICE_MODEL_MAP:
        # Office Converter 直接产出完整 model_list，固定按 Flash TXT 模式处理。
        effort = "flash"
        parse_mode = "txt"
        file_stream = BytesIO(file_bytes)
        office_model = _OFFICE_MODEL_MAP[file_suffix]()
        infer_started_at = time.perf_counter()
        model_list = office_model.predict(file_stream)
        infer_elapsed = time.perf_counter() - infer_started_at
    else:
        document = PDFDocument(file_bytes)
        document_closed = False
        model_list: list[list[dict[str, Any]]] = []
        try:
            if parse_mode == "auto":
                parse_mode = document.classify()
            if parse_mode not in ["txt", "ocr"]:
                raise ValueError(f"parse_mode {parse_mode} is not supported")

            # Flash 只处理原生文本，OCR 文档继续复用 Hybrid low 流程。
            if effort == "flash" and parse_mode == "ocr":
                effort = "low"

            flash_txt_mode = effort == "flash" and parse_mode == "txt"
            page_count = document.page_count
            hybrid_model = None

            if not flash_txt_mode:
                hybrid_model_singleton = HybridLocalModelContextSingleton()
                hybrid_model = hybrid_model_singleton.get_model()

                if effort in ["high", "xhigh"]:
                    vlm_runtime = _load_vlm_runtime()
                    vlm_backend = get_vlm_engine(inference_engine="auto", is_async=False)
                    vlm_predictor = vlm_runtime["ModelSingleton"]().get_model(
                        backend=vlm_backend,
                        model_path=None,
                        server_url=None,
                    )
                    vlm_predictor = vlm_runtime["_maybe_enable_serial_execution"](vlm_predictor, vlm_backend)
                else:
                    vlm_predictor = None

            infer_started_at = time.perf_counter()
            if flash_txt_mode:
                # Flash 先对整份 PDF 生成完整 model_list，不依赖页面渲染和处理窗口。
                from mineru.model.flash import PdfModel

                model_list = PdfModel().predict(document)

            configured_window_size = get_processing_window_size(default=64)
            windows = _build_processing_windows(page_count, configured_window_size)
            _log_processing_window_plan(page_count, configured_window_size, len(windows))

            for window in windows:
                window_pages = _get_window_pdf_pages(document, window)
                images_list = load_images_from_pdf_bytes_range(
                    pdf_bytes=file_bytes,
                    start_page_id=window.start,
                    end_page_id=window.end,
                    image_type="pil_img",
                )
                try:
                    if len(window_pages) != len(images_list):
                        raise ValueError("Hybrid processing window PDF page count does not match image count")
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    _log_processing_window(window, page_count, len(images_pil_list))

                    if flash_txt_mode:
                        # Flash 仅切割当前窗口的外层列表，用于页图释放前原地补充视觉块裁图。
                        window_model_list = model_list[window.start : window.end + 1]
                    else:
                        np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
                        images_layout_res = hybrid_model.layout_model.batch_predict(
                            images_pil_list, batch_size=min(8, BATCH_RATIO * LAYOUT_BASE_BATCH_SIZE)
                        )

                        # 使用小模型layout时对layout的表格做旋转检测
                        if effort in ["low", "medium", "high"]:
                            table_items = _collect_table_items(images_layout_res, np_images)
                            if table_items:
                                _apply_table_orientations(
                                    table_items,
                                    parse_mode,
                                    window_pages,
                                    images_list,
                                    hybrid_model,
                                )

                        vl_style_layout_blocks = _build_vl_style_layout_blocks(images_layout_res, images_pil_list)

                        if parse_mode == "txt":
                            if effort in ["low", "medium"]:
                                window_model_list = vl_style_layout_blocks
                            elif effort == "high":
                                window_model_list = vlm_predictor.batch_extract_with_layout(
                                    images=images_pil_list,
                                    blocks_list=vl_style_layout_blocks,
                                    not_extract_list=NOT_EXTRACT_TYPES,
                                    image_analysis=False,
                                )
                            elif effort == "xhigh":
                                window_model_list = vlm_predictor.batch_two_step_extract(
                                    images=images_pil_list,
                                    not_extract_list=NOT_EXTRACT_TYPES,
                                    image_analysis=image_analysis,
                                )
                                _apply_layout_title_split(
                                    window_model_list,
                                    images_layout_res,
                                    [_normalize_page_size(image) for image in images_pil_list],
                                )
                            else:
                                raise ValueError(f"Unsupported analyze effort: {effort}")
                        elif parse_mode == "ocr":
                            if effort in ["low", "medium"]:
                                window_model_list = vl_style_layout_blocks
                            elif effort == "high":
                                window_model_list = vlm_predictor.batch_extract_with_layout(
                                    images=images_pil_list,
                                    blocks_list=vl_style_layout_blocks,
                                    image_analysis=False,
                                )
                            elif effort == "xhigh":
                                window_model_list = vlm_predictor.batch_two_step_extract(
                                    images=images_pil_list,
                                    image_analysis=image_analysis,
                                )
                                _apply_layout_title_split(
                                    window_model_list,
                                    images_layout_res,
                                    [_normalize_page_size(image) for image in images_pil_list],
                                )
                            else:
                                raise ValueError(f"Unsupported analyze effort: {effort}")
                        else:
                            raise ValueError(f"Unsupported parse mode: {parse_mode}")

                        if effort == "low":
                            window_model_list = _process_low_text(
                                images_list,
                                window_pages,
                                window_model_list,
                                parse_mode,
                                hybrid_model,
                                images_layout_res,
                            )
                        else:
                            window_model_list = _process_text_and_formulas(
                                images_list,
                                window_pages,
                                window_model_list,
                                parse_mode,
                                effort,
                                hybrid_model,
                                images_layout_res,
                            )

                        if effort in {"medium", "high"}:
                            _apply_seal_ocr(hybrid_model, window_model_list, np_images)

                    _attach_visual_block_images(
                        window_model_list,
                        images_list,
                        page_start_index=window.start,
                    )
                    if not flash_txt_mode:
                        model_list.extend(window_model_list)
                finally:
                    _close_images(images_list)

            # 仅 PDF 模型结果需要将行内公式定界符统一替换为 eq 标签。
            _replace_inline_formula_delimiters(model_list)
            infer_elapsed = time.perf_counter() - infer_started_at

            document.close()
            document_closed = True
            if hybrid_model is not None:
                clean_memory(hybrid_model.device)
        finally:
            if not document_closed:
                document.close()

    _log_infer_performance(file_suffix, len(model_list), infer_elapsed)

    # PDF 与 Office 的 model_list 在此汇合，后续统一生产 Middle JSON。
    middle_json: list[PageInfo] = []
    return middle_json, model_list


if __name__ == "__main__":
    from mineru.cli_old.common import read_fn



    # 根据当前文件位置定位项目根目录，读取 demo 下的 PDF 和 Office 样例。
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    demo_file_paths = [
        os.path.join(project_root, "demo", "pdfs", "demo1.pdf"),
        os.path.join(project_root, "demo", "office_docs", "docx_01.docx"),
    ]

    for file_path in demo_file_paths:
        file_bytes = read_fn(file_path)
        file_suffix = os.path.splitext(file_path)[1].lstrip(".").lower()
        middle_json, model_list = doc_analyze(file_bytes, effort="medium", file_suffix=file_suffix)
        logger.info(f"file_path: {file_path}")
        logger.info(f"middle_json: {middle_json}")
        logger.info(f"model_list: {model_list}")
