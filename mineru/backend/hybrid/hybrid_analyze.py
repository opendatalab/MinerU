# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import html
import os
import time
from collections import defaultdict
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
from ...utils.config_reader import get_device, get_processing_window_size, get_table_enable
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


def ocr_classify(pdf_doc: PDFDocument, parse_method: str = "auto") -> bool:
    # 确定OCR设置
    _ocr_enable = False
    if parse_method == "auto":
        if pdf_doc.classify() == "ocr":
            _ocr_enable = True
    elif parse_method == "ocr":
        _ocr_enable = True
    return _ocr_enable


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
    def _set_temp_pixel_bbox(res: dict[str, Any], pixel_bbox: list[int]) -> None:
        """临时切换为像素 bbox，便于复用已有裁剪逻辑。"""
        res["_normalized_bbox"] = list(res["bbox"])
        res["bbox"] = pixel_bbox

    def _restore_normalized_bbox(res: dict[str, Any]) -> None:
        """恢复归一化 bbox，避免 OCR det 过程污染 Hybrid 输出。"""
        normalized_bbox = res.pop("_normalized_bbox", None)
        if normalized_bbox is not None:
            res["bbox"] = normalized_bbox

    ocr_res_list: list[list[dict[str, Any]]] = []

    if not local_context.enable_ocr_det_batch:
        # 非批处理模式 - 逐页处理
        for np_image, page_mfd_res, page_results in tqdm(
            zip(np_images, mfd_res, model_list), total=len(np_images), desc="OCR-det"
        ):
            ocr_res_list.append([])
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
                ocr_res = run_ocr_inference(
                    local_context.ocr_model.ocr,
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
                        local_context.lang,
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
                all_cropped_images_info.append((bgr_image, useful_list, adjusted_mfdetrec_res, ocr_res_list[-1]))

        batch_images = [crop_info[0] for crop_info in all_cropped_images_info]
        det_batch_size = min(len(batch_images), batch_ratio * OCR_DET_BASE_BATCH_SIZE)
        batch_results = run_ocr_inference(
            local_context.ocr_model.text_detector.batch_predict,
            batch_images,
            det_batch_size,
            tqdm_enable=True,
            tqdm_desc="OCR-det",
        )

        for crop_info, (dt_boxes, _) in zip(all_cropped_images_info, batch_results):
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
                        local_context.lang,
                    )
                    ocr_page_res_list.extend(ocr_result_list)
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


def _is_medium_table_enabled() -> bool:
    """读取 Hybrid medium 的表格开关，复用 Hybrid 当前通过环境变量传入的表格配置。"""
    return get_table_enable(os.getenv("MINERU_VLM_TABLE_ENABLE", "True").lower() == "true")


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
    formula_enable: bool,
    batch_ratio: int,
) -> list[list[dict[str, Any]]]:
    """执行 Hybrid medium 的本地公式识别，并返回供 OCR det 遮罩使用的公式框。"""
    if not formula_enable:
        for layout_res in images_layout_res:
            layout_res[:] = [item for item in layout_res if item.get("label") != "inline_formula"]
        return [[] for _ in images_layout_res]

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
                    "lang": lang,
                    "table_img": table_img,
                    "wired_table_img": wired_table_img,
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
    det_images = [cv2.cvtColor(item["table_img"], cv2.COLOR_RGB2BGR) for item in table_items]
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
    if len(batch_results) != len(table_items):
        raise ValueError("Hybrid medium table OCR det batch result count mismatch")

    for table_id, (table_item, bgr_image, (dt_boxes, _)) in enumerate(zip(table_items, det_images, batch_results)):
        if dt_boxes is None or len(dt_boxes) == 0:
            continue
        for dt_box in sorted_boxes(dt_boxes):
            dt_box_array = np.asarray(dt_box, dtype=np.float32)
            rec_img_lang_group[table_item["lang"]].append(
                {
                    "cropped_img": get_rotate_crop_image_for_text_rec(bgr_image, dt_box_array.copy()),
                    "dt_box": dt_box_array.copy(),
                    "table_id": table_id,
                }
            )
    return rec_img_lang_group


def _apply_medium_table_recognition(
    local_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
    lang_list: list[str | None],
    batch_ratio: int,
) -> None:
    """执行 Hybrid medium 的本地表格识别，结果写回 layout table 项的 html 字段。"""
    if not _is_medium_table_enabled():
        return

    table_items = _collect_medium_table_items(images_layout_res, np_images, lang_list)
    if not table_items:
        return

    atom_model_manager = local_context.atom_model_manager
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

    table_cls_model = atom_model_manager.get_atom_model(atom_model_name=AtomicModel.TableCls)
    try:
        table_cls_model.batch_predict(table_items)
    except Exception as exc:
        logger.warning(f"Hybrid medium table classification failed: {exc}, using default model")

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
            ocr_result_item = [img_dict["dt_box"], html.escape(str(ocr_res[0])), ocr_res[1]]
            table_item.setdefault("ocr_result", []).append(ocr_result_item)

    wireless_table_model = atom_model_manager.get_atom_model(atom_model_name=AtomicModel.WirelessTable)
    wireless_table_model.batch_predict(table_items)

    wired_table_items = []
    for table_item in table_items:
        cls_label = table_item["table_res"].get("cls_label")
        cls_score = table_item["table_res"].get("cls_score", 0.0)
        if (cls_label == AtomicModel.WirelessTable and cls_score < 0.9) or cls_label == AtomicModel.WiredTable:
            wired_table_items.append(table_item)
        table_item["table_res"].pop("cls_label", None)
        table_item["table_res"].pop("cls_score", None)

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

    for table_item in table_items:
        html_code = table_item["table_res"].get("html", "") or ""
        if "<table>" in html_code and "</table>" in html_code:
            start_index = html_code.find("<table>")
            end_index = html_code.rfind("</table>") + len("</table>")
            table_item["table_res"]["html"] = html_code[start_index:end_index]


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
    inline_formula_enable: bool,
    _ocr_enable: bool,
    batch_ratio: int,
) -> tuple[list[list[dict[str, Any]]], HybridLocalModelContext]:
    """Hybrid medium 路径：单文件窗口内执行本地 layout/OCR/table/formula，不跨文件 batch。"""
    local_context_singleton = HybridLocalModelContextSingleton()
    local_context = local_context_singleton.get_model(
        lang=language,
        formula_enable=inline_formula_enable,
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
        inline_formula_enable,
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
    _normalize_bbox([[] for _ in images_pil_list], ocr_res_list, images_pil_list)
    merged_model_list = _merge_page_sidecar_items(
        medium_model_list,
        [[] for _ in images_pil_list],
        ocr_res_list,
    )
    return merged_model_list, local_context


def _process_ocr_and_formulas(
    images_pil_list: list[Image.Image],
    model_list: list[list[dict[str, Any]]],
    language: str | None,
    inline_formula_enable: bool,
    _ocr_enable: bool,
    batch_ratio: int = 1,
) -> tuple[list[list[dict[str, Any]]], HybridLocalModelContext]:
    """处理OCR和公式识别"""

    # 遍历model_list,对文本块截图交由OCR识别
    # 根据_ocr_enable决定ocr只开det还是det+rec
    # 根据inline_formula_enable决定是使用mfd和ocr结合的方式,还是纯ocr方式

    # 将PIL图片转换为numpy数组
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]

    # 获取混合模型实例
    local_context_singleton = HybridLocalModelContextSingleton()
    local_context = local_context_singleton.get_model(
        lang=language,
        formula_enable=inline_formula_enable,
    )

    # 在进行`行内`公式检测和识别前，先将图像中的图片、表格、`行间`公式区域mask掉
    layout_images = _mask_image_regions(np_images, model_list) if inline_formula_enable else np_images
    images_layout_res = _predict_layout_for_title_split(local_context, layout_images, batch_ratio)

    if inline_formula_enable:
        images_mfd_res = _build_inline_formula_inputs(images_layout_res)
        # 公式识别
        inline_formula_list = run_mfr_inference(
            local_context.mfr_model.batch_predict,
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
        need_ocr_list = []
        img_crop_list = []
        for page_ocr_res_list in ocr_res_list:
            for ocr_res in page_ocr_res_list:
                if "np_img" in ocr_res:
                    need_ocr_list.append((page_ocr_res_list, ocr_res))
                    img_crop_list.append(ocr_res.pop("np_img"))
        if len(img_crop_list) > 0:
            # Process OCR
            ocr_result_list = run_ocr_inference(
                local_context.ocr_model.ocr,
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
    return merged_model_list, local_context


def _extract_high_with_local_layout(
    predictor: Any,
    images_pil_list: list[Image.Image],
    language: str | None,
    inline_formula_enable: bool,
    _ocr_enable: bool,
    batch_ratio: int,
) -> tuple[list[list[dict[str, Any]]], HybridLocalModelContext]:
    """Hybrid high 路径：用本地 layout 小模型约束 VLM 抽取，再补 OCR/formula sidecar。"""
    local_context_singleton = HybridLocalModelContextSingleton()
    local_context = local_context_singleton.get_model(
        lang=language,
        formula_enable=inline_formula_enable,
    )
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    images_layout_res = _predict_layout_for_title_split(local_context, np_images, batch_ratio)
    layout_blocks = _build_high_layout_blocks(images_layout_res, images_pil_list)
    window_model_list = predictor.batch_extract_with_layout(
        images=images_pil_list,
        blocks_list=layout_blocks,
        not_extract_list=list(NOT_EXTRACT_TYPES),
        image_analysis=False,
    )
    window_model_list, local_context = _process_ocr_and_formulas(
        images_pil_list,
        window_model_list,
        language,
        inline_formula_enable,
        _ocr_enable,
        batch_ratio=batch_ratio,
    )
    return window_model_list, local_context


async def _aio_extract_high_with_local_layout(
    predictor: Any,
    images_pil_list: list[Image.Image],
    language: str | None,
    inline_formula_enable: bool,
    _ocr_enable: bool,
    batch_ratio: int,
) -> tuple[list[list[dict[str, Any]]], HybridLocalModelContext]:
    """Hybrid high 异步路径：异步调用 VLM 的本地 layout 约束抽取，其余本地步骤放线程执行。"""
    local_context_singleton = HybridLocalModelContextSingleton()
    local_context = local_context_singleton.get_model(
        lang=language,
        formula_enable=inline_formula_enable,
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
        not_extract_list=list(NOT_EXTRACT_TYPES),
        image_analysis=False,
    )
    window_model_list, local_context = await asyncio.to_thread(
        _process_ocr_and_formulas,
        images_pil_list,
        window_model_list,
        language,
        inline_formula_enable,
        _ocr_enable,
        batch_ratio,
    )
    return window_model_list, local_context


def _apply_layout_title_split_for_window(
    images_pil_list: list[Image.Image],
    model_list: list[list[dict[str, Any]]],
    language: str | None,
    batch_ratio: int,
) -> HybridLocalModelContext:
    """为VLM-OCR路径补跑layout小模型，先基于VLM原始title做OCR det，再拆分标题。"""
    local_context_singleton = HybridLocalModelContextSingleton()
    local_context = local_context_singleton.get_model(
        lang=language,
        formula_enable=False,
    )
    images_layout_res = _predict_layout_for_title_split(
        local_context,
        images_pil_list,
        batch_ratio,
    )
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    ocr_res_list = _ocr_det(
        local_context,
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
    return local_context


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


def _should_enable_vlm_ocr(ocr_enable: bool, language: str, inline_formula_enable: bool) -> bool:
    """判断 extra_high effort 是否启用 VLM OCR 两阶段抽取。"""
    force_enable = os.getenv("MINERU_FORCE_VLM_OCR_ENABLE", "0").lower() in ("1", "true", "yes")
    if force_enable:
        return True

    return ocr_enable and language in ["ch", "en"] and inline_formula_enable


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
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    effort: Literal["medium", "high", "extra_high"] = DEFAULT_HYBRID_EFFORT,
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    image_cache: ImagePayloadCache | None = None,
    **kwargs: object,
) -> tuple[list[PageInfo], list[list[dict[str, Any]]], bool]:
    client_side_output_generation = bool(kwargs.pop("client_side_output_generation", False))
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
    _vlm_ocr_enable = effort == MAX_HYBRID_EFFORT and _should_enable_vlm_ocr(
        _ocr_enable,
        language,
        inline_formula_enable,
    )

    middle_json: list[PageInfo] = []
    model_list: list[list[dict[str, Any]]] = []
    doc_closed = False
    local_context = None
    try:
        page_count = pdf_doc.page_count
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
                images_list = load_images_from_pdf_bytes_range(
                    pdf_bytes=pdf_bytes,
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
                    if effort == LOCAL_HYBRID_EFFORT:
                        window_model_list, local_context = _extract_with_local_layout(
                            images_pil_list,
                            language,
                            inline_formula_enable,
                            _ocr_enable,
                            batch_ratio,
                        )
                    elif effort == LAYOUT_HYBRID_EFFORT:
                        with vlm_runtime["predictor_execution_guard"](predictor):
                            window_model_list, local_context = _extract_high_with_local_layout(
                                predictor,
                                images_pil_list,
                                language,
                                inline_formula_enable,
                                _ocr_enable,
                                batch_ratio,
                            )
                    elif _vlm_ocr_enable:
                        with vlm_runtime["predictor_execution_guard"](predictor):
                            window_model_list = predictor.batch_two_step_extract(
                                images=images_pil_list,
                                image_analysis=image_analysis,
                            )
                        local_context = _apply_layout_title_split_for_window(
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
                        page_cvt_fn=blocks_to_page_info,
                        page_start_index=window_start,
                        page_index_map=page_index_map,
                        _ocr_enable=_ocr_enable,
                        _vlm_ocr_enable=_vlm_ocr_enable,
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

        if client_side_output_generation:
            apply_server_side_postprocess(
                middle_json,
                local_context,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        else:
            finalize_middle_json(
                middle_json,
                local_context,
                _ocr_enable,
                _vlm_ocr_enable,
                effort=effort,
            )
        pdf_doc.close()
        doc_closed = True
        clean_memory(device)
        return middle_json, model_list, _vlm_ocr_enable
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
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    effort: Literal["medium", "high", "extra_high"] = DEFAULT_HYBRID_EFFORT,
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    image_cache: ImagePayloadCache | None = None,
    **kwargs: object,
) -> tuple[list[PageInfo], list[list[dict[str, Any]]], bool]:
    client_side_output_generation = bool(kwargs.pop("client_side_output_generation", False))
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
    _vlm_ocr_enable = effort == MAX_HYBRID_EFFORT and _should_enable_vlm_ocr(
        _ocr_enable,
        language,
        inline_formula_enable,
    )

    middle_json: list[PageInfo] = []
    model_list = []
    doc_closed = False
    local_context = None
    try:
        page_count = pdf_doc.page_count
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
                    if effort == LOCAL_HYBRID_EFFORT:
                        window_model_list, local_context = await asyncio.to_thread(
                            _extract_with_local_layout,
                            images_pil_list,
                            language,
                            inline_formula_enable,
                            _ocr_enable,
                            batch_ratio,
                        )
                    elif effort == LAYOUT_HYBRID_EFFORT:
                        async with vlm_runtime["aio_predictor_execution_guard"](predictor):
                            window_model_list, local_context = await _aio_extract_high_with_local_layout(
                                predictor,
                                images_pil_list,
                                language,
                                inline_formula_enable,
                                _ocr_enable,
                                batch_ratio,
                            )
                    elif _vlm_ocr_enable:
                        async with vlm_runtime["aio_predictor_execution_guard"](predictor):
                            window_model_list = await predictor.aio_batch_two_step_extract(
                                images=images_pil_list,
                                image_analysis=image_analysis,
                            )
                        local_context = await asyncio.to_thread(
                            _apply_layout_title_split_for_window,
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
                        page_cvt_fn=blocks_to_page_info,
                        page_start_index=window_start,
                        page_index_map=page_index_map,
                        _ocr_enable=_ocr_enable,
                        _vlm_ocr_enable=_vlm_ocr_enable,
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

        if client_side_output_generation:
            await asyncio.to_thread(
                apply_server_side_postprocess,
                middle_json,
                local_context,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        else:
            await asyncio.to_thread(
                finalize_middle_json,
                middle_json,
                local_context,
                _ocr_enable,
                _vlm_ocr_enable,
                effort=effort,
            )
        pdf_doc.close()
        doc_closed = True
        clean_memory(device)
        return middle_json, model_list, _vlm_ocr_enable
    finally:
        if not doc_closed:
            pdf_doc.close()
