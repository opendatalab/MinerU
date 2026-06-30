# Copyright (c) Opendatalab. All rights reserved.

from typing import Any

import cv2
import numpy as np
from loguru import logger

from ...types import BBox
from ...utils.pdf_image_tools import get_crop_img

OCR_DET_PADDING = 50


def _get_ch_ocr_det_model() -> Any:
    """获取默认中文 OCR 检测模型，当前 ch 已对应轻量 PP-OCRv6 配置。"""
    try:
        from ..pipeline.model_init import AtomModelSingleton
    except Exception as e:
        logger.error(
            "Failed to import AtomModelSingleton, OCR detection will not work. If you want to use OCR features, "
            "please execute `pip install mineru[core]` to install the required packages."
        )
        raise e

    atom_model_manager = AtomModelSingleton()

    return atom_model_manager.get_atom_model(
        atom_model_name="ocr",
        ocr_show_log=False,
        det_db_box_thresh=0.3,
        lang="ch",
    )


def _detect_ocr_boxes_from_padded_crop(
    bbox: BBox,
    page_pil_img: Any,
    scale: float,
    ocr_model: Any = None,
    padding: int = OCR_DET_PADDING,
) -> tuple[list[dict[str, Any]], int]:
    if not bbox:
        return [], padding

    crop_pil_img = get_crop_img(bbox, page_pil_img, scale)
    crop_np_img = np.array(crop_pil_img)
    if crop_np_img.size == 0:
        return [], padding

    if padding > 0:
        crop_np_img = cv2.copyMakeBorder(
            crop_np_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

    crop_img = cv2.cvtColor(crop_np_img, cv2.COLOR_RGB2BGR)
    if ocr_model is None:
        ocr_model = _get_ch_ocr_det_model()

    ocr_det_res = ocr_model.ocr(crop_img, rec=False)[0]
    return ocr_det_res or [], padding
