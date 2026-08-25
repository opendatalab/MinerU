# Copyright (c) Opendatalab. All rights reserved.
"""OCR 原始输出到 Analyze 检测结果的坐标回投。"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ...utils.geometry import normalize_to_int_bbox
from .geometry import calculate_is_angle
from .image import get_rotate_crop_image_for_text_rec


class OcrConfidence:
    """集中保存 OCR 结果过滤阈值。"""

    min_confidence = 0.5
    min_width = 3


def get_adjusted_mfdetrec_res(
    single_page_mfdetrec_res: list[dict[str, Any]],
    useful_list: list[int],
) -> list[dict[str, Any]]:
    """把整页公式框换算为当前 OCR 裁图坐标。"""
    paste_x, paste_y, xmin, ymin, _xmax, _ymax, new_width, new_height = useful_list
    adjusted_mfdetrec_res = []
    for mf_res in single_page_mfdetrec_res:
        mf_xmin, mf_ymin, mf_xmax, mf_ymax = mf_res["bbox"]
        x0 = mf_xmin - xmin + paste_x
        y0 = mf_ymin - ymin + paste_y
        x1 = mf_xmax - xmin + paste_x
        y1 = mf_ymax - ymin + paste_y
        if any([x1 < 0, y1 < 0]) or any([x0 > new_width, y0 > new_height]):
            continue
        adjusted_mfdetrec_res.append({"bbox": [x0, y0, x1, y1]})
    return adjusted_mfdetrec_res


def get_ocr_result_list(
    ocr_res: list[Any],
    useful_list: list[int],
    ocr_enable: bool,
    bgr_image: np.ndarray,
) -> list[dict[str, Any]]:
    """过滤 OCR 输出并将坐标回投到原始页面。"""
    paste_x, paste_y, xmin, ymin, _xmax, _ymax, _new_width, _new_height = useful_list
    ocr_result_list = []
    ori_im = bgr_image.copy()
    for box_ocr_res in ocr_res:
        img_crop = None
        need_ocr_rec = False
        if len(box_ocr_res) == 2:
            p1, p2, p3, p4 = box_ocr_res[0]
            text, score = box_ocr_res[1]
            if score < OcrConfidence.min_confidence:
                continue
        else:
            p1, p2, p3, p4 = box_ocr_res
            text, score = "", 1
            if ocr_enable:
                tmp_box = copy.deepcopy(np.array([p1, p2, p3, p4]).astype("float32"))
                img_crop = get_rotate_crop_image_for_text_rec(ori_im, tmp_box)
                need_ocr_rec = True

        poly = [p1, p2, p3, p4]
        if (p3[0] - p1[0]) < OcrConfidence.min_width:
            continue
        if calculate_is_angle(poly):
            x_center = sum(point[0] for point in poly) / 4
            y_center = sum(point[1] for point in poly) / 4
            new_height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
            new_width = p3[0] - p1[0]
            p1 = [x_center - new_width / 2, y_center - new_height / 2]
            p2 = [x_center + new_width / 2, y_center - new_height / 2]
            p3 = [x_center + new_width / 2, y_center + new_height / 2]
            p4 = [x_center - new_width / 2, y_center + new_height / 2]

        p1 = [p1[0] - paste_x + xmin, p1[1] - paste_y + ymin]
        p2 = [p2[0] - paste_x + xmin, p2[1] - paste_y + ymin]
        p3 = [p3[0] - paste_x + xmin, p3[1] - paste_y + ymin]
        p4 = [p4[0] - paste_x + xmin, p4[1] - paste_y + ymin]
        bbox = normalize_to_int_bbox([p1, p2, p3, p4])
        if bbox is None:
            continue
        ocr_item = {
            "label": "ocr_text",
            "bbox": bbox,
            "score": 1.0 if ocr_enable else float(round(score, 2)),
            "text": text,
        }
        if need_ocr_rec:
            ocr_item["np_img"] = img_crop
        ocr_result_list.append(ocr_item)
    return ocr_result_list


__all__ = ["OcrConfidence", "get_adjusted_mfdetrec_res", "get_ocr_result_list"]
