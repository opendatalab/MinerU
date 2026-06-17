# Copyright (c) Opendatalab. All rights reserved.
"""表格内联对象与 OCR 结果处理的纯工具函数。

这些函数原本属于 BatchAnalyze，不依赖 torch/onnxruntime 等重型推理库，
提取到独立模块后可独立进行单元测试与复用。
"""
import base64
import html
from collections import defaultdict

import cv2
import numpy as np

from .bbox_utils import normalize_to_int_bbox

TABLE_OCR_REC_SINGLE_CHAR_REPLACEMENTS = {
    "香": "否",
    "哦樂": "哦",
}
TABLE_OCR_REC_REGEX_REPLACEMENTS = (
    # 仅规范化完整的“单个数字 + 號”，避免影响“10號”“第6號”等普通文本。
    (__import__("re").compile(r"^([0-9])號$"), r"\1"),
)


def sort_table_ocr_result(ocr_result: list[list]) -> None:
    """按表格单元格空间位置对 OCR 结果排序。

    先按行分组（10px 容差），同一行内按 x 升序排列。
    """
    if not ocr_result:
        return

    ocr_result.sort(
        key=lambda item: (
            int(float(np.asarray(item[0])[0][1]) / 10),
            float(np.asarray(item[0])[0][0]),
        )
    )


def normalize_table_ocr_rec_text(text):
    """规范化表格 OCR rec 的已知误识别，避免后续表格模型消费错误文本。"""
    if not isinstance(text, str):
        return text
    if text in TABLE_OCR_REC_SINGLE_CHAR_REPLACEMENTS:
        return TABLE_OCR_REC_SINGLE_CHAR_REPLACEMENTS[text]
    for pattern, replacement in TABLE_OCR_REC_REGEX_REPLACEMENTS:
        match = pattern.fullmatch(text)
        if match:
            return match.expand(replacement)
    return text


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def is_point_in_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def bbox_intersection(bbox1: list[float], bbox2: list[float]) -> list[float] | None:
    x0 = max(float(bbox1[0]), float(bbox2[0]))
    y0 = max(float(bbox1[1]), float(bbox2[1]))
    x1 = min(float(bbox1[2]), float(bbox2[2]))
    y1 = min(float(bbox1[3]), float(bbox2[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def bbox_intersection_area(bbox1: list[float], bbox2: list[float]) -> float:
    overlap_bbox = bbox_intersection(bbox1, bbox2)
    if overlap_bbox is None:
        return 0.0
    return float(overlap_bbox[2] - overlap_bbox[0]) * float(overlap_bbox[3] - overlap_bbox[1])


def bbox_to_relative_bbox(bbox: list[float], base_bbox: list[float]) -> list[float]:
    return [
        float(bbox[0]) - float(base_bbox[0]),
        float(bbox[1]) - float(base_bbox[1]),
        float(bbox[2]) - float(base_bbox[0]),
        float(bbox[3]) - float(base_bbox[1]),
    ]


def bbox_to_quad(bbox: list[float]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def encode_table_inline_image(np_img: np.ndarray, bbox: list[float]) -> str:
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
    return f"data:image/jpg;base64,{b64_str}"


def get_virtual_image_bbox(bbox: list[float], box_size: float = 10.0) -> list[float]:
    center_x, center_y = bbox_center(bbox)
    half_size = box_size / 2.0
    return [
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size,
    ]


def table_supports_inline_objects(table_res_dict: dict) -> bool:
    return str(table_res_dict.get("rotate_label", "0")) == "0"


def extract_table_inline_objects(
    layout_res: list[dict],
    np_img: np.ndarray,
    formula_enable: bool,
) -> dict[int, list[dict]]:
    """从 layout 结果中提取位于表格内部的内联图片与公式对象。"""
    image_h, image_w = np_img.shape[:2]
    image_size = (image_h, image_w)

    tables = []
    for res in layout_res:
        if res.get("label") != "table":
            continue
        table_bbox = normalize_to_int_bbox(res.get("bbox"), image_size=image_size)
        if table_bbox is None:
            continue
        tables.append((res, table_bbox))

    if not tables:
        return {}

    # 构建网格空间索引，加速 item 与 table 的匹配
    grid_size = max(50, min(image_h, image_w) // 20)
    grid_size = max(1, grid_size)
    table_grid = defaultdict(list)
    for idx, (_, table_bbox) in enumerate(tables):
        x0, y0, x1, y1 = table_bbox
        for gx in range(int(x0) // grid_size, int(x1) // grid_size + 1):
            for gy in range(int(y0) // grid_size, int(y1) // grid_size + 1):
                table_grid[(gx, gy)].append(idx)

    table_inline_objects = {id(table_res): [] for table_res, _ in tables}
    remove_ids = set()
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

        item_center = bbox_center(item_bbox)
        cx, cy = item_center
        cell = (int(cx) // grid_size, int(cy) // grid_size)
        candidate_indices = table_grid.get(cell, [])

        matched_tables = []
        for idx in candidate_indices:
            table_res, table_bbox = tables[idx]
            if not is_point_in_bbox(item_center, table_bbox):
                continue
            overlap_area = bbox_intersection_area(item_bbox, table_bbox)
            matched_tables.append((overlap_area, table_res, table_bbox))

        if not matched_tables:
            continue

        matched_tables.sort(key=lambda item: item[0], reverse=True)
        _, table_res, table_bbox = matched_tables[0]
        overlap_bbox = bbox_intersection(item_bbox, table_bbox)
        if overlap_bbox is None:
            continue

        rel_overlap_bbox = bbox_to_relative_bbox(overlap_bbox, table_bbox)
        score = float(layout_item.get("score", 1.0))

        if label == "image":
            image_src = encode_table_inline_image(np_img, item_bbox)
            if not image_src:
                continue
            content = f'<img src="{image_src}"/>'
            token_bbox = get_virtual_image_bbox(rel_overlap_bbox)
            kind = "image"
        else:
            latex = layout_item.get("latex", "")
            if not latex:
                continue
            content = f"<eq>{html.escape(latex)}</eq>"
            token_bbox = rel_overlap_bbox
            kind = "formula"

        table_inline_objects[id(table_res)].append(
            {
                "kind": kind,
                "page_bbox": item_bbox,
                "table_rel_mask_bbox": rel_overlap_bbox,
                "table_token_bbox": token_bbox,
                "content": content,
                "score": score,
            }
        )
        remove_ids.add(id(layout_item))

    if remove_ids:
        layout_res[:] = [item for item in layout_res if id(item) not in remove_ids]

    return table_inline_objects
