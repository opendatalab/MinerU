# Copyright (c) Opendatalab. All rights reserved.
"""PP-DocLayoutV2 的后处理与可视化逻辑（与推理引擎无关）。

本模块只依赖 numpy / cv2 / PIL，不依赖 torch / transformers，
供 transformers 版（`pp_doclayoutv2.py`）和 ONNX 版（`pp_doclayout_v2_onnx.py`）
共同复用，保证两个后端的输出后处理完全一致。
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...utils.bbox_utils import normalize_to_int_bbox

DEFAULT_IMAGE_SIZE = (800, 800)
DEFAULT_RESCALE_FACTOR = 1.0 / 255.0

PP_DOCLAYOUT_V2_LABELS = [
    "abstract",  # 0 论文摘要
    "algorithm",  # 1 算法
    "aside_text",  # 2 页边注文本，通常位于页面边缘，提供补充信息或注释，与主内容相关但不直接包含在内
    "chart",  # 3 图表，通常包含数据可视化元素，如柱状图、折线图、饼图等，用于展示数据关系和趋势
    "content",  # 4 只在大的目录块中出现，其他地方未见
    "display_formula",  # 5 独立展示的公式，通常占据整行或多行，具有较大字体和清晰的布局，以突出其重要性和可读性
    "doc_title",  # 6 文章标题，一篇文章的主标题
    "figure_title",  # 7 image/chart/table的caption
    "footer",  # 8 页脚文本
    "footer_image",  # 9 页脚图片
    "footnote",  # 10 page footnote，通常位于页面底部，提供对正文中特定内容的补充说明、引用来源或其他相关信息
    "formula_number",  # 11 公式编号，通常与display_formula配合使用，标识独立展示的公式在文档中的位置和顺序，便于引用和索引
    "header",  # 12 页眉文本
    "header_image",  # 13 页眉图片
    "image",  # 14 图片
    "inline_formula",  # 15 行内公式
    "number",  # 16 页码
    "paragraph_title",  # 17 段落标题，有别与文章标题
    "reference",  # 18 参考文献，list外框
    "reference_content",  # 19 参考文献内容，list item
    "seal",  # 20 印章
    "table",  # 21 表格
    "text",  # 22 一般文本
    "vertical_text",  # 23 竖排文本
    "vision_footnote",  # 24 image/chart/table的footnote
]

PP_DOCLAYOUT_V2_LABEL_TO_ID = {label: index for index, label in enumerate(PP_DOCLAYOUT_V2_LABELS)}

# Per-class confidence threshold used before reading-order decoding.
DEFAULT_CLASS_THRESHOLDS = [
    0.5,  # 0  abstract
    0.5,  # 1  algorithm
    0.5,  # 2  aside_text
    0.5,  # 3  chart
    0.5,  # 4  content
    0.4,  # 5  display_formula
    0.4,  # 6  doc_title
    0.5,  # 7  figure_title
    0.5,  # 8  footer
    0.5,  # 9  footer_image
    0.5,  # 10 footnote
    0.5,  # 11 formula_number
    0.5,  # 12 header
    0.5,  # 13 header_image
    0.5,  # 14 image
    0.4,  # 15 inline_formula
    0.5,  # 16 number
    0.4,  # 17 paragraph_title
    0.5,  # 18 reference
    0.5,  # 19 reference_content
    0.45,  # 20 seal
    0.5,  # 21 table
    0.4,  # 22 text
    0.4,  # 23 vertical_text
    0.5,  # 24 vision_footnote
]

# Reading-order head class remap used by the original upstream model.
DEFAULT_CLASS_ORDER = [
    4,  # 0  abstract
    2,  # 1  algorithm
    14,  # 2  aside_text
    1,  # 3  chart
    5,  # 4  content
    7,  # 5  display_formula
    8,  # 6  doc_title
    6,  # 7  figure_title
    11,  # 8  footer
    11,  # 9  footer
    9,  # 10 footnote
    13,  # 11 formula_number
    10,  # 12 header
    10,  # 13 header_image
    1,  # 14 image
    2,  # 15 inline_formula
    3,  # 16 number
    0,  # 17 paragraph_title
    2,  # 18 reference
    2,  # 19 reference_content
    12,  # 20 seal
    1,  # 21 table
    2,  # 22 text
    15,  # 23 vertical_text
    6,  # 24 vision_footnote
]


def load_preprocess_config(model_dir: str) -> Dict:
    config_path = os.path.join(model_dir, "preprocessor_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_to_color(label: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    saturation = 0.65 + (digest[1] / 255.0) * 0.2
    value = 0.85 + (digest[2] / 255.0) * 0.1
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


class PPDocLayoutV2PostProcessor:
    """PP-DocLayoutV2 检测结果的后处理逻辑。

    所有方法都是纯 numpy/Python 实现，不依赖 torch/transformers。
    子类只需提供 ``conf`` / ``use_paddlex_filter_boxes`` 等实例属性，
    并实现 ``predict`` / ``batch_predict`` 即可。
    """

    HEADER_FOOTER_BOUNDARY_EXEMPT_LABELS = {"aside_text", "footnote", "number"}
    VISUAL_BODY_LABELS = {"image", "chart", "table", "seal"}
    PAGE_REGION_LABELS = {
        "header",
        "header_image",
        "footer",
        "footer_image",
        "footnote",
        "number",
        "aside_text",
    }

    # ------------------------------------------------------------------
    # bbox 几何计算
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_bbox(box: Sequence[float], image_size: Tuple[int, int]) -> Optional[List[int]]:
        return normalize_to_int_bbox(box, image_size=image_size)

    @staticmethod
    def _calculate_bbox_area(box: Sequence[float]) -> float:
        xmin, ymin, xmax, ymax = [float(v) for v in box]
        return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

    @classmethod
    def _calculate_intersection_area(cls, box1: Sequence[float], box2: Sequence[float]) -> float:
        x1_min, y1_min, x1_max, y1_max = [float(v) for v in box1]
        x2_min, y2_min, x2_max, y2_max = [float(v) for v in box2]
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        return cls._calculate_bbox_area((inter_xmin, inter_ymin, inter_xmax, inter_ymax))

    @classmethod
    def _calculate_overlap_ratio(cls, box1: Sequence[float], box2: Sequence[float]) -> float:
        inter_area = cls._calculate_intersection_area(box1, box2)
        ref_area = min(cls._calculate_bbox_area(box1), cls._calculate_bbox_area(box2))
        if ref_area <= 0.0:
            return 0.0
        return inter_area / ref_area

    @classmethod
    def _calculate_iou(cls, box1: Sequence[float], box2: Sequence[float]) -> float:
        inter_area = cls._calculate_intersection_area(box1, box2)
        union_area = cls._calculate_bbox_area(box1) + cls._calculate_bbox_area(box2) - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    @classmethod
    def _calculate_cover_ratio(cls, box1: Sequence[float], box2: Sequence[float]) -> float:
        box1_area = cls._calculate_bbox_area(box1)
        if box1_area <= 0.0:
            return 0.0
        return cls._calculate_intersection_area(box1, box2) / box1_area

    @staticmethod
    def _calculate_x_overlap_ratio(box1: Sequence[float], box2: Sequence[float]) -> float:
        """计算两个 bbox 在横向上的重叠比例，用于判断是否属于同一栏。"""
        box1_xmin, _, box1_xmax, _ = [float(v) for v in box1]
        box2_xmin, _, box2_xmax, _ = [float(v) for v in box2]
        box1_width = max(0.0, box1_xmax - box1_xmin)
        box2_width = max(0.0, box2_xmax - box2_xmin)
        ref_width = min(box1_width, box2_width)
        if ref_width <= 0.0:
            return 0.0
        overlap_width = max(0.0, min(box1_xmax, box2_xmax) - max(box1_xmin, box2_xmin))
        return overlap_width / ref_width

    @staticmethod
    def _calculate_x_cover_ratio(anchor_box: Sequence[float], candidate_box: Sequence[float]) -> float:
        """计算 anchor 在横向上覆盖 candidate 的比例。"""
        anchor_xmin, _, anchor_xmax, _ = [float(v) for v in anchor_box]
        candidate_xmin, _, candidate_xmax, _ = [float(v) for v in candidate_box]
        candidate_width = max(0.0, candidate_xmax - candidate_xmin)
        if candidate_width <= 0.0:
            return 0.0
        overlap_width = max(
            0.0,
            min(anchor_xmax, candidate_xmax) - max(anchor_xmin, candidate_xmin),
        )
        return overlap_width / candidate_width

    @classmethod
    def _is_footer_x_scope(
        cls,
        anchor_box: Dict,
        candidate_box: Dict,
        image_size: Optional[Tuple[int, int]],
        full_width_threshold: float = 0.7,
        x_overlap_threshold: float = 0.3,
    ) -> bool:
        """判断候选块是否在页脚文本锚点的横向作用范围内。"""
        anchor_bbox = anchor_box.get("bbox")
        candidate_bbox = candidate_box.get("bbox")
        if not anchor_bbox or not candidate_bbox:
            return False

        if image_size is not None and len(image_size) >= 2:
            page_width = float(image_size[1])
            anchor_width = max(0.0, float(anchor_bbox[2]) - float(anchor_bbox[0]))
            if page_width > 0.0 and anchor_width / page_width >= full_width_threshold:
                return True

        return cls._calculate_x_overlap_ratio(anchor_bbox, candidate_bbox) >= x_overlap_threshold

    @classmethod
    def _is_covered_by_footnote(
        cls,
        footnote_box: Dict,
        candidate_box: Dict,
        x_cover_threshold: float = 0.7,
    ) -> bool:
        """判断候选块是否位于 footnote 区域内或其下方，并被其横向覆盖。"""
        footnote_bbox = footnote_box.get("bbox")
        candidate_bbox = candidate_box.get("bbox")
        if not footnote_bbox or not candidate_bbox:
            return False
        if candidate_bbox[1] < footnote_bbox[1]:
            return False
        return cls._calculate_x_cover_ratio(footnote_bbox, candidate_bbox) >= x_cover_threshold

    # ------------------------------------------------------------------
    # box 类型判断
    # ------------------------------------------------------------------
    @classmethod
    def _is_header_footer_boundary_candidate(cls, box: Dict, anchor_labels: set[str]) -> bool:
        """判断普通块是否可被页眉/页脚/页码边界规则改标。"""
        label = box.get("label")
        if label in cls.HEADER_FOOTER_BOUNDARY_EXEMPT_LABELS:
            return False
        return label not in anchor_labels

    @classmethod
    def _is_footnote_relabel_candidate(cls, box: Dict) -> bool:
        """排除页眉、页脚、页码、页边注等非正文区域块，保留正文内容块。"""
        return box.get("label") not in cls.PAGE_REGION_LABELS

    @staticmethod
    def _is_reference_box(box: Dict) -> bool:
        return box.get("label") == "reference" or int(box.get("cls_id", -1)) == 18

    @staticmethod
    def _is_display_formula_box(box: Dict) -> bool:
        return box.get("label") == "display_formula" or int(box.get("cls_id", -1)) == 5

    @staticmethod
    def _is_inline_formula_box(box: Dict) -> bool:
        return box.get("label") == "inline_formula" or int(box.get("cls_id", -1)) == 15

    @staticmethod
    def _is_formula_box(box: Dict) -> bool:
        return PPDocLayoutV2PostProcessor._is_display_formula_box(box) or PPDocLayoutV2PostProcessor._is_inline_formula_box(box)

    @staticmethod
    def _is_formula_number_box(box: Dict) -> bool:
        return box.get("label") == "formula_number" or int(box.get("cls_id", -1)) == 11

    # ------------------------------------------------------------------
    # label 修改
    # ------------------------------------------------------------------
    @staticmethod
    def _set_box_label(box: Dict, label: str) -> None:
        """统一同步设置 layout 检测框的标签名和类别编号。"""
        if label not in PP_DOCLAYOUT_V2_LABEL_TO_ID:
            raise ValueError(f"Unsupported PP-DocLayoutV2 label: {label}")
        box["label"] = label
        box["cls_id"] = PP_DOCLAYOUT_V2_LABEL_TO_ID[label]

    @staticmethod
    def _set_formula_label(box: Dict, label: str) -> None:
        if label not in {"inline_formula", "display_formula"}:
            raise ValueError(f"Unsupported formula label: {label}")
        PPDocLayoutV2PostProcessor._set_box_label(box, label)

    @staticmethod
    def _set_header_footer_label(box: Dict, label: str) -> None:
        """同步设置页眉/页脚相关标签及其类别编号。"""
        if label not in {"footer", "footer_image", "header", "header_image"}:
            raise ValueError(f"Unsupported header/footer label: {label}")
        PPDocLayoutV2PostProcessor._set_box_label(box, label)

    @staticmethod
    def _set_footnote_label(box: Dict) -> None:
        """同步设置 page footnote 标签及其类别编号。"""
        PPDocLayoutV2PostProcessor._set_box_label(box, "footnote")

    def _label_id_to_label_name(self, label_id: int) -> str:
        if 0 <= label_id < len(PP_DOCLAYOUT_V2_LABELS):
            return PP_DOCLAYOUT_V2_LABELS[label_id]
        # 兜底：调用方可能传入了非预设 id，这里返回字符串形式。
        return str(label_id)

    # ------------------------------------------------------------------
    # 后处理主流程
    # ------------------------------------------------------------------
    @classmethod
    def _filter_internal_visual_caption_boxes(
        cls,
        boxes: List[Dict],
        cover_threshold: float = 0.8,
    ) -> List[Dict]:
        """过滤落在图、表、印章等视觉主体内部的 figure_title。

        这类块通常是图内的 (a)/(b) 标号，不应作为外部 caption 参与后续视觉分组。
        """
        visual_boxes = [box for box in boxes if box.get("label") in cls.VISUAL_BODY_LABELS]
        if not visual_boxes:
            return boxes

        filtered_boxes = []
        for box in boxes:
            if box.get("label") != "figure_title":
                filtered_boxes.append(box)
                continue

            caption_bbox = box.get("bbox")
            if not caption_bbox or len(caption_bbox) < 4:
                filtered_boxes.append(box)
                continue

            caption_center_x = (float(caption_bbox[0]) + float(caption_bbox[2])) / 2
            caption_center_y = (float(caption_bbox[1]) + float(caption_bbox[3])) / 2
            is_internal_caption = False
            for visual_box in visual_boxes:
                visual_bbox = visual_box.get("bbox")
                if not visual_bbox or len(visual_bbox) < 4:
                    continue

                visual_xmin, visual_ymin, visual_xmax, visual_ymax = [float(v) for v in visual_bbox]
                if not (visual_xmin <= caption_center_x <= visual_xmax and visual_ymin <= caption_center_y <= visual_ymax):
                    continue

                if cls._calculate_cover_ratio(caption_bbox, visual_bbox) >= cover_threshold:
                    is_internal_caption = True
                    break

            if not is_internal_caption:
                filtered_boxes.append(box)

        return filtered_boxes

    @classmethod
    def _reclassify_header_footer_by_page_half(
        cls,
        boxes: List[Dict],
        image_size: Optional[Tuple[int, int]],
    ) -> List[Dict]:
        """按页面上下半区重新校正页眉/页脚锚点，避免跨半页误触发边界规则。"""
        if image_size is None:
            return boxes

        page_height = float(image_size[0])
        if page_height <= 0:
            return boxes

        page_middle = page_height * 0.5
        upper_half_labels = {
            "footer": "header",
            "footer_image": "header_image",
        }
        lower_half_labels = {
            "header": "footer",
            "header_image": "footer_image",
        }
        for box in boxes:
            bbox = box.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            label = box.get("label")
            y_mid = (float(bbox[1]) + float(bbox[3])) / 2
            if y_mid < page_middle and label in upper_half_labels:
                cls._set_header_footer_label(box, upper_half_labels[label])
            elif y_mid >= page_middle and label in lower_half_labels:
                cls._set_header_footer_label(box, lower_half_labels[label])

        return boxes

    @staticmethod
    def _union_bbox(box1: Sequence[float], box2: Sequence[float]) -> List[int]:
        x1_min, y1_min, x1_max, y1_max = [float(v) for v in box1]
        x2_min, y2_min, x2_max, y2_max = [float(v) for v in box2]
        return [
            math.floor(min(x1_min, x2_min)),
            math.floor(min(y1_min, y2_min)),
            math.ceil(max(x1_max, x2_max)),
            math.ceil(max(y1_max, y2_max)),
        ]

    @staticmethod
    def _renumber_indices(boxes: List[Dict]) -> List[Dict]:
        for index, box in enumerate(boxes, start=1):
            box["index"] = index
        return boxes

    @classmethod
    def _deduplicate_boxes_by_iou(
        cls,
        boxes: List[Dict],
        iou_threshold: float = 0.9,
    ) -> List[Dict]:
        if len(boxes) <= 1:
            return boxes

        sorted_candidates = sorted(
            enumerate(boxes),
            key=lambda item: (-float(item[1].get("score", 0.0)), item[0]),
        )
        suppressed_indexes = set()
        kept_indexes = []

        for candidate_pos, (current_index, current_box) in enumerate(sorted_candidates):
            if current_index in suppressed_indexes:
                continue
            kept_indexes.append(current_index)
            for other_index, other_box in sorted_candidates[candidate_pos + 1 :]:
                if other_index in suppressed_indexes:
                    continue
                if cls._calculate_iou(current_box["bbox"], other_box["bbox"]) > iou_threshold:
                    suppressed_indexes.add(other_index)

        kept_indexes.sort()
        return [boxes[index] for index in kept_indexes]

    @classmethod
    def _merge_nested_formula_boxes(
        cls,
        boxes: List[Dict],
        overlap_threshold: float = 0.7,
    ) -> List[Dict]:
        if len(boxes) <= 1:
            return boxes

        changed = True
        while changed:
            changed = False
            formula_indexes = [index for index, box in enumerate(boxes) if cls._is_formula_box(box)]
            for formula_pos, left_index in enumerate(formula_indexes):
                for right_index in formula_indexes[formula_pos + 1 :]:
                    left_box = boxes[left_index]
                    right_box = boxes[right_index]
                    if cls._calculate_overlap_ratio(left_box["bbox"], right_box["bbox"]) < overlap_threshold:
                        continue

                    left_area = cls._calculate_bbox_area(left_box["bbox"])
                    right_area = cls._calculate_bbox_area(right_box["bbox"])
                    if left_area > right_area:
                        keep_index, drop_index = left_index, right_index
                    elif right_area > left_area:
                        keep_index, drop_index = right_index, left_index
                    else:
                        left_score = float(left_box.get("score", 0.0))
                        right_score = float(right_box.get("score", 0.0))
                        keep_index, drop_index = (
                            (left_index, right_index) if left_score >= right_score else (right_index, left_index)
                        )

                    keep_box = boxes[keep_index]
                    drop_box = boxes[drop_index]
                    keep_box["bbox"] = cls._union_bbox(keep_box["bbox"], drop_box["bbox"])
                    keep_box["score"] = round(
                        max(float(keep_box.get("score", 0.0)), float(drop_box.get("score", 0.0))),
                        4,
                    )
                    del boxes[drop_index]
                    changed = True
                    break
                if changed:
                    break

        return boxes

    @classmethod
    def _relabel_formula_boxes(
        cls,
        boxes: List[Dict],
        overlap_threshold: float = 0.7,
    ) -> List[Dict]:
        parent_candidates = [
            box
            for box in boxes
            if (not cls._is_formula_box(box) and not cls._is_formula_number_box(box) and not cls._is_reference_box(box))
        ]

        for box in boxes:
            if not cls._is_formula_box(box):
                continue
            target_label = "display_formula"
            for parent_box in parent_candidates:
                if cls._calculate_cover_ratio(box["bbox"], parent_box["bbox"]) >= overlap_threshold:
                    target_label = "inline_formula"
                    break
            cls._set_formula_label(box, target_label)

        return boxes

    @classmethod
    def _relabel_header_footer_boundary_blocks(
        cls,
        boxes: List[Dict],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        """按视觉坐标用页眉/页脚锚点修正边界区域的普通块标签。"""
        if len(boxes) <= 1:
            return boxes

        header_labels = {"header", "header_image"}
        footer_labels = {"footer", "footer_image"}
        ordered_boxes = sorted(boxes, key=lambda box: box["index"])
        ordered_boxes = cls._reclassify_header_footer_by_page_half(
            ordered_boxes,
            image_size=image_size,
        )
        boundary_anchor_ids = {
            id(box) for box in ordered_boxes if box.get("label") in header_labels or box.get("label") in footer_labels
        }

        header_anchor = max(
            (box for box in ordered_boxes if box.get("label") in header_labels),
            key=lambda box: (box["bbox"][3], box["index"]),
            default=None,
        )
        footer_anchor = min(
            (box for box in ordered_boxes if box.get("label") in footer_labels),
            key=lambda box: (box["bbox"][1], box["index"]),
            default=None,
        )

        # 先按最后一个页眉锚点的下边界修正，后续页脚修正可覆盖重叠区间。
        if header_anchor is not None:
            header_boundary = header_anchor["bbox"][3]
            for box in ordered_boxes:
                if not cls._is_header_footer_boundary_candidate(box, header_labels):
                    continue
                if box["bbox"][3] <= header_boundary:
                    cls._set_box_label(box, "header")

        footnote_anchors = [box for box in ordered_boxes if box.get("label") == "footnote"]
        if footnote_anchors:
            for box in ordered_boxes:
                if not cls._is_footnote_relabel_candidate(box):
                    continue
                for footnote_anchor in footnote_anchors:
                    if cls._is_covered_by_footnote(footnote_anchor, box):
                        cls._set_footnote_label(box)
                        break

        if footer_anchor is not None:
            footer_boundary = footer_anchor["bbox"][1]
            for box in ordered_boxes:
                if not cls._is_header_footer_boundary_candidate(box, footer_labels):
                    continue
                if box["bbox"][1] >= footer_boundary and cls._is_footer_x_scope(footer_anchor, box, image_size):
                    cls._set_box_label(box, "footer")

        if image_size is None:
            return ordered_boxes

        page_height = float(image_size[0])
        if page_height <= 0:
            return ordered_boxes

        top_boundary = page_height * 0.3
        bottom_boundary = page_height * 0.7
        top_numbers = []
        bottom_numbers = []
        for box in ordered_boxes:
            if box.get("label") != "number":
                continue
            y_mid = (float(box["bbox"][1]) + float(box["bbox"][3])) / 2
            if y_mid <= top_boundary:
                top_numbers.append(box)
            elif y_mid >= bottom_boundary:
                bottom_numbers.append(box)

        top_number_anchor = max(
            top_numbers,
            key=lambda box: (box["bbox"][3], box["index"]),
            default=None,
        )
        bottom_number_anchor = min(
            bottom_numbers,
            key=lambda box: (box["bbox"][1], box["index"]),
            default=None,
        )

        # number 自身不改标签，仅用上下 30% 区域中的 number 作为辅助分割线。
        if top_number_anchor is not None:
            header_boundary = top_number_anchor["bbox"][1]
            for box in ordered_boxes:
                if id(box) in boundary_anchor_ids or not cls._is_header_footer_boundary_candidate(box, set()):
                    continue
                if box["bbox"][3] <= header_boundary:
                    cls._set_box_label(box, "header")

        if bottom_number_anchor is not None:
            footer_boundary = bottom_number_anchor["bbox"][3]
            for box in ordered_boxes:
                if id(box) in boundary_anchor_ids or not cls._is_header_footer_boundary_candidate(box, set()):
                    continue
                if box["bbox"][1] >= footer_boundary:
                    cls._set_box_label(box, "footer")

        return ordered_boxes

    @classmethod
    def _apply_layout_post_process(
        cls,
        boxes: List[Dict],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        processed_boxes = [{**box, "bbox": list(box["bbox"])} for box in boxes]
        processed_boxes = cls._deduplicate_boxes_by_iou(processed_boxes, iou_threshold=0.9)
        processed_boxes = cls._merge_nested_formula_boxes(processed_boxes, overlap_threshold=0.7)
        processed_boxes = cls._relabel_formula_boxes(processed_boxes, overlap_threshold=0.7)
        processed_boxes = cls._relabel_header_footer_boundary_blocks(
            processed_boxes,
            image_size=image_size,
        )
        processed_boxes = cls._filter_internal_visual_caption_boxes(
            processed_boxes,
            cover_threshold=0.8,
        )
        return cls._renumber_indices(processed_boxes)

    @classmethod
    def _apply_paddlex_filter_boxes(
        cls,
        boxes: List[Dict],
        drop_inline_formula: bool = True,
    ) -> List[Dict]:
        filtered_boxes = [dict(box) for box in boxes if not cls._is_reference_box(box)]
        dropped_indexes = set()

        for i in range(len(filtered_boxes)):
            if i in dropped_indexes:
                continue
            x1, y1, x2, y2 = filtered_boxes[i]["bbox"]
            width = float(x2) - float(x1)
            height = float(y2) - float(y1)
            if (width < 6.0 or height < 6.0) and (drop_inline_formula or not cls._is_inline_formula_box(filtered_boxes[i])):
                dropped_indexes.add(i)
                continue

            for j in range(i + 1, len(filtered_boxes)):
                if i in dropped_indexes or j in dropped_indexes:
                    continue

                if not drop_inline_formula and (
                    cls._is_inline_formula_box(filtered_boxes[i]) or cls._is_inline_formula_box(filtered_boxes[j])
                ):
                    continue

                overlap_ratio = cls._calculate_overlap_ratio(
                    filtered_boxes[i]["bbox"],
                    filtered_boxes[j]["bbox"],
                )
                if drop_inline_formula and (
                    cls._is_inline_formula_box(filtered_boxes[i]) or cls._is_inline_formula_box(filtered_boxes[j])
                ):
                    if overlap_ratio > 0.5:
                        if cls._is_inline_formula_box(filtered_boxes[i]):
                            dropped_indexes.add(i)
                        if cls._is_inline_formula_box(filtered_boxes[j]):
                            dropped_indexes.add(j)
                        continue

                if overlap_ratio > 0.7:
                    box_area_i = cls._calculate_bbox_area(filtered_boxes[i]["bbox"])
                    box_area_j = cls._calculate_bbox_area(filtered_boxes[j]["bbox"])
                    labels = {filtered_boxes[i]["label"], filtered_boxes[j]["label"]}
                    if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                        if "table" not in labels or labels <= {"table", "image", "seal", "chart"}:
                            continue
                    if box_area_i >= box_area_j:
                        dropped_indexes.add(j)
                    else:
                        dropped_indexes.add(i)

        kept_boxes = [box for index, box in enumerate(filtered_boxes) if index not in dropped_indexes]
        return cls._renumber_indices(kept_boxes)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def visualize(
        self,
        image: Union[np.ndarray, Image.Image],
        results: List[Dict],
    ) -> Image.Image:
        """在图像上绘制检测结果，返回 PIL.Image。"""
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB").copy()
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB").copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()
        for res in sorted(
            results,
            key=lambda item: (item.get("index", 10**9), item.get("bbox", [0, 0, 0, 0])[1]),
        ):
            xmin, ymin, xmax, ymax = res["bbox"]
            color = label_to_color(res["label"])
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

            text = f"{res['index']}: {res['label']} {res['score']:.2f}"
            text_top = int(round(ymin))
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            pad = 3

            if text_top - text_height - pad * 2 >= 0:
                text_bg_top = text_top - text_height - pad * 2
            else:
                text_bg_top = text_top
            text_bg_bottom = text_bg_top + text_height + pad * 2
            text_bg_right = int(round(xmax))
            text_bg_left = text_bg_right - text_width - pad * 2

            draw.rectangle(
                [text_bg_left, text_bg_top, text_bg_right, text_bg_bottom],
                fill=color,
            )
            draw.text(
                (text_bg_left + pad, text_bg_top + pad),
                text,
                fill="white",
                font=font,
            )
        return pil_image


__all__ = [
    "DEFAULT_CLASS_ORDER",
    "DEFAULT_CLASS_THRESHOLDS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_RESCALE_FACTOR",
    "PP_DOCLAYOUT_V2_LABEL_TO_ID",
    "PP_DOCLAYOUT_V2_LABELS",
    "PPDocLayoutV2PostProcessor",
    "label_to_color",
    "load_preprocess_config",
]
