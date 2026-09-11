# Copyright (c) Opendatalab. All rights reserved.
"""复用 Transformers 的官方 PP-DocLayoutV2，保留 MinerU 的预处理和版面后处理。"""

import argparse
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from torch import nn
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import PPDocLayoutV2Config, PPDocLayoutV2ForObjectDetection
from transformers.models.pp_doclayout_v2.configuration_pp_doclayout_v2 import PPDocLayoutV2ReadingOrderConfig
from transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2 import (
    PPDocLayoutV2ForObjectDetectionOutput,
    PPDocLayoutV2ReadingOrder,
)

from .pp_doclayout_v2_base import (
    DEFAULT_CLASS_ORDER,
    DEFAULT_CLASS_THRESHOLDS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RESCALE_FACTOR,
    PP_DOCLAYOUT_V2_LABEL_TO_ID,
    PP_DOCLAYOUT_V2_LABELS,
    PPDocLayoutV2PostProcessor,
    label_to_color,
    load_preprocess_config,
)


class _CpuSinePositionEmbedding(nn.Module):
    """在 CPU 计算官方实现所需的 float64 频率，再把位置编码送回模型设备。"""

    def __init__(self, embedding: nn.Module) -> None:
        """组合无参数的位置编码模块，保持预训练权重名称和数学实现不变。"""
        super().__init__()
        self.embedding = embedding

    def forward(
        self,
        width: int,
        height: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """避开 MPS 不支持的 float64 运算，复用官方 CPU 位置编码缓存。"""
        return self.embedding(width=width, height=height, device=torch.device("cpu"), dtype=dtype).to(device=device)


class PPDocLayoutV2LayoutModel(PPDocLayoutV2PostProcessor):
    def __init__(
        self,
        weight: str,
        device: Optional[str] = "cuda",
        imgsz: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        conf: float = 0.45,
        use_paddlex_filter_boxes: bool = True,
    ):
        self.device = device or "cpu"
        self.conf = conf
        self.use_paddlex_filter_boxes = use_paddlex_filter_boxes
        self.model_dir = weight
        self.preprocess_config = load_preprocess_config(self.model_dir)
        size = self.preprocess_config.get("size", {})
        self.imgsz = (
            int(size.get("width", imgsz[0])),
            int(size.get("height", imgsz[1])),
        )
        self.rescale_factor = float(self.preprocess_config.get("rescale_factor", DEFAULT_RESCALE_FACTOR))
        self.config = PPDocLayoutV2Config.from_pretrained(self.model_dir)
        # 保留原有 FP32 推理精度，直接加载到目标设备以减少 CPU 权重中转。
        self.model = PPDocLayoutV2ForObjectDetection.from_pretrained(
            self.model_dir,
            config=self.config,
            device_map={"": self.device},
            dtype=torch.float32,
            attn_implementation="eager" if torch.device(self.device).type == "mps" else None,
        )
        if torch.device(self.device).type == "mps":
            # 官方正弦编码内部使用 float64；只适配这个无参数模块，不切换整个模型的设备。
            for layer in self.model.model.encoder.aifi:
                layer.position_embedding = _CpuSinePositionEmbedding(layer.position_embedding)
        # 同步加载器生成的非持久化 buffer；已经位于目标设备的参数不会再次复制。
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _get_order_seqs(order_logits: torch.Tensor) -> torch.Tensor:
        order_scores = torch.sigmoid(order_logits)
        batch_size, sequence_length, _ = order_scores.shape
        order_votes = order_scores.triu(diagonal=1).sum(dim=1) + (1.0 - order_scores.transpose(1, 2)).tril(diagonal=-1).sum(
            dim=1
        )
        order_pointers = torch.argsort(order_votes, dim=1)
        order_seq = torch.empty_like(order_pointers)
        ranks = torch.arange(sequence_length, device=order_pointers.device, dtype=order_pointers.dtype).expand(batch_size, -1)
        order_seq.scatter_(1, order_pointers, ranks)
        return order_seq

    def _preprocess_single_image(self, image: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type for PP-DocLayoutV2: {type(image)}")

        pil_image = pil_image.convert("RGB")
        target_size = pil_image.size[1], pil_image.size[0]
        pixel_values = tvF.pil_to_tensor(pil_image)
        pixel_values = tvF.resize(
            pixel_values,
            size=[self.imgsz[1], self.imgsz[0]],
            interpolation=InterpolationMode.BICUBIC,
            antialias=False,
        )
        pixel_values = pixel_values.to(dtype=torch.float32) * self.rescale_factor
        return pixel_values, target_size

    def _post_process_object_detection(
        self,
        outputs: PPDocLayoutV2ForObjectDetectionOutput,
        target_sizes: Sequence[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        boxes = outputs.pred_boxes
        logits = outputs.logits
        order_logits = outputs.order_logits
        order_seqs = self._get_order_seqs(order_logits)

        box_centers, box_dims = torch.split(boxes, 2, dim=-1)
        boxes = torch.cat([box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims], dim=-1)

        img_height, img_width = torch.as_tensor(target_sizes, device=boxes.device).unbind(1)
        scale_factor = torch.stack([img_width, img_height, img_width, img_height], dim=1).to(boxes.device)
        boxes = boxes * scale_factor[:, None, :]

        num_top_queries = logits.shape[1]
        num_classes = logits.shape[2]
        scores = torch.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
        labels = index % num_classes
        index = index // num_classes
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        order_seqs = order_seqs.gather(dim=1, index=index)

        results = []
        for score, label, box, order_seq in zip(scores, labels, boxes, order_seqs):
            keep = score >= self.conf
            order_seq = order_seq[keep]
            order_seq, indices = torch.sort(order_seq)
            results.append(
                {
                    "scores": score[keep][indices],
                    "labels": label[keep][indices],
                    "boxes": box[keep][indices],
                    "order_seq": order_seq,
                }
            )
        return results

    def _parse_prediction(self, result: Dict[str, torch.Tensor], image_size: Tuple[int, int]) -> List[Dict]:
        layout_res = []
        for index, (score, label_id, box, _order_seq) in enumerate(
            zip(
                result["scores"],
                result["labels"],
                result["boxes"],
                result["order_seq"],
            ),
            start=1,
        ):
            bbox = self._clip_bbox(box.tolist(), image_size)
            if bbox is None:
                continue

            cls_id = int(label_id.item())
            layout_res.append(
                {
                    "cls_id": cls_id,
                    "label": self._label_id_to_label_name(cls_id),
                    "score": round(float(score.item()), 4),
                    "bbox": bbox,
                    "index": index,
                }
            )
        return layout_res

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

    @classmethod
    def _is_header_footer_boundary_candidate(cls, box: Dict, anchor_labels: set[str]) -> bool:
        """判断普通块是否可被页眉/页脚/页码边界规则改标。"""
        label = box.get("label")
        # 未被明确页眉页脚父框覆盖的公式仍保留公式身份，避免仅按页边位置误改标。
        if cls._is_formula_box(box):
            return False
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
        return PPDocLayoutV2LayoutModel._is_display_formula_box(box) or PPDocLayoutV2LayoutModel._is_inline_formula_box(box)

    @staticmethod
    def _is_formula_number_box(box: Dict) -> bool:
        return box.get("label") == "formula_number" or int(box.get("cls_id", -1)) == 11

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
        PPDocLayoutV2LayoutModel._set_box_label(box, label)

    @staticmethod
    def _set_header_footer_label(box: Dict, label: str) -> None:
        """同步设置页眉/页脚相关标签及其类别编号。"""
        if label not in {"footer", "footer_image", "header", "header_image"}:
            raise ValueError(f"Unsupported header/footer label: {label}")
        PPDocLayoutV2LayoutModel._set_box_label(box, label)

    @staticmethod
    def _set_footnote_label(box: Dict) -> None:
        """同步设置 page footnote 标签及其类别编号。"""
        PPDocLayoutV2LayoutModel._set_box_label(box, "footnote")

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
    def _filter_formula_boxes_inside_page_marginals(
        cls,
        boxes: List[Dict],
        cover_threshold: float = 0.8,
    ) -> List[Dict]:
        """删除被明确页眉页脚父框覆盖的公式框，避免页边小公式生成重叠文本 span。"""

        marginal_labels = {"header", "header_image", "footer", "footer_image"}
        marginal_parents = [box for box in boxes if box.get("label") in marginal_labels]
        if not marginal_parents:
            return boxes

        filtered_boxes = []
        for box in boxes:
            if not cls._is_formula_box(box):
                filtered_boxes.append(box)
                continue
            # 仅按公式自身被明确父框覆盖的比例过滤，不使用页面上下坐标带猜测归属。
            if any(cls._calculate_cover_ratio(box["bbox"], parent["bbox"]) >= cover_threshold for parent in marginal_parents):
                continue
            filtered_boxes.append(box)
        return filtered_boxes

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
        processed_boxes = cls._filter_formula_boxes_inside_page_marginals(
            processed_boxes,
            cover_threshold=0.8,
        )
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

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        use_paddlex_filter_boxes: Optional[bool] = None,
    ) -> List[Dict]:
        return self.batch_predict(
            [image],
            batch_size=1,
            use_paddlex_filter_boxes=use_paddlex_filter_boxes,
        )[0]

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 1,
        use_paddlex_filter_boxes: Optional[bool] = None,
    ) -> List[List[Dict]]:
        if len(images) == 0:
            return []

        use_paddlex_filter_boxes = (
            self.use_paddlex_filter_boxes if use_paddlex_filter_boxes is None else use_paddlex_filter_boxes
        )
        results: List[List[Dict]] = []
        with torch.no_grad():
            with tqdm(total=len(images), desc="Layout Predict") as pbar:
                for start in range(0, len(images), batch_size):
                    batch_images = images[start : start + batch_size]
                    pixel_values_list = []
                    target_sizes = []
                    for image in batch_images:
                        pixel_values, target_size = self._preprocess_single_image(image)
                        pixel_values_list.append(pixel_values)
                        target_sizes.append(target_size)

                    batch_tensor = torch.stack(pixel_values_list, dim=0).to(self.device)
                    outputs = self.model(pixel_values=batch_tensor)
                    predictions = self._post_process_object_detection(outputs, target_sizes)
                    for prediction, image_size in zip(predictions, target_sizes):
                        layout_res = self._parse_prediction(prediction, image_size)
                        if use_paddlex_filter_boxes:
                            layout_res = self._apply_paddlex_filter_boxes(layout_res, drop_inline_formula=False)
                        layout_res = self._apply_layout_post_process(layout_res, image_size=image_size)
                        results.append(layout_res)
                    pbar.update(len(batch_images))
        return results


__all__ = [
    "DEFAULT_CLASS_ORDER",
    "DEFAULT_CLASS_THRESHOLDS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_RESCALE_FACTOR",
    "PP_DOCLAYOUT_V2_LABELS",
    "PP_DOCLAYOUT_V2_LABEL_TO_ID",
    "PPDocLayoutV2Config",
    "PPDocLayoutV2ForObjectDetection",
    "PPDocLayoutV2LayoutModel",
    "PPDocLayoutV2ReadingOrder",
    "PPDocLayoutV2ReadingOrderConfig",
    "label_to_color",
    "load_preprocess_config",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP-DocLayoutV2 local inference smoke test")
    parser.add_argument("image", nargs="?", help="Path to an input image. If omitted, only model loading is tested.")
    parser.add_argument("--model", default=None, help="Model name or path.")
    parser.add_argument("--device", default=None, help="Runtime device, e.g. cpu/mps/cuda.")
    parser.add_argument("--output", default=None, help="Optional path to save the visualization image.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the visualization window.")
    args = parser.parse_args()

    if args.device is None:
        from ..runtime.device import get_device

        args.device = get_device()

    if args.model is None:
        from ..registry import MINERU_4_MODELS_TORCH

        args.model = str(MINERU_4_MODELS_TORCH.pp_doclayout_v2.ensure())

    args.image = "/Users/myhloli/pdf/png/index.png"

    model = PPDocLayoutV2LayoutModel(
        weight=args.model,
        device=args.device,
    )
    print(f"model loaded on {model.device}")

    if args.image:
        with Image.open(args.image) as img:
            results = model.predict(img)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            vis_img = model.visualize(img, results)
            if args.output:
                vis_img.save(args.output)
            if not args.no_show:
                vis_img.show()
