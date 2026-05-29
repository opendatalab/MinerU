# Copyright (c) Opendatalab. All rights reserved.
import re
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np


def calculate_iou(
    box1: Union[np.ndarray, List], box2: Union[np.ndarray, List]
) -> float:
    """
    :param box1: Iterable [xmin,ymin,xmax,ymax]
    :param box2: Iterable [xmin,ymin,xmax,ymax]
    :return: iou: float 0-1
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # 不相交直接退出检测
    if b1_x2 < b2_x1 or b1_x1 > b2_x2 or b1_y2 < b2_y1 or b1_y1 > b2_y2:
        return 0.0
    # 计算交集
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    i_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算并集
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    u_area = b1_area + b2_area - i_area

    # 避免除零错误，如果区域小到乘积为0,认为是错误识别，直接去掉
    if u_area == 0:
        return 1
        # 检查完全包含
    iou = i_area / u_area
    return iou



def is_box_contained(
    box1: Union[np.ndarray, List], box2: Union[np.ndarray, List], threshold=0.2
) -> Union[int, None]:
    """
    :param box1: Iterable [xmin,ymin,xmax,ymax]
    :param box2: Iterable [xmin,ymin,xmax,ymax]
    :return: 1: box1 is contained 2: box2 is contained None: no contain these
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # 不相交直接退出检测
    if b1_x2 < b2_x1 or b1_x1 > b2_x2 or b1_y2 < b2_y1 or b1_y1 > b2_y2:
        return None
    # 计算box2的总面积
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)

    # 计算box1和box2的交集
    intersect_x1 = max(b1_x1, b2_x1)
    intersect_y1 = max(b1_y1, b2_y1)
    intersect_x2 = min(b1_x2, b2_x2)
    intersect_y2 = min(b1_y2, b2_y2)

    # 计算交集的面积
    intersect_area = max(0, intersect_x2 - intersect_x1) * max(
        0, intersect_y2 - intersect_y1
    )

    # 计算外面的面积
    b1_outside_area = b1_area - intersect_area
    b2_outside_area = b2_area - intersect_area

    # 计算外面的面积占box2总面积的比例
    ratio_b1 = b1_outside_area / b1_area if b1_area > 0 else 0
    ratio_b2 = b2_outside_area / b2_area if b2_area > 0 else 0

    if ratio_b1 < threshold:
        return 1
    if ratio_b2 < threshold:
        return 2
    # 判断比例是否大于阈值
    return None


def is_single_axis_contained(
    box1: Union[np.ndarray, List],
    box2: Union[np.ndarray, List],
    axis="x",
    threhold: float = 0.2,
) -> Union[int, None]:
    """
    :param box1: Iterable [xmin,ymin,xmax,ymax]
    :param box2: Iterable [xmin,ymin,xmax,ymax]
    :return: 1: box1 is contained 2: box2 is contained None: no contain these
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # 计算轴重叠大小
    if axis == "x":
        b1_area = b1_x2 - b1_x1
        b2_area = b2_x2 - b2_x1
        i_area = min(b1_x2, b2_x2) - max(b1_x1, b2_x1)
    else:
        b1_area = b1_y2 - b1_y1
        b2_area = b2_y2 - b2_y1
        i_area = min(b1_y2, b2_y2) - max(b1_y1, b2_y1)
        # 计算外面的面积
    b1_outside_area = b1_area - i_area
    b2_outside_area = b2_area - i_area

    ratio_b1 = b1_outside_area / b1_area if b1_area > 0 else 0
    ratio_b2 = b2_outside_area / b2_area if b2_area > 0 else 0
    if ratio_b1 < threhold:
        return 1
    if ratio_b2 < threhold:
        return 2
    return None


def sorted_ocr_boxes(
    dt_boxes: Union[np.ndarray, list], threhold: float = 0.2
) -> Tuple[Union[np.ndarray, list], List[int]]:
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with (xmin, ymin, xmax, ymax)
    return:
        sorted boxes(array) with (xmin, ymin, xmax, ymax)
    """
    num_boxes = len(dt_boxes)
    if num_boxes <= 0:
        return dt_boxes, []
    indexed_boxes = [(box, idx) for idx, box in enumerate(dt_boxes)]
    sorted_boxes_with_idx = sorted(indexed_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes, indices = zip(*sorted_boxes_with_idx)
    indices = list(indices)
    _boxes = [dt_boxes[i] for i in indices]
    threahold = 20
    # 避免输出和输入格式不对应，与函数功能不符合
    if isinstance(dt_boxes, np.ndarray):
        _boxes = np.array(_boxes)
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            c_idx = is_single_axis_contained(
                _boxes[j], _boxes[j + 1], axis="y", threhold=threhold
            )
            if (
                c_idx is not None
                and _boxes[j + 1][0] < _boxes[j][0]
                and abs(_boxes[j][1] - _boxes[j + 1][1]) < threahold
            ):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1].copy(), _boxes[j].copy()
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
            else:
                break
    return _boxes, indices


def box_4_1_poly_to_box_4_2(poly_box: Union[list, np.ndarray]) -> List[List[float]]:
    xmin, ymin, xmax, ymax = tuple(poly_box)
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def box_4_2_poly_to_box_4_1(poly_box: Union[list, np.ndarray]) -> List[Any]:
    """
    将poly_box转换为box_4_1
    :param poly_box:
    :return:
    """
    return [poly_box[0][0], poly_box[0][1], poly_box[2][0], poly_box[2][1]]


def _ocr_boxes_to_array(dt_rec_boxes: List[List[Union[Any, str]]]) -> np.ndarray:
    """将 OCR 四点框批量转换为 [xmin, ymin, xmax, ymax]，保持旧逻辑使用第 0/2 点。"""
    if len(dt_rec_boxes) == 0:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(
        [
            [
                gt_box[0][0][0],
                gt_box[0][0][1],
                gt_box[0][2][0],
                gt_box[0][2][1],
            ]
            for gt_box in dt_rec_boxes
        ],
        dtype=np.float64,
    )


def _pred_boxes_to_array(pred_bboxes: np.ndarray) -> np.ndarray:
    """将预测 cell 四点框批量转换为 [xmin, ymin, xmax, ymax]，保持旧逻辑使用第 0/2 点。"""
    pred_bboxes = np.asarray(pred_bboxes)
    if pred_bboxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(
        [
            [pred_box[0][0], pred_box[0][1], pred_box[2][0], pred_box[2][1]]
            for pred_box in pred_bboxes
        ],
        dtype=np.float64,
    )


_TABLE_CELL_TOKEN_RE = re.compile(
    r"""
    \d{4}[-/]\d{1,2}[-/]\d{1,2}
    |[-+]?\d{1,3}(?:,\d{3})+(?:\.\d{2})?%?
    |[-+]?\d+(?:\.\d+)?%?
    |[A-Za-z]+
    |[\u4e00-\u9fff]+
    """,
    re.VERBOSE,
)


def _char_visual_weight(char: str) -> float:
    """按字符视觉宽度估算拆分位置，避免简单按字符数切分造成明显偏移。"""
    if "\u4e00" <= char <= "\u9fff":
        return 1.8
    if char.isdigit() or char.isalpha():
        return 1.0
    if char.isspace():
        return 0.35
    if char in {",", ".", "-", "+", "%", "/", ":", ";", "'", '"'}:
        return 0.45
    return 0.8


def _intersect_box(box1: np.ndarray, box2: np.ndarray) -> List[float]:
    """生成 OCR 子框与目标 cell 的交集框，作为拆分后伪 OCR 的 bbox。"""
    x0 = max(float(box1[0]), float(box2[0]))
    y0 = max(float(box1[1]), float(box2[1]))
    x1 = min(float(box1[2]), float(box2[2]))
    y1 = min(float(box1[3]), float(box2[3]))
    if x1 <= x0:
        x0, x1 = float(box2[0]), float(box2[2])
    if y1 <= y0:
        y0, y1 = float(box1[1]), float(box1[3])
    return [x0, y0, x1, y1]


def _make_split_ocr_result(gt_box: List[Union[Any, str]], bbox: List[float], text: str):
    """复制原 OCR 结果，只替换 bbox 和文本，保持 score 等后续字段不变。"""
    split_result = list(gt_box)
    split_result[0] = box_4_1_poly_to_box_4_2(bbox)
    split_result[1] = text
    return split_result


def _extract_cell_tokens(text: str, cell_count: int) -> List[str]:
    """优先用金额、日期、百分比和短 token 拆分跨格文本。"""
    stripped_text = text.strip()
    if not stripped_text:
        return []

    matches = list(_TABLE_CELL_TOKEN_RE.finditer(stripped_text))
    tokens = [match.group(0) for match in matches]
    if len(tokens) != cell_count:
        return []

    covered = "".join(tokens)
    uncovered = _TABLE_CELL_TOKEN_RE.sub("", stripped_text)
    if covered and not uncovered.strip():
        return tokens
    return []


def _split_text_by_cell_boundaries(
    text: str,
    ocr_box: np.ndarray,
    candidate_boxes: List[np.ndarray],
) -> List[str]:
    """按 cell 边界在 OCR 框中的相对位置，结合字符宽度估算文本切点。"""
    stripped_text = text.strip()
    if len(stripped_text) < len(candidate_boxes):
        return []

    ocr_width = float(ocr_box[2] - ocr_box[0])
    if ocr_width <= 0:
        return []

    weights = [_char_visual_weight(char) for char in stripped_text]
    total_weight = sum(weights)
    if total_weight <= 0:
        return []

    cumulative_weights = np.cumsum(weights)
    split_indices = []
    split_boundaries = []
    previous_idx = 0
    for left_box, right_box in zip(candidate_boxes, candidate_boxes[1:]):
        boundary_x = (float(left_box[2]) + float(right_box[0])) / 2
        ratio = (boundary_x - float(ocr_box[0])) / ocr_width
        target_weight = total_weight * min(max(ratio, 0.0), 1.0)
        split_idx = int(np.searchsorted(cumulative_weights, target_weight, side="left") + 1)

        remaining_boundaries = len(candidate_boxes) - len(split_indices) - 1
        min_idx = previous_idx + 1
        max_idx = len(stripped_text) - remaining_boundaries
        split_idx = min(max(split_idx, min_idx), max_idx)
        if split_idx <= previous_idx:
            return []
        split_indices.append(split_idx)
        split_boundaries.append(min(max(boundary_x, float(ocr_box[0])), float(ocr_box[2])))
        previous_idx = split_idx

    segments = []
    start = 0
    for split_idx in split_indices + [len(stripped_text)]:
        segment = stripped_text[start:split_idx].strip()
        if not segment:
            return []
        segments.append(segment)
        start = split_idx
    split_positions = [float(ocr_box[0])] + split_boundaries + [float(ocr_box[2])]
    for segment, left_x, right_x in zip(segments, split_positions, split_positions[1:]):
        segment_weight = sum(_char_visual_weight(char) for char in segment)
        expected_width = ocr_width * segment_weight / total_weight
        # 检测框轻微越过单元格边界时也会被强制分到至少一个字符；这里要求切片的
        # 实际投影宽度能支撑该片段的加权字符宽度，避免把 bbox padding 误当成跨格文本。
        if right_x - left_x < expected_width * 0.6:
            return []
    return segments


def _min_projection_segment_width(text: str, ocr_box: np.ndarray) -> float:
    """估算一个可拆文本片段的最小像素宽度，用于过滤 OCR 检测框的轻微越界。"""
    stripped_text = text.strip()
    ocr_width = float(ocr_box[2] - ocr_box[0])
    if not stripped_text or ocr_width <= 0:
        return float("inf")

    weights = [_char_visual_weight(char) for char in stripped_text]
    total_weight = sum(weights)
    if total_weight <= 0:
        return float("inf")

    # 允许 OCR 框/字符宽度有一定误差，但相邻 cell 的投影宽度至少要接近一个最窄字符；
    # 同时要求达到 OCR 框宽度的一小段比例，过滤长 OCR 框轻微擦到行标题 cell 的情况。
    return max(ocr_width * min(weights) / total_weight * 0.7, ocr_width * 0.04)


def _same_row_adjacent_cells(candidate_boxes: List[np.ndarray]) -> bool:
    """确认候选 cell 位于同一行且横向相邻，避免拆分跨行或非相邻文本。"""
    if len(candidate_boxes) < 2:
        return False

    for left_box, right_box in zip(candidate_boxes, candidate_boxes[1:]):
        y_overlap = min(float(left_box[3]), float(right_box[3])) - max(
            float(left_box[1]), float(right_box[1])
        )
        min_height = min(
            float(left_box[3] - left_box[1]),
            float(right_box[3] - right_box[1]),
        )
        if min_height <= 0 or y_overlap / min_height < 0.6:
            return False

        gap = float(right_box[0] - left_box[2])
        min_width = min(
            float(left_box[2] - left_box[0]),
            float(right_box[2] - right_box[0]),
        )
        if abs(gap) > max(2.0, min_width * 0.3):
            return False
    return True


def _split_cross_cell_ocr_result(
    gt_box: List[Union[Any, str]],
    ocr_box: np.ndarray,
    pred_boxes: np.ndarray,
    candidate_indices: np.ndarray,
    allow_weighted_split: bool = True,
) -> List[Tuple[int, List[Union[Any, str]]]]:
    """把横跨同一行相邻 cell 的 OCR 结果拆成多个伪 OCR 结果。"""
    if len(candidate_indices) < 2:
        return []

    sorted_indices = sorted(candidate_indices.tolist(), key=lambda idx: pred_boxes[idx][0])
    candidate_boxes = [pred_boxes[idx] for idx in sorted_indices]
    if not _same_row_adjacent_cells(candidate_boxes):
        return []

    text = str(gt_box[1])
    segments = _extract_cell_tokens(text, len(sorted_indices))
    if not segments and allow_weighted_split:
        segments = _split_text_by_cell_boundaries(text, ocr_box, candidate_boxes)
    if len(segments) != len(sorted_indices):
        return []

    split_results = []
    for cell_idx, cell_box, segment in zip(sorted_indices, candidate_boxes, segments):
        bbox = _intersect_box(ocr_box, cell_box)
        split_results.append((cell_idx, _make_split_ocr_result(gt_box, bbox, segment)))
    return split_results


def _select_clear_best_cell(
    candidate_indices: np.ndarray,
    coverage_scores: np.ndarray,
    iou_scores: np.ndarray,
    ocr_box: np.ndarray,
    pred_boxes: np.ndarray,
) -> Union[int, None]:
    """当某个 cell 明显优于其他候选时直接归属，避免不必要拆分。"""
    if len(candidate_indices) == 1:
        return int(candidate_indices[0])

    ranked = sorted(
        candidate_indices.tolist(),
        key=lambda idx: (coverage_scores[idx], iou_scores[idx]),
        reverse=True,
    )
    best_idx = ranked[0]
    second_idx = ranked[1]
    best_coverage = float(coverage_scores[best_idx])
    second_coverage = float(coverage_scores[second_idx])

    center_x = (float(ocr_box[0]) + float(ocr_box[2])) / 2
    center_y = (float(ocr_box[1]) + float(ocr_box[3])) / 2
    center_hits = [
        idx
        for idx in candidate_indices.tolist()
        if (
            float(pred_boxes[idx][0]) <= center_x < float(pred_boxes[idx][2])
            and float(pred_boxes[idx][1]) <= center_y <= float(pred_boxes[idx][3])
        )
    ]
    if len(center_hits) == 1 and center_hits[0] == best_idx:
        if best_coverage >= 0.55 and best_coverage - second_coverage >= 0.15:
            return int(best_idx)

    if best_coverage >= 0.65 and best_coverage - second_coverage >= 0.2:
        return int(best_idx)
    return None


def match_ocr_cell(dt_rec_boxes: List[List[Union[Any, str]]], pred_bboxes: np.ndarray):
    """
    :param dt_rec_boxes: [[(4.2), text, score]]
    :param pred_bboxes: shap (4,2)
    :return:
    """
    ocr_boxes = _ocr_boxes_to_array(dt_rec_boxes)
    pred_boxes = _pred_boxes_to_array(pred_bboxes)
    if ocr_boxes.size == 0 or pred_boxes.size == 0:
        return {}, []

    ocr = ocr_boxes[:, None, :]
    pred = pred_boxes[None, :, :]

    no_intersection = (
        (ocr[..., 2] < pred[..., 0])
        | (ocr[..., 0] > pred[..., 2])
        | (ocr[..., 3] < pred[..., 1])
        | (ocr[..., 1] > pred[..., 3])
    )
    intersects = ~no_intersection

    inter_x1 = np.maximum(ocr[..., 0], pred[..., 0])
    inter_y1 = np.maximum(ocr[..., 1], pred[..., 1])
    inter_x2 = np.minimum(ocr[..., 2], pred[..., 2])
    inter_y2 = np.minimum(ocr[..., 3], pred[..., 3])
    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    ocr_width = ocr[..., 2] - ocr[..., 0]
    ocr_height = ocr[..., 3] - ocr[..., 1]
    pred_width = pred[..., 2] - pred[..., 0]
    pred_height = pred[..., 3] - pred[..., 1]
    ocr_area = ocr_width * ocr_height
    pred_area = pred_width * pred_height

    # 等价实现 is_box_contained(ocr_box, pred_box, 0.6) == 1。
    ocr_outside_ratio = np.divide(
        ocr_area - inter_area,
        ocr_area,
        out=np.zeros_like(inter_area, dtype=np.float64),
        where=ocr_area > 0,
    )
    contained = intersects & (ocr_outside_ratio < 0.6)

    # 等价实现 calculate_iou()，包括 union 为 0 时返回 1 的历史行为。
    union_area = ocr_area + pred_area - inter_area
    iou = np.divide(
        inter_area,
        union_area,
        out=np.ones_like(inter_area, dtype=np.float64),
        where=union_area != 0,
    )
    iou[no_intersection] = 0.0
    matched_mask = contained | (iou > 0.8)
    coverage = np.divide(
        inter_area,
        ocr_area,
        out=np.zeros_like(inter_area, dtype=np.float64),
        where=ocr_area > 0,
    )
    # 低覆盖率横向长 OCR 需要按投影找同一行候选。这里复用上面已经算出的交集矩阵，
    # 避免每个 unmatched OCR 再用 Python 循环扫描所有 cell。
    projection_y_ratio = np.divide(
        inter_height,
        ocr_height,
        out=np.zeros_like(inter_height, dtype=np.float64),
        where=ocr_height > 0,
    )
    projection_mask = intersects & (projection_y_ratio >= 0.6) & (inter_width > 0)

    matched = {}
    not_match_orc_boxes = []
    for i, gt_box in enumerate(dt_rec_boxes):
        projection_indices = np.flatnonzero(projection_mask[i])
        if len(projection_indices) > 1:
            min_segment_width = _min_projection_segment_width(str(gt_box[1]), ocr_boxes[i])
            projection_indices = projection_indices[
                inter_width[i][projection_indices] >= min_segment_width
            ]
            split_results = _split_cross_cell_ocr_result(
                gt_box,
                ocr_boxes[i],
                pred_boxes,
                projection_indices,
            )
            if split_results:
                for cell_idx, split_gt_box in split_results:
                    matched.setdefault(cell_idx, []).append(split_gt_box)
                continue

        matched_indices = np.flatnonzero(matched_mask[i])
        if len(matched_indices) == 0:
            not_match_orc_boxes.append(gt_box)
            continue

        best_cell_idx = _select_clear_best_cell(
            matched_indices,
            coverage[i],
            iou[i],
            ocr_boxes[i],
            pred_boxes,
        )
        if best_cell_idx is not None:
            matched.setdefault(best_cell_idx, []).append(gt_box)
            continue

        split_results = _split_cross_cell_ocr_result(
            gt_box,
            ocr_boxes[i],
            pred_boxes,
            matched_indices,
        )
        if split_results:
            for cell_idx, split_gt_box in split_results:
                matched.setdefault(cell_idx, []).append(split_gt_box)
            continue

        # 无法确定归属时不要把同一 OCR 重复塞进多个 cell，交给空 cell 裁剪 OCR 兜底。
        not_match_orc_boxes.append(gt_box)

    return matched, not_match_orc_boxes


def gather_ocr_list_by_row(ocr_list: List[Any], threhold: float = 0.2) -> List[Any]:
    """
    :param ocr_list: [[[xmin,ymin,xmax,ymax], text]]
    :return:
    """
    threshold = 10
    for i in range(len(ocr_list)):
        if not ocr_list[i]:
            continue

        for j in range(i + 1, len(ocr_list)):
            if not ocr_list[j]:
                continue
            cur = ocr_list[i]
            next = ocr_list[j]
            cur_box = cur[0]
            next_box = next[0]
            c_idx = is_single_axis_contained(
                cur[0], next[0], axis="y", threhold=threhold
            )
            if c_idx:
                dis = max(next_box[0] - cur_box[2], 0)
                blank_str = int(dis / threshold) * " "
                cur[1] = cur[1] + blank_str + next[1]
                xmin = min(cur_box[0], next_box[0])
                xmax = max(cur_box[2], next_box[2])
                ymin = min(cur_box[1], next_box[1])
                ymax = max(cur_box[3], next_box[3])
                cur_box[0] = xmin
                cur_box[1] = ymin
                cur_box[2] = xmax
                cur_box[3] = ymax
                ocr_list[j] = None
    ocr_list = [x for x in ocr_list if x]
    return ocr_list


def _normalize_logic_points(logi_points: Union[np.ndarray, List]) -> np.ndarray:
    """把逻辑坐标统一成 N x 4 数组，方便后续按结构网格渲染空 cell。"""
    points = np.asarray(logi_points, dtype=np.int32)
    if points.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    return points.reshape(-1, 4)


def _normalize_cell_text_map(cell_box_map: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """把 OCR 文本映射规整成 cell_index -> 文本列表，缺失的 cell 后续按空文本处理。"""
    normalized = {}
    for cell_idx, values in (cell_box_map or {}).items():
        if values is None:
            continue
        if isinstance(values, str):
            normalized[int(cell_idx)] = [values]
            continue
        try:
            normalized[int(cell_idx)] = [
                str(value) for value in values if value is not None
            ]
        except TypeError:
            normalized[int(cell_idx)] = [str(values)]
    return normalized


def _normalize_cell_bboxes(
    cell_bboxes: Optional[Union[np.ndarray, List]]
) -> Optional[np.ndarray]:
    """把单元格物理框统一成 N x 4 x 2，用于判断边缘空列是否是结构噪声。"""
    if cell_bboxes is None:
        return None
    bboxes = np.asarray(cell_bboxes, dtype=np.float64)
    if bboxes.size == 0:
        return None
    if bboxes.ndim == 3 and bboxes.shape[1:] == (4, 2):
        return bboxes
    if bboxes.ndim == 2 and bboxes.shape[1] == 8:
        return bboxes.reshape(-1, 4, 2)
    if bboxes.ndim == 2 and bboxes.shape[1] == 4:
        x0, y0, x1, y1 = bboxes.T
        return np.stack(
            [
                np.stack([x0, y0], axis=1),
                np.stack([x1, y0], axis=1),
                np.stack([x1, y1], axis=1),
                np.stack([x0, y1], axis=1),
            ],
            axis=1,
        )
    return None


def _build_table_grid(logic_points: np.ndarray):
    """按完整结构坐标填充网格，空文本 cell 也保留结构占位。"""
    max_row = int(logic_points[:, 1].max() + 1)
    max_col = int(logic_points[:, 3].max() + 1)
    grid = [[None] * max_col for _ in range(max_row)]
    for i, logic_point in enumerate(logic_points):
        row_start, row_end, col_start, col_end = [int(v) for v in logic_point]
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                grid[row][col] = (i, row_start, row_end, col_start, col_end)
    return grid, max_row, max_col


def _cell_text(cell_text_map: Dict[int, List[str]], cell_idx: int) -> str:
    """获取 cell 文本；没有 OCR 命中的结构 cell 返回空字符串。"""
    return "".join(cell_text_map.get(cell_idx, []))


def _cell_has_visible_text(cell_text_map: Dict[int, List[str]], cell_idx: int) -> bool:
    """判断 cell 是否有可见文本，避免仅靠空白字符阻止边缘噪声裁剪。"""
    return bool(_cell_text(cell_text_map, cell_idx).strip())


def _cell_bbox_rect(cell_bboxes: Optional[np.ndarray], cell_idx: int):
    """读取 cell 外接矩形，缺少物理框时返回 None 并退化为结构判断。"""
    if cell_bboxes is None or cell_idx >= len(cell_bboxes):
        return None
    bbox = cell_bboxes[cell_idx]
    return (
        float(np.min(bbox[:, 0])),
        float(np.min(bbox[:, 1])),
        float(np.max(bbox[:, 0])),
        float(np.max(bbox[:, 1])),
    )


def _estimate_axis_sizes(
    logic_points: np.ndarray,
    cell_bboxes: Optional[np.ndarray],
    axis: str,
    axis_count: int,
) -> List[Optional[float]]:
    """估算每行/列的几何尺寸，用来区分真实完整空列和异常外围噪声列。"""
    if cell_bboxes is None:
        return [None] * axis_count
    axis_sizes = [[] for _ in range(axis_count)]
    for cell_idx, logic_point in enumerate(logic_points):
        rect = _cell_bbox_rect(cell_bboxes, cell_idx)
        if rect is None:
            continue
        row_start, row_end, col_start, col_end = [int(v) for v in logic_point]
        x0, y0, x1, y1 = rect
        if axis == "col":
            span = max(col_end - col_start + 1, 1)
            size = max((x1 - x0) / span, 0)
            target_range = range(col_start, col_end + 1)
        else:
            span = max(row_end - row_start + 1, 1)
            size = max((y1 - y0) / span, 0)
            target_range = range(row_start, row_end + 1)
        if size <= 0:
            continue
        for axis_idx in target_range:
            if 0 <= axis_idx < axis_count:
                axis_sizes[axis_idx].append(size)
    return [
        float(np.median(sizes)) if sizes else None
        for sizes in axis_sizes
    ]


def _axis_reference_size(
    axis_sizes: List[Optional[float]], axis_idx: int
) -> Optional[float]:
    """用其它行/列的中位尺寸作为参照，避免单个边缘噪声框主导判断。"""
    sizes = [
        size
        for i, size in enumerate(axis_sizes)
        if i != axis_idx and size is not None and size > 0
    ]
    if not sizes:
        return None
    return float(np.median(sizes))


def _axis_size_is_abnormal(
    axis_sizes: List[Optional[float]], axis_idx: int
) -> bool:
    """判断空边缘行/列尺寸是否明显异常；正常尺寸的完整空列需要保留。"""
    axis_size = axis_sizes[axis_idx]
    reference_size = _axis_reference_size(axis_sizes, axis_idx)
    if axis_size is None or reference_size is None or reference_size <= 0:
        return False
    ratio = axis_size / reference_size
    return ratio < 0.35 or ratio > 2.5


def _edge_axis_has_text(
    grid: List[List[Optional[Tuple[int, int, int, int, int]]]],
    cell_text_map: Dict[int, List[str]],
    axis: str,
    axis_idx: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> bool:
    """检查当前边缘行/列是否有文本；有文本的边缘不能被裁剪。"""
    if axis == "col":
        positions = (grid[row][axis_idx] for row in range(row_start, row_end + 1))
    else:
        positions = (grid[axis_idx][col] for col in range(col_start, col_end + 1))
    return any(
        cell is not None and _cell_has_visible_text(cell_text_map, cell[0])
        for cell in positions
    )


def _edge_axis_coverage(
    grid: List[List[Optional[Tuple[int, int, int, int, int]]]],
    axis: str,
    axis_idx: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> Tuple[int, int]:
    """统计边缘行/列在当前保留范围内的结构覆盖度，覆盖不完整通常是外围噪声。"""
    if axis == "col":
        covered = sum(
            grid[row][axis_idx] is not None
            for row in range(row_start, row_end + 1)
        )
        total = row_end - row_start + 1
    else:
        covered = sum(
            grid[axis_idx][col] is not None
            for col in range(col_start, col_end + 1)
        )
        total = col_end - col_start + 1
    return covered, total


def _is_noise_edge_axis(
    grid: List[List[Optional[Tuple[int, int, int, int, int]]]],
    cell_text_map: Dict[int, List[str]],
    axis_sizes: List[Optional[float]],
    axis: str,
    axis_idx: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> bool:
    """只裁剪无文本且结构不完整或尺寸异常的边缘，保留真实完整空行/空列。"""
    if _edge_axis_has_text(
        grid, cell_text_map, axis, axis_idx, row_start, row_end, col_start, col_end
    ):
        return False
    covered, total = _edge_axis_coverage(
        grid, axis, axis_idx, row_start, row_end, col_start, col_end
    )
    if covered == 0 or covered < total:
        return True
    return _axis_size_is_abnormal(axis_sizes, axis_idx)


def _trim_noise_edges(
    grid: List[List[Optional[Tuple[int, int, int, int, int]]]],
    cell_text_map: Dict[int, List[str]],
    row_sizes: List[Optional[float]],
    col_sizes: List[Optional[float]],
    max_row: int,
    max_col: int,
) -> Tuple[int, int, int, int]:
    """裁剪外围噪声行列；完整覆盖且尺寸正常的空边缘会被当作真实表格结构保留。"""
    row_start, row_end = 0, max_row - 1
    col_start, col_end = 0, max_col - 1

    while row_start <= row_end and _is_noise_edge_axis(
        grid, cell_text_map, row_sizes, "row", row_start,
        row_start, row_end, col_start, col_end,
    ):
        row_start += 1
    while row_end >= row_start and _is_noise_edge_axis(
        grid, cell_text_map, row_sizes, "row", row_end,
        row_start, row_end, col_start, col_end,
    ):
        row_end -= 1
    while col_start <= col_end and _is_noise_edge_axis(
        grid, cell_text_map, col_sizes, "col", col_start,
        row_start, row_end, col_start, col_end,
    ):
        col_start += 1
    while col_end >= col_start and _is_noise_edge_axis(
        grid, cell_text_map, col_sizes, "col", col_end,
        row_start, row_end, col_start, col_end,
    ):
        col_end -= 1

    return row_start, row_end, col_start, col_end


def plot_html_table(
    logi_points: Union[np.ndarray, List],
    cell_box_map: Dict[int, List[str]],
    cell_bboxes: Optional[Union[np.ndarray, List]] = None,
) -> str:
    logic_points = _normalize_logic_points(logi_points)
    if logic_points.size == 0:
        return "<html><body><table></table></body></html>"

    cell_text_map = _normalize_cell_text_map(cell_box_map)
    normalized_bboxes = _normalize_cell_bboxes(cell_bboxes)
    grid, max_row, max_col = _build_table_grid(logic_points)
    row_sizes = _estimate_axis_sizes(logic_points, normalized_bboxes, "row", max_row)
    col_sizes = _estimate_axis_sizes(logic_points, normalized_bboxes, "col", max_col)
    row_start, row_end, col_start, col_end = _trim_noise_edges(
        grid, cell_text_map, row_sizes, col_sizes, max_row, max_col
    )

    table_html = "<html><body><table>"
    if row_start > row_end or col_start > col_end:
        return table_html + "</table></body></html>"

    for row in range(row_start, row_end + 1):
        temp = "<tr>"
        for col in range(col_start, col_end + 1):
            cell = grid[row][col]
            if cell is None:
                temp += "<td></td>"
                continue

            cell_idx, origin_row_start, origin_row_end, origin_col_start, origin_col_end = cell
            clipped_row_start = max(origin_row_start, row_start)
            clipped_col_start = max(origin_col_start, col_start)
            if row == clipped_row_start and col == clipped_col_start:
                row_span = min(origin_row_end, row_end) - clipped_row_start + 1
                col_span = min(origin_col_end, col_end) - clipped_col_start + 1
                text = _cell_text(cell_text_map, cell_idx)
                temp += f"<td rowspan={row_span} colspan={col_span}>{text}</td>"
        table_html = table_html + temp + "</tr>"

    table_html += "</table></body></html>"
    return table_html
