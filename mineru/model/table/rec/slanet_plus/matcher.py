# Copyright (c) Opendatalab. All rights reserved.
# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from .matcher_utils import compute_iou, distance


TABLE_MATCH_CHUNK_SIZE = 256


def _normalize_cell_bboxes(cell_bboxes):
    """将4点或8点单元格框统一为[x0, y0, x1, y1]，便于后续向量化匹配。"""
    if cell_bboxes is None:
        return np.empty((0, 4), dtype=np.float64)

    cell_array = np.asarray(cell_bboxes, dtype=object)
    if cell_array.size == 0:
        return np.empty((0, 4), dtype=np.float64)

    try:
        numeric_array = np.asarray(cell_bboxes, dtype=np.float64)
    except ValueError:
        numeric_array = None

    if numeric_array is not None and numeric_array.ndim == 2:
        if numeric_array.shape[1] == 4:
            return numeric_array
        if numeric_array.shape[1] == 8:
            x_coords = numeric_array[:, 0::2]
            y_coords = numeric_array[:, 1::2]
            return np.stack(
                [
                    np.min(x_coords, axis=1),
                    np.min(y_coords, axis=1),
                    np.max(x_coords, axis=1),
                    np.max(y_coords, axis=1),
                ],
                axis=1,
            ).astype(np.float64, copy=False)

    normalized = []
    for cell_bbox in cell_bboxes:
        bbox = np.asarray(cell_bbox, dtype=np.float64).reshape(-1)
        if bbox.size == 8:
            normalized.append(
                [
                    np.min(bbox[0::2]),
                    np.min(bbox[1::2]),
                    np.max(bbox[0::2]),
                    np.max(bbox[1::2]),
                ]
            )
        elif bbox.size == 4:
            normalized.append(bbox.tolist())
        else:
            raise ValueError(f"Unsupported table cell bbox shape: {bbox.shape}")

    return np.asarray(normalized, dtype=np.float64)


def _pairwise_iou_and_distance(dt_boxes, cell_bboxes):
    """批量计算OCR框到所有单元格框的IoU和原有distance指标，保持旧排序语义。"""
    dt = dt_boxes[:, None, :]
    cells = cell_bboxes[None, :, :]

    dt_area = (dt[..., 2] - dt[..., 0]) * (dt[..., 3] - dt[..., 1])
    cell_area = (cells[..., 2] - cells[..., 0]) * (cells[..., 3] - cells[..., 1])
    sum_area = dt_area + cell_area

    left_line = np.maximum(dt[..., 1], cells[..., 1])
    right_line = np.minimum(dt[..., 3], cells[..., 3])
    top_line = np.maximum(dt[..., 0], cells[..., 0])
    bottom_line = np.minimum(dt[..., 2], cells[..., 2])
    intersect = (right_line - left_line) * (bottom_line - top_line)
    has_intersection = (left_line < right_line) & (top_line < bottom_line)
    union = sum_area - intersect
    iou = np.zeros_like(intersect, dtype=np.float64)
    np.divide(intersect, union, out=iou, where=has_intersection & (union != 0))

    dis = (
        np.abs(cells[..., 0] - dt[..., 0])
        + np.abs(cells[..., 1] - dt[..., 1])
        + np.abs(cells[..., 2] - dt[..., 2])
        + np.abs(cells[..., 3] - dt[..., 3])
    )
    dis_2 = np.abs(cells[..., 0] - dt[..., 0]) + np.abs(cells[..., 1] - dt[..., 1])
    dis_3 = np.abs(cells[..., 2] - dt[..., 2]) + np.abs(cells[..., 3] - dt[..., 3])
    distance_score = dis + np.minimum(dis_2, dis_3)

    return iou, distance_score


def _select_best_cell_indices(iou_scores, distance_scores):
    """按旧实现的(1-IoU, distance)排序规则，为每个OCR框选择最优单元格下标。"""
    best_indices = []
    inverse_iou_scores = 1.0 - iou_scores
    for row_index in range(inverse_iou_scores.shape[0]):
        row_inverse_iou = inverse_iou_scores[row_index]
        min_inverse_iou = np.min(row_inverse_iou)
        iou_candidates = np.flatnonzero(row_inverse_iou == min_inverse_iou)
        candidate_distances = distance_scores[row_index, iou_candidates]
        min_distance = np.min(candidate_distances)
        distance_candidates = np.flatnonzero(candidate_distances == min_distance)
        best_indices.append(int(iou_candidates[distance_candidates[0]]))
    return best_indices


class TableMatch:
    def __init__(self, filter_ocr_result=True, use_master=False):
        self.filter_ocr_result = filter_ocr_result
        self.use_master = use_master

    def __call__(self, pred_structures, cell_bboxes, dt_boxes, rec_res):
        if self.filter_ocr_result:
            dt_boxes, rec_res = self._filter_ocr_result(cell_bboxes, dt_boxes, rec_res)
        matched_index = self.match_result(dt_boxes, cell_bboxes)
        pred_html, pred = self.get_pred_html(pred_structures, matched_index, rec_res)
        return pred_html

    def match_result(self, dt_boxes, cell_bboxes, min_iou=0.1**8):
        matched = {}
        dt_boxes = np.asarray(dt_boxes, dtype=np.float64)
        if dt_boxes.size == 0:
            return matched
        dt_boxes = dt_boxes.reshape(-1, 4)

        cell_bboxes = _normalize_cell_bboxes(cell_bboxes)
        if cell_bboxes.size == 0:
            return matched

        for start in range(0, len(dt_boxes), TABLE_MATCH_CHUNK_SIZE):
            end = min(start + TABLE_MATCH_CHUNK_SIZE, len(dt_boxes))
            iou_scores, distance_scores = _pairwise_iou_and_distance(
                dt_boxes[start:end],
                cell_bboxes,
            )
            best_cell_indices = _select_best_cell_indices(iou_scores, distance_scores)

            for offset, best_cell_index in enumerate(best_cell_indices):
                ocr_index = start + offset
                best_inverse_iou = 1.0 - iou_scores[offset, best_cell_index]
                # 保持旧实现的阈值语义：最佳IoU过低时不分配到任何单元格。
                if best_inverse_iou >= 1 - min_iou:
                    continue

                matched.setdefault(best_cell_index, []).append(ocr_index)
        return matched

    def get_pred_html(self, pred_structures, matched_index, ocr_contents):
        end_html = []
        td_index = 0
        for tag in pred_structures:
            if "</td>" not in tag:
                end_html.append(tag)
                continue

            if "<td></td>" == tag:
                end_html.extend("<td>")

            if td_index in matched_index.keys():
                b_with = False
                if (
                    "<b>" in ocr_contents[matched_index[td_index][0]]
                    and len(matched_index[td_index]) > 1
                ):
                    b_with = True
                    end_html.extend("<b>")

                for i, td_index_index in enumerate(matched_index[td_index]):
                    content = ocr_contents[td_index_index][0]
                    if len(matched_index[td_index]) > 1:
                        if len(content) == 0:
                            continue

                        if content[0] == " ":
                            content = content[1:]

                        if "<b>" in content:
                            content = content[3:]

                        if "</b>" in content:
                            content = content[:-4]

                        if len(content) == 0:
                            continue

                        if i != len(matched_index[td_index]) - 1 and " " != content[-1]:
                            content += " "
                    end_html.extend(content)

                if b_with:
                    end_html.extend("</b>")

            if "<td></td>" == tag:
                end_html.append("</td>")
            else:
                end_html.append(tag)

            td_index += 1

        # Filter <thead></thead><tbody></tbody> elements
        filter_elements = ["<thead>", "</thead>", "<tbody>", "</tbody>"]
        end_html = [v for v in end_html if v not in filter_elements]
        return "".join(end_html), end_html

    def decode_logic_points(self, pred_structures):
        logic_points = []
        current_row = 0
        current_col = 0
        max_rows = 0
        max_cols = 0
        occupied_cells = {}  # 用于记录已经被占用的单元格

        def is_occupied(row, col):
            return (row, col) in occupied_cells

        def mark_occupied(row, col, rowspan, colspan):
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    occupied_cells[(r, c)] = True

        i = 0
        while i < len(pred_structures):
            token = pred_structures[i]

            if token == "<tr>":
                current_col = 0  # 每次遇到 <tr> 时，重置当前列号
            elif token == "</tr>":
                current_row += 1  # 行结束，行号增加
            elif token.startswith("<td"):
                colspan = 1
                rowspan = 1
                j = i
                if token != "<td></td>":
                    j += 1
                    # 提取 colspan 和 rowspan 属性
                    while j < len(pred_structures) and not pred_structures[
                        j
                    ].startswith(">"):
                        if "colspan=" in pred_structures[j]:
                            colspan = int(pred_structures[j].split("=")[1].strip("\"'"))
                        elif "rowspan=" in pred_structures[j]:
                            rowspan = int(pred_structures[j].split("=")[1].strip("\"'"))
                        j += 1

                # 跳过已经处理过的属性 token
                i = j

                # 找到下一个未被占用的列
                while is_occupied(current_row, current_col):
                    current_col += 1

                # 计算逻辑坐标
                r_start = current_row
                r_end = current_row + rowspan - 1
                col_start = current_col
                col_end = current_col + colspan - 1

                # 记录逻辑坐标
                logic_points.append([r_start, r_end, col_start, col_end])

                # 标记占用的单元格
                mark_occupied(r_start, col_start, rowspan, colspan)

                # 更新当前列号
                current_col += colspan

                # 更新最大行数和列数
                max_rows = max(max_rows, r_end + 1)
                max_cols = max(max_cols, col_end + 1)

            i += 1

        return logic_points

    def _filter_ocr_result(self, cell_bboxes, dt_boxes, rec_res):
        y1 = cell_bboxes[:, 1::2].min()
        new_dt_boxes = []
        new_rec_res = []

        for box, rec in zip(dt_boxes, rec_res):
            if np.max(box[1::2]) < y1:
                continue
            new_dt_boxes.append(box)
            new_rec_res.append(rec)
        return new_dt_boxes, new_rec_res
