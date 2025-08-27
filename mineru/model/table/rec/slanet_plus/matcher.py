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
        for i, gt_box in enumerate(dt_boxes):
            distances = []
            for j, pred_box in enumerate(cell_bboxes):
                if len(pred_box) == 8:
                    pred_box = [
                        np.min(pred_box[0::2]),
                        np.min(pred_box[1::2]),
                        np.max(pred_box[0::2]),
                        np.max(pred_box[1::2]),
                    ]
                distances.append(
                    (distance(gt_box, pred_box), 1.0 - compute_iou(gt_box, pred_box))
                )  # compute iou and l1 distance
            sorted_distances = distances.copy()
            # select det box by iou and l1 distance
            sorted_distances = sorted(
                sorted_distances, key=lambda item: (item[1], item[0])
            )
            # must > min_iou
            if sorted_distances[0][1] >= 1 - min_iou:
                continue

            if distances.index(sorted_distances[0]) not in matched:
                matched[distances.index(sorted_distances[0])] = [i]
            else:
                matched[distances.index(sorted_distances[0])].append(i)
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
