from typing import Any, Dict, List, Union, Tuple

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


def match_ocr_cell(dt_rec_boxes: List[List[Union[Any, str]]], pred_bboxes: np.ndarray):
    """
    :param dt_rec_boxes: [[(4.2), text, score]]
    :param pred_bboxes: shap (4,2)
    :return:
    """
    matched = {}
    not_match_orc_boxes = []
    for i, gt_box in enumerate(dt_rec_boxes):
        for j, pred_box in enumerate(pred_bboxes):
            pred_box = [pred_box[0][0], pred_box[0][1], pred_box[2][0], pred_box[2][1]]
            ocr_boxes = gt_box[0]
            # xmin,ymin,xmax,ymax
            ocr_box = (
                ocr_boxes[0][0],
                ocr_boxes[0][1],
                ocr_boxes[2][0],
                ocr_boxes[2][1],
            )
            contained = is_box_contained(ocr_box, pred_box, 0.6)
            if contained == 1 or calculate_iou(ocr_box, pred_box) > 0.8:
                if j not in matched:
                    matched[j] = [gt_box]
                else:
                    matched[j].append(gt_box)
            else:
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


def plot_html_table(
    logi_points: Union[Union[np.ndarray, List]], cell_box_map: Dict[int, List[str]]
) -> str:
    # 初始化最大行数和列数
    max_row = 0
    max_col = 0
    # 计算最大行数和列数
    for point in logi_points:
        max_row = max(max_row, point[1] + 1)  # 加1是因为结束下标是包含在内的
        max_col = max(max_col, point[3] + 1)  # 加1是因为结束下标是包含在内的

    # 创建一个二维数组来存储 sorted_logi_points 中的元素
    grid = [[None] * max_col for _ in range(max_row)]

    valid_start_row = (1 << 16) - 1
    valid_start_col = (1 << 16) - 1
    valid_end_col = 0
    # 将 sorted_logi_points 中的元素填充到 grid 中
    for i, logic_point in enumerate(logi_points):
        row_start, row_end, col_start, col_end = (
            logic_point[0],
            logic_point[1],
            logic_point[2],
            logic_point[3],
        )
        ocr_rec_text_list = cell_box_map.get(i)
        if ocr_rec_text_list and "".join(ocr_rec_text_list):
            valid_start_row = min(row_start, valid_start_row)
            valid_start_col = min(col_start, valid_start_col)
            valid_end_col = max(col_end, valid_end_col)
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                grid[row][col] = (i, row_start, row_end, col_start, col_end)

    # 创建表格
    table_html = "<html><body><table>"

    # 遍历每行
    for row in range(max_row):
        if row < valid_start_row:
            continue
        temp = "<tr>"
        # 遍历每一列
        for col in range(max_col):
            if col < valid_start_col or col > valid_end_col:
                continue
            if not grid[row][col]:
                temp += "<td></td>"
            else:
                i, row_start, row_end, col_start, col_end = grid[row][col]
                if not cell_box_map.get(i):
                    continue
                if row == row_start and col == col_start:
                    ocr_rec_text = cell_box_map.get(i)
                    # text = "<br>".join(ocr_rec_text)
                    text = "".join(ocr_rec_text)
                    # 如果是起始单元格
                    row_span = row_end - row_start + 1
                    col_span = col_end - col_start + 1
                    cell_content = (
                        f"<td rowspan={row_span} colspan={col_span}>{text}</td>"
                    )
                    temp += cell_content

        table_html = table_html + temp + "</tr>"

    table_html += "</table></body></html>"
    return table_html



