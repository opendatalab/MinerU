# Copyright (c) Opendatalab. All rights reserved.
import copy

import cv2
import numpy as np
from magic_pdf.pre_proc.ocr_dict_merge import merge_spans_to_line
from magic_pdf.libs.boxbase import __is_overlaps_y_exceeds_threshold


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img


def preprocess_image(_image):
    alpha_color = (255, 255, 255)
    _image = alpha_to_color(_image, alpha_color)
    return _image


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def bbox_to_points(bbox):
    """ 将bbox格式转换为四个顶点的数组 """
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype('float32')


def points_to_bbox(points):
    """ 将四个顶点的数组转换为bbox格式 """
    x0, y0 = points[0]
    x1, _ = points[1]
    _, y1 = points[2]
    return [x0, y0, x1, y1]


def merge_intervals(intervals):
    # Sort the intervals based on the start value
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # If the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Otherwise, there is overlap, so we merge the current and previous intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def remove_intervals(original, masks):
    # Merge all mask intervals
    merged_masks = merge_intervals(masks)

    result = []
    original_start, original_end = original

    for mask in merged_masks:
        mask_start, mask_end = mask

        # If the mask starts after the original range, ignore it
        if mask_start > original_end:
            continue

        # If the mask ends before the original range starts, ignore it
        if mask_end < original_start:
            continue

        # Remove the masked part from the original range
        if original_start < mask_start:
            result.append([original_start, mask_start - 1])

        original_start = max(mask_end + 1, original_start)

    # Add the remaining part of the original range, if any
    if original_start <= original_end:
        result.append([original_start, original_end])

    return result


def update_det_boxes(dt_boxes, mfd_res):
    new_dt_boxes = []
    angle_boxes_list = []
    for text_box in dt_boxes:

        if calculate_is_angle(text_box):
            angle_boxes_list.append(text_box)
            continue

        text_bbox = points_to_bbox(text_box)
        masks_list = []
        for mf_box in mfd_res:
            mf_bbox = mf_box['bbox']
            if __is_overlaps_y_exceeds_threshold(text_bbox, mf_bbox):
                masks_list.append([mf_bbox[0], mf_bbox[2]])
        text_x_range = [text_bbox[0], text_bbox[2]]
        text_remove_mask_range = remove_intervals(text_x_range, masks_list)
        temp_dt_box = []
        for text_remove_mask in text_remove_mask_range:
            temp_dt_box.append(bbox_to_points([text_remove_mask[0], text_bbox[1], text_remove_mask[1], text_bbox[3]]))
        if len(temp_dt_box) > 0:
            new_dt_boxes.extend(temp_dt_box)

    new_dt_boxes.extend(angle_boxes_list)

    return new_dt_boxes


def merge_overlapping_spans(spans):
    """
    Merges overlapping spans on the same line.

    :param spans: A list of span coordinates [(x1, y1, x2, y2), ...]
    :return: A list of merged spans
    """
    # Return an empty list if the input spans list is empty
    if not spans:
        return []

    # Sort spans by their starting x-coordinate
    spans.sort(key=lambda x: x[0])

    # Initialize the list of merged spans
    merged = []
    for span in spans:
        # Unpack span coordinates
        x1, y1, x2, y2 = span
        # If the merged list is empty or there's no horizontal overlap, add the span directly
        if not merged or merged[-1][2] < x1:
            merged.append(span)
        else:
            # If there is horizontal overlap, merge the current span with the previous one
            last_span = merged.pop()
            # Update the merged span's top-left corner to the smaller (x1, y1) and bottom-right to the larger (x2, y2)
            x1 = min(last_span[0], x1)
            y1 = min(last_span[1], y1)
            x2 = max(last_span[2], x2)
            y2 = max(last_span[3], y2)
            # Add the merged span back to the list
            merged.append((x1, y1, x2, y2))

    # Return the list of merged spans
    return merged


def merge_det_boxes(dt_boxes):
    """
    Merge detection boxes.

    This function takes a list of detected bounding boxes, each represented by four corner points.
    The goal is to merge these bounding boxes into larger text regions.

    Parameters:
    dt_boxes (list): A list containing multiple text detection boxes, where each box is defined by four corner points.

    Returns:
    list: A list containing the merged text regions, where each region is represented by four corner points.
    """
    # Convert the detection boxes into a dictionary format with bounding boxes and type
    dt_boxes_dict_list = []
    angle_boxes_list = []
    for text_box in dt_boxes:
        text_bbox = points_to_bbox(text_box)

        if calculate_is_angle(text_box):
            angle_boxes_list.append(text_box)
            continue

        text_box_dict = {
            'bbox': text_bbox,
            'type': 'text',
        }
        dt_boxes_dict_list.append(text_box_dict)

    # Merge adjacent text regions into lines
    lines = merge_spans_to_line(dt_boxes_dict_list)

    # Initialize a new list for storing the merged text regions
    new_dt_boxes = []
    for line in lines:
        line_bbox_list = []
        for span in line:
            line_bbox_list.append(span['bbox'])

        # Merge overlapping text regions within the same line
        merged_spans = merge_overlapping_spans(line_bbox_list)

        # Convert the merged text regions back to point format and add them to the new detection box list
        for span in merged_spans:
            new_dt_boxes.append(bbox_to_points(span))

    new_dt_boxes.extend(angle_boxes_list)

    return new_dt_boxes


def get_adjusted_mfdetrec_res(single_page_mfdetrec_res, useful_list):
    paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
    # Adjust the coordinates of the formula area
    adjusted_mfdetrec_res = []
    for mf_res in single_page_mfdetrec_res:
        mf_xmin, mf_ymin, mf_xmax, mf_ymax = mf_res["bbox"]
        # Adjust the coordinates of the formula area to the coordinates relative to the cropping area
        x0 = mf_xmin - xmin + paste_x
        y0 = mf_ymin - ymin + paste_y
        x1 = mf_xmax - xmin + paste_x
        y1 = mf_ymax - ymin + paste_y
        # Filter formula blocks outside the graph
        if any([x1 < 0, y1 < 0]) or any([x0 > new_width, y0 > new_height]):
            continue
        else:
            adjusted_mfdetrec_res.append({
                "bbox": [x0, y0, x1, y1],
            })
    return adjusted_mfdetrec_res


def get_ocr_result_list(ocr_res, useful_list, ocr_enable, new_image, lang):
    paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
    ocr_result_list = []
    ori_im = new_image.copy()
    for box_ocr_res in ocr_res:

        if len(box_ocr_res) == 2:
            p1, p2, p3, p4 = box_ocr_res[0]
            text, score = box_ocr_res[1]
            # logger.info(f"text: {text}, score: {score}")
            if score < 0.6:  # 过滤低置信度的结果
                continue
        else:
            p1, p2, p3, p4 = box_ocr_res
            text, score = "", 1

            if ocr_enable:
                tmp_box = copy.deepcopy(np.array([p1, p2, p3, p4]).astype('float32'))
                img_crop = get_rotate_crop_image(ori_im, tmp_box)

        # average_angle_degrees = calculate_angle_degrees(box_ocr_res[0])
        # if average_angle_degrees > 0.5:
        poly = [p1, p2, p3, p4]
        if calculate_is_angle(poly):
            # logger.info(f"average_angle_degrees: {average_angle_degrees}, text: {text}")
            # 与x轴的夹角超过0.5度，对边界做一下矫正
            # 计算几何中心
            x_center = sum(point[0] for point in poly) / 4
            y_center = sum(point[1] for point in poly) / 4
            new_height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
            new_width = p3[0] - p1[0]
            p1 = [x_center - new_width / 2, y_center - new_height / 2]
            p2 = [x_center + new_width / 2, y_center - new_height / 2]
            p3 = [x_center + new_width / 2, y_center + new_height / 2]
            p4 = [x_center - new_width / 2, y_center + new_height / 2]

        # Convert the coordinates back to the original coordinate system
        p1 = [p1[0] - paste_x + xmin, p1[1] - paste_y + ymin]
        p2 = [p2[0] - paste_x + xmin, p2[1] - paste_y + ymin]
        p3 = [p3[0] - paste_x + xmin, p3[1] - paste_y + ymin]
        p4 = [p4[0] - paste_x + xmin, p4[1] - paste_y + ymin]

        if ocr_enable:
            ocr_result_list.append({
                'category_id': 15,
                'poly': p1 + p2 + p3 + p4,
                'score': 1,
                'text': text,
                'np_img': img_crop,
                'lang': lang,
            })
        else:
            ocr_result_list.append({
                'category_id': 15,
                'poly': p1 + p2 + p3 + p4,
                'score': float(round(score, 2)),
                'text': text,
            })

    return ocr_result_list


def calculate_is_angle(poly):
    p1, p2, p3, p4 = poly
    height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
    if 0.8 * height <= (p3[1] - p1[1]) <= 1.2 * height:
        return False
    else:
        # logger.info((p3[1] - p1[1])/height)
        return True


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img