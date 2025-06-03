# Copyright (c) Opendatalab. All rights reserved.
import cv2
import numpy as np

from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio, calculate_iou, \
    get_minbox_if_overlap_by_ratio
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.pdf_image_tools import get_crop_img


def remove_outside_spans(spans, all_bboxes, all_discarded_blocks):
    def get_block_bboxes(blocks, block_type_list):
        return [block[0:4] for block in blocks if block[7] in block_type_list]

    image_bboxes = get_block_bboxes(all_bboxes, [BlockType.IMAGE_BODY])
    table_bboxes = get_block_bboxes(all_bboxes, [BlockType.TABLE_BODY])
    other_block_type = []
    for block_type in BlockType.__dict__.values():
        if not isinstance(block_type, str):
            continue
        if block_type not in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY]:
            other_block_type.append(block_type)
    other_block_bboxes = get_block_bboxes(all_bboxes, other_block_type)
    discarded_block_bboxes = get_block_bboxes(all_discarded_blocks, [BlockType.DISCARDED])

    new_spans = []

    for span in spans:
        span_bbox = span['bbox']
        span_type = span['type']

        if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.4 for block_bbox in
               discarded_block_bboxes):
            new_spans.append(span)
            continue

        if span_type == ContentType.IMAGE:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   image_bboxes):
                new_spans.append(span)
        elif span_type == ContentType.TABLE:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   table_bboxes):
                new_spans.append(span)
        else:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   other_block_bboxes):
                new_spans.append(span)

    return new_spans


def remove_overlaps_low_confidence_spans(spans):
    dropped_spans = []
    #  删除重叠spans中置信度低的的那些
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                # span1 或 span2 任何一个都不应该在 dropped_spans 中
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue
                else:
                    if calculate_iou(span1['bbox'], span2['bbox']) > 0.9:
                        if span1['score'] < span2['score']:
                            span_need_remove = span1
                        else:
                            span_need_remove = span2
                        if (
                            span_need_remove is not None
                            and span_need_remove not in dropped_spans
                        ):
                            dropped_spans.append(span_need_remove)

    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)

    return spans, dropped_spans


def remove_overlaps_min_spans(spans):
    dropped_spans = []
    #  删除重叠spans中较小的那些
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                # span1 或 span2 任何一个都不应该在 dropped_spans 中
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue
                else:
                    overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.65)
                    if overlap_box is not None:
                        span_need_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                        if span_need_remove is not None and span_need_remove not in dropped_spans:
                            dropped_spans.append(span_need_remove)
    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)

    return spans, dropped_spans


def txt_spans_extract(pdf_page, spans, pil_img, scale):

    textpage = pdf_page.get_textpage()
    width, height = pdf_page.get_size()
    cropbox = pdf_page.get_cropbox()
    need_ocr_spans = []
    for span in spans:
        if span['type'] in [ContentType.INTERLINE_EQUATION, ContentType.IMAGE, ContentType.TABLE]:
            continue
        span_bbox = span['bbox']
        rect_box = [span_bbox[0] + cropbox[0],
                    height - span_bbox[3] + cropbox[1],
                    span_bbox[2] + cropbox[0],
                    height - span_bbox[1] + cropbox[1]]
        text = textpage.get_text_bounded(left=rect_box[0], top=rect_box[1],
                                         right=rect_box[2], bottom=rect_box[3])
        if text and len(text) > 0:
            span['content'] = text.strip()
            span['score'] = 1.0
        else:
            need_ocr_spans.append(span)

    if len(need_ocr_spans) > 0:

        for span in need_ocr_spans:
            # 对span的bbox截图再ocr
            span_pil_img = get_crop_img(span['bbox'], pil_img, scale)
            span_img = cv2.cvtColor(np.array(span_pil_img), cv2.COLOR_RGB2BGR)
            # 计算span的对比度，低于0.20的span不进行ocr
            if calculate_contrast(span_img, img_mode='bgr') <= 0.17:
                spans.remove(span)
                continue

            span['content'] = ''
            span['score'] = 1.0
            span['np_img'] = span_img

    return spans


def calculate_contrast(img, img_mode) -> float:
    """
    计算给定图像的对比度。
    :param img: 图像，类型为numpy.ndarray
    :Param img_mode = 图像的色彩通道，'rgb' 或 'bgr'
    :return: 图像的对比度值
    """
    if img_mode == 'rgb':
        # 将RGB图像转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_mode == 'bgr':
        # 将BGR图像转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image mode. Please provide 'rgb' or 'bgr'.")

    # 计算均值和标准差
    mean_value = np.mean(gray_img)
    std_dev = np.std(gray_img)
    # 对比度定义为标准差除以平均值（加上小常数避免除零错误）
    contrast = std_dev / (mean_value + 1e-6)
    # logger.debug(f"contrast: {contrast}")
    return round(contrast, 2)