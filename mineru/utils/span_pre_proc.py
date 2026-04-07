# Copyright (c) Opendatalab. All rights reserved.
import collections
import math
import re
import statistics

import cv2
import numpy as np
from loguru import logger

from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.pdf_image_tools import get_crop_img
from mineru.utils.pdf_text_tool import get_page

MAX_NATIVE_TEXT_CHARS_PER_PAGE = 65535


def __replace_ligatures(text: str):
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'ft', 'ﬆ': 'st'
    }
    return re.sub('|'.join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)

def __replace_unicode(text: str):
    ligatures = {
        '\r\n': '', '\u0002': '-',
    }
    return re.sub('|'.join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)


"""pdf_text dict方案 char级别"""
def txt_spans_extract(pdf_page, spans, pil_img, scale, all_bboxes, all_discarded_blocks):
    page_char_count = None
    try:
        page_char_count = pdf_page.get_textpage().count_chars()
    except Exception as exc:
        logger.debug(f"Failed to get page char count before txt extraction: {exc}")

    if page_char_count is not None and page_char_count > MAX_NATIVE_TEXT_CHARS_PER_PAGE:
        logger.info(
            "Fallback to post-OCR in txt_spans_extract due to high char count: "
            f"count_chars={page_char_count}"
        )
        need_ocr_spans = [
            span for span in spans if span.get('type') == ContentType.TEXT
        ]
        return _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale)

    page_dict = get_page(pdf_page)

    page_all_chars = []
    page_all_lines = []
    for block in page_dict['blocks']:
        for line in block['lines']:
            rotation_degrees = math.degrees(line['rotation'])
            # 旋转角度不为0, 90, 180, 270的行，直接跳过（rotation_degrees的值可能不为整数）
            if not any(abs(rotation_degrees - angle) < 0.1 for angle in [0, 90, 180, 270]):
                continue
            page_all_lines.append(line)
            for span in line['spans']:
                for char in span['chars']:
                    page_all_chars.append(char)

    # 计算所有sapn的高度的中位数
    span_height_list = []
    for span in spans:
        if span['type'] in [ContentType.TEXT]:
            span_height = span['bbox'][3] - span['bbox'][1]
            span['height'] = span_height
            span['width'] = span['bbox'][2] - span['bbox'][0]
            span_height_list.append(span_height)
    if len(span_height_list) == 0:
        return spans
    else:
        median_span_height = statistics.median(span_height_list)

    useful_spans = []
    unuseful_spans = []
    # 纵向span的两个特征：1. 高度超过多个line 2. 高宽比超过某个值
    vertical_spans = []
    for span in spans:
        if span['type'] in [ContentType.TEXT]:
            for block in all_bboxes + all_discarded_blocks:
                if block[7] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.INTERLINE_EQUATION]:
                    continue
                if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block[0:4]) > 0.5:
                    if span['height'] > median_span_height * 2.3 and span['height'] > span['width'] * 2.3:
                        vertical_spans.append(span)
                    elif block in all_bboxes:
                        useful_spans.append(span)
                    else:
                        unuseful_spans.append(span)
                    break

    """垂直的span框直接用line进行填充"""
    if len(vertical_spans) > 0:
        for pdfium_line in page_all_lines:
            for span in vertical_spans:
                if calculate_overlap_area_in_bbox1_area_ratio(pdfium_line['bbox'].bbox, span['bbox']) > 0.5:
                    for pdfium_span in pdfium_line['spans']:
                        span['content'] += pdfium_span['text']
                    break

        for span in vertical_spans:
            if len(span['content']) == 0:
                spans.remove(span)

    """水平的span框先用char填充，再用ocr填充空的span框"""
    new_spans = []

    for span in useful_spans + unuseful_spans:
        if span['type'] in [ContentType.TEXT]:
            span['chars'] = []
            new_spans.append(span)

    need_ocr_spans = fill_char_in_spans(new_spans, page_all_chars, median_span_height)

    return _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale)


def _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale):
    if len(need_ocr_spans) == 0:
        return spans

    for span in need_ocr_spans:
        # 对span的bbox截图再ocr
        span_pil_img = get_crop_img(span['bbox'], pil_img, scale)
        span_img = cv2.cvtColor(np.array(span_pil_img), cv2.COLOR_RGB2BGR)
        # 计算span的对比度，低于0.17的span不进行ocr
        if calculate_contrast(span_img, img_mode='bgr') <= 0.17:
            if span in spans:
                spans.remove(span)
            continue

        span['content'] = ''
        span['score'] = 1.0
        span['np_img'] = span_img

    return spans


def fill_char_in_spans(spans, all_chars, median_span_height):
    # 简单从上到下排一下序
    spans = sorted(spans, key=lambda x: x['bbox'][1])

    grid_size = median_span_height
    grid = collections.defaultdict(list)
    for i, span in enumerate(spans):
        start_cell = int(span['bbox'][1] / grid_size)
        end_cell = int(span['bbox'][3] / grid_size)
        for cell_idx in range(start_cell, end_cell + 1):
            grid[cell_idx].append(i)

    for char in all_chars:
        char_center_y = (char['bbox'][1] + char['bbox'][3]) / 2
        cell_idx = int(char_center_y / grid_size)

        candidate_span_indices = grid.get(cell_idx, [])

        for span_idx in candidate_span_indices:
            span = spans[span_idx]
            if calculate_char_in_span(char['bbox'], span['bbox'], char['char']):
                span['chars'].append(char)
                break

    need_ocr_spans = []
    for span in spans:
        chars_to_content(span)
        # 有的span中虽然没有字但有一两个空的占位符，用宽高和content长度过滤
        if len(span['content']) * span['height'] < span['width'] * 0.5:
            # logger.info(f"maybe empty span: {len(span['content'])}, {span['height']}, {span['width']}")
            need_ocr_spans.append(span)
        del span['height'], span['width']
    return need_ocr_spans


LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；', ']', '】', '}', '}', '>', '》', '、', ',', '，', '-', '—', '–',)
LINE_START_FLAG = ('(', '（', '"', '“', '【', '{', '《', '<', '「', '『', '【', '[',)

Span_Height_Ratio = 0.33  # 字符的中轴和span的中轴高度差不能超过1/3span高度
def calculate_char_in_span(char_bbox, span_bbox, char, span_height_ratio=Span_Height_Ratio):
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    span_center_y = (span_bbox[1] + span_bbox[3]) / 2
    span_height = span_bbox[3] - span_bbox[1]

    if (
        span_bbox[0] < char_center_x < span_bbox[2]
        and span_bbox[1] < char_center_y < span_bbox[3]
        and abs(char_center_y - span_center_y) < span_height * span_height_ratio  # 字符的中轴和span的中轴高度差不能超过Span_Height_Ratio
    ):
        return True
    else:
        # 如果char是LINE_STOP_FLAG，就不用中心点判定，换一种方案（左边界在span区域内，高度判定和之前逻辑一致）
        # 主要是给结尾符号一个进入span的机会，这个char还应该离span右边界较近
        if char in LINE_STOP_FLAG:
            if (
                (span_bbox[2] - span_height) < char_bbox[0] < span_bbox[2]
                and char_center_x > span_bbox[0]
                and span_bbox[1] < char_center_y < span_bbox[3]
                and abs(char_center_y - span_center_y) < span_height * span_height_ratio
            ):
                return True
        elif char in LINE_START_FLAG:
            if (
                span_bbox[0] < char_bbox[2] < (span_bbox[0] + span_height)
                and char_center_x < span_bbox[2]
                and span_bbox[1] < char_center_y < span_bbox[3]
                and abs(char_center_y - span_center_y) < span_height * span_height_ratio
            ):
                return True
        else:
            return False


def chars_to_content(span):
    # 检查span中的char是否为空
    if len(span['chars']) != 0:
        # 给chars按char_idx排序
        chars = sorted(span['chars'], key=lambda x: x['char_idx'])

        # Calculate the width of each character
        char_widths = [char['bbox'][2] - char['bbox'][0] for char in chars]
        # Calculate the median width
        median_width = statistics.median(char_widths)

        parts = []
        for idx, char1 in enumerate(chars):
            char2 = chars[idx + 1] if idx + 1 < len(chars) else None

            # 如果下一个char的x0和上一个char的x1距离超过0.25个字符宽度，则需要在中间插入一个空格
            if (
                char2
                and char2['bbox'][0] - char1['bbox'][2] > median_width * 0.25
                and char1['char'] != ' '
                and char2['char'] != ' '
            ):
                parts.append(char1['char'])
                parts.append(' ')
            else:
                parts.append(char1['char'])

        content = ''.join(parts)
        content = __replace_unicode(content)
        content = __replace_ligatures(content)
        content = __replace_ligatures(content)
        span['content'] = content.strip()

    del span['chars']


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
