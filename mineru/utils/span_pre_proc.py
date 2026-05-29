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
from mineru.utils.pdf_text_tool import get_lines_from_chars, get_page_chars
from mineru.utils.pdfium_guard import pdfium_guard

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
    textpage = None
    try:
        with pdfium_guard():
            textpage = pdf_page.get_textpage()
            page_char_count = textpage.count_chars()
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

    page_chars = get_page_chars(
        pdf_page,
        textpage=textpage,
        page_char_count=page_char_count,
    )
    page_all_chars = [
        char for char in page_chars['chars']
        if _is_supported_rotation(char['rotation'])
    ]

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
        page_all_lines = [
            line for line in get_lines_from_chars(page_chars['chars'])
            if _is_supported_rotation(line['rotation'])
        ]
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


def _is_supported_rotation(rotation) -> bool:
    """判断 pdftext 旋转角是否属于当前可回填的四个标准方向。"""
    rotation_degrees = math.degrees(rotation)
    return any(abs(rotation_degrees - angle) < 0.1 for angle in [0, 90, 180, 270])


def _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale):
    if len(need_ocr_spans) == 0:
        return spans

    for span in need_ocr_spans:
        # 对span的bbox截图再ocr
        span_pil_img = get_crop_img(span['bbox'], pil_img, scale)
        span_img = cv2.cvtColor(np.array(span_pil_img), cv2.COLOR_RGB2BGR)
        # 计算span的对比度，低于0.17的span不进行ocr，等于0.17的临界框保留给后置OCR。
        if calculate_contrast(span_img, img_mode='bgr') < 0.17:
            if span in spans:
                spans.remove(span)
            continue

        span['content'] = ''
        span['score'] = 1.0
        span['np_img'] = span_img

    return spans


class SpanBlockMatcher:
    """按 block 顺序消费 span，并用 y 方向索引减少无效重叠计算。"""

    def __init__(self, spans):
        self.spans = list(spans)
        self.used_span_indices = set()
        self.grid_size = self._get_grid_size(self.spans)
        self.grid = self._build_grid(self.spans)

    @staticmethod
    def _get_grid_size(spans):
        """根据 span 高度估算索引网格大小，避免过细或过粗。"""
        heights = [
            span['bbox'][3] - span['bbox'][1]
            for span in spans
            if span.get('bbox') and span['bbox'][3] > span['bbox'][1]
        ]
        if not heights:
            return 1
        return max(1, statistics.median(heights))

    def _build_grid(self, spans):
        """将 span 按 y 方向网格登记，后续按 block bbox 快速取候选。"""
        grid = collections.defaultdict(list)
        for index, span in enumerate(spans):
            bbox = span.get('bbox')
            if not bbox:
                continue
            start_cell, end_cell = self._cell_range(bbox)
            for cell_idx in range(start_cell, end_cell + 1):
                grid[cell_idx].append(index)
        return grid

    def _cell_range(self, bbox):
        """计算 bbox 覆盖的 y 方向网格范围。"""
        return (
            int(bbox[1] / self.grid_size),
            int(bbox[3] / self.grid_size),
        )

    def _candidate_indices_for_block(self, block_bbox):
        """取出与 block 纵向范围可能相交的 span 原始索引。"""
        start_cell, end_cell = self._cell_range(block_bbox)
        candidate_indices = set()
        for cell_idx in range(start_cell, end_cell + 1):
            candidate_indices.update(self.grid.get(cell_idx, []))
        return sorted(candidate_indices)

    def collect_for_block(self, block_bbox, overlap_ratio_getter=None, threshold=0.5):
        """返回当前 block 命中的 span，并标记为已消费以保持旧归属语义。"""
        if overlap_ratio_getter is None:
            overlap_ratio_getter = self._default_overlap_ratio

        block_spans = []
        for span_idx in self._candidate_indices_for_block(block_bbox):
            if span_idx in self.used_span_indices:
                continue
            span = self.spans[span_idx]
            if overlap_ratio_getter(span, block_bbox) > threshold:
                block_spans.append(span)
                self.used_span_indices.add(span_idx)
        return block_spans

    def remaining_spans(self):
        """返回尚未归属到任何 block 的 span，方便保持后续兼容。"""
        return [
            span
            for index, span in enumerate(self.spans)
            if index not in self.used_span_indices
        ]

    @staticmethod
    def _default_overlap_ratio(span, block_bbox):
        """默认沿用旧逻辑：计算 span 面积中落入 block 的比例。"""
        return calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block_bbox)


def fill_char_in_spans(spans, all_chars, median_span_height):
    # 简单从上到下排一下序
    spans = sorted(spans, key=lambda x: x['bbox'][1])

    grid_size = max(1, median_span_height)
    grid = collections.defaultdict(list)
    span_bboxes = []
    for i, span in enumerate(spans):
        span_bbox = span['bbox']
        span_bboxes.append(span_bbox)
        start_cell = int(span_bbox[1] / grid_size)
        end_cell = int(span_bbox[3] / grid_size)
        for cell_idx in range(start_cell, end_cell + 1):
            grid[cell_idx].append(i)

    for char in all_chars:
        char_bbox = char['bbox']
        char_center_x = (char_bbox[0] + char_bbox[2]) / 2
        char_center_y = (char_bbox[1] + char_bbox[3]) / 2
        cell_idx = int(char_center_y / grid_size)

        candidate_span_indices = grid.get(cell_idx, [])

        for span_idx in candidate_span_indices:
            span = spans[span_idx]
            span_bbox = span_bboxes[span_idx]
            if (
                char['char'] not in LINE_STOP_FLAG
                and char['char'] not in LINE_START_FLAG
                and not span_bbox[0] < char_center_x < span_bbox[2]
            ):
                continue
            if calculate_char_in_span(char_bbox, span_bbox, char['char']):
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
        chars = span['chars']
        # 大多数情况下 char 已按 PDF 原始顺序进入，只有乱序时才排序。
        if any(
            chars[idx]['char_idx'] > chars[idx + 1]['char_idx']
            for idx in range(len(chars) - 1)
        ):
            chars = sorted(chars, key=lambda x: x['char_idx'])

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
