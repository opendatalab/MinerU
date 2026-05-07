# Copyright (c) Opendatalab. All rights reserved.
from mineru.utils.enum_class import ContentType
from mineru.utils.ocr_utils import _is_overlaps_y_exceeds_threshold, _is_overlaps_x_exceeds_threshold

VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 2
VERTICAL_SPAN_IN_BLOCK_THRESHOLD = 0.8


def is_vertical_text_block_by_spans(spans):
    """根据块内文本 span 的高宽比判断文本块是否更像竖排文本。"""
    valid_span_count = 0
    vertical_span_count = 0
    for span in spans:
        bbox = span.get('bbox')
        if not bbox or len(bbox) < 4:
            continue

        span_width = bbox[2] - bbox[0]
        span_height = bbox[3] - bbox[1]
        if span_width <= 0 or span_height <= 0:
            continue

        valid_span_count += 1
        if span_height / span_width > VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD:
            vertical_span_count += 1

    if valid_span_count == 0:
        return False

    return vertical_span_count / valid_span_count > VERTICAL_SPAN_IN_BLOCK_THRESHOLD


def fix_text_block(block):
    # 文本block中的公式span都应该转换成行内type
    for span in block['spans']:
        if span['type'] == ContentType.INTERLINE_EQUATION:
            span['type'] = ContentType.INLINE_EQUATION

    if is_vertical_text_block_by_spans(block['spans']):
        # 如果是纵向文本块，则按纵向lines处理
        block_lines = merge_spans_to_vertical_line(block['spans'])
        sort_block_lines = vertical_line_sort_spans_from_top_to_bottom(block_lines)
    else:
        block_lines = merge_spans_to_line(block['spans'])
        sort_block_lines = line_sort_spans_by_left_to_right(block_lines)

    block['lines'] = sort_block_lines
    del block['spans']
    return block


def merge_spans_to_line(spans, threshold=0.6):
    if len(spans) == 0:
        return []
    else:
        # 按照y0坐标排序
        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # 如果当前的span类型为"interline_equation" 或者 当前行中已经有"interline_equation"
            # image和table类型，同上
            if span['type'] in [
                    ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                    ContentType.TABLE
            ] or any(s['type'] in [
                    ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                    ContentType.TABLE
            ] for s in current_line):
                # 则开始新行
                lines.append(current_line)
                current_line = [span]
                continue

            # 如果当前的span与当前行的最后一个span在y轴上重叠，则添加到当前行
            if _is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                # 否则，开始新行
                lines.append(current_line)
                current_line = [span]

        # 添加最后一行
        if current_line:
            lines.append(current_line)

        return lines


def merge_spans_to_vertical_line(spans, threshold=0.6):
    """将纵向文本的spans合并成纵向lines（从右向左阅读）"""
    if len(spans) == 0:
        return []
    else:
        # 按照x2坐标从大到小排序（从右向左）
        spans.sort(key=lambda span: span['bbox'][2], reverse=True)

        vertical_lines = []
        current_line = [spans[0]]

        for span in spans[1:]:
            # 特殊类型元素单独成列
            if span['type'] in [
                ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                ContentType.TABLE
            ] or any(s['type'] in [
                ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                ContentType.TABLE
            ] for s in current_line):
                vertical_lines.append(current_line)
                current_line = [span]
                continue

            # 如果当前的span与当前行的最后一个span在y轴上重叠，则添加到当前行
            if _is_overlaps_x_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                vertical_lines.append(current_line)
                current_line = [span]

        # 添加最后一列
        if current_line:
            vertical_lines.append(current_line)

        return vertical_lines


# 将每一个line中的span从左到右排序
def line_sort_spans_by_left_to_right(lines):
    line_objects = []
    for line in lines:
        #  按照x0坐标排序
        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects


def vertical_line_sort_spans_from_top_to_bottom(vertical_lines):
    line_objects = []
    for line in vertical_lines:
        # 按照y0坐标排序（从上到下）
        line.sort(key=lambda span: span['bbox'][1])

        # 计算整个列的边界框
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]

        # 组装结果
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects
