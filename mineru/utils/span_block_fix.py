# Copyright (c) Opendatalab. All rights reserved.
from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.ocr_utils import __is_overlaps_y_exceeds_threshold


def fill_spans_in_blocks(blocks, spans, radio):
    """将allspans中的span按位置关系，放入blocks中."""
    block_with_spans = []
    for block in blocks:
        block_type = block[7]
        block_bbox = block[0:4]
        block_dict = {
            'type': block_type,
            'bbox': block_bbox,
        }
        if block_type in [
            BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
        ]:
            block_dict['group_id'] = block[-1]
        block_spans = []
        for span in spans:
            span_bbox = span['bbox']
            if calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > radio and span_block_type_compatible(
                    span['type'], block_type):
                block_spans.append(span)

        block_dict['spans'] = block_spans
        block_with_spans.append(block_dict)

        # 从spans删除已经放入block_spans中的span
        if len(block_spans) > 0:
            for span in block_spans:
                spans.remove(span)

    return block_with_spans, spans


def span_block_type_compatible(span_type, block_type):
    if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
        return block_type in [
            BlockType.TEXT,
            BlockType.TITLE,
            BlockType.IMAGE_CAPTION,
            BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_CAPTION,
            BlockType.TABLE_FOOTNOTE,
            BlockType.DISCARDED
        ]
    elif span_type == ContentType.INTERLINE_EQUATION:
        return block_type in [BlockType.INTERLINE_EQUATION, BlockType.TEXT]
    elif span_type == ContentType.IMAGE:
        return block_type in [BlockType.IMAGE_BODY]
    elif span_type == ContentType.TABLE:
        return block_type in [BlockType.TABLE_BODY]
    else:
        return False


def fix_discarded_block(discarded_block_with_spans):
    fix_discarded_blocks = []
    for block in discarded_block_with_spans:
        block = fix_text_block(block)
        fix_discarded_blocks.append(block)
    return fix_discarded_blocks


def fix_text_block(block):
    # 文本block中的公式span都应该转换成行内type
    for span in block['spans']:
        if span['type'] == ContentType.INTERLINE_EQUATION:
            span['type'] = ContentType.INLINE_EQUATION
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
            if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                # 否则，开始新行
                lines.append(current_line)
                current_line = [span]

        # 添加最后一行
        if current_line:
            lines.append(current_line)

        return lines


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


def fix_block_spans(block_with_spans):
    fix_blocks = []
    for block in block_with_spans:
        block_type = block['type']

        if block_type in [BlockType.TEXT, BlockType.TITLE,
                          BlockType.IMAGE_CAPTION, BlockType.IMAGE_CAPTION,
                          BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
                          ]:
            block = fix_text_block(block)
        elif block_type in [BlockType.INTERLINE_EQUATION, BlockType.IMAGE_BODY, BlockType.TABLE_BODY]:
            block = fix_interline_block(block)
        else:
            continue
        fix_blocks.append(block)
    return fix_blocks


def fix_interline_block(block):
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block