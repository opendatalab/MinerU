from loguru import logger

from magic_pdf.libs.boxbase import __is_overlaps_y_exceeds_threshold, get_minbox_if_overlap_by_ratio, \
    calculate_overlap_area_in_bbox1_area_ratio, _is_in_or_part_overlap_with_area_ratio
from magic_pdf.libs.drop_tag import DropTag
from magic_pdf.libs.ocr_content_type import ContentType, BlockType
from magic_pdf.pre_proc.ocr_span_list_modify import modify_y_axis, modify_inline_equation
from magic_pdf.pre_proc.remove_bbox_overlap import remove_overlap_between_bbox_for_span


# 将每一个line中的span从左到右排序
def line_sort_spans_by_left_to_right(lines):
    line_objects = []
    for line in lines:
        # 按照x0坐标排序
        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            "bbox": line_bbox,
            "spans": line,
        })
    return line_objects


def merge_spans_to_line(spans):
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
            if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] or any(
                    s['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] for s in
                    current_line):
                # 则开始新行
                lines.append(current_line)
                current_line = [span]
                continue

            # 如果当前的span与当前行的最后一个span在y轴上重叠，则添加到当前行
            if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox']):
                current_line.append(span)
            else:
                # 否则，开始新行
                lines.append(current_line)
                current_line = [span]

        # 添加最后一行
        if current_line:
            lines.append(current_line)

        return lines


def merge_spans_to_line_by_layout(spans, layout_bboxes):
    lines = []
    new_spans = []
    dropped_spans = []
    for item in layout_bboxes:
        layout_bbox = item['layout_bbox']
        # 遍历spans,将每个span放入对应的layout中
        layout_sapns = []
        for span in spans:
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], layout_bbox) > 0.6:
                layout_sapns.append(span)
        # 如果layout_sapns不为空，则放入new_spans中
        if len(layout_sapns) > 0:
            new_spans.append(layout_sapns)
            # 从spans删除已经放入layout_sapns中的span
            for layout_sapn in layout_sapns:
                spans.remove(layout_sapn)

    if len(new_spans) > 0:
        for layout_sapns in new_spans:
            layout_lines = merge_spans_to_line(layout_sapns)
            lines.extend(layout_lines)

    # 对line中的span进行排序
    lines = line_sort_spans_by_left_to_right(lines)

    for span in spans:
        span['tag'] = DropTag.NOT_IN_LAYOUT
        dropped_spans.append(span)

    return lines, dropped_spans


def merge_lines_to_block(lines):
    # 目前不做block拼接,先做个结构,每个block中只有一个line,block的bbox就是line的bbox
    blocks = []
    for line in lines:
        blocks.append(
            {
                "bbox": line["bbox"],
                "lines": [line],
            }
        )
    return blocks


def sort_blocks_by_layout(all_bboxes, layout_bboxes):
    new_blocks = []
    sort_blocks = []
    for item in layout_bboxes:
        layout_bbox = item['layout_bbox']

        # 遍历blocks,将每个blocks放入对应的layout中
        layout_blocks = []
        for block in all_bboxes:
            # 如果是footnote则跳过
            if block[7] == BlockType.Footnote:
                continue
            block_bbox = block[:4]
            if calculate_overlap_area_in_bbox1_area_ratio(block_bbox, layout_bbox) > 0.8:
                layout_blocks.append(block)

        # 如果layout_blocks不为空，则放入new_blocks中
        if len(layout_blocks) > 0:
            new_blocks.append(layout_blocks)
            # 从all_bboxes删除已经放入layout_blocks中的block
            for layout_block in layout_blocks:
                all_bboxes.remove(layout_block)

    # 如果new_blocks不为空，则对new_blocks中每个block进行排序
    if len(new_blocks) > 0:
        for bboxes_in_layout_block in new_blocks:
            bboxes_in_layout_block.sort(key=lambda x: x[1])  # 一个layout内部的box，按照y0自上而下排序
            sort_blocks.extend(bboxes_in_layout_block)

    # sort_blocks中已经包含了当前页面所有最终留下的block，且已经排好了顺序
    return sort_blocks


def fill_spans_in_blocks(blocks, spans, radio):
    '''
    将allspans中的span按位置关系，放入blocks中
    '''
    block_with_spans = []
    for block in blocks:
        block_type = block[7]
        block_bbox = block[0:4]
        block_dict = {
            'type': block_type,
            'bbox': block_bbox,
        }
        block_spans = []
        for span in spans:
            span_bbox = span['bbox']
            if calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > radio:
                block_spans.append(span)

        '''行内公式调整, 高度调整至与同行文字高度一致(优先左侧, 其次右侧)'''
        # displayed_list = []
        # text_inline_lines = []
        # modify_y_axis(block_spans, displayed_list, text_inline_lines)

        '''模型识别错误的行间公式, type类型转换成行内公式'''
        # block_spans = modify_inline_equation(block_spans, displayed_list, text_inline_lines)

        '''bbox去除粘连'''  # 去粘连会影响span的bbox，导致后续fill的时候出错
        # block_spans = remove_overlap_between_bbox_for_span(block_spans)

        block_dict['spans'] = block_spans
        block_with_spans.append(block_dict)

        # 从spans删除已经放入block_spans中的span
        if len(block_spans) > 0:
            for span in block_spans:
                spans.remove(span)

    return block_with_spans, spans


def fix_block_spans(block_with_spans, img_blocks, table_blocks):
    '''
    1、img_block和table_block因为包含caption和footnote的关系，存在block的嵌套关系
        需要将caption和footnote的text_span放入相应img_block和table_block内的
        caption_block和footnote_block中
    2、同时需要删除block中的spans字段
    '''
    fix_blocks = []
    for block in block_with_spans:
        block_type = block['type']

        if block_type == BlockType.Image:
            block = fix_image_block(block, img_blocks)
        elif block_type == BlockType.Table:
            block = fix_table_block(block, table_blocks)
        elif block_type in [BlockType.Text, BlockType.Title]:
            block = fix_text_block(block)
        elif block_type == BlockType.InterlineEquation:
            block = fix_interline_block(block)
        else:
            continue
        fix_blocks.append(block)
    return fix_blocks


def fix_discarded_block(discarded_block_with_spans):
    fix_discarded_blocks = []
    for block in discarded_block_with_spans:
        block = fix_text_block(block)
        fix_discarded_blocks.append(block)
    return fix_discarded_blocks


def merge_spans_to_block(spans: list, block_bbox: list, block_type: str):
    block_spans = []
    # 如果有img_caption，则将img_block中的text_spans放入img_caption_block中
    for span in spans:
        if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block_bbox) > 0.6:
            block_spans.append(span)
    block_lines = merge_spans_to_line(block_spans)
    # 对line中的span进行排序
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block = {
        'bbox': block_bbox,
        'type': block_type,
        'lines': sort_block_lines
    }
    return block, block_spans


def make_body_block(span: dict, block_bbox: list, block_type: str):
    # 创建body_block
    body_line = {
        'bbox': block_bbox,
        'spans': [span],
    }
    body_block = {
        'bbox': block_bbox,
        'type': block_type,
        'lines': [body_line]
    }
    return body_block


def fix_image_block(block, img_blocks):
    block['blocks'] = []
    # 遍历img_blocks,找到与当前block匹配的img_block
    for img_block in img_blocks:
        if _is_in_or_part_overlap_with_area_ratio(block['bbox'], img_block['bbox'], 0.95):

            # 创建img_body_block
            for span in block['spans']:
                if span['type'] == ContentType.Image and img_block['img_body_bbox'] == span['bbox']:
                    # 创建img_body_block
                    img_body_block = make_body_block(span, img_block['img_body_bbox'], BlockType.ImageBody)
                    block['blocks'].append(img_body_block)

                    # 从spans中移除img_body_block中已经放入的span
                    block['spans'].remove(span)
                    break

            # 根据list长度，判断img_block中是否有img_caption
            if img_block['img_caption_bbox'] is not None:
                img_caption_block, img_caption_spans = merge_spans_to_block(
                    block['spans'], img_block['img_caption_bbox'], BlockType.ImageCaption
                )
                block['blocks'].append(img_caption_block)

            break
    del block['spans']
    return block


def fix_table_block(block, table_blocks):
    block['blocks'] = []
    # 遍历table_blocks,找到与当前block匹配的table_block
    for table_block in table_blocks:
        if _is_in_or_part_overlap_with_area_ratio(block['bbox'], table_block['bbox'], 0.95):

            # 创建table_body_block
            for span in block['spans']:
                if span['type'] == ContentType.Table and table_block['table_body_bbox'] == span['bbox']:
                    # 创建table_body_block
                    table_body_block = make_body_block(span, table_block['table_body_bbox'], BlockType.TableBody)
                    block['blocks'].append(table_body_block)

                    # 从spans中移除img_body_block中已经放入的span
                    block['spans'].remove(span)
                    break

            # 根据list长度，判断table_block中是否有caption
            if table_block['table_caption_bbox'] is not None:
                table_caption_block, table_caption_spans = merge_spans_to_block(
                    block['spans'], table_block['table_caption_bbox'], BlockType.TableCaption
                )
                block['blocks'].append(table_caption_block)

                # 如果table_caption_block_spans不为空
                if len(table_caption_spans) > 0:
                    #  一些span已经放入了caption_block中，需要从block['spans']中删除
                    for span in table_caption_spans:
                        block['spans'].remove(span)

            # 根据list长度，判断table_block中是否有table_note
            if table_block['table_footnote_bbox'] is not None:
                table_footnote_block, table_footnote_spans = merge_spans_to_block(
                    block['spans'], table_block['table_footnote_bbox'], BlockType.TableFootnote
                )
                block['blocks'].append(table_footnote_block)

            break
    del block['spans']
    return block


def fix_text_block(block):
    # 文本block中的公式span都应该转换成行内type
    for span in block['spans']:
        if span['type'] == ContentType.InterlineEquation:
            span['type'] = ContentType.InlineEquation
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block


def fix_interline_block(block):
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block
