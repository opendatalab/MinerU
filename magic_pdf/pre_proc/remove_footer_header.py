import re

from magic_pdf.libs.boxbase import _is_in_or_part_overlap
from magic_pdf.libs.drop_tag import CONTENT_IN_FOOT_OR_HEADER, PAGE_NO


def remove_headder_footer_one_page(text_raw_blocks, image_bboxes, table_bboxes, header_bboxs, footer_bboxs,
                                   page_no_bboxs, page_w, page_h):
    """
    删除页眉页脚，页码
    从line级别进行删除，删除之后观察这个text-block是否是空的，如果是空的，则移动到remove_list中
    """
    header = []
    footer = []
    if len(header) == 0:
        model_header = header_bboxs
        if model_header:
            x0 = min([x for x, _, _, _ in model_header])
            y0 = min([y for _, y, _, _ in model_header])
            x1 = max([x1 for _, _, x1, _ in model_header])
            y1 = max([y1 for _, _, _, y1 in model_header])
            header = [x0, y0, x1, y1]
    if len(footer) == 0:
        model_footer = footer_bboxs
        if model_footer:
            x0 = min([x for x, _, _, _ in model_footer])
            y0 = min([y for _, y, _, _ in model_footer])
            x1 = max([x1 for _, _, x1, _ in model_footer])
            y1 = max([y1 for _, _, _, y1 in model_footer])
            footer = [x0, y0, x1, y1]

    header_y0 = 0 if len(header) == 0 else header[3]
    footer_y0 = page_h if len(footer) == 0 else footer[1]
    if page_no_bboxs:
        top_part = [b for b in page_no_bboxs if b[3] < page_h / 2]
        btn_part = [b for b in page_no_bboxs if b[1] > page_h / 2]

        top_max_y0 = max([b[1] for b in top_part]) if top_part else 0
        btn_min_y1 = min([b[3] for b in btn_part]) if btn_part else page_h

        header_y0 = max(header_y0, top_max_y0)
        footer_y0 = min(footer_y0, btn_min_y1)

    content_boundry = [0, header_y0, page_w, footer_y0]

    header = [0, 0, page_w, header_y0]
    footer = [0, footer_y0, page_w, page_h]

    """以上计算出来了页眉页脚的边界，下面开始进行删除"""
    text_block_to_remove = []
    # 首先检查每个textblock
    for blk in text_raw_blocks:
        if len(blk['lines']) > 0:
            for line in blk['lines']:
                line_del = []
                for span in line['spans']:
                    span_del = []
                    if span['bbox'][3] < header_y0:
                        span_del.append(span)
                    elif _is_in_or_part_overlap(span['bbox'], header) or _is_in_or_part_overlap(span['bbox'], footer):
                        span_del.append(span)
                for span in span_del:
                    line['spans'].remove(span)
                if not line['spans']:
                    line_del.append(line)

            for line in line_del:
                blk['lines'].remove(line)
        else:
            # if not blk['lines']:
            blk['tag'] = CONTENT_IN_FOOT_OR_HEADER
            text_block_to_remove.append(blk)

    """有的时候由于pageNo太小了，总是会有一点和content_boundry重叠一点，被放入正文，因此对于pageNo，进行span粒度的删除"""
    page_no_block_2_remove = []
    if page_no_bboxs:
        for pagenobox in page_no_bboxs:
            for block in text_raw_blocks:
                if _is_in_or_part_overlap(pagenobox, block['bbox']):  # 在span级别删除页码
                    for line in block['lines']:
                        for span in line['spans']:
                            if _is_in_or_part_overlap(pagenobox, span['bbox']):
                                # span['text'] = ''
                                span['tag'] = PAGE_NO
                                # 检查这个block是否只有这一个span，如果是，那么就把这个block也删除
                                if len(line['spans']) == 1 and len(block['lines']) == 1:
                                    page_no_block_2_remove.append(block)
    else:
        # 测试最后一个是不是页码：规则是，最后一个block仅有1个line,一个span,且text是数字，空格，符号组成，不含字母,并且包含数字
        if len(text_raw_blocks) > 0:
            text_raw_blocks.sort(key=lambda x: x['bbox'][1], reverse=True)
            last_block = text_raw_blocks[0]
            if len(last_block['lines']) == 1:
                last_line = last_block['lines'][0]
                if len(last_line['spans']) == 1:
                    last_span = last_line['spans'][0]
                    if last_span['text'].strip() and not re.search('[a-zA-Z]', last_span['text']) and re.search('[0-9]',
                                                                                                                last_span[
                                                                                                                    'text']):
                        last_span['tag'] = PAGE_NO
                        page_no_block_2_remove.append(last_block)

    for b in page_no_block_2_remove:
        text_block_to_remove.append(b)

    for blk in text_block_to_remove:
        if blk in text_raw_blocks:
            text_raw_blocks.remove(blk)

    text_block_remain = text_raw_blocks
    image_bbox_to_remove = [bbox for bbox in image_bboxes if not _is_in_or_part_overlap(bbox, content_boundry)]

    image_bbox_remain = [bbox for bbox in image_bboxes if _is_in_or_part_overlap(bbox, content_boundry)]
    table_bbox_to_remove = [bbox for bbox in table_bboxes if not _is_in_or_part_overlap(bbox, content_boundry)]
    table_bbox_remain = [bbox for bbox in table_bboxes if _is_in_or_part_overlap(bbox, content_boundry)]

    return image_bbox_remain, table_bbox_remain, text_block_remain, text_block_to_remove, image_bbox_to_remove, table_bbox_to_remove
