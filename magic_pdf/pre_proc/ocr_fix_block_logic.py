from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from magic_pdf.libs.ocr_content_type import ContentType, BlockType
from magic_pdf.pre_proc.ocr_dict_merge import merge_spans_to_line, line_sort_spans_by_left_to_right


def merge_spans_to_block(spans: list, block_bbox: list, block_type: str):
    block_spans = []
    # 如果有img_caption，则将img_block中的text_spans放入img_caption_block中
    for span in spans:
        if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block_bbox) > 0.8:
            block_spans.append(span)
    block_lines = merge_spans_to_line(block_spans)
    # 对line中的span进行排序
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block = {
        'bbox': block_bbox,
        'block_type': block_type,
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
        'block_type': block_type,
        'lines': [body_line]
    }
    return body_block


def fix_image_block(block, img_blocks):
    block['blocks'] = []
    # 遍历img_blocks,找到与当前block匹配的img_block
    for img_block in img_blocks:
        if img_block['bbox'] == block['bbox']:
            # 创建img_body_block
            for span in block['spans']:
                if span['type'] == ContentType.Image and span['bbox'] == img_block['img_body_bbox']:
                    # 创建img_body_block
                    img_body_block = make_body_block(span, img_block['img_body_bbox'], BlockType.ImageBody)
                    block['blocks'].append(img_body_block)

                    # 从spans中移除img_body_block中已经放入的span
                    block['spans'].remove(span)
                    break

            # 根据list长度，判断img_block中是否有img_caption
            if len(img_block['img_caption_bbox']) > 0:
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
        if table_block['bbox'] == block['bbox']:
            # 创建table_body_block
            for span in block['spans']:
                if span['type'] == ContentType.Table and span['bbox'] == table_block['table_body_bbox']:
                    # 创建table_body_block
                    table_body_block = make_body_block(span, table_block['table_body_bbox'], BlockType.TableBody)
                    block['blocks'].append(table_body_block)

                    # 从spans中移除img_body_block中已经放入的span
                    block['spans'].remove(span)
                    break

            # 根据list长度，判断table_block中是否有caption
            if len(table_block['table_caption_bbox']) > 0:
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
            if len(table_block['table_footnote_bbox']) > 0:
                table_footnote_block, table_footnote_spans = merge_spans_to_block(
                    block['spans'], table_block['table_footnote_bbox'], BlockType.TableFootnote
                )
                block['blocks'].append(table_footnote_block)

            break
    del block['spans']
    return block


def fix_text_block(block):
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block


