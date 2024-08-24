from loguru import logger

from magic_pdf.libs.boxbase import get_minbox_if_overlap_by_ratio, calculate_overlap_area_in_bbox1_area_ratio, \
    calculate_iou
from magic_pdf.libs.drop_tag import DropTag
from magic_pdf.libs.ocr_content_type import BlockType
from magic_pdf.pre_proc.remove_bbox_overlap import remove_overlap_between_bbox_for_block


def ocr_prepare_bboxes_for_layout_split(img_blocks, table_blocks, discarded_blocks, text_blocks,
                                        title_blocks, interline_equation_blocks, page_w, page_h):
    all_bboxes = []
    all_discarded_blocks = []
    for image in img_blocks:
        x0, y0, x1, y1 = image['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Image, None, None, None, None, image["score"]])

    for table in table_blocks:
        x0, y0, x1, y1 = table['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Table, None, None, None, None, table["score"]])

    for text in text_blocks:
        x0, y0, x1, y1 = text['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Text, None, None, None, None, text["score"]])

    for title in title_blocks:
        x0, y0, x1, y1 = title['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Title, None, None, None, None, title["score"]])

    for interline_equation in interline_equation_blocks:
        x0, y0, x1, y1 = interline_equation['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.InterlineEquation, None, None, None, None, interline_equation["score"]])

    '''block嵌套问题解决'''
    '''文本框与标题框重叠，优先信任文本框'''
    all_bboxes = fix_text_overlap_title_blocks(all_bboxes)
    '''任何框体与舍弃框重叠，优先信任舍弃框'''
    all_bboxes = remove_need_drop_blocks(all_bboxes, discarded_blocks)

    # interline_equation 与title或text框冲突的情况，分两种情况处理
    '''interline_equation框与文本类型框iou比较接近1的时候，信任行间公式框'''
    all_bboxes = fix_interline_equation_overlap_text_blocks_with_hi_iou(all_bboxes)
    '''interline_equation框被包含在文本类型框内，且interline_equation比文本区块小很多时信任文本框，这时需要舍弃公式框'''
    # 通过后续大框套小框逻辑删除

    '''discarded_blocks中只保留宽度超过1/3页面宽度的，高度超过10的，处于页面下半50%区域的（限定footnote）'''
    for discarded in discarded_blocks:
        x0, y0, x1, y1 = discarded['bbox']
        all_discarded_blocks.append([x0, y0, x1, y1, None, None, None, BlockType.Discarded, None, None, None, None, discarded["score"]])
        # 将footnote加入到all_bboxes中，用来计算layout
        if (x1 - x0) > (page_w / 3) and (y1 - y0) > 10 and y0 > (page_h / 2):
            all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Footnote, None, None, None, None, discarded["score"]])

    '''经过以上处理后，还存在大框套小框的情况，则删除小框'''
    all_bboxes = remove_overlaps_min_blocks(all_bboxes)
    all_discarded_blocks = remove_overlaps_min_blocks(all_discarded_blocks)
    '''将剩余的bbox做分离处理，防止后面分layout时出错'''
    all_bboxes, drop_reasons = remove_overlap_between_bbox_for_block(all_bboxes)

    return all_bboxes, all_discarded_blocks, drop_reasons


def fix_interline_equation_overlap_text_blocks_with_hi_iou(all_bboxes):
    # 先提取所有text和interline block
    text_blocks = []
    for block in all_bboxes:
        if block[7] == BlockType.Text:
            text_blocks.append(block)
    interline_equation_blocks = []
    for block in all_bboxes:
        if block[7] == BlockType.InterlineEquation:
            interline_equation_blocks.append(block)

    need_remove = []

    for interline_equation_block in interline_equation_blocks:
        for text_block in text_blocks:
            interline_equation_block_bbox = interline_equation_block[:4]
            text_block_bbox = text_block[:4]
            if calculate_iou(interline_equation_block_bbox, text_block_bbox) > 0.8:
                if text_block not in need_remove:
                    need_remove.append(text_block)

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)

    return all_bboxes


def fix_text_overlap_title_blocks(all_bboxes):
    # 先提取所有text和title block
    text_blocks = []
    for block in all_bboxes:
        if block[7] == BlockType.Text:
            text_blocks.append(block)
    title_blocks = []
    for block in all_bboxes:
        if block[7] == BlockType.Title:
            title_blocks.append(block)

    need_remove = []

    for text_block in text_blocks:
        for title_block in title_blocks:
            text_block_bbox = text_block[:4]
            title_block_bbox = title_block[:4]
            if calculate_iou(text_block_bbox, title_block_bbox) > 0.8:
                if title_block not in need_remove:
                    need_remove.append(title_block)

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)

    return all_bboxes


def remove_need_drop_blocks(all_bboxes, discarded_blocks):
    need_remove = []
    for block in all_bboxes:
        for discarded_block in discarded_blocks:
            block_bbox = block[:4]
            if calculate_overlap_area_in_bbox1_area_ratio(block_bbox, discarded_block['bbox']) > 0.6:
                if block not in need_remove:
                    need_remove.append(block)
                    break

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)
    return all_bboxes


def remove_overlaps_min_blocks(all_bboxes):
    #  删除重叠blocks中较小的那些
    need_remove = []
    for block1 in all_bboxes:
        for block2 in all_bboxes:
            if block1 != block2:
                block1_bbox = block1[:4]
                block2_bbox = block2[:4]
                overlap_box = get_minbox_if_overlap_by_ratio(block1_bbox, block2_bbox, 0.8)
                if overlap_box is not None:
                    bbox_to_remove = next((block for block in all_bboxes if block[:4] == overlap_box), None)
                    if bbox_to_remove is not None and bbox_to_remove not in need_remove:
                        need_remove.append(bbox_to_remove)

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)

    return all_bboxes
