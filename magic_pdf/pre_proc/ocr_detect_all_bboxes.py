from magic_pdf.libs.boxbase import get_minbox_if_overlap_by_ratio
from magic_pdf.libs.drop_tag import DropTag
from magic_pdf.libs.ocr_content_type import BlockType


def ocr_prepare_bboxes_for_layout_split(img_blocks, table_blocks, discarded_blocks, text_blocks,
                                        title_blocks, interline_equation_blocks, page_w, page_h):
    all_bboxes = []

    for image in img_blocks:
        x0, y0, x1, y1 = image['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Image, None, None, None, None])

    for table in table_blocks:
        x0, y0, x1, y1 = table['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Table, None, None, None, None])

    for text in text_blocks:
        x0, y0, x1, y1 = text['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Text, None, None, None, None])

    for title in title_blocks:
        x0, y0, x1, y1 = title['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Title, None, None, None, None])

    for interline_equation in interline_equation_blocks:
        x0, y0, x1, y1 = interline_equation['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.InterlineEquation, None, None, None, None])

    '''discarded_blocks中只保留宽度超过1/3页面宽度的，高度超过10的，处于页面下半50%区域的（限定footnote）'''
    for discarded in discarded_blocks:
        x0, y0, x1, y1 = discarded['bbox']
        if (x1 - x0) > (page_w / 3) and (y1 - y0) > 10 and y0 > (page_h / 2):
            all_bboxes.append([x0, y0, x1, y1, None, None, None, BlockType.Footnote, None, None, None, None])

    '''block嵌套问题解决'''
    # @todo 1. text block大框套小框，删除小框 2. 图片或文本框与舍弃框重叠，优先信任舍弃框 3. 文本框与标题框重叠，优先信任文本框
    all_bboxes, dropped_blocks = remove_overlaps_min_blocks(all_bboxes)

    return all_bboxes


def remove_overlaps_min_blocks(all_bboxes):
    dropped_blocks = []
    #  删除重叠blocks中较小的那些
    for block1 in all_bboxes.copy():
        for block2 in all_bboxes.copy():
            if block1 != block2:
                block1_box = block1[0], block1[1], block1[2], block1[3]
                block2_box = block2[0], block2[1], block2[2], block2[3]
                overlap_box = get_minbox_if_overlap_by_ratio(block1_box, block2_box, 0.8)
                if overlap_box is not None:
                    bbox_to_remove = next(
                        (block for block in all_bboxes if [block[0], block[1], block[2], block[3]] == overlap_box),
                        None)
                    if bbox_to_remove is not None:
                        all_bboxes.remove(bbox_to_remove)
                        bbox_to_remove['tag'] = DropTag.BLOCK_OVERLAP
                        dropped_blocks.append(bbox_to_remove)
    return all_bboxes, dropped_blocks
