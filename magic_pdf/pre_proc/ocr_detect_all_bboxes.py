from magic_pdf.config.ocr_content_type import BlockType
from magic_pdf.libs.boxbase import (
    calculate_iou,
    calculate_overlap_area_in_bbox1_area_ratio,
    calculate_vertical_projection_overlap_ratio,
    get_minbox_if_overlap_by_ratio
)
from magic_pdf.pre_proc.remove_bbox_overlap import remove_overlap_between_bbox_for_block


def add_bboxes(blocks, block_type, bboxes):
    for block in blocks:
        x0, y0, x1, y1 = block['bbox']
        if block_type in [
            BlockType.ImageBody,
            BlockType.ImageCaption,
            BlockType.ImageFootnote,
            BlockType.TableBody,
            BlockType.TableCaption,
            BlockType.TableFootnote,
        ]:
            bboxes.append(
                [
                    x0,
                    y0,
                    x1,
                    y1,
                    None,
                    None,
                    None,
                    block_type,
                    None,
                    None,
                    None,
                    None,
                    block['score'],
                    block['group_id'],
                ]
            )
        else:
            bboxes.append(
                [
                    x0,
                    y0,
                    x1,
                    y1,
                    None,
                    None,
                    None,
                    block_type,
                    None,
                    None,
                    None,
                    None,
                    block['score'],
                ]
            )


def ocr_prepare_bboxes_for_layout_split_v2(
    img_body_blocks,
    img_caption_blocks,
    img_footnote_blocks,
    table_body_blocks,
    table_caption_blocks,
    table_footnote_blocks,
    discarded_blocks,
    text_blocks,
    title_blocks,
    interline_equation_blocks,
    page_w,
    page_h,
):
    all_bboxes = []

    add_bboxes(img_body_blocks, BlockType.ImageBody, all_bboxes)
    add_bboxes(img_caption_blocks, BlockType.ImageCaption, all_bboxes)
    add_bboxes(img_footnote_blocks, BlockType.ImageFootnote, all_bboxes)
    add_bboxes(table_body_blocks, BlockType.TableBody, all_bboxes)
    add_bboxes(table_caption_blocks, BlockType.TableCaption, all_bboxes)
    add_bboxes(table_footnote_blocks, BlockType.TableFootnote, all_bboxes)
    add_bboxes(text_blocks, BlockType.Text, all_bboxes)
    add_bboxes(title_blocks, BlockType.Title, all_bboxes)
    add_bboxes(interline_equation_blocks, BlockType.InterlineEquation, all_bboxes)

    # 🚀 **NEW: Add discarded blocks as normal text**
    # ADD:
    add_bboxes(discarded_blocks, BlockType.Text, all_bboxes)

    """block嵌套问题解决"""
    """文本框与标题框重叠，优先信任文本框"""
    all_bboxes = fix_text_overlap_title_blocks(all_bboxes)
    """任何框体与舍弃框重叠，优先信任舍弃框"""

        # 🚀 **NEW: Remove filtering of discarded blocks**
    # Previously, remove_need_drop_blocks() prioritized discarded blocks.
    # We now treat them normally, so we skip this step.
    # SKIP:
    # all_bboxes = remove_need_drop_blocks(all_bboxes, discarded_blocks)

    # interline_equation 与title或text框冲突的情况，分两种情况处理
    """interline_equation框与文本类型框iou比较接近1的时候，信任行间公式框"""
    all_bboxes = fix_interline_equation_overlap_text_blocks_with_hi_iou(all_bboxes)
    """interline_equation框被包含在文本类型框内，且interline_equation比文本区块小很多时信任文本框，这时需要舍弃公式框"""
    # 通过后续大框套小框逻辑删除

    """discarded_blocks"""

    # 🚀 **NEW: Instead of keeping discarded blocks separate, merge them**
    # Previously:
    # all_discarded_blocks = []
    # add_bboxes(discarded_blocks, BlockType.Discarded, all_discarded_blocks)
    # Now, they're already in all_bboxes as text, so we remove this.
    # SKIP:
    # all_discarded_blocks = []
    # add_bboxes(discarded_blocks, BlockType.Discarded, all_discarded_blocks)

    """footnote识别：宽度超过1/3页面宽度的，高度超过10的，处于页面下半50%区域的"""
    footnote_blocks = []
    for discarded in discarded_blocks:
        x0, y0, x1, y1 = discarded['bbox']
        if (x1 - x0) > (page_w / 3) and (y1 - y0) > 10 and y0 > (page_h / 2):
            footnote_blocks.append([x0, y0, x1, y1])

    """移除在footnote下面的任何框"""
    need_remove_blocks = find_blocks_under_footnote(all_bboxes, footnote_blocks)
    if len(need_remove_blocks) > 0:
        for block in need_remove_blocks:
            all_bboxes.remove(block)
            # 🚀 **NEW: Don't add to discarded list, just remove them**
            # all_discarded_blocks.append(block)  # Removed
            # SKIP:
            # all_discarded_blocks.append(block)

    """经过以上处理后，还存在大框套小框的情况，则删除小框"""
    all_bboxes = remove_overlaps_min_blocks(all_bboxes)
    all_discarded_blocks = []
    all_discarded_blocks = remove_overlaps_min_blocks(all_discarded_blocks)

    """将剩余的bbox做分离处理，防止后面分layout时出错"""
    # all_bboxes, drop_reasons = remove_overlap_between_bbox_for_block(all_bboxes)
    all_bboxes.sort(key=lambda x: x[0]+x[1])
    
    # 🚀 **NEW: Don't return discarded blocks separately**
    # SKIP:
    # return all_bboxes, all_discarded_blocks
    # ADD:
    return all_bboxes, []


def find_blocks_under_footnote(all_bboxes, footnote_blocks):
    need_remove_blocks = []
    for block in all_bboxes:
        block_x0, block_y0, block_x1, block_y1 = block[:4]
        for footnote_bbox in footnote_blocks:
            footnote_x0, footnote_y0, footnote_x1, footnote_y1 = footnote_bbox
            # 如果footnote的纵向投影覆盖了block的纵向投影的80%且block的y0大于等于footnote的y1
            if (
                block_y0 >= footnote_y1
                and calculate_vertical_projection_overlap_ratio(
                    (block_x0, block_y0, block_x1, block_y1), footnote_bbox
                )
                >= 0.8
            ):
                if block not in need_remove_blocks:
                    need_remove_blocks.append(block)
                    break
    return need_remove_blocks


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
            if (
                calculate_overlap_area_in_bbox1_area_ratio(
                    block_bbox, discarded_block['bbox']
                )
                > 0.6
            ):
                if block not in need_remove:
                    need_remove.append(block)
                    break

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)
    return all_bboxes


def remove_overlaps_min_blocks(all_bboxes):
    #  重叠block，小的不能直接删除，需要和大的那个合并成一个更大的。
    #  删除重叠blocks中较小的那些
    need_remove = []
    for block1 in all_bboxes:
        for block2 in all_bboxes:
            if block1 != block2:
                block1_bbox = block1[:4]
                block2_bbox = block2[:4]
                overlap_box = get_minbox_if_overlap_by_ratio(
                    block1_bbox, block2_bbox, 0.8
                )
                if overlap_box is not None:
                    block_to_remove = next(
                        (block for block in all_bboxes if block[:4] == overlap_box),
                        None,
                    )
                    if (
                        block_to_remove is not None
                        and block_to_remove not in need_remove
                    ):
                        large_block = block1 if block1 != block_to_remove else block2
                        x1, y1, x2, y2 = large_block[:4]
                        sx1, sy1, sx2, sy2 = block_to_remove[:4]
                        x1 = min(x1, sx1)
                        y1 = min(y1, sy1)
                        x2 = max(x2, sx2)
                        y2 = max(y2, sy2)
                        large_block[:4] = [x1, y1, x2, y2]
                        need_remove.append(block_to_remove)

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)

    return all_bboxes
