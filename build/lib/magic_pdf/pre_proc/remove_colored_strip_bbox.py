from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap, calculate_overlap_area_2_minbox_area_ratio
from loguru import logger

from magic_pdf.libs.drop_tag import COLOR_BG_HEADER_TXT_BLOCK


def __area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def rectangle_position_determination(rect, p_width):
    """
    判断矩形是否在页面中轴线附近。

    Args:
        rect (list): 矩形坐标，格式为[x1, y1, x2, y2]。
        p_width (int): 页面宽度。

    Returns:
        bool: 若矩形在页面中轴线附近则返回True，否则返回False。
    """
    # 页面中轴线x坐标
    x_axis = p_width / 2
    # 矩形是否跨越中轴线
    is_span = rect[0] < x_axis and rect[2] > x_axis
    if is_span:
        return True
    else:
        # 矩形与中轴线的距离，只算近的那一边
        distance = rect[0] - x_axis if rect[0] > x_axis else x_axis - rect[2]
        # 判断矩形与中轴线的距离是否小于页面宽度的20%
        if distance < p_width * 0.2:
            return True
        else:
            return False

def remove_colored_strip_textblock(remain_text_blocks, page):
    """
    根据页面中特定颜色和大小过滤文本块，将符合条件的文本块从remain_text_blocks中移除，并返回移除的文本块列表colored_strip_textblock。

    Args:
        remain_text_blocks (list): 剩余文本块列表。
        page (Page): 页面对象。

    Returns:
        tuple: 剩余文本块列表和移除的文本块列表。
    """
    colored_strip_textblocks = []  # 先构造一个空的返回
    if len(remain_text_blocks) > 0:
        p_width, p_height = page.rect.width, page.rect.height
        blocks = page.get_cdrawings()
        colored_strip_bg_rect = []
        for block in blocks:
            is_filled = 'fill' in block and block['fill'] and block['fill'] != (1.0, 1.0, 1.0)  # 过滤掉透明的
            rect = block['rect']
            area_is_large_enough = __area(rect) > 100  # 过滤掉特别小的矩形
            rectangle_position_determination_result = rectangle_position_determination(rect, p_width)
            in_upper_half_page = rect[3] < p_height * 0.3  # 找到位于页面上半部分的矩形，下边界小于页面高度的30%
            aspect_ratio_exceeds_4 = (rect[2] - rect[0]) > (rect[3] - rect[1]) * 4  # 找到长宽比超过4的矩形

            if is_filled and area_is_large_enough and rectangle_position_determination_result and in_upper_half_page and aspect_ratio_exceeds_4:
                colored_strip_bg_rect.append(rect)

        if len(colored_strip_bg_rect) > 0:
            for colored_strip_block_bbox in colored_strip_bg_rect:
                for text_block in remain_text_blocks:
                    text_bbox = text_block['bbox']
                    if _is_in(text_bbox, colored_strip_block_bbox) or (_is_in_or_part_overlap(text_bbox, colored_strip_block_bbox) and calculate_overlap_area_2_minbox_area_ratio(text_bbox, colored_strip_block_bbox) > 0.6):
                        logger.info(f'remove_colored_strip_textblock: {text_bbox}, {colored_strip_block_bbox}')
                        text_block['tag'] = COLOR_BG_HEADER_TXT_BLOCK
                        colored_strip_textblocks.append(text_block)

                if len(colored_strip_textblocks) > 0:
                    for colored_strip_textblock in colored_strip_textblocks:
                        if colored_strip_textblock in remain_text_blocks:
                            remain_text_blocks.remove(colored_strip_textblock)

    return remain_text_blocks, colored_strip_textblocks

