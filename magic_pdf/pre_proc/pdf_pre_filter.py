from magic_pdf.libs.commons import fitz
from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap
from magic_pdf.libs.drop_reason import DropReason


def __area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def __is_contain_color_background_rect(page:fitz.Page, text_blocks, image_bboxes) -> bool:
    """
    检查page是包含有颜色背景的矩形
    """
    color_bg_rect = []
    p_width, p_height = page.rect.width, page.rect.height
    
    # 先找到最大的带背景矩形
    blocks = page.get_cdrawings()
    for block in blocks:
        
        if 'fill' in block and block['fill']: # 过滤掉透明的
            fill = list(block['fill'])
            fill[0], fill[1], fill[2] = int(fill[0]), int(fill[1]), int(fill[2])
            if fill==(1.0,1.0,1.0):
                continue
            rect = block['rect']
            # 过滤掉特别小的矩形
            if __area(rect) < 10*10:
                continue
            # 为了防止是svg图片上的色块，这里过滤掉这类
            
            if any([_is_in_or_part_overlap(rect, img_bbox) for img_bbox in image_bboxes]):
                continue
            color_bg_rect.append(rect)
            
    # 找到最大的背景矩形
    if len(color_bg_rect) > 0:
        max_rect = max(color_bg_rect, key=lambda x:__area(x))
        max_rect_int = (int(max_rect[0]), int(max_rect[1]), int(max_rect[2]), int(max_rect[3]))
        # 判断最大的背景矩形是否包含超过3行文字，或者50个字 TODO
        if max_rect[2]-max_rect[0] > 0.2*p_width and  max_rect[3]-max_rect[1] > 0.1*p_height:#宽度符合
            #看是否有文本块落入到这个矩形中
            for text_block in text_blocks:
                box = text_block['bbox']
                box_int = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                if _is_in(box_int, max_rect_int):
                    return True
    
    return False


def __is_table_overlap_text_block(text_blocks, table_bbox):
    """
    检查table_bbox是否覆盖了text_blocks里的文本块
    TODO
    """
    for text_block in text_blocks:
        box = text_block['bbox']
        if _is_in_or_part_overlap(table_bbox, box):
            return True
    return False


def pdf_filter(page:fitz.Page, text_blocks, table_bboxes, image_bboxes) -> tuple:
    """
    return:(True|False, err_msg)
        True, 如果pdf符合要求
        False, 如果pdf不符合要求
        
    """
    if __is_contain_color_background_rect(page, text_blocks, image_bboxes):
        return False, {"_need_drop": True, "_drop_reason": DropReason.COLOR_BACKGROUND_TEXT_BOX}

    
    return True, None