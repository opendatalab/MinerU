"""
找到能分割布局的水平的横线、色块
"""

import os
from magic_pdf.libs.commons import fitz
from magic_pdf.libs.boxbase import _is_in_or_part_overlap


def __rect_filter_by_width(rect, page_w, page_h):
    mid_x = page_w/2
    if rect[0]< mid_x < rect[2]:
        return True
    return False


def __rect_filter_by_pos(rect, image_bboxes, table_bboxes):
    """
    不能出现在table和image的位置
    """
    for box in image_bboxes:
        if _is_in_or_part_overlap(rect, box):
            return False
    
    for box in table_bboxes:
        if _is_in_or_part_overlap(rect, box):
            return False
    
    return True


def __debug_show_page(page, bboxes1: list,bboxes2: list,bboxes3: list,):
    save_path = "./tmp/debug.pdf"
    if os.path.exists(save_path):
        # 删除已经存在的文件
        os.remove(save_path)
    # 创建一个新的空白 PDF 文件
    doc = fitz.open('')

    width = page.rect.width
    height = page.rect.height
    new_page = doc.new_page(width=width, height=height)
    
    shape = new_page.new_shape()
    for bbox in bboxes1:
        # 原始box画上去
        rect = fitz.Rect(*bbox[0:4])
        shape = new_page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=fitz.pdfcolor['red'], fill=fitz.pdfcolor['blue'], fill_opacity=0.2)
        shape.finish()
        shape.commit()
        
    for bbox in bboxes2:
        # 原始box画上去
        rect = fitz.Rect(*bbox[0:4])
        shape = new_page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=None, fill=fitz.pdfcolor['yellow'], fill_opacity=0.2)
        shape.finish()
        shape.commit()
        
    for bbox in bboxes3:
        # 原始box画上去
        rect = fitz.Rect(*bbox[0:4])
        shape = new_page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=fitz.pdfcolor['red'], fill=None)
        shape.finish()
        shape.commit()
        
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    doc.save(save_path)
    doc.close() 
    
def get_spilter_of_page(page, image_bboxes, table_bboxes):
    """
    获取到色块和横线
    """
    cdrawings = page.get_cdrawings()
    
    spilter_bbox = []
    for block in cdrawings:
        if 'fill' in block:
            fill = block['fill']
        if 'fill' in block and block['fill'] and block['fill']!=(1.0,1.0,1.0):
            rect = block['rect']
            if __rect_filter_by_width(rect, page.rect.width, page.rect.height) and __rect_filter_by_pos(rect, image_bboxes, table_bboxes):
                spilter_bbox.append(list(rect))
    
    """过滤、修正一下这些box。因为有时候会有一些矩形，高度为0或者为负数，造成layout计算无限循环。如果是负高度或者0高度，统一修正为高度为1"""
    for box in spilter_bbox:
        if box[3]-box[1] <= 0:
            box[3] = box[1] + 1
            
    #__debug_show_page(page, spilter_bbox, [], [])
    
    return spilter_bbox
