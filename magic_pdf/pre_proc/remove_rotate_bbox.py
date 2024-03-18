import math

from magic_pdf.libs.boxbase import is_vbox_on_side
from magic_pdf.libs.drop_tag import EMPTY_SIDE_BLOCK, ROTATE_TEXT, VERTICAL_TEXT


def detect_non_horizontal_texts(result_dict):
    """
    This function detects watermarks and vertical margin notes in the document.

    Watermarks are identified by finding blocks with the same coordinates and frequently occurring identical texts across multiple pages.
    If these conditions are met, the blocks are highly likely to be watermarks, as opposed to headers or footers, which can change from page to page.
    If the direction of these blocks is not horizontal, they are definitely considered to be watermarks.

    Vertical margin notes are identified by finding blocks with the same coordinates and frequently occurring identical texts across multiple pages.
    If these conditions are met, the blocks are highly likely to be vertical margin notes, which typically appear on the left and right sides of the page.
    If the direction of these blocks is vertical, they are definitely considered to be vertical margin notes.


    Parameters
    ----------
    result_dict : dict
        The result dictionary.

    Returns
    -------
    result_dict : dict
        The updated result dictionary.
    """
    # Dictionary to store information about potential watermarks
    potential_watermarks = {}
    potential_margin_notes = {}

    for page_id, page_content in result_dict.items():
        if page_id.startswith("page_"):
            for block_id, block_data in page_content.items():
                if block_id.startswith("block_"):
                    if "dir" in block_data:
                        coordinates_text = (block_data["bbox"], block_data["text"])  # Tuple of coordinates and text

                        angle = math.atan2(block_data["dir"][1], block_data["dir"][0])
                        angle = abs(math.degrees(angle))

                        if angle > 5 and angle < 85:  # Check if direction is watermarks
                            if coordinates_text in potential_watermarks:
                                potential_watermarks[coordinates_text] += 1
                            else:
                                potential_watermarks[coordinates_text] = 1

                        if angle > 85 and angle < 105:  # Check if direction is vertical
                            if coordinates_text in potential_margin_notes:
                                potential_margin_notes[coordinates_text] += 1  # Increment count
                            else:
                                potential_margin_notes[coordinates_text] = 1  # Initialize count

    # Identify watermarks by finding entries with counts higher than a threshold (e.g., appearing on more than half of the pages)
    watermark_threshold = len(result_dict) // 2
    watermarks = {k: v for k, v in potential_watermarks.items() if v > watermark_threshold}

    # Identify margin notes by finding entries with counts higher than a threshold (e.g., appearing on more than half of the pages)
    margin_note_threshold = len(result_dict) // 2
    margin_notes = {k: v for k, v in potential_margin_notes.items() if v > margin_note_threshold}

    # Add watermark information to the result dictionary
    for page_id, blocks in result_dict.items():
        if page_id.startswith("page_"):
            for block_id, block_data in blocks.items():
                coordinates_text = (block_data["bbox"], block_data["text"])
                if coordinates_text in watermarks:
                    block_data["is_watermark"] = 1
                else:
                    block_data["is_watermark"] = 0

                if coordinates_text in margin_notes:
                    block_data["is_vertical_margin_note"] = 1
                else:
                    block_data["is_vertical_margin_note"] = 0

    return result_dict


"""
1. 当一个block里全部文字都不是dir=(1,0)，这个block整体去掉
2. 当一个block里全部文字都是dir=(1,0)，但是每行只有一个字，这个block整体去掉。这个block必须出现在页面的四周，否则不去掉
"""
import re

def __is_a_word(sentence):
    # 如果输入是中文并且长度为1，则返回True
    if re.fullmatch(r'[\u4e00-\u9fa5]', sentence):
        return True
    # 判断是否为单个英文单词或字符（包括ASCII标点）
    elif re.fullmatch(r'[a-zA-Z0-9]+', sentence) and len(sentence) <=2:
        return True
    else:
        return False


def __get_text_color(num):
    """获取字体的颜色RGB值"""
    blue = num & 255
    green = (num >> 8) & 255
    red = (num >> 16) & 255
    return red, green, blue


def __is_empty_side_box(text_block):
    """
    是否是边缘上的空白没有任何内容的block
    """
    for line in text_block['lines']:
        for span in line['spans']:
            font_color = span['color']
            r,g,b = __get_text_color(font_color)
            if len(span['text'].strip())>0 and (r,g,b)!=(255,255,255):
                return False
            
    return True


def remove_rotate_side_textblock(pymu_text_block, page_width, page_height):
    """
    返回删除了垂直，水印，旋转的textblock
    删除的内容打上tag返回
    """
    removed_text_block = []
    
    for i, block in enumerate(pymu_text_block): # 格式参考test/assets/papre/pymu_textblocks.json
        lines = block['lines']
        block_bbox = block['bbox']
        if not is_vbox_on_side(block_bbox, page_width, page_height, 0.2): # 保证这些box必须在页面的两边
           continue
        
        if all([__is_a_word(line['spans'][0]["text"]) for line in lines if len(line['spans'])>0]) and len(lines)>1 and all([len(line['spans'])==1 for line in lines]):
            is_box_valign = (len(set([int(line['spans'][0]['bbox'][0] ) for line in lines if len(line['spans'])>0]))==1) and (len([int(line['spans'][0]['bbox'][0] ) for line in lines if len(line['spans'])>0])>1)  # 测试bbox在垂直方向是不是x0都相等，也就是在垂直方向排列.同时必须大于等于2个字
            
            if is_box_valign:
                block['tag'] = VERTICAL_TEXT
                removed_text_block.append(block)
                continue
        
        for line in lines:
            if line['dir']!=(1,0):
                block['tag'] = ROTATE_TEXT
                removed_text_block.append(block) # 只要有一个line不是dir=(1,0)，就把整个block都删掉
                break
        
    for block in removed_text_block:
        pymu_text_block.remove(block)
    
    return pymu_text_block, removed_text_block

def get_side_boundry(rotate_bbox, page_width, page_height):
    """
    根据rotate_bbox，返回页面的左右正文边界
    """
    left_x = 0
    right_x = page_width
    for x in rotate_bbox:
        box = x['bbox']
        if box[2]<page_width/2:
            left_x = max(left_x, box[2])
        else:
            right_x = min(right_x, box[0])
            
    return left_x+1, right_x-1


def remove_side_blank_block(pymu_text_block, page_width, page_height):
    """
    删除页面两侧的空白block
    """
    removed_text_block = []
    
    for i, block in enumerate(pymu_text_block): # 格式参考test/assets/papre/pymu_textblocks.json
        block_bbox = block['bbox']
        if not is_vbox_on_side(block_bbox, page_width, page_height, 0.2): # 保证这些box必须在页面的两边
           continue
            
        if __is_empty_side_box(block):
            block['tag'] = EMPTY_SIDE_BLOCK
            removed_text_block.append(block)
            continue
        
    for block in removed_text_block:
        pymu_text_block.remove(block)
    
    return pymu_text_block, removed_text_block