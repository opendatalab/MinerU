


import re    
from magic_pdf.libs.boxbase import  _is_in_or_part_overlap, _is_part_overlap, find_bottom_nearest_text_bbox, find_left_nearest_text_bbox, find_right_nearest_text_bbox, find_top_nearest_text_bbox

from magic_pdf.libs.textbase import get_text_block_base_info

def fix_image_vertical(image_bboxes:list, text_blocks:list):
    """
    修正图片的位置
    如果图片与文字block发生一定重叠（也就是图片切到了一部分文字），那么减少图片边缘，让文字和图片不再重叠。
    只对垂直方向进行。
    """
    for image_bbox in image_bboxes:
        for text_block in text_blocks:
            text_bbox = text_block["bbox"]
            if _is_part_overlap(text_bbox, image_bbox) and any([text_bbox[0]>=image_bbox[0] and text_bbox[2]<=image_bbox[2], text_bbox[0]<=image_bbox[0] and text_bbox[2]>=image_bbox[2]]):
                if text_bbox[1] < image_bbox[1]:#在图片上方
                    image_bbox[1] = text_bbox[3]+1
                elif text_bbox[3]>image_bbox[3]:#在图片下方
                    image_bbox[3] = text_bbox[1]-1
                
    return image_bboxes

def __merge_if_common_edge(bbox1, bbox2):
    x_min_1, y_min_1, x_max_1, y_max_1 = bbox1
    x_min_2, y_min_2, x_max_2, y_max_2 = bbox2

    # 检查是否有公共的水平边
    if y_min_1 == y_min_2 or y_max_1 == y_max_2:
        # 确保一个框的x范围在另一个框的x范围内
        if max(x_min_1, x_min_2) <= min(x_max_1, x_max_2):
            return [min(x_min_1, x_min_2), min(y_min_1, y_min_2), max(x_max_1, x_max_2), max(y_max_1, y_max_2)]

    # 检查是否有公共的垂直边
    if x_min_1 == x_min_2 or x_max_1 == x_max_2:
        # 确保一个框的y范围在另一个框的y范围内
        if max(y_min_1, y_min_2) <= min(y_max_1, y_max_2):
            return [min(x_min_1, x_min_2), min(y_min_1, y_min_2), max(x_max_1, x_max_2), max(y_max_1, y_max_2)]

    # 如果没有公共边
    return None

def fix_seperated_image(image_bboxes:list):
    """
    如果2个图片有一个边重叠，那么合并2个图片
    """
    new_images = []
    droped_img_idx = []
            
    for i in range(0, len(image_bboxes)):
        for j in range(i+1, len(image_bboxes)):
            new_img = __merge_if_common_edge(image_bboxes[i], image_bboxes[j])
            if new_img is not None:
                new_images.append(new_img)
                droped_img_idx.append(i)
                droped_img_idx.append(j)
                break
            
    for i in range(0, len(image_bboxes)):
        if i not in droped_img_idx:
            new_images.append(image_bboxes[i])
            
    return new_images


def __check_img_title_pattern(text):
    """
    检查文本段是否是表格的标题
    """
    patterns = [r"^(fig|figure).*", r"^(scheme).*"]
    text = text.strip()
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return True
    return False

def __get_fig_caption_text(text_block):
    txt = " ".join(span['text'] for line in text_block['lines'] for span in line['spans'])
    line_cnt = len(text_block['lines'])
    txt = txt.replace("Ž . ", '')
    return txt, line_cnt


def __find_and_extend_bottom_caption(text_block, pymu_blocks, image_box):
    """
    继续向下方寻找和图片caption字号，字体，颜色一样的文字框，合并入caption。
    text_block是已经找到的图片catpion（这个caption可能不全，多行被划分到多个pymu block里了）
    """
    combined_image_caption_text_block = list(text_block.copy()['bbox'])
    base_font_color, base_font_size, base_font_type = get_text_block_base_info(text_block)
    while True:
        tb_add = find_bottom_nearest_text_bbox(pymu_blocks, combined_image_caption_text_block)
        if not tb_add:
            break
        tb_font_color, tb_font_size, tb_font_type = get_text_block_base_info(tb_add)
        if tb_font_color==base_font_color and tb_font_size==base_font_size and tb_font_type==base_font_type:
            combined_image_caption_text_block[0] = min(combined_image_caption_text_block[0], tb_add['bbox'][0])
            combined_image_caption_text_block[2] = max(combined_image_caption_text_block[2], tb_add['bbox'][2])
            combined_image_caption_text_block[3] = tb_add['bbox'][3]
        else:
            break
            
    image_box[0] = min(image_box[0], combined_image_caption_text_block[0])
    image_box[1] = min(image_box[1], combined_image_caption_text_block[1])
    image_box[2] = max(image_box[2], combined_image_caption_text_block[2])
    image_box[3] = max(image_box[3], combined_image_caption_text_block[3])
    text_block['_image_caption'] = True
        

def include_img_title(pymu_blocks, image_bboxes: list):
    """
    向上方和下方寻找符合图片title的文本block，合并到图片里
    如果图片上下都有fig的情况怎么办？寻找标题距离最近的那个。
    ---
    增加对左侧和右侧图片标题的寻找
    """

    
    for tb in image_bboxes:
        # 优先找下方的
        max_find_cnt = 3 # 向上，向下最多找3个就停止
        temp_box = tb.copy()
        while max_find_cnt>0:
            text_block_btn = find_bottom_nearest_text_bbox(pymu_blocks, temp_box)
            if text_block_btn:
                txt, line_cnt = __get_fig_caption_text(text_block_btn)
                if len(txt.strip())>0:
                    if not __check_img_title_pattern(txt) and max_find_cnt>0 and line_cnt<3: # 设置line_cnt<=2目的是为了跳过子标题，或者有时候图片下方文字没有被图片识别模型放入图片里
                        max_find_cnt = max_find_cnt - 1
                        temp_box[3] = text_block_btn['bbox'][3]
                        continue
                    else:
                        break
                else:
                    temp_box[3] = text_block_btn['bbox'][3] # 宽度不变，扩大
                    max_find_cnt = max_find_cnt - 1
            else:
                break
        
        max_find_cnt = 3 # 向上，向下最多找3个就停止
        temp_box = tb.copy()
        while max_find_cnt>0:
            text_block_top = find_top_nearest_text_bbox(pymu_blocks, temp_box)
            if text_block_top:
                txt, line_cnt = __get_fig_caption_text(text_block_top)
                if len(txt.strip())>0:
                    if not __check_img_title_pattern(txt) and max_find_cnt>0 and line_cnt <3:
                        max_find_cnt = max_find_cnt - 1
                        temp_box[1] = text_block_top['bbox'][1]
                        continue
                    else:
                        break
                else:
                    b = text_block_top['bbox']
                    temp_box[1] = b[1] # 宽度不变，扩大
                    max_find_cnt = max_find_cnt - 1
            else:
                break
        
        if text_block_btn and text_block_top and text_block_btn.get("_image_caption", False) is False and text_block_top.get("_image_caption", False) is False :
            btn_text, _ = __get_fig_caption_text(text_block_btn)
            top_text, _ = __get_fig_caption_text(text_block_top)
            if __check_img_title_pattern(btn_text) and __check_img_title_pattern(top_text):
                # 取距离图片最近的
                btn_text_distance = text_block_btn['bbox'][1] - tb[3]
                top_text_distance = tb[1] - text_block_top['bbox'][3]
                if btn_text_distance<top_text_distance: # caption在下方
                    __find_and_extend_bottom_caption(text_block_btn, pymu_blocks, tb)
                else:
                    text_block = text_block_top
                    tb[0] = min(tb[0], text_block['bbox'][0])
                    tb[1] = min(tb[1], text_block['bbox'][1])
                    tb[2] = max(tb[2], text_block['bbox'][2])
                    tb[3] = max(tb[3], text_block['bbox'][3])
                    text_block_btn['_image_caption'] = True
                continue
            
        text_block = text_block_btn # find_bottom_nearest_text_bbox(pymu_blocks, tb)
        if text_block and text_block.get("_image_caption", False) is False:
            first_text_line, _ = __get_fig_caption_text(text_block)
            if __check_img_title_pattern(first_text_line):
                # 发现特征之后，继续向相同方向寻找（想同颜色，想同大小，想同字体）的textblock
                __find_and_extend_bottom_caption(text_block, pymu_blocks, tb)
                continue
            
        text_block = text_block_top # find_top_nearest_text_bbox(pymu_blocks, tb)
        if text_block  and text_block.get("_image_caption", False) is False:
            first_text_line, _ = __get_fig_caption_text(text_block)
            if __check_img_title_pattern(first_text_line):
                tb[0] = min(tb[0], text_block['bbox'][0])
                tb[1] = min(tb[1], text_block['bbox'][1])
                tb[2] = max(tb[2], text_block['bbox'][2])
                tb[3] = max(tb[3], text_block['bbox'][3])
                text_block['_image_caption'] = True
                continue
            
        """向左、向右寻找，暂时只寻找一次"""
        left_text_block = find_left_nearest_text_bbox(pymu_blocks, tb)
        if left_text_block and left_text_block.get("_image_caption", False) is False:
            first_text_line, _ = __get_fig_caption_text(left_text_block)
            if __check_img_title_pattern(first_text_line):
                tb[0] = min(tb[0], left_text_block['bbox'][0])
                tb[1] = min(tb[1], left_text_block['bbox'][1])
                tb[2] = max(tb[2], left_text_block['bbox'][2])
                tb[3] = max(tb[3], left_text_block['bbox'][3])
                left_text_block['_image_caption'] = True
                continue
            
        right_text_block = find_right_nearest_text_bbox(pymu_blocks, tb)
        if right_text_block and right_text_block.get("_image_caption", False) is False:
            first_text_line, _ = __get_fig_caption_text(right_text_block)
            if __check_img_title_pattern(first_text_line):
                tb[0] = min(tb[0], right_text_block['bbox'][0])
                tb[1] = min(tb[1], right_text_block['bbox'][1])
                tb[2] = max(tb[2], right_text_block['bbox'][2])
                tb[3] = max(tb[3], right_text_block['bbox'][3])
                right_text_block['_image_caption'] = True
                continue

    return image_bboxes


def combine_images(image_bboxes:list):
    """
    合并图片，如果图片有重叠，那么合并
    """
    new_images = []
    droped_img_idx = []
            
    for i in range(0, len(image_bboxes)):
        for j in range(i+1, len(image_bboxes)):
            if j not in droped_img_idx and _is_in_or_part_overlap(image_bboxes[i], image_bboxes[j]):
                # 合并
                image_bboxes[i][0], image_bboxes[i][1],image_bboxes[i][2],image_bboxes[i][3] = min(image_bboxes[i][0], image_bboxes[j][0]), min(image_bboxes[i][1], image_bboxes[j][1]), max(image_bboxes[i][2], image_bboxes[j][2]), max(image_bboxes[i][3], image_bboxes[j][3])
                droped_img_idx.append(j)
            
    for i in range(0, len(image_bboxes)):
        if i not in droped_img_idx:
            new_images.append(image_bboxes[i])
            
    return new_images