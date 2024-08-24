from magic_pdf.libs.commons import fitz             # pyMuPDF库
import re

from magic_pdf.libs.boxbase import _is_in_or_part_overlap, _is_part_overlap, find_bottom_nearest_text_bbox, find_left_nearest_text_bbox, find_right_nearest_text_bbox, find_top_nearest_text_bbox             # json


## version 2
def get_merged_line(page):
    """
    这个函数是为了从pymuPDF中提取出的矢量里筛出水平的横线，并且将断开的线段进行了合并。
    :param page :fitz读取的当前页的内容
    """
    drawings_bbox = []
    drawings_line = []
    drawings = page.get_drawings()  # 提取所有的矢量
    for p in drawings:
        drawings_bbox.append(p["rect"].irect)  # (L, U, R, D)

    lines = []
    for L, U, R, D in drawings_bbox:
        if abs(D - U) <= 3: # 筛出水平的横线
            lines.append((L, U, R, D))
    U_groups = []
    visited = [False for _ in range(len(lines))]
    for i, (L1, U1, R1, D1) in enumerate(lines):
        if visited[i] == True:
            continue
        tmp_g = [(L1, U1, R1, D1)]
        for j, (L2, U2, R2, D2) in enumerate(lines):
            if i == j:
                continue
            if visited[j] == True:
                continue
            if max(U1, D1, U2, D2) - min(U1, D1, U2, D2) <= 5:   # 把高度一致的线放进一个group
                tmp_g.append((L2, U2, R2, D2))
                visited[j] = True
        U_groups.append(tmp_g)
        
    res = []
    for group in U_groups:
        group.sort(key = lambda LURD: (LURD[0], LURD[2]))
        LL, UU, RR, DD = group[0]
        for i, (L1, U1, R1, D1) in enumerate(group):
            if (L1 - RR) >= 5:
                cur_line = (LL, UU, RR, DD)
                res.append(cur_line)
                LL = L1
            else:
                RR = max(RR, R1)
        cur_line = (LL, UU, RR, DD)
        res.append(cur_line)
    return res

def fix_tables(page: fitz.Page, table_bboxes: list, include_table_title: bool, scan_line_num: int):
    """
    :param page :fitz读取的当前页的内容
    :param table_bboxes: list类型，每一个元素是一个元祖 (L, U, R, D)
    :param include_table_title: 是否将表格的标题也圈进来
    :param scan_line_num: 在与表格框临近的上下几个文本框里扫描搜索标题
    """
    
    drawings_lines = get_merged_line(page)
    fix_table_bboxes = []
    
    for table in table_bboxes:
        (L, U, R, D) = table
        fix_table_L = []
        fix_table_U = []
        fix_table_R = []
        fix_table_D = []
        width = R - L
        width_range = width * 0.1 # 只看距离表格整体宽度10%之内偏差的线
        height = D - U
        height_range = height * 0.1 # 只看距离表格整体高度10%之内偏差的线
        for line in drawings_lines:
            if (L - width_range) <= line[0] <= (L + width_range) and (R - width_range) <= line[2] <= (R + width_range): # 相近的宽度
                if (U - height_range) < line[1] < (U + height_range): # 上边界，在一定的高度范围内
                    fix_table_U.append(line[1])
                    fix_table_L.append(line[0])
                    fix_table_R.append(line[2])
                elif (D - height_range) < line[1] < (D + height_range): # 下边界，在一定的高度范围内
                    fix_table_D.append(line[1])
                    fix_table_L.append(line[0])
                    fix_table_R.append(line[2])

        if fix_table_U:
            U = min(fix_table_U)
        if fix_table_D:
            D = max(fix_table_D)
        if fix_table_L:
            L = min(fix_table_L)
        if fix_table_R:
            R = max(fix_table_R)
            
        if include_table_title:   # 需要将表格标题包括
            text_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]   # 所有的text的block
            incolumn_text_blocks = [block for block in text_blocks if not ((block['bbox'][0] < L and block['bbox'][2] < L) or (block['bbox'][0] > R and block['bbox'][2] > R))]  # 将与表格完全没有任何遮挡的文字筛除掉（比如另一栏的文字）
            upper_text_blocks = [block for block in incolumn_text_blocks if (U - block['bbox'][3]) > 0]  # 将在表格线以上的text block筛选出来
            sorted_filtered_text_blocks = sorted(upper_text_blocks, key=lambda x: (U - x['bbox'][3], x['bbox'][0])) # 按照text block的下边界距离表格上边界的距离升序排序，如果是同一个高度，则先左再右
            
            for idx in range(scan_line_num):   
                if idx+1 <= len(sorted_filtered_text_blocks):
                    line_temp = sorted_filtered_text_blocks[idx]['lines']
                    if line_temp:
                        text = line_temp[0]['spans'][0]['text'] # 提取出第一个span里的text内容
                        check_en = re.match('Table', text) # 检查是否有Table开头的(英文）
                        check_ch = re.match('表', text) # 检查是否有Table开头的(中文）
                        if check_en or check_ch:
                            if sorted_filtered_text_blocks[idx]['bbox'][1] < D: # 以防出现负的bbox
                                U = sorted_filtered_text_blocks[idx]['bbox'][1]
                                  
        fix_table_bboxes.append([L-2, U-2, R+2, D+2])
    
    return fix_table_bboxes

def __check_table_title_pattern(text):
    """
    检查文本段是否是表格的标题
    """
    patterns = [r'^table\s\d+']
    
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return True
        else:
            return False
         
         
def fix_table_text_block(pymu_blocks, table_bboxes: list):
    """
    调整table, 如果table和上下的text block有相交区域，则将table的上下边界调整到text block的上下边界
    例如 tmp/unittest/unittest_pdf/纯2列_ViLT_6_文字 表格.pdf
    """
    for tb in table_bboxes:
        (L, U, R, D) = tb
        for block in pymu_blocks:
            if _is_in_or_part_overlap((L, U, R, D), block['bbox']):
                txt = " ".join(span['text'] for line in block['lines'] for span in line['spans'])
                if not __check_table_title_pattern(txt) and block.get("_table", False) is False: # 如果是table的title，那么不调整。因为下一步会统一调整，如果这里进行了调整，后面的调整会造成调整到其他table的title上（在连续出现2个table的情况下）。
                    tb[0] = min(tb[0], block['bbox'][0])
                    tb[1] = min(tb[1], block['bbox'][1])
                    tb[2] = max(tb[2], block['bbox'][2])
                    tb[3] = max(tb[3], block['bbox'][3])
                    block['_table'] = True # 占位，防止其他table再次占用
                    
                """如果是个table的title，但是有部分重叠，那么修正这个title,使得和table不重叠"""
                if _is_part_overlap(tb, block['bbox']) and __check_table_title_pattern(txt):
                    block['bbox'] = list(block['bbox'])
                    if block['bbox'][3] > U:
                        block['bbox'][3] = U-1
                    if block['bbox'][1] < D:
                        block['bbox'][1] = D+1
                
                
    return table_bboxes


def __get_table_caption_text(text_block):
    txt = " ".join(span['text'] for line in text_block['lines'] for span in line['spans'])
    line_cnt = len(text_block['lines'])
    txt = txt.replace("Ž . ", '')
    return txt, line_cnt


def include_table_title(pymu_blocks, table_bboxes: list):
    """
    把表格的title也包含进来，扩展到table_bbox上
    """
    for tb in table_bboxes:
        max_find_cnt = 3 # 上上最多找3次
        temp_box = tb.copy()
        while max_find_cnt>0:
            text_block_top = find_top_nearest_text_bbox(pymu_blocks, temp_box)
            if text_block_top:
                txt, line_cnt = __get_table_caption_text(text_block_top)
                if len(txt.strip())>0:
                    if not __check_table_title_pattern(txt) and max_find_cnt>0 and line_cnt<3:
                        max_find_cnt = max_find_cnt -1
                        temp_box[1] = text_block_top['bbox'][1]
                        continue
                    else:
                        break
                else:
                    temp_box[1] = text_block_top['bbox'][1] # 宽度不变，扩大
                    max_find_cnt = max_find_cnt - 1
            else:
                break
            
        max_find_cnt = 3 # 向下找
        temp_box = tb.copy()
        while max_find_cnt>0:
            text_block_bottom = find_bottom_nearest_text_bbox(pymu_blocks, temp_box)
            if text_block_bottom:
                txt, line_cnt = __get_table_caption_text(text_block_bottom)
                if len(txt.strip())>0:
                    if not __check_table_title_pattern(txt) and max_find_cnt>0 and line_cnt<3:
                        max_find_cnt = max_find_cnt - 1
                        temp_box[3] = text_block_bottom['bbox'][3]
                        continue
                    else:
                        break
                else:
                    temp_box[3] = text_block_bottom['bbox'][3]
                    max_find_cnt = max_find_cnt - 1
            else:
                break
        
        if text_block_top and text_block_bottom and text_block_top.get("_table_caption", False) is False and text_block_bottom.get("_table_caption", False) is False :
            btn_text, _ = __get_table_caption_text(text_block_bottom)
            top_text, _ = __get_table_caption_text(text_block_top)
            if __check_table_title_pattern(btn_text) and __check_table_title_pattern(top_text): # 上下都有一个tbale的caption
                # 取距离最近的
                btn_text_distance = text_block_bottom['bbox'][1] - tb[3]
                top_text_distance = tb[1] - text_block_top['bbox'][3]
                text_block = text_block_bottom if btn_text_distance<top_text_distance else text_block_top
                tb[0] = min(tb[0], text_block['bbox'][0])
                tb[1] = min(tb[1], text_block['bbox'][1])
                tb[2] = max(tb[2], text_block['bbox'][2])
                tb[3] = max(tb[3], text_block['bbox'][3])
                text_block_bottom['_table_caption'] = True
                continue

        # 如果以上条件都不满足，那么就向下找
        text_block = text_block_top
        if text_block and text_block.get("_table_caption", False) is False:
            first_text_line = " ".join(span['text'] for line in text_block['lines'] for span in line['spans'])
            if __check_table_title_pattern(first_text_line) and text_block.get("_table", False) is False:
                tb[0] = min(tb[0], text_block['bbox'][0])
                tb[1] = min(tb[1], text_block['bbox'][1])
                tb[2] = max(tb[2], text_block['bbox'][2])
                tb[3] = max(tb[3], text_block['bbox'][3])
                text_block['_table_caption'] = True
                continue
            
        text_block = text_block_bottom
        if text_block and text_block.get("_table_caption", False) is False:
            first_text_line, _ = __get_table_caption_text(text_block)
            if __check_table_title_pattern(first_text_line) and text_block.get("_table", False) is False:
                tb[0] = min(tb[0], text_block['bbox'][0])
                tb[1] = min(tb[1], text_block['bbox'][1])
                tb[2] = max(tb[2], text_block['bbox'][2])
                tb[3] = max(tb[3], text_block['bbox'][3])
                text_block['_table_caption'] = True
                continue
        
        """向左、向右寻找，暂时只寻找一次"""
        left_text_block = find_left_nearest_text_bbox(pymu_blocks, tb)
        if left_text_block and left_text_block.get("_image_caption", False) is False:
            first_text_line, _ = __get_table_caption_text(left_text_block)
            if __check_table_title_pattern(first_text_line):
                tb[0] = min(tb[0], left_text_block['bbox'][0])
                tb[1] = min(tb[1], left_text_block['bbox'][1])
                tb[2] = max(tb[2], left_text_block['bbox'][2])
                tb[3] = max(tb[3], left_text_block['bbox'][3])
                left_text_block['_image_caption'] = True
                continue
            
        right_text_block = find_right_nearest_text_bbox(pymu_blocks, tb)
        if right_text_block and right_text_block.get("_image_caption", False) is False:
            first_text_line, _ = __get_table_caption_text(right_text_block)
            if __check_table_title_pattern(first_text_line):
                tb[0] = min(tb[0], right_text_block['bbox'][0])
                tb[1] = min(tb[1], right_text_block['bbox'][1])
                tb[2] = max(tb[2], right_text_block['bbox'][2])
                tb[3] = max(tb[3], right_text_block['bbox'][3])
                right_text_block['_image_caption'] = True
                continue
                
    return table_bboxes