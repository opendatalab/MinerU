import re
from pdf_tools.libs import _is_in_or_part_overlap
from pdf_tools.libs import fitz
import collections


def calculate_overlapRatio_between_rect1_and_rect2(L1: float, U1: float, R1: float, D1: float, L2: float, U2: float, R2: float, D2: float) -> (float, float):
    # 计算两个rect，重叠面积各占2个rect面积的比例
    if min(R1, R2) < max(L1, L2) or min(D1, D2) < max(U1, U2):
        return 0, 0
    square_1 = (R1 - L1) * (D1 - U1)
    square_2 = (R2 - L2) * (D2 - U2)
    if square_1 == 0 or square_2 == 0:
        return 0, 0
    square_overlap = (min(R1, R2) - max(L1, L2)) * (min(D1, D2) - max(U1, U2))
    return square_overlap / square_1, square_overlap / square_2

def calculate_overlapRatio_between_line1_and_line2(L1: float, R1: float, L2: float, R2: float) -> (float, float):
    # 计算两个line，重叠区间各占2个line长度的比例
    if max(L1, L2) > min(R1, R2):
        return 0, 0
    if L1 == R1 or L2 == R2:
        return 0, 0
    overlap_line = min(R1, R2) - max(L1, L2)
    return overlap_line / (R1 - L1), overlap_line / (R2 - L2)


def parse_footnoteLine(page_ID: int, page: fitz.Page, json_from_DocXchain_obj, exclude_bboxes):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """
    DPI = 72  # use this resolution
    pix = page.get_pixmap(dpi=DPI)
    pageL = 0
    pageR = int(pix.w)
    pageU = 0
    pageD = int(pix.h)

    #---------------------- PyMuPDF解析text --------------------#
    textSize_freq = collections.defaultdict(float)        # text块中，textSize的频率
    textBlock_bboxs = []
    textLine_bboxs = []
    text_blocks = page.get_text(
            "dict",
            flags=fitz.TEXTFLAGS_TEXT,
            #clip=clip,
        )["blocks"]
    totText_list = []
    for i in range(len(text_blocks)):
        # print(blocks[i])                #### print
        bbox = text_blocks[i]['bbox']
        textBlock_bboxs.append(bbox)
        # print(bbox) 
        cur_block_text_list = []
        for tt in text_blocks[i]['lines']:
            # 当前line
            cur_line_text_list = []
            cur_line_bbox = None                            # 当前line，最右侧的section的bbox
            for xf in tt['spans']:
                L, U, R, D = xf['bbox']
                L, R = min(L, R), max(L, R)
                U, D = min(U, D), max(U, D)
                textLine_bboxs.append((L, U, R, D))
                cur_line_text_list.append(xf['text'])
                textSize_freq[xf['size']] += len(xf['text'])
            cur_lines_text = ' '.join(cur_line_text_list)
            cur_block_text_list.append(cur_lines_text)
        totText_list.append('\n'.join(cur_block_text_list))
    totText = '\n'.join(totText_list)
    # print(totText)                              # 打印Text

    textLine_bboxs.sort(key = lambda LURD: (LURD[0], LURD[1]))
    textBlock_bboxs.sort(key = lambda LURD: (LURD[0], LURD[1]))
    
    # print('------------ textSize_freq -----------')
    max_sizeFreq = 0                        # 出现频率最高的textSize
    textSize_withMaxFreq = 0
    for x, f in textSize_freq.items():
        # print(x, f)
        if f > max_sizeFreq:
            max_sizeFreq = f
            textSize_withMaxFreq = x
    #**********************************************************#

    #------------------ PyMuPDF读取drawings -----------------#
    horizon_lines = []
    drawings = page.get_cdrawings()
    for drawing in drawings:
        try:
            rect = drawing['rect']
            L, U, R, D = rect
            # if (L, U, R, D) in exclude_bboxes:
            #     continue        # 如果是Fiugre, Table, Equation。注释掉是因为，可以暂时先不消，先自我对消。最后再判读需不需要排除。
            # 如果是水平线
            if U <= D and D - U <= 3:
                # 如果长度够
                if (pageR - pageL) / 15 <= R - L:
                    if not(80/800 * pageD <= U <= 750/800 * pageD):
                        continue    # 很可能是页眉和页脚的线
                    horizon_lines.append((L, U, R, D))
                    # print((L, U, R, D))
        except:
            pass
    horizon_lines.sort(key = lambda LURD: (LURD[1]))
    #********************************************************#
    
    #----------------- 两条线可能是在表格中 ------------------#
    def has_text_below_line(L: float, U: float, R: float, D: float, inLowerArea: bool) -> bool:
        """
        检查线下是否紧挨着text
        """
        Uu, Du = U - textSize_withMaxFreq, U        # 线上的一个矩形
        Lu, Ru = L, R
        Ud, Dd = U, U + textSize_withMaxFreq        # 线下的一个矩形
        Ld, Rd = L, R
        find = 0                        # 在线下的文字。统计面积。
        leftTextCnt = 0                 # 不在线底下的文字（整体在线左侧的文字），说明不是个脚注线。统计面积。
        English_alpha_cnt = 0           # 英文字母个数
        nonEnglish_alpha_cnt = 0        # 非英文字母个数
        punctuation_mark_cnt = 0        # 常见标点符号个数
        digit_cnt = 0                   # 数字个数

        distance_nearest_up_line = None
        distance_nearest_down_line = None

        for i in range(len(text_blocks)):
            # print(blocks[i])                #### print
            bbox = text_blocks[i]['bbox']
            L0, U0, R0, D0 = bbox
            if 0< (R0 - L0) < pageR / 6 and (D0 - U0) / (R0 - L0) > 10 :
                continue                # 一个很窄的，竖直的长条。比如，arXiv预印本，左侧的arXiv标志信息。
            textBlock_bboxs.append(bbox)
            # print(bbox) 
            cur_block_text_list = []
            for tt in text_blocks[i]['lines']:
                # 当前line
                cur_line_text_list = []
                cur_line_bbox = None                            # 当前line，最右侧的section的bbox
                for xf in tt['spans']:
                    L2, U2, R2, D2 = xf['bbox']
                    L2, R2 = min(L2, R2), max(L2, R2)
                    U2, D2 = min(U2, D2), max(U2, D2)
                    textLine = xf['text']
                    if L>0 and L2 < L and (L - L2) / L > 0.2:                        
                        leftTextCnt += abs(R2 - L2) * abs(D2 - U2)
                    else:
                        ## 线下的部分
                        ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(Ud, Dd, U2, D2)
                        ratio_3, ratio_4 = calculate_overlapRatio_between_line1_and_line2(Ld, Rd, L2, R2)
                        if U < (U2 + D2) / 2 and ratio_1 > 0 and ratio_2 > 0:
                            if max(ratio_3, ratio_4) > 0.8:
                                # if 444 <= U1 < 445 and 55 <= L2 < 56:
                                #     print('匹配的框', L2, U2, R2, D2)
                                # if xf['size'] > 1.2 * textSize_withMaxFreq:
                                #     return False        # 可能是个标题。不能这样卡
                                find += abs(R2 - L2) * abs(D2 - U2)
                                distance_nearest_down_line = (U2 + D2) / 2 - U
                                for c in textLine:
                                    if c == ' ':
                                        continue
                                    elif c.isdigit() == True:
                                        digit_cnt += 1
                                    elif c in ',.:!?[]()%，。、！？：【】（）《》-':
                                        punctuation_mark_cnt += 1
                                    elif c.isalpha() == True:
                                        English_alpha_cnt += 1
                                    else:
                                        nonEnglish_alpha_cnt += 1
                        ## 线上的部分
                        ratio_5, ratio_6 = calculate_overlapRatio_between_line1_and_line2(Uu, Du, U2, D2)
                        ratio_7, ratio_8 = calculate_overlapRatio_between_line1_and_line2(Lu, Ru, L2, R2)
                        if (U2 + D2) / 2 < U and ratio_5 > 0 and ratio_6 > 0:
                            if max(ratio_7, ratio_8) > 0.8:
                                distance_nearest_up_line = U - (U2 + D2) / 2
                                # if distance_nearest_up_line < 0:
                                #     print(Lu, Uu, Ru, Du, L2, U2, R2, D2)
        # print(distance_nearest_up_line, distance_nearest_down_line)
        if distance_nearest_up_line != None and distance_nearest_down_line != None:
            if distance_nearest_up_line * 1.5 < distance_nearest_down_line:
                return False                        # 如果，一根线。距离上面的文字line更近。说明是个下划线，而不是footnoteLine
                        
        ## 在上面的线条，要考虑左侧的text块儿。在很靠下的线条，就暂时不考虑左侧text块儿了。
        if inLowerArea == False:
            if leftTextCnt >= 2000/500000 * pageR * pageD:
                return False
            return find >= 0 and (English_alpha_cnt + nonEnglish_alpha_cnt + digit_cnt) >= 10
        ## 最下面区域的线条，判断时。
        # print(English_alpha_cnt, nonEnglish_alpha_cnt, digit_cnt)
        if (English_alpha_cnt + nonEnglish_alpha_cnt + digit_cnt) == 0:
            return False
        if (English_alpha_cnt + digit_cnt) / (English_alpha_cnt + nonEnglish_alpha_cnt + digit_cnt) > 0.5:
            if nonEnglish_alpha_cnt / (English_alpha_cnt + nonEnglish_alpha_cnt + digit_cnt) > 0.4:
                return False
            else:
                return True
        return True
            
    
    visited = [False for _ in range(len(horizon_lines))]
    for i, b1 in enumerate(horizon_lines):
        for j in range(i + 1, len(horizon_lines)):
            L1, U1, R1, D1 = horizon_lines[i]
            L2, U2, R2, D2 = horizon_lines[j]
            
            ## 在一条水平线，且挨着
            if L1 > L2:
                L1, U1, R1, D1, L2, U2, R2, D2 = L2, U2, R2, D2, L1, U1, R1, D1
            in_horizontal_line_flag = (max(U1, D1, U2, D2) - min(U1, D1, U2, D2) <= 5) and (L2 - R1 <= pageR/10)
            if in_horizontal_line_flag == True:
                visited[i] = True
                visited[j] = True
                
            ## 在竖直方向上是一致的。(表格，或者有的文章就是喜欢划线）
            L1, U1, R1, D1 = horizon_lines[i]
            L2, U2, R2, D2 = horizon_lines[j]            
            ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(L1, R1, L2, R2)
            # print(L1, U1, R1, D1, L2, U2, R2, D2, ratio_1, ratio_2)
            in_vertical_line_flag = (ratio_1 > 0.9 and ratio_2 > 0.9) or (max(ratio_1, ratio_2) > 0.95)
            if in_vertical_line_flag == True:
                visited[i] = True         
                # if (U2 < pageD * 0.8 or (U2 - U1) < pageD * 0.3) and has_text_below_line(L2, U2, R2, D2, False) == False:
                #     visited[j] = True             # 最最底下的线先不要动
            else:
                if ratio_1 > 0 and (R2 - L2) / (R1 - L1) > 1:
                    visited[i] = True
    # print(horizon_lines)
    horizon_lines = [horizon_lines[i] for i in range(len(horizon_lines)) if visited[i] == False]
    # print(horizon_lines)
    #*****************************************************************#    

    #------- 靠上的，就不是脚注。用一个THRESHOLD直接卡掉位于上半页的 -------#
    visited = [False for _ in range(len(horizon_lines))]
    THRESHOLD = (pageD - pageU) * 0.5
    for i, (L, U, R, D) in enumerate(horizon_lines):
        if U < THRESHOLD:
            visited[i] = True
    horizon_lines = [horizon_lines[i] for i in range(len(horizon_lines)) if visited[i] == False]
    #******************************************************#
    
    #--------------- 此时，还有遮挡的，上面的丢弃 ---------------#
    visited = [False for _ in range(len(horizon_lines))]
    for i, (L1, U1, R1, D1) in enumerate(horizon_lines):
        for j in range(i + 1, len(horizon_lines)):
            L2, U2, R2, D2 = horizon_lines[j]
            ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(L1, R1, L2, R2)
            if (ratio_1 > 0.2 and ratio_2 > 0.2) or max(ratio_1, ratio_2) > 0.7:
                visited[i] = True
    horizon_lines = [horizon_lines[i] for i in range(len(horizon_lines)) if visited[i] == False]
    #********************************************************#
    # print(horizon_lines)
    ## 检查，线下面有没有紧挨着的text
    horizon_lines = [LURD for LURD in horizon_lines if has_text_below_line(*(LURD), True) == True]
    # print(horizon_lines)
    ## 卡一下长度
    # horizon_lines = [LURD for LURD in horizon_lines if (LURD[2] - LURD[0] >= pageR / 10)]
    
    ## 上面最多保留2条
    horizon_lines = horizon_lines[max(-2, -len(horizon_lines)) :]
    
    
    #----------------------------------------------------- 第2段 -----------------------------------------------------------#
    #----------------------------------- 最下面的情形，用距离硬卡。还有在右侧的情形就被包含了 -----------------------------------#
    #------------------ PyMuPDF读取drawings -----------------#
    down_horizon_lines = []
        
    drawings = page.get_cdrawings()
    for drawing in drawings:
        try:
            rect = drawing['rect']
            L, U, R, D = rect
            # if (L, U, R, D) in exclude_bboxes:
            #     continue        # 如果是Fiugre, Table, Equation。目前是Figure识别的比较好。但是Table和Equation识别的不好
            # 如果是水平线
            if U <= D and D - U <= 3 and U > pageD * 0.85:
                # 如果长度够
                if (pageR - pageL) / 15 <= R - L:
                    down_horizon_lines.append((L, U, R, D))
                    # print((L, U, R, D))
        except:
            pass
                
    down_horizon_lines.sort(key = lambda LURD: (LURD[0], LURD[2], LURD[1]))
    visited = [False for _ in range(len(down_horizon_lines))]
    for i in range(len(down_horizon_lines) - 1):
        L1, U1, R1, D1 = down_horizon_lines[i]
        L2, U2, R2, D2 = down_horizon_lines[i + 1]
        ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(L1, R1, L2, R2)
        if ratio_1 <= 0.1 and ratio_2 <= 0.1:
            if L2 - R1 <= pageR / 3:
                visited[i] = True
                visited[i + 1] = True
    down_horizon_lines = [down_horizon_lines[i] for i in range(len(down_horizon_lines)) if visited[i] == False]
    
    down_horizon_lines = [LURD for LURD in down_horizon_lines if has_text_below_line(*(LURD), True) == True]
    # for LURD in down_horizon_lines:
    #     print('第2阶段，LURD是： ', LURD)
    #     print(has_text_below_line(*(LURD), True))

    footnoteLines = horizon_lines + down_horizon_lines
    footnoteLines = list(set(footnoteLines))
    footnoteLines = footnoteLines[max(-2, -len(footnoteLines)) : ]
    
    #-------------------------- 最后再检查一遍。是否在图片、表格、公式中。 ------------------------------#
    def line_in_specialBboxes(L: float, U: float, R: float, D: float, specialBboxes) -> bool:
        L2, U2, R2, D2 = L, U, R, D     # 当前这根线
        for L1, U1, R1, D1 in specialBboxes:
            if U1 <= U2 <= D2 < D1:
                ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(L1, R1, L2, R2)
                if ratio_1 > 0 and ratio_2 > 0.6:
                    return True
            # else:
                # U1 -= min(textSize_withMaxFreq * 2, 20)
                # D1 += min(textSize_withMaxFreq * 2, 20)
                # if U1 <= U2 <= D2 < D1:
                #     ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(L1, R1, L2, R2)
                #     if ratio_1 > 0 and ratio_2 > 0.8:
                #         return True
        return False                
        
    footnoteLines = [LURD for LURD in footnoteLines if line_in_specialBboxes(*(LURD), exclude_bboxes) == False]
    
    #-------------------------- 检查，线，是否在当前column的左侧，而不是在一段文字的中间 （通过DocXChain识别的column或者徐超老师写的Layout识别）------------------------------#
    # #--------- 通过json_from_DocXchain来获取 column ---------#
    # column_bbox_from_DocXChain = []

    # xf_json = json_from_DocXchain_obj
    # width_from_json = xf_json['page_info']['width']
    # height_from_json = xf_json['page_info']['height']
    # LR_scaleRatio = width_from_json / (pageR - pageL)
    # UD_scaleRatio = height_from_json / (pageD - pageU)

    # # {0: 'title',  # 标题
    # # 1: 'figure', # 图片
    # #  2: 'plain text',  # 文本
    # #  3: 'header',      # 页眉
    # #  4: 'page number', # 页码
    # #  5: 'footnote',    # 脚注
    # #  6: 'footer',      # 页脚
    # #  7: 'table',       # 表格
    # #  8: 'table caption',  # 表格描述
    # #  9: 'figure caption', # 图片描述
    # #  10: 'equation',      # 公式
    # #  11: 'full column',   # 单栏
    # #  12: 'sub column',    # 多栏
    # #  13: 'embedding',     # 嵌入公式
    # #  14: 'isolated'}      # 单行公式
    # for xf in xf_json['layout_dets']:
    #     L = xf['poly'][0] / LR_scaleRatio
    #     U = xf['poly'][1] / UD_scaleRatio
    #     R = xf['poly'][2] / LR_scaleRatio
    #     D = xf['poly'][5] / UD_scaleRatio
    #     # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
    #     # R += pageL
    #     # U += pageU
    #     # D += pageU
    #     L, R = min(L, R), max(L, R)
    #     U, D = min(U, D), max(U, D)
    #     if (xf['category_id'] == 11 or xf['category_id'] == 12) and xf['score'] >= 0.3:
    #         column_bbox_from_DocXChain.append((L, U, R, D))
    
    #---------------手写，检查，线是否是与某个column的左端对齐 ------------------#
    def check_isOnTheLeftOfColumn(L: float, U: float, R: float, D: float) -> bool:
        LL = L - textSize_withMaxFreq
        RR = LL
        UU = max(pageD * 0.02, U - 100/800 * pageD)
        DD = min(U + 50/800 * pageD, pageD * 0.98)
        
        # print(LL, UU, RR, DD)
        cnt = 0
        for bbox in textLine_bboxs:
            L2, U2, R2, D2 = bbox
            ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(UU, DD, U2, D2)
            ratio_3, ratio_4 = calculate_overlapRatio_between_line1_and_line2(L, R, L2, R2)
            if ratio_1 > 0 and ratio_2 > 0:
                if max(ratio_3, ratio_4) > 0.8:
                    if abs(LL - L2) <= 20/700 * pageR:
                        cnt += 1
                    # else:
                    #     if (R2 - L2) >= 30/700 * pageR:
                    #         print(LL, UU, RR, DD, L2, U2, R2, D2)
                    #         return False                  # 不能这样卡。有些注释里面，单独的特殊符号就是一个textLineBbox
        # print('cnt: ', cnt)
        return cnt >= 4
    
    # def check_isOnTheLeftOfColumn_considerLayout(L0: float, U0: float, R0: float, D0: float) -> bool:
    #     LL = L0 - textSize_withMaxFreq * 1.5
    #     RR = LL
    #     UU = 100/800 * pageD
    #     DD = 700/800 * pageD
        
    #     STEP = textSize_withMaxFreq / 2
        
    #     def check_ok(L: float, U: float, R: float, D: float) -> bool:
    #         for bbox in textBlock_bboxs:
    #             L2, U2, R2, D2 = bbox
    #             ratio_3, ratio_4 = calculate_overlapRatio_between_line1_and_line2(L, R, L2, R2)
    #             if max(ratio_3, ratio_4) > 0.8:
    #                 if (R2 - L2) > 1/4 * pageR and L2 < LL <= RR < R2:
    #                     if abs(LL - L2) < 50/700 * pageR or abs(RR - R2) < 50/700 * pageR:
    #                         continue
    #                     else:
    #                         return False
    #         return True
                             
    #     ## 先探上面
    #     u = UU
    #     d = U0
    #     while u + STEP/2 < d:
    #         mid = (u + d) / 2
    #         if check_ok(L0, mid, R0, U0) == True:
    #             d = mid
    #         else:
    #             u = mid + STEP
    #             print(mid)
    #     dist_up = U0 - u
    #     print(u)
    #     ## 再探下面
    #     u = D0
    #     d = DD
    #     while u + STEP/2 < d:
    #         mid = (u + d) / 2
    #         if check_ok(L0, mid, R0, D0) == True:
    #             u = mid
    #         else:
    #             d = mid - STEP
    #     print(u)
    #     print('^^^^^^^^^^^^^^')
    #     dist_down = u - D0
        
    #     if dist_up + dist_down < textSize_withMaxFreq * 10:
    #         return False
    #     return True
    
    
    footnoteLines = [LURD for LURD in footnoteLines if check_isOnTheLeftOfColumn(*(LURD)) == True]
    # footnoteLines = [LURD for LURD in footnoteLines if check_isOnTheLeftOfColumn_considerLayout(*(LURD)) == True]     # 不具有泛化性。不用了。
    
    #--------------------------------- 通过footnoteLine获取bbox -------------------------------#
    def get_footnoteBbox(L: float, U: float, R: float, D: float) -> (float, float, float, float):
        """
        检查线下是否紧挨着text
        """
        L1, U1, R1, D1 = L, U, R, D
        raw_bboxes = []
        for i in range(len(text_blocks)):
            bbox = text_blocks[i]['bbox']
            L2, U2, R2, D2 = bbox
            if (D2 - U2) / (R2 - L2) > 10 and (R2 - L2) < pageR / 6:
                continue                # 一个很窄的，竖直的长条。比如，arXiv预印本，左侧的arXiv标志信息。
            if U2 < D2 < U1:
                continue                # 在线上面
            under_THRESHOLD = min(D1 + textSize_withMaxFreq * 20, pageD * 0.98)
            if U2 < under_THRESHOLD:
                ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(L1, R1, L2, R2)
                if max(ratio_1, ratio_2) > 0.8:
                    raw_bboxes.append((L2, U2, R2, D2))
        # print(L1, U1, R1, D1)
        # print(raw_bboxes)
        if len(raw_bboxes) == 0:
            return []
        
        raw_bboxes.sort(key = lambda LURD: (LURD[1], LURD[0]))
        raw_bboxes = [LURD for LURD in raw_bboxes if (abs(LURD[0] - L1) < textSize_withMaxFreq * 6 or L1 < LURD[0])]  # footnote的bbox，应该都是左端对齐的
        if len(raw_bboxes) == 0:
            return []
        #------------------ full column和sub column混合，肯定也不行 ------------------#
        LL, UU, RR, DD = raw_bboxes[0]
        for L, U, R, D in raw_bboxes:
            LL, UU, RR, DD = min(LL, L), min(UU, U), max(RR, R), max(DD, D)
        for L, U, R, D in raw_bboxes:
            if (RR - LL) > pageR*0.8 and (R - L) > pageR * 0.15 and (RR - LL) / (R - L) > 2:
                return []
            if abs(LL - L) > textSize_withMaxFreq * 3:
                return []       
        
        #-------------------- 太高了的，full column的框。不行 ----------------------#
        if UU < 650/800 * pageD and (RR - LL) > 0.5 * pageR:
            return []
        
        #-------------- 第一段字数很少。后面的段字数很多，也不行 ----------------#
        if len(raw_bboxes) > 1:
            bbox_square = []
            for L, U, R, D in raw_bboxes:
                cur_s = abs(R - L) * abs(D - U)
                bbox_square.append(cur_s)
            
            s0 = bbox_square[0]
            s1n = sum(bbox_square[1: ]) / len(bbox_square[1: ])
            if s1n / s0 > 10 or max(bbox_square) / s0 > 15:
                return []
        
        raw_bboxes += [(LL, UU, RR, DD)]
        return raw_bboxes            
                                
    # print(footnoteLines)
    footnoteBboxes = []
    for L, U, R, D in footnoteLines:
        cur = get_footnoteBbox(L, U, R, D)
        if len(cur) > 0:
            footnoteBboxes.append((L, U, R, D))
            footnoteBboxes += cur
    
    footnoteBboxes = list(set(footnoteBboxes))
    return footnoteBboxes
    

def __bbox_in(box1, box2):
    """
    box1是否在box2中
    """
    L1, U1, R1, D1 = box1
    L2, U2, R2, D2 = box2
    if int(L2) <= int(L1) and int(U2) <= int(U1) and int(R1) <= int(R2) and int(D1) <= int(D2):
        return True
    return False
    
def remove_footnote_text(raw_text_block, footnote_bboxes):
    """
    :param raw_text_block: str类型，是当前页的文本内容
    :param footnoteBboxes: list类型，是当前页的脚注bbox
    """
    footnote_text_blocks = []
    for block in raw_text_block:
        text_bbox = block['bbox']
        # TODO 更严谨点在line级别做
        if any([_is_in_or_part_overlap(text_bbox, footnote_bbox) for footnote_bbox in footnote_bboxes]):
        #if any([text_bbox[3]>=footnote_bbox[1] for footnote_bbox in footnote_bboxes]):
            block['tag'] = 'footnote'
            footnote_text_blocks.append(block)
            #raw_text_block.remove(block)
            
    # 移除，不能再内部移除，否则会出错
    for block in footnote_text_blocks:
        raw_text_block.remove(block)
        
    return raw_text_block, footnote_text_blocks

def remove_footnote_image(image_blocks, footnote_bboxes):
    """
    :param image_bboxes: list类型，是当前页的图片bbox(结构体)
    :param footnoteBboxes: list类型，是当前页的脚注bbox
    """
    footnote_imgs_blocks = []
    for image_block in image_blocks:
        if any([__bbox_in(image_block['bbox'], footnote_bbox) for footnote_bbox in footnote_bboxes]):
            footnote_imgs_blocks.append(image_block)
            
    for footnote_imgs_block in footnote_imgs_blocks:
        image_blocks.remove(footnote_imgs_block)
            
    return image_blocks, footnote_imgs_blocks


def remove_headder_footer_one_page(text_raw_blocks, image_bboxes, table_bboxes, header_bboxs, footer_bboxs, page_no_bboxs, page_w, page_h):
    """
    删除页眉页脚，页码
    从line级别进行删除，删除之后观察这个text-block是否是空的，如果是空的，则移动到remove_list中
    """
    header = []
    footer = []
    if len(header)==0:
        model_header = header_bboxs
        if model_header:
            x0 = min([x for x,_,_,_ in model_header])
            y0 = min([y for _,y,_,_ in model_header])
            x1 = max([x1 for _,_,x1,_ in model_header])
            y1 = max([y1 for _,_,_,y1 in model_header])
            header = [x0, y0, x1, y1]
    if len(footer)==0:
        model_footer = footer_bboxs
        if model_footer:
            x0 = min([x for x,_,_,_ in model_footer])
            y0 = min([y for _,y,_,_ in model_footer])
            x1 = max([x1 for _,_,x1,_ in model_footer])
            y1 = max([y1 for _,_,_,y1 in model_footer])
            footer = [x0, y0, x1, y1]


    header_y0 = 0 if len(header) == 0 else header[3]
    footer_y0 = page_h if len(footer) == 0 else footer[1]
    if page_no_bboxs:
        top_part = [b for b in page_no_bboxs if b[3] < page_h/2]
        btn_part = [b for b in page_no_bboxs if b[1] > page_h/2]
        
        top_max_y0 = max([b[1] for b in top_part]) if top_part else 0
        btn_min_y1 = min([b[3] for b in btn_part]) if btn_part else page_h
        
        header_y0 = max(header_y0, top_max_y0)
        footer_y0 = min(footer_y0, btn_min_y1)
        
    content_boundry = [0, header_y0, page_w, footer_y0]
    
    header = [0,0, page_w, header_y0]
    footer = [0, footer_y0, page_w, page_h]
        
    """以上计算出来了页眉页脚的边界，下面开始进行删除"""
    text_block_to_remove = []
    # 首先检查每个textblock
    for blk in text_raw_blocks:
        if len(blk['lines']) > 0:
            for line in blk['lines']:
                line_del = []
                for span in line['spans']:
                    span_del = []
                    if span['bbox'][3] < header_y0:
                        span_del.append(span)
                    elif _is_in_or_part_overlap(span['bbox'], header) or _is_in_or_part_overlap(span['bbox'], footer):
                        span_del.append(span)
                for span in span_del:
                    line['spans'].remove(span)
                if not line['spans']:
                    line_del.append(line)

            for line in line_del:
                blk['lines'].remove(line)
        else:
        # if not blk['lines']:
            blk['tag'] = 'in-foot-header-area'
            text_block_to_remove.append(blk)
        
    """有的时候由于pageNo太小了，总是会有一点和content_boundry重叠一点，被放入正文，因此对于pageNo，进行span粒度的删除"""
    page_no_block_2_remove = []
    if page_no_bboxs:
        for pagenobox in page_no_bboxs:
            for block in text_raw_blocks:
                if _is_in_or_part_overlap(pagenobox, block['bbox']): # 在span级别删除页码
                    for line in block['lines']:
                        for span in line['spans']:
                            if _is_in_or_part_overlap(pagenobox, span['bbox']):
                                #span['text'] = ''
                                span['tag'] = "page-no"
                                # 检查这个block是否只有这一个span，如果是，那么就把这个block也删除
                                if len(line['spans']) == 1 and len(block['lines'])==1:
                                    page_no_block_2_remove.append(block)
    else:
        # 测试最后一个是不是页码：规则是，最后一个block仅有1个line,一个span,且text是数字，空格，符号组成，不含字母,并且包含数字
        if len(text_raw_blocks) > 0:
            text_raw_blocks.sort(key=lambda x: x['bbox'][1], reverse=True)
            last_block = text_raw_blocks[0]
            if len(last_block['lines']) == 1:
                last_line = last_block['lines'][0]
                if len(last_line['spans']) == 1:
                    last_span = last_line['spans'][0]
                    if last_span['text'].strip() and not re.search('[a-zA-Z]', last_span['text']) and re.search('[0-9]', last_span['text']):
                        last_span['tag'] = "page-no"
                        page_no_block_2_remove.append(last_block)
                        
                                    
    for b in page_no_block_2_remove:
        text_block_to_remove.append(b)
        
    for blk in text_block_to_remove:
        if blk in text_raw_blocks:
            text_raw_blocks.remove(blk)
    
    text_block_remain = text_raw_blocks
    image_bbox_to_remove = [bbox for bbox in image_bboxes if not _is_in_or_part_overlap(bbox, content_boundry)]

    image_bbox_remain = [bbox for bbox in image_bboxes if _is_in_or_part_overlap(bbox, content_boundry)]
    table_bbox_to_remove = [bbox for bbox in table_bboxes if not _is_in_or_part_overlap(bbox, content_boundry)]
    table_bbox_remain = [bbox for bbox in table_bboxes if _is_in_or_part_overlap(bbox, content_boundry)]
    
    return image_bbox_remain, table_bbox_remain, text_block_remain, text_block_to_remove, image_bbox_to_remove, table_bbox_to_remove
