import collections      # 统计库
import re
from magic_pdf.libs.commons import fitz             # pyMuPDF库


#--------------------------------------- Tool Functions --------------------------------------#
# 正则化，输入文本，输出只保留a-z,A-Z,0-9
def remove_special_chars(s: str) -> str:
    pattern = r"[^a-zA-Z0-9]"
    res = re.sub(pattern, "", s)
    return res

def check_rect1_sameWith_rect2(L1: float, U1: float, R1: float, D1: float, L2: float, U2: float, R2: float, D2: float) -> bool:
    # 判断rect1和rect2是否一模一样
    return L1 == L2 and U1 == U2 and R1 == R2 and D1 == D2

def check_rect1_contains_rect2(L1: float, U1: float, R1: float, D1: float, L2: float, U2: float, R2: float, D2: float) -> bool:
    # 判断rect1包含了rect2
    return (L1 <= L2 <= R2 <= R1) and (U1 <= U2 <= D2 <= D1)

def check_rect1_overlaps_rect2(L1: float, U1: float, R1: float, D1: float, L2: float, U2: float, R2: float, D2: float) -> bool:
    # 判断rect1与rect2是否存在重叠（只有一条边重叠，也算重叠）
    return max(L1, L2) <= min(R1, R2) and max(U1, U2) <= min(D1, D2)

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


# 判断rect其实是一条line
def check_rect_isLine(L: float, U: float, R: float, D: float) -> bool:
    width = R - L
    height = D - U
    if width <= 3 or height <= 3:
        return True
    if width / height >= 30 or height / width >= 30:
        return True



def parse_images(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict, junk_img_bojids=[]):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """
    #### 通过fitz获取page信息
    ## 超越边界
    DPI = 72  # use this resolution
    pix = page.get_pixmap(dpi=DPI)
    pageL = 0
    pageR = int(pix.w)
    pageU = 0
    pageD = int(pix.h)
    
    #----------------- 保存每一个文本块的LURD ------------------#
    textLine_blocks = []
    blocks = page.get_text(
            "dict",
            flags=fitz.TEXTFLAGS_TEXT,
            #clip=clip,
        )["blocks"]
    for i in range(len(blocks)):
        bbox = blocks[i]['bbox']
        # print(bbox)
        for tt in blocks[i]['lines']:
            # 当前line
            cur_line_bbox = None                            # 当前line，最右侧的section的bbox
            for xf in tt['spans']:
                L, U, R, D = xf['bbox']
                L, R = min(L, R), max(L, R)
                U, D = min(U, D), max(U, D)
                textLine_blocks.append((L, U, R, D))
    textLine_blocks.sort(key = lambda LURD: (LURD[1], LURD[0]))
    

    #---------------------------------------------- 保存img --------------------------------------------------#
    raw_imgs = page.get_images()                    # 获取所有的图片
    imgs = []
    img_names = []                              # 保存图片的名字，方便在md中插入引用
    img_bboxs = []                              # 保存图片的location信息。
    img_visited = [] # 记忆化，记录该图片是否在md中已经插入过了
    img_ID = 0

    ## 获取、保存每张img的location信息(x1, y1, x2, y2， UL, DR坐标)
    for i in range(len(raw_imgs)):
        # 如果图片在junklist中则跳过
        if raw_imgs[i][0] in junk_img_bojids:
            continue
        else:
            try:
                tt = page.get_image_rects(raw_imgs[i][0], transform = True)

                rec = tt[0][0]
                L, U, R, D = int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3])

                L, R = min(L, R), max(L, R)
                U, D = min(U, D), max(U, D)
                if not(pageL <= L < R <= pageR and pageU <= U < D <= pageD):
                    continue
                if pageL == L and R == pageR:
                    continue
                if pageU == U and D == pageD:
                    continue
                # pix1 = page.get_Pixmap(clip=(L,U,R,D))
                new_img_name = "{}_{}.png".format(page_ID, i)      # 图片name
                # pix1.save(res_dir_path + '/' + new_img_name)        # 把图片存出在新建的文件夹，并命名
                img_names.append(new_img_name)
                img_bboxs.append((L, U, R, D))
                img_visited.append(False)
                imgs.append(raw_imgs[i])
            except:
                continue
    
    #-------- 如果img之间有重叠。说明获取的img大小有问题，位置也不一定对。就扔掉--------#
    imgs_ok = [True for _ in range(len(imgs))]
    for i in range(len(imgs)):
        L1, U1, R1, D1 = img_bboxs[i]
        for j in range(i + 1, len(imgs)):
            L2, U2, R2, D2 = img_bboxs[j]
            ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
            s1 = abs(R1 - L1) * abs(D1 - U1)
            s2 = abs(R2 - L2) * abs(D2 - U2)
            if ratio_1 > 0 and ratio_2 > 0:
                if ratio_1 == 1 and ratio_2 > 0.8:
                    imgs_ok[i] = False
                elif ratio_1 > 0.8 and ratio_2 == 1:
                    imgs_ok[j] = False 
                elif s1 > 20000 and s2 > 20000 and ratio_1 > 0.4 and ratio_2 > 0.4:
                    imgs_ok[i] = False
                    imgs_ok[j] = False
                elif s1 / s2 > 5 and ratio_2 > 0.5:
                    imgs_ok[j] = False
                elif s2 / s1 > 5 and ratio_1 > 0.5:
                    imgs_ok[i] = False
                    
    imgs = [imgs[i] for i in range(len(imgs)) if imgs_ok[i] == True]
    img_names = [img_names[i] for i in range(len(imgs)) if imgs_ok[i] == True]
    img_bboxs = [img_bboxs[i] for i in range(len(imgs)) if imgs_ok[i] == True]
    img_visited = [img_visited[i] for i in range(len(imgs)) if imgs_ok[i] == True]
    #*******************************************************************************#
    
    #---------------------------------------- 通过fitz提取svg的信息 -----------------------------------------#
    #
    svgs = page.get_drawings()
    #------------ preprocess, check一些大框，看是否是合理的 ----------#
    ## 去重。有时候会遇到rect1和rect2是完全一样的情形。
    svg_rect_visited = set()
    available_svgIdx = []
    for i in range(len(svgs)):
        L, U, R, D = svgs[i]['rect'].irect
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        tt = (L, U, R, D)
        if tt not in svg_rect_visited:
            svg_rect_visited.add(tt)
            available_svgIdx.append(i)
        
    svgs = [svgs[i] for i in available_svgIdx]                  # 去重后，有效的svgs
    svg_childs = [[] for _ in range(len(svgs))]
    svg_parents = [[] for _ in range(len(svgs))]
    svg_overlaps = [[] for _ in range(len(svgs))]            #svg_overlaps[i]是一个list，存的是与svg_i有重叠的svg的index。e.g., svg_overlaps[0] = [1, 2, 7, 9]
    svg_visited = [False for _ in range(len(svgs))]
    svg_exceedPage = [0 for _ in range(len(svgs))]       # 是否超越边界（artbox），很大，但一般是一个svg的底。  
        
    
    for i in range(len(svgs)):
        L, U, R, D = svgs[i]['rect'].irect
        ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L, U, R, D, pageL, pageU, pageR, pageD)
        if (pageL + 20 < L <= R < pageR - 20) and (pageU + 20 < U <= D < pageD - 20):
            if ratio_2 >= 0.7:
                svg_exceedPage[i] += 4
        else:
            if L <= pageL:
                svg_exceedPage[i] += 1
            if pageR <= R:
                svg_exceedPage[i] += 1
            if U <= pageU:
                svg_exceedPage[i] += 1
            if pageD <= D:
                svg_exceedPage[i] += 1
            
    #### 如果有≥2个的超边界的框，就不要手写规则判断svg了。很难写对。
    if len([x for x in svg_exceedPage if x >= 1]) >= 2:
        svgs = []
        svg_childs = []
        svg_parents = []
        svg_overlaps = []
        svg_visited = []
        svg_exceedPage = []  
            
    #---------------------------- build graph ----------------------------#
    for i, p in enumerate(svgs):
        L1, U1, R1, D1 = svgs[i]["rect"].irect
        for j in range(len(svgs)):
            if i == j:
                continue
            L2, U2, R2, D2 = svgs[j]["rect"].irect
            ## 包含
            if check_rect1_contains_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                svg_childs[i].append(j)
                svg_parents[j].append(i)
            else:
                ## 交叉
                if check_rect1_overlaps_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                    svg_overlaps[i].append(j)

    #---------------- 确定最终的svg。连通块儿的外围 -------------------#
    eps_ERROR = 5                      # 给识别出的svg，四周留白（为了防止pyMuPDF的rect不准）
    svg_ID = 0        
    svg_final_names = []
    svg_final_bboxs = []
    svg_final_visited = []              # 为下面，text识别左准备。作用同img_visited
    
    svg_idxs = [i for i in range(len(svgs))]
    svg_idxs.sort(key = lambda i: -(svgs[i]['rect'].irect[2] - svgs[i]['rect'].irect[0]) * (svgs[i]['rect'].irect[3] - svgs[i]['rect'].irect[1]))   # 按照面积，从大到小排序
     
    for i in svg_idxs:
        if svg_visited[i] == True:
            continue
        svg_visited[i] = True
        L, U, R, D = svgs[i]['rect'].irect
        width = R - L
        height = D - U
        if check_rect_isLine(L, U, R, D) == True:
            svg_visited[i] = False
            continue
        # if i == 4:
        #     print(i, L, U, R, D)
        #     print(svg_parents[i])
        
        cur_block_element_cnt = 0               # 当前要判定为svg的区域中，有多少elements，最外围的最大svg框除外。
        if len(svg_parents[i]) == 0:
            ## 是个普通框的情形
            cur_block_element_cnt += len(svg_childs[i])
            if svg_exceedPage[i] == 0:
                ## 误差。可能已经包含在某个框里面了
                neglect_flag = False
                for pL, pU, pR, pD in svg_final_bboxs:
                    if pL <= L <= R <= pR and pU <= U <= D <= pD:
                        neglect_flag = True
                        break
                if neglect_flag == True:
                    continue
                
                ## 搜索连通域, bfs+记忆化
                q = collections.deque()
                for j in svg_overlaps[i]:
                    q.append(j)
                while q:
                    j = q.popleft()
                    svg_visited[j] = True
                    L2, U2, R2, D2 = svgs[j]['rect'].irect
                    # width2 = R2 - L2
                    # height2 = D2 - U2
                    # if width2 <= 2 or height2 <= 2 or (height2 / width2) >= 30 or (width2 / height2) >= 30:
                    #     continue
                    L = min(L, L2)
                    R = max(R, R2)
                    U = min(U, U2)
                    D = max(D, D2)
                    cur_block_element_cnt += 1
                    cur_block_element_cnt += len(svg_childs[j])
                    for k in svg_overlaps[j]:
                        if svg_visited[k] == False and svg_exceedPage[k] == 0:
                            svg_visited[k] = True
                            q.append(k)
            elif svg_exceedPage[i] <= 2:
                ## 误差。可能已经包含在某个svg_final_bbox框里面了
                neglect_flag = False
                for sL, sU, sR, sD in svg_final_bboxs:
                    if sL <= L <= R <= sR and sU <= U <= D <= sD:
                        neglect_flag = True
                        break
                if neglect_flag == True:
                    continue
                
                L, U, R, D = pageR, pageD, pageL, pageU
                ## 所有孩子元素的最大边界
                for j in svg_childs[i]:
                    if svg_visited[j] == True:
                        continue
                    if svg_exceedPage[j] >= 1:
                        continue
                    svg_visited[j] = True                       #### 这个位置考虑一下
                    L2, U2, R2, D2 = svgs[j]['rect'].irect
                    L = min(L, L2)
                    R = max(R, R2)
                    U = min(U, U2)
                    D = max(D, D2)
                    cur_block_element_cnt += 1
                    
            # 如果是条line，就不用保存了
            if check_rect_isLine(L, U, R, D) == True:
                continue
            # 如果当前的svg，连2个elements都没有，就不用保存了
            if cur_block_element_cnt < 3:
                continue
            
            ## 当前svg，框住了多少文本框。如果框多了，可能就是错了
            contain_textLineBlock_cnt = 0
            for L2, U2, R2, D2 in textLine_blocks:
                if check_rect1_contains_rect2(L, U, R, D, L2, U2, R2, D2) == True:
                    contain_textLineBlock_cnt += 1
            if contain_textLineBlock_cnt >= 10:
                continue
            
            # L -= eps_ERROR * 2
            # U -= eps_ERROR
            # R += eps_ERROR * 2
            # D += eps_ERROR
            # # cur_svg = page.get_pixmap(matrix=fitz.Identity, dpi=None, colorspace=fitz.csRGB, clip=(U,L,R,D), alpha=False, annots=True)
            # cur_svg = page.get_pixmap(clip=(L,U,R,D))
            new_svg_name = "svg_{}_{}.png".format(page_ID, svg_ID)      # 图片name
            # cur_svg.save(res_dir_path + '/' + new_svg_name)        # 把图片存出在新建的文件夹，并命名
            svg_final_names.append(new_svg_name)                      # 把图片的名字存在list中，方便在md中插入引用
            svg_final_bboxs.append((L, U, R, D))
            svg_final_visited.append(False)
            svg_ID += 1
    
    ## 识别出的svg，可能有 包含，相邻的情形。需要进一步合并
    svg_idxs = [i for i in range(len(svg_final_bboxs))]
    svg_idxs.sort(key = lambda i: (svg_final_bboxs[i][1], svg_final_bboxs[i][0]))   # (U, L)
    svg_final_names_2 = []
    svg_final_bboxs_2 = []
    svg_final_visited_2 = []              # 为下面，text识别左准备。作用同img_visited
    svg_ID_2 = 0
    for i in range(len(svg_final_bboxs)):
        L1, U1, R1, D1 = svg_final_bboxs[i]
        for j in range(i + 1, len(svg_final_bboxs)):
            L2, U2, R2, D2 = svg_final_bboxs[j]
            # 如果 rect1包含了rect2
            if check_rect1_contains_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                svg_final_visited[j] = True
                continue
            # 水平并列
            ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(U1, D1, U2, D2)
            if ratio_1 >= 0.7 and ratio_2 >= 0.7:
                if abs(L2 - R1) >= 20:
                    continue
                LL = min(L1, L2)
                UU = min(U1, U2)
                RR = max(R1, R2)
                DD = max(D1, D2)
                svg_final_bboxs[i] = (LL, UU, RR, DD)
                svg_final_visited[j] = True
                continue
            # 竖直并列
            ratio_1, ratio_2 = calculate_overlapRatio_between_line1_and_line2(L1, R2, L2, R2)
            if ratio_1 >= 0.7 and ratio_2 >= 0.7:
                if abs(U2 - D1) >= 20:
                    continue
                LL = min(L1, L2)
                UU = min(U1, U2)
                RR = max(R1, R2)
                DD = max(D1, D2)
                svg_final_bboxs[i] = (LL, UU, RR, DD)
                svg_final_visited[j] = True
    
    for i in range(len(svg_final_bboxs)):
        if svg_final_visited[i] == False:
            L, U, R, D = svg_final_bboxs[i]
            svg_final_bboxs_2.append((L, U, R, D))
            
            L -= eps_ERROR * 2
            U -= eps_ERROR
            R += eps_ERROR * 2
            D += eps_ERROR
            # cur_svg = page.get_pixmap(clip=(L,U,R,D))
            new_svg_name = "svg_{}_{}.png".format(page_ID, svg_ID_2)      # 图片name
            # cur_svg.save(res_dir_path + '/' + new_svg_name)        # 把图片存出在新建的文件夹，并命名
            svg_final_names_2.append(new_svg_name)                      # 把图片的名字存在list中，方便在md中插入引用
            svg_final_bboxs_2.append((L, U, R, D))
            svg_final_visited_2.append(False)
            svg_ID_2 += 1
       
    ## svg收尾。识别为drawing，但是在上面没有拼成一张图的。
    # 有收尾才comprehensive
    # xxxx
    # xxxx
    # xxxx
    # xxxx
    
    
    #--------- 通过json_from_DocXchain来获取，figure, table, equation的bbox ---------#
    figure_bbox_from_DocXChain = []
    
    figure_from_DocXChain_visited = []          # 记忆化
    figure_bbox_from_DocXChain_overlappedRatio = []
    
    figure_only_from_DocXChain_bboxs = []     # 存储
    figure_only_from_DocXChain_names = []
    figure_only_from_DocXChain_visited = []
    figure_only_ID = 0
    
    xf_json = json_from_DocXchain_obj
    width_from_json = xf_json['page_info']['width']
    height_from_json = xf_json['page_info']['height']
    LR_scaleRatio = width_from_json / (pageR - pageL)
    UD_scaleRatio = height_from_json / (pageD - pageU)
    
    for xf in xf_json['layout_dets']:
    # {0: 'title', 1: 'figure', 2: 'plain text', 3: 'header', 4: 'page number', 5: 'footnote', 6: 'footer', 7: 'table', 8: 'table caption', 9: 'figure caption', 10: 'equation', 11: 'full column', 12: 'sub column'}
        L = xf['poly'][0] / LR_scaleRatio
        U = xf['poly'][1] / UD_scaleRatio
        R = xf['poly'][2] / LR_scaleRatio
        D = xf['poly'][5] / UD_scaleRatio
        # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
        # R += pageL
        # U += pageU
        # D += pageU
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        # figure
        if xf["category_id"] == 1 and xf['score'] >= 0.3:
            figure_bbox_from_DocXChain.append((L, U, R, D))
            figure_from_DocXChain_visited.append(False)
            figure_bbox_from_DocXChain_overlappedRatio.append(0.0)

    #---------------------- 比对上面识别出来的img,svg 与DocXChain给的figure -----------------------#
    
    ## 比对imgs
    for i, b1 in enumerate(figure_bbox_from_DocXChain):
        # print('--------- DocXChain的图片', b1)
        L1, U1, R1, D1 = b1
        for b2 in img_bboxs:
            # print('-------- igms得到的图', b2)
            L2, U2, R2, D2 = b2
            s1 = abs(R1 - L1) * abs(D1 - U1)
            s2 = abs(R2 - L2) * abs(D2 - U2)
            # 相同
            if check_rect1_sameWith_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                figure_from_DocXChain_visited[i] = True
            # 包含
            elif check_rect1_contains_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                if s2 / s1 > 0.8:
                    figure_from_DocXChain_visited[i] = True
            elif check_rect1_contains_rect2(L2, U2, R2, D2, L1, U1, R1, D1) == True:
                if s1 / s2 > 0.8:
                    figure_from_DocXChain_visited[i] = True 
            else:
                # 重叠了相当一部分
                # print('进入第3部分')
                ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                if (ratio_1 >= 0.6 and ratio_2 >= 0.6) or (ratio_1 >= 0.8 and s1/s2>0.8) or (ratio_2 >= 0.8 and s2/s1>0.8):
                    figure_from_DocXChain_visited[i] = True
                else:
                    figure_bbox_from_DocXChain_overlappedRatio[i] += ratio_1
                    # print('图片的重叠率是{}'.format(ratio_1))


    ## 比对svgs
    svg_final_bboxs_2_badIdxs = []
    for i, b1 in enumerate(figure_bbox_from_DocXChain):
        L1, U1, R1, D1 = b1
        for j, b2 in enumerate(svg_final_bboxs_2):
            L2, U2, R2, D2 = b2
            s1 = abs(R1 - L1) * abs(D1 - U1)
            s2 = abs(R2 - L2) * abs(D2 - U2)
            # 相同
            if check_rect1_sameWith_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                figure_from_DocXChain_visited[i] = True
            # 包含
            elif check_rect1_contains_rect2(L1, U1, R1, D1, L2, U2, R2, D2) == True:
                figure_from_DocXChain_visited[i] = True
            elif check_rect1_contains_rect2(L2, U2, R2, D2, L1, U1, R1, D1) == True:
                if s1 / s2 > 0.7:
                    figure_from_DocXChain_visited[i] = True
                else:
                    svg_final_bboxs_2_badIdxs.append(j)     # svg丢弃。用DocXChain的结果。
            else:
                # 重叠了相当一部分
                ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                if (ratio_1 >= 0.5 and ratio_2 >= 0.5) or (min(ratio_1, ratio_2) >= 0.4 and max(ratio_1, ratio_2) >= 0.6):
                    figure_from_DocXChain_visited[i] = True
                else:
                    figure_bbox_from_DocXChain_overlappedRatio[i] += ratio_1
                    
    # 丢掉错误的svg
    svg_final_bboxs_2 = [svg_final_bboxs_2[i] for i in range(len(svg_final_bboxs_2)) if i not in set(svg_final_bboxs_2_badIdxs)]
    
    for i in range(len(figure_from_DocXChain_visited)):
        if figure_bbox_from_DocXChain_overlappedRatio[i] >= 0.7:
            figure_from_DocXChain_visited[i] = True
    
    # DocXChain识别出来的figure，但是没被保存的。
    for i in range(len(figure_from_DocXChain_visited)):
        if figure_from_DocXChain_visited[i] == False:
            figure_from_DocXChain_visited[i] = True
            cur_bbox = figure_bbox_from_DocXChain[i]
            # cur_figure = page.get_pixmap(clip=cur_bbox)
            new_figure_name = "figure_only_{}_{}.png".format(page_ID, figure_only_ID)      # 图片name
            # cur_figure.save(res_dir_path + '/' + new_figure_name)        # 把图片存出在新建的文件夹，并命名
            figure_only_from_DocXChain_names.append(new_figure_name)                      # 把图片的名字存在list中，方便在md中插入引用
            figure_only_from_DocXChain_bboxs.append(cur_bbox)
            figure_only_from_DocXChain_visited.append(False)
            figure_only_ID += 1
    
    img_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    svg_final_bboxs_2.sort(key = lambda LURD: (LURD[1], LURD[0]))
    figure_only_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    curPage_all_fig_bboxs = img_bboxs + svg_final_bboxs + figure_only_from_DocXChain_bboxs
    
    #--------------------------- 最后统一去重 -----------------------------------#
    curPage_all_fig_bboxs.sort(key = lambda LURD: ( (LURD[2]-LURD[0])*(LURD[3]-LURD[1]) , LURD[0], LURD[1]) )
    
    #### 先考虑包含关系的小块
    final_duplicate = set()
    for i in range(len(curPage_all_fig_bboxs)):
        L1, U1, R1, D1 = curPage_all_fig_bboxs[i]
        for j in range(len(curPage_all_fig_bboxs)):
            if i == j:
                continue
            L2, U2, R2, D2 = curPage_all_fig_bboxs[j]
            s1 = abs(R1 - L1) * abs(D1 - U1)
            s2 = abs(R2 - L2) * abs(D2 - U2)
            if check_rect1_contains_rect2(L2, U2, R2, D2, L1, U1, R1, D1) == True:
                final_duplicate.add((L1, U1, R1, D1))
            else:
                ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                if ratio_1 >= 0.8 and ratio_2 <= 0.6:
                    final_duplicate.add((L1, U1, R1, D1))

    curPage_all_fig_bboxs = [LURD for LURD in curPage_all_fig_bboxs if LURD not in final_duplicate]
    
    #### 再考虑重叠关系的块
    final_duplicate = set()
    final_synthetic_bboxs = []
    for i in range(len(curPage_all_fig_bboxs)):
        L1, U1, R1, D1 = curPage_all_fig_bboxs[i]
        for j in range(len(curPage_all_fig_bboxs)):
            if i == j:
                continue
            L2, U2, R2, D2 = curPage_all_fig_bboxs[j]
            s1 = abs(R1 - L1) * abs(D1 - U1)
            s2 = abs(R2 - L2) * abs(D2 - U2)
            ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
            union_ok = False
            if (ratio_1 >= 0.8 and ratio_2 <= 0.6) or (ratio_1 > 0.6 and ratio_2 > 0.6): 
                union_ok = True
            if (ratio_1 > 0.2 and s2 / s1 > 5):
                union_ok = True
            if (L1 <= (L2+R2)/2 <= R1) and (U1 <= (U2+D2)/2 <= D1):
                union_ok = True
            if (L2 <= (L1+R1)/2 <= R2) and (U2 <= (U1+D1)/2 <= D2):
                union_ok = True
            if union_ok == True:
                final_duplicate.add((L1, U1, R1, D1))
                final_duplicate.add((L2, U2, R2, D2))
                L3, U3, R3, D3 = min(L1, L2), min(U1, U2), max(R1, R2), max(D1, D2)
                final_synthetic_bboxs.append((L3, U3, R3, D3))

    # print('---------- curPage_all_fig_bboxs ---------')
    # print(curPage_all_fig_bboxs)
    curPage_all_fig_bboxs = [b for b in curPage_all_fig_bboxs if b not in final_duplicate]    
    final_synthetic_bboxs = list(set(final_synthetic_bboxs))


    ## 再再考虑重叠关系。极端情况下会迭代式地2进1
    new_images = []
    droped_img_idx = []
    image_bboxes = [[b[0], b[1], b[2], b[3]] for b in final_synthetic_bboxs]        
    for i in range(0, len(image_bboxes)):
        for j in range(i+1, len(image_bboxes)):
            if j not in droped_img_idx:
                L2, U2, R2, D2 = image_bboxes[j]
                s1 = abs(R1 - L1) * abs(D1 - U1)
                s2 = abs(R2 - L2) * abs(D2 - U2)
                ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
                union_ok = False
                if (ratio_1 >= 0.8 and ratio_2 <= 0.6) or (ratio_1 > 0.6 and ratio_2 > 0.6): 
                    union_ok = True
                if (ratio_1 > 0.2 and s2 / s1 > 5):
                    union_ok = True
                if (L1 <= (L2+R2)/2 <= R1) and (U1 <= (U2+D2)/2 <= D1):
                    union_ok = True
                if (L2 <= (L1+R1)/2 <= R2) and (U2 <= (U1+D1)/2 <= D2):
                    union_ok = True
                if union_ok == True:
                    # 合并
                    image_bboxes[i][0], image_bboxes[i][1],image_bboxes[i][2],image_bboxes[i][3] = min(image_bboxes[i][0], image_bboxes[j][0]), min(image_bboxes[i][1], image_bboxes[j][1]), max(image_bboxes[i][2], image_bboxes[j][2]), max(image_bboxes[i][3], image_bboxes[j][3])
                    droped_img_idx.append(j)
            
    for i in range(0, len(image_bboxes)):
        if i not in droped_img_idx:
            new_images.append(image_bboxes[i])
    
    
    # find_union_FLAG = True
    # while find_union_FLAG == True:
    #     find_union_FLAG = False
    #     final_duplicate = set()
    #     tmp = []
    #     for i in range(len(final_synthetic_bboxs)):
    #         L1, U1, R1, D1 = final_synthetic_bboxs[i]
    #         for j in range(len(final_synthetic_bboxs)):
    #             if i == j:
    #                 continue
    #             L2, U2, R2, D2 = final_synthetic_bboxs[j]
    #             s1 = abs(R1 - L1) * abs(D1 - U1)
    #             s2 = abs(R2 - L2) * abs(D2 - U2)
    #             ratio_1, ratio_2 = calculate_overlapRatio_between_rect1_and_rect2(L1, U1, R1, D1, L2, U2, R2, D2)
    #             union_ok = False
    #             if (ratio_1 >= 0.8 and ratio_2 <= 0.6) or (ratio_1 > 0.6 and ratio_2 > 0.6): 
    #                 union_ok = True
    #             if (ratio_1 > 0.2 and s2 / s1 > 5):
    #                 union_ok = True
    #             if (L1 <= (L2+R2)/2 <= R1) and (U1 <= (U2+D2)/2 <= D1):
    #                 union_ok = True
    #             if (L2 <= (L1+R1)/2 <= R2) and (U2 <= (U1+D1)/2 <= D2):
    #                 union_ok = True
    #             if union_ok == True:
    #                 find_union_FLAG = True
    #                 final_duplicate.add((L1, U1, R1, D1))
    #                 final_duplicate.add((L2, U2, R2, D2))
    #                 L3, U3, R3, D3 = min(L1, L2), min(U1, U2), max(R1, R2), max(D1, D2)
    #                 tmp.append((L3, U3, R3, D3)) 
    #     if find_union_FLAG == True:
    #         tmp = list(set(tmp))
    #         final_synthetic_bboxs = tmp[:]
    

    # curPage_all_fig_bboxs += final_synthetic_bboxs
    # print('--------- final synthetic')
    # print(final_synthetic_bboxs)
    #**************************************************************************#
    images1 = [[img[0], img[1], img[2], img[3]] for img in curPage_all_fig_bboxs]
    images = images1 + new_images
    return images

