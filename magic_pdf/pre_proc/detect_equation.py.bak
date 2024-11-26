from magic_pdf.libs.boxbase import _is_in, calculate_overlap_area_2_minbox_area_ratio              # 正则
from magic_pdf.libs.commons import fitz             # pyMuPDF库


def __solve_contain_bboxs(all_bbox_list: list):

    """将两个公式的bbox做判断是否有包含关系，若有的话则删掉较小的bbox"""

    dump_list = []
    for i in range(len(all_bbox_list)):
        for j in range(i + 1, len(all_bbox_list)):
            # 获取当前两个值
            bbox1 = all_bbox_list[i][:4]
            bbox2 = all_bbox_list[j][:4]
            
            # 删掉较小的框
            if _is_in(bbox1, bbox2):
                dump_list.append(all_bbox_list[i])
            elif _is_in(bbox2, bbox1):
                dump_list.append(all_bbox_list[j])
            else:
                ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
                if ratio > 0.7:
                    s1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) 
                    s2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    if s2 > s1:  
                        dump_list.append(all_bbox_list[i])
                    else:
                        dump_list.append(all_bbox_list[i]) 

    # 遍历需要删除的列表中的每个元素
    for item in dump_list:
        
        while item in all_bbox_list:
            all_bbox_list.remove(item)
    return all_bbox_list


def parse_equations(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
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
    

    #--------- 通过json_from_DocXchain来获取 table ---------#
    equationEmbedding_from_DocXChain_bboxs = []
    equationIsolated_from_DocXChain_bboxs = []
    
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
        # equation
        img_suffix = f"{page_ID}_{int(L)}_{int(U)}_{int(R)}_{int(D)}"
        if xf['category_id'] == 13 and xf['score'] >= 0.3:      
            latex_text = xf.get("latex", "EmptyInlineEquationResult")
            debugable_latex_text = f"{latex_text}|{img_suffix}"
            equationEmbedding_from_DocXChain_bboxs.append((L, U, R, D, latex_text))
        if xf['category_id'] == 14 and xf['score'] >= 0.3:
            latex_text = xf.get("latex", "EmptyInterlineEquationResult")
            debugable_latex_text = f"{latex_text}|{img_suffix}"
            equationIsolated_from_DocXChain_bboxs.append((L, U, R, D, latex_text))
    
    #---------------------------------------- 排序，编号，保存 -----------------------------------------#
    equationIsolated_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    equationIsolated_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    
    equationEmbedding_from_DocXChain_names = []
    equationEmbedding_ID = 0
    
    equationIsolated_from_DocXChain_names = []
    equationIsolated_ID = 0
    
    for L, U, R, D, _ in equationEmbedding_from_DocXChain_bboxs:
        if not(L < R and U < D):
            continue
        try:
            # cur_equation = page.get_pixmap(clip=(L,U,R,D))
            new_equation_name = "equationEmbedding_{}_{}.png".format(page_ID, equationEmbedding_ID)        # 公式name
            # cur_equation.save(res_dir_path + '/' + new_equation_name)                       # 把公式存出在新建的文件夹，并命名
            equationEmbedding_from_DocXChain_names.append(new_equation_name)                         # 把公式的名字存在list中，方便在md中插入引用
            equationEmbedding_ID += 1
        except:
            pass

    for L, U, R, D, _ in equationIsolated_from_DocXChain_bboxs:
        if not(L < R and U < D):
            continue
        try:
            # cur_equation = page.get_pixmap(clip=(L,U,R,D))
            new_equation_name = "equationEmbedding_{}_{}.png".format(page_ID, equationIsolated_ID)        # 公式name
            # cur_equation.save(res_dir_path + '/' + new_equation_name)                       # 把公式存出在新建的文件夹，并命名
            equationIsolated_from_DocXChain_names.append(new_equation_name)                         # 把公式的名字存在list中，方便在md中插入引用
            equationIsolated_ID += 1
        except:
            pass
    
    equationEmbedding_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    equationIsolated_from_DocXChain_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    
    
    """根据pdf可视区域，调整bbox的坐标"""
    cropbox = page.cropbox
    if cropbox[0]!=page.rect[0] or cropbox[1]!=page.rect[1]:
        for eq_box in equationEmbedding_from_DocXChain_bboxs:
            eq_box = [eq_box[0]+cropbox[0], eq_box[1]+cropbox[1], eq_box[2]+cropbox[0], eq_box[3]+cropbox[1], eq_box[4]]
        for eq_box in equationIsolated_from_DocXChain_bboxs:
            eq_box = [eq_box[0]+cropbox[0], eq_box[1]+cropbox[1], eq_box[2]+cropbox[0], eq_box[3]+cropbox[1], eq_box[4]]
        
    deduped_embedding_eq_bboxes = __solve_contain_bboxs(equationEmbedding_from_DocXChain_bboxs)
    return deduped_embedding_eq_bboxes, equationIsolated_from_DocXChain_bboxs
