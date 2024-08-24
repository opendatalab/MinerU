from magic_pdf.libs.commons import fitz             # pyMuPDF库


def parse_tables(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
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
    table_bbox_from_DocXChain = []

    xf_json = json_from_DocXchain_obj
    width_from_json = xf_json['page_info']['width']
    height_from_json = xf_json['page_info']['height']
    LR_scaleRatio = width_from_json / (pageR - pageL)
    UD_scaleRatio = height_from_json / (pageD - pageU)

    
    for xf in xf_json['layout_dets']:
    # {0: 'title', 1: 'figure', 2: 'plain text', 3: 'header', 4: 'page number', 5: 'footnote', 6: 'footer', 7: 'table', 8: 'table caption', 9: 'figure caption', 10: 'equation', 11: 'full column', 12: 'sub column'}
    #  13: 'embedding',     # 嵌入公式
    #  14: 'isolated'}      # 单行公式
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
        if xf['category_id'] == 7 and xf['score'] >= 0.3:
            table_bbox_from_DocXChain.append((L, U, R, D))
            
    
    table_final_names = []
    table_final_bboxs = []
    table_ID = 0
    for L, U, R, D in table_bbox_from_DocXChain:
        # cur_table = page.get_pixmap(clip=(L,U,R,D))
        new_table_name = "table_{}_{}.png".format(page_ID, table_ID)      # 表格name
        # cur_table.save(res_dir_path + '/' + new_table_name)        # 把表格存出在新建的文件夹，并命名
        table_final_names.append(new_table_name)                      # 把表格的名字存在list中，方便在md中插入引用
        table_final_bboxs.append((L, U, R, D))
        table_ID += 1
        

    table_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    curPage_all_table_bboxs = table_final_bboxs
    return curPage_all_table_bboxs

