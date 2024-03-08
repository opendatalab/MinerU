from magic_pdf.libs.commons import fitz             # pyMuPDF库
from magic_pdf.libs.coordinate_transform import get_scale_ratio


def parse_pageNos(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """

    #--------- 通过json_from_DocXchain来获取 pageNo ---------#
    pageNo_bbox_from_DocXChain = []

    xf_json = json_from_DocXchain_obj
    horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(xf_json, page)

    # {0: 'title',  # 标题
    # 1: 'figure', # 图片
    #  2: 'plain text',  # 文本
    #  3: 'header',      # 页眉
    #  4: 'page number', # 页码
    #  5: 'footnote',    # 脚注
    #  6: 'footer',      # 页脚
    #  7: 'table',       # 表格
    #  8: 'table caption',  # 表格描述
    #  9: 'figure caption', # 图片描述
    #  10: 'equation',      # 公式
    #  11: 'full column',   # 单栏
    #  12: 'sub column',    # 多栏
    #  13: 'embedding',     # 嵌入公式
    #  14: 'isolated'}      # 单行公式
    for xf in xf_json['layout_dets']:
        L = xf['poly'][0] / horizontal_scale_ratio
        U = xf['poly'][1] / vertical_scale_ratio
        R = xf['poly'][2] / horizontal_scale_ratio
        D = xf['poly'][5] / vertical_scale_ratio
        # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
        # R += pageL
        # U += pageU
        # D += pageU
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        if xf['category_id'] == 4 and xf['score'] >= 0.3:
            pageNo_bbox_from_DocXChain.append((L, U, R, D))
            
    
    pageNo_final_names = []
    pageNo_final_bboxs = []
    pageNo_ID = 0
    for L, U, R, D in pageNo_bbox_from_DocXChain:
        # cur_pageNo = page.get_pixmap(clip=(L,U,R,D))
        new_pageNo_name = "pageNo_{}_{}.png".format(page_ID, pageNo_ID)    # 页码name
        # cur_pageNo.save(res_dir_path + '/' + new_pageNo_name)           # 把页码存储在新建的文件夹，并命名
        pageNo_final_names.append(new_pageNo_name)                        # 把页码的名字存在list中
        pageNo_final_bboxs.append((L, U, R, D))
        pageNo_ID += 1
        

    pageNo_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    curPage_all_pageNo_bboxs = pageNo_final_bboxs
    return curPage_all_pageNo_bboxs

