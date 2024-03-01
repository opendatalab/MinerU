from pdf_tools.libs import fitz             # pyMuPDF库


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


def evaluate_pdf_layout(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
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

    #--------- 通过json_from_DocXchain来获取 title ---------#
    title_bbox_from_DocXChain = []

    xf_json = json_from_DocXchain_obj
    width_from_json = xf_json['page_info']['width']
    height_from_json = xf_json['page_info']['height']
    LR_scaleRatio = width_from_json / (pageR - pageL)
    UD_scaleRatio = height_from_json / (pageD - pageU)

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
    LOSS_THRESHOLD = 2000               # 经验值
    fullColumn_bboxs = []
    subColumn_bboxs = []
    plainText_bboxs = []
    #### read information of plain text
    for xf in xf_json['layout_dets']:
        L = xf['poly'][0] / LR_scaleRatio
        U = xf['poly'][1] / UD_scaleRatio
        R = xf['poly'][2] / LR_scaleRatio
        D = xf['poly'][5] / UD_scaleRatio
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        if xf['category_id'] == 2:
            plainText_bboxs.append((L, U, R, D))
    #### read information of column
    for xf in xf_json['subfield_dets']:
        L = xf['poly'][0] / LR_scaleRatio
        U = xf['poly'][1] / UD_scaleRatio
        R = xf['poly'][2] / LR_scaleRatio
        D = xf['poly'][5] / UD_scaleRatio
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        if xf['category_id'] == 11:
            fullColumn_bboxs.append((L, U, R, D))
        elif xf['category_id'] == 12:
            subColumn_bboxs.append((L, U, R, D))
            

    curPage_loss = 0        # 当前页的loss
    fail_cnt = 0            # Text文本块没被圈到的情形。
    for L, U, R, D in plainText_bboxs:
        find = False
        for L2, U2, R2, D2 in (fullColumn_bboxs + subColumn_bboxs):
            ratio_1, _ = calculate_overlapRatio_between_rect1_and_rect2(L, U, R, D, L2, U2, R2, D2)
            if ratio_1 >= 0.9:
                loss_1 = (L + R) / 2 - (L2 + R2) / 2
                loss_2 = L - L2
                cur_loss = min(abs(loss_1), abs(loss_2))
                curPage_loss += cur_loss
                find = True
                break
        if find == False:
            fail_cnt += 1

    isSimpleLayout_flag = False
    if fail_cnt == 0 and len(fullColumn_bboxs) <= 1 and len(subColumn_bboxs) <= 2:
        if curPage_loss <= LOSS_THRESHOLD:
            isSimpleLayout_flag  = True
    
    return isSimpleLayout_flag, len(fullColumn_bboxs), len(subColumn_bboxs), curPage_loss
