from libs.commons import fitz
from typing import List


def show_image(item, title=""):
    """Display a pixmap.

    Just to display Pixmap image of "item" - ignore the man behind the curtain.

    Args:
        item: any PyMuPDF object having a "get_pixmap" method.
        title: a string to be used as image title

    Generates an RGB Pixmap from item using a constant DPI and using matplotlib
    to show it inline of the notebook.
    """
    DPI = 150  # use this resolution
    import numpy as np
    import matplotlib.pyplot as plt

    # %matplotlib inline
    pix = item.get_pixmap(dpi=DPI)
    img = np.ndarray([pix.h, pix.w, 3], dtype=np.uint8, buffer=pix.samples_mv)
    plt.figure(dpi=DPI)  # set the figure's DPI
    plt.title(title)  # set title of image
    _ = plt.imshow(img, extent=(0, pix.w * 72 / DPI, pix.h * 72 / DPI, 0))


def calculate_overlapRatio_between_line1_and_line2(L1: float, R1: float, L2: float, R2: float) -> (float, float):
    # 计算两个line，重叠line各占2个line长度的比例
    if max(L1, L2) > min(R1, R2):
        return 0, 0
    if L1 == R1 or L2 == R2:
        return 0, 0
    overlap_line = min(R1, R2) - max(L1, L2)
    return overlap_line / (R1 - L1), overlap_line / (R2 - L2)


def get_targetAxis_and_splitAxis(page_ID: int, page: fitz.Page, columnNumber: int, textBboxs: List[(float, float, float, float)]) -> (List[float], List[float]):
    """
    param: page: fitz解析出来的格式
    param: columnNumber: Text的列数
    param: textBboxs: 文本块list。 [(L, U, R, D), ... ]
    return: 
    
    """
    INF = 10 ** 9
    pageL, pageU, pageR, pageD = INF, INF, 0, 0
    for L, U, R, D in textBboxs:
        assert L <= R and U <= D
        pageL = min(pageL, L)
        pageR = max(pageR, R)
        pageU = min(pageU, U)
        pageD = max(pageD, D)

    pageWidth = pageR - pageL
    pageHeight = pageD - pageU
    pageL -= pageWidth / 10  # 10是经验值
    pageR += pageWidth / 10
    pageU -= pageHeight / 10
    pageD += pageHeight / 10
    pageWidth = pageR - pageL
    pageHeight = pageD - pageU

    x_targetAxis = []
    x_splitAxis = []
    for i in range(0, columnNumber * 2 + 1):
        if i & 1:
            x_targetAxis.append(pageL + pageWidth / (2 * columnNumber) * i)
        else:
            x_splitAxis.append(pageL + pageWidth / (2 * columnNumber) * i)

    # # 可视化：分列的外框
    # path_bbox = []
    # N = len(x_targetAxis)
    # for i in range(N):
    #     L, R = x_splitAxis[i], x_splitAxis[i + 1]
    #     path_bbox.append((L, pageU, R, pageD))
    # shape = page.new_shape()
    # # iterate over the bboxes
    # color_map = [fitz.pdfcolor["red"], fitz.pdfcolor["blue"], fitz.pdfcolor["yellow"], fitz.pdfcolor["black"], fitz.pdfcolor["green"], fitz.pdfcolor["brown"]]
    # for i, rect in enumerate(path_bbox):
    #     # if i < 20:
    #     #     continue
    #     shape.draw_rect(rect)  # draw a border
    #     shape.insert_text(Point(rect[0], rect[1])+(5, 15), str(i), color=fitz.pdfcolor["blue"])
    #     shape.finish(color=color_map[i%len(color_map)])
    #     # shape.finish(color=fitz.pdfcolor["blue"])
    #     shape.commit()  # store to the page

    #     # if i == 3:
    #     #     print(rect)
    #     #     break
    #     # print(rect)
    # show_image(page, f"Table & Header BBoxes")            

    return x_targetAxis, x_splitAxis


def calculate_loss(page_ID: int, x_targetAxis: List[float], x_splitAxis: List[float], textBboxs: List[(float, float, float, float)]) -> (float, bool):
    INF = 10 ** 9

    # page_artbox = page.artbox
    # pageL, pageU, pageR, pageD = page_artbox[0], page_artbox[1], page_artbox[2], page_artbox[3]

    pageL, pageU, pageR, pageD = INF, INF, 0, 0
    for L, U, R, D in textBboxs:
        assert L <= R and U <= D
        pageL = min(pageL, L)
        pageR = max(pageR, R)
        pageU = min(pageU, U)
        pageD = max(pageD, D)

    pageWidth = pageR - pageL
    pageHeight = pageD - pageU
    pageL -= pageWidth / 10
    pageR += pageWidth / 10
    pageU -= pageHeight / 10
    pageD += pageHeight / 10
    pageWidth = pageR - pageL
    pageHeight = pageD - pageU

    col_N = len(x_targetAxis)  # 列数
    col_texts_mid = [[] for _ in range(col_N)]
    col_texts_LR = [[] for _ in range(col_N)]

    oneLocateLoss_mid = 0
    oneLocateLoss_LR = 0
    oneLocateCnt_mid = 0  # 完美在一列中的个数
    oneLocateCnt_LR = 0
    oneLocateSquare_mid = 0.0  # 完美在一列的面积
    oneLocateSquare_LR = 0.0

    multiLocateLoss_mid = 0
    multiLocateLoss_LR = 0
    multiLocateCnt_mid = 0  # 在多列中的个数
    multiLocateCnt_LR = 0
    multiLocateSquare_mid = 0.0  # 在多列中的面积
    multiLocateSquare_LR = 0.0

    allLocateLoss_mid = 0
    allLocateLoss_LR = 0
    allLocateCnt_mid = 0  # 横跨页面的大框的个数
    allLocateCnt_LR = 0
    allLocateSquare_mid = 0.0  # 横跨整个页面的个数
    allLocateSquare_LR = 0.0

    isSimpleCondition = True  # 就1个。2种方式，只要有一种情况不规整，就是不规整。
    colID_Textcnt_mid = [0 for _ in range(col_N)]  # 每一列中有多少个Text块，根据mid判断的
    colID_Textcnt_LR = [0 for _ in range(col_N)]  # 每一列中有多少个Text块，根据区间边界判断

    allLocateBboxs_mid = []  # 跨整页的，bbox
    allLocateBboxs_LR = []
    non_allLocateBboxs_mid = []
    non_allLocateBboxs_LR = []  # 不在单独某一列，但又不是全列
    for L, U, R, D in textBboxs:
        if D - U < 40:  # 现在还没拼接好。先简单这样过滤页眉。也会牺牲一些很窄的长条
            continue
        if R - L < 40:
            continue
        located_cols_mid = []
        located_cols_LR = []
        for col_ID in range(col_N):
            if col_N == 1:
                located_cols_mid.append(col_ID)
                located_cols_LR.append(col_ID)
            else:
                if L <= x_targetAxis[col_ID] <= R:
                    located_cols_mid.append(col_ID)
                if calculate_overlapRatio_between_line1_and_line2(x_splitAxis[col_ID], x_splitAxis[col_ID + 1], L, R)[0] >= 0.2:
                    located_cols_LR.append(col_ID)

        if len(located_cols_mid) == col_N:
            allLocateBboxs_mid.append((L, U, R, D))
        else:
            non_allLocateBboxs_mid.append((L, U, R, D))
        if len(located_cols_LR) == col_N:
            allLocateBboxs_LR.append((L, U, R, D))
        else:
            non_allLocateBboxs_LR.append((L, U, R, D))

    allLocateBboxs_mid.sort(key=lambda LURD: (LURD[1], LURD[0]))
    non_allLocateBboxs_mid.sort(key=lambda LURD: (LURD[1], LURD[0]))
    allLocateBboxs_LR.sort(key=lambda LURD: (LURD[1], LURD[0]))
    non_allLocateBboxs_LR.sort(key=lambda LURD: (LURD[1], LURD[0]))

    # --------------------判断，是不是有标题类的小块，掺杂在一列的pdf页面里。-------------#
    isOneClumn = False
    under_cnt = 0
    under_square = 0.0
    before_cnt = 0
    before_square = 0.0
    for nL, nU, nR, nD in non_allLocateBboxs_mid:
        cnt = 0
        for L, U, R, D in allLocateBboxs_mid:
            if nD <= U:
                cnt += 1
        if cnt >= 1:
            before_cnt += cnt
            before_square += (R - L) * (D - U) * cnt
        else:
            under_cnt += 1
            under_square += (R - L) * (D - U) * cnt

    if (before_square + under_square) != 0 and before_square / (before_square + under_square) >= 0.2:
        isOneClumn = True

    if isOneClumn == True and col_N != 1:
        return INF, False
    if isOneClumn == True and col_N == 1:
        return 0, True
    #### 根据边界的统计情况，再判断一次
    isOneClumn = False
    under_cnt = 0
    under_square = 0.0
    before_cnt = 0
    before_square = 0.0
    for nL, nU, nR, nD in non_allLocateBboxs_LR:
        cnt = 0
        for L, U, R, D in allLocateBboxs_LR:
            if nD <= U:
                cnt += 1
        if cnt >= 1:
            before_cnt += cnt
            before_square += (R - L) * (D - U) * cnt
        else:
            under_cnt += 1
            under_square += (R - L) * (D - U) * cnt

    if (before_square + under_square) != 0 and before_square / (before_square + under_square) >= 0.2:
        isOneClumn = True

    if isOneClumn == True and col_N != 1:
        return INF, False
    if isOneClumn == True and col_N == 1:
        return 0, True

    for L, U, R, D in textBboxs:
        assert L < R and U < D, 'There is an error on bbox of text when calculate loss!'

        # 简单排除页眉、迷你小块
        # if (D - U) < pageHeight / 15 < 40 or (R - L) < pageWidth / 8:
        if (D - U) < 40:
            continue
        if (R - L) < 40:
            continue
        mid = (L + R) / 2
        located_cols_mid = []  # 在哪一列里，根据中点来判断
        located_cols_LR = []  # 在哪一列里，根据边界判断
        for col_ID in range(col_N):
            if col_N == 1:
                located_cols_mid.append(col_ID)
            else:
                # 根据中点判断
                if L <= x_targetAxis[col_ID] <= R:
                    located_cols_mid.append(col_ID)
                # 根据边界判断
                if calculate_overlapRatio_between_line1_and_line2(x_splitAxis[col_ID], x_splitAxis[col_ID + 1], L, R)[0] >= 0.2:
                    located_cols_LR.append(col_ID)

        ## 1列的情形
        if col_N == 1:
            oneLocateLoss_mid += abs(mid - x_targetAxis[located_cols_mid[0]]) * (D - U) * (R - L)
            # oneLocateLoss_mid += abs(L - x_splitAxis[located_cols[0]]) * (D - U) * (R - L)
            oneLocateLoss_LR += abs(L - x_splitAxis[located_cols_mid[0]]) * (D - U) * (R - L)
            oneLocateCnt_mid += 1
            oneLocateSquare_mid += (D - U) * (R - L)
        ## 多列的情形
        else:
            ######## 根据mid判断
            if len(located_cols_mid) == 1:
                oneLocateLoss_mid += abs(mid - x_targetAxis[located_cols_mid[0]]) * (D - U) * (R - L)
                # oneLocateLoss_mid += abs(L - x_splitAxis[located_cols[0]]) * (D - U) * (R - L)
                oneLocateCnt_mid += 1
                oneLocateSquare_mid += (D - U) * (R - L)
            elif 1 <= len(located_cols_mid) < col_N:
                ll, rr = located_cols_mid[0], located_cols_mid[-1]
                # multiLocateLoss_mid += abs(mid - (x_targetAxis[ll] + x_targetAxis[rr]) / 2) * (D - U) * (R - L)
                multiLocateLoss_mid += abs(mid - x_targetAxis[ll]) * (D - U) * (R - L)
                # multiLocateLoss_mid += abs(mid - (pageL + pageR) / 2) * (D - U) * (R - L)
                multiLocateCnt_mid += 1
                multiLocateSquare_mid += (D - U) * (R - L)
                isSimpleCondition = False
            else:
                allLocateLoss_mid += abs(mid - (pageR + pageL) / 2) * (D - U) * (R - L)
                allLocateCnt_mid += 1
                allLocateSquare_mid += (D - U) * (R - L)
                isSimpleCondition = False

            ######## 根据区间的边界判断
            if len(located_cols_LR) == 1:
                oneLocateLoss_LR += abs(mid - x_targetAxis[located_cols_LR[0]]) * (D - U) * (R - L)
                # oneLocateLoss_LR += abs(L - x_splitAxis[located_cols_LR[0]]) * (D - U) * (R - L)
                oneLocateCnt_LR += 1
                oneLocateSquare_LR += (D - U) * (R - L)
            elif 1 <= len(located_cols_LR) < col_N:
                ll, rr = located_cols_LR[0], located_cols_LR[-1]
                # multiLocateLoss_LR += abs(mid - (x_targetAxis[ll] + x_targetAxis[rr]) / 2) * (D - U) * (R - L)
                multiLocateLoss_LR += abs(mid - x_targetAxis[ll]) * (D - U) * (R - L)
                # multiLocateLoss_LR += abs(mid - (pageL + pageR) / 2) * (D - U) * (R - L)
                multiLocateCnt_LR += 1
                multiLocateSquare_LR += (D - U) * (R - L)
                isSimpleCondition = False
            else:
                allLocateLoss_LR += abs(mid - (pageR + pageL) / 2) * (D - U) * (R - L)
                allLocateCnt_LR += 1
                allLocateSquare_LR += (D - U) * (R - L)
                isSimpleCondition = False

    tot_TextCnt = oneLocateCnt_mid + multiLocateCnt_mid + allLocateCnt_mid
    tot_TextSquare = oneLocateSquare_mid + multiLocateSquare_mid + allLocateSquare_mid

    # 1列的情形
    if tot_TextSquare != 0 and allLocateSquare_mid / tot_TextSquare >= 0.85 and col_N == 1:
        return 0, True

    # 多列的情形

    # if col_N >= 2:
    #     if allLocateCnt >= 1:
    #         oneLocateLoss_mid += ((pageR - pageL)) * oneLocateCnt_mid
    #         multiLocateLoss_mid += ((pageR - pageL) ) * multiLocateCnt_mid
    #     else:
    #         if multiLocateCnt_mid >= 1:
    #             oneLocateLoss_mid += ((pageR - pageL)) * oneLocateCnt_mid
    totLoss_mid = oneLocateLoss_mid + multiLocateLoss_mid + allLocateLoss_mid
    totLoss_LR = oneLocateCnt_LR + multiLocateCnt_LR + allLocateLoss_LR
    return totLoss_mid + totLoss_LR, isSimpleCondition


def get_columnNumber(page_ID: int, page: fitz.Page, textBboxs) -> (int, float):
    columnNumber_loss = dict()
    columnNumber_isSimpleCondition = dict()
    #### 枚举列数
    for columnNumber in range(1, 5):
        # print('---------{}--------'.format(columnNumber))
        x_targetAxis, x_splitAxis = get_targetAxis_and_splitAxis(page_ID, page, columnNumber, textBboxs)
        loss, isSimpleCondition = calculate_loss(page_ID, x_targetAxis, x_splitAxis, textBboxs)
        columnNumber_loss[columnNumber] = loss
        columnNumber_isSimpleCondition[columnNumber] = isSimpleCondition

    col_idxs = [i for i in range(1, len(columnNumber_loss) + 1)]
    col_idxs.sort(key=lambda i: (columnNumber_loss[i], i))

    return col_idxs, columnNumber_loss, columnNumber_isSimpleCondition
