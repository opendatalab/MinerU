from collections import Counter
from magic_pdf.libs.commons import fitz             # pyMuPDF库
from magic_pdf.libs.coordinate_transform import get_scale_ratio


def parse_footnotes_by_model(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict, md_bookname_save_path=None, debug_mode=False):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """

    #--------- 通过json_from_DocXchain来获取 footnote ---------#
    footnote_bbox_from_DocXChain = []

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
        # if xf['category_id'] == 5 and xf['score'] >= 0.3:
        if xf['category_id'] == 5 and xf['score'] >= 0.43:  # 新的footnote阈值
            footnote_bbox_from_DocXChain.append((L, U, R, D))
            
    
    footnote_final_names = []
    footnote_final_bboxs = []
    footnote_ID = 0
    for L, U, R, D in footnote_bbox_from_DocXChain:
        if debug_mode:
            # cur_footnote = page.get_pixmap(clip=(L,U,R,D))
            new_footnote_name = "footnote_{}_{}.png".format(page_ID, footnote_ID)    # 脚注name
            # cur_footnote.save(md_bookname_save_path + '/' + new_footnote_name)           # 把脚注存储在新建的文件夹，并命名
            footnote_final_names.append(new_footnote_name)                        # 把脚注的名字存在list中
        footnote_final_bboxs.append((L, U, R, D))
        footnote_ID += 1
        

    footnote_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    curPage_all_footnote_bboxs = footnote_final_bboxs
    return curPage_all_footnote_bboxs


def need_remove(block):
    if 'lines' in block and len(block['lines']) > 0:
        # block中只有一行，且该行文本全是大写字母，或字体为粗体bold关键词，SB关键词，把这个block捞回来
        if len(block['lines']) == 1:
            if 'spans' in block['lines'][0] and len(block['lines'][0]['spans']) == 1:
                font_keywords = ['SB', 'bold', 'Bold']
                if block['lines'][0]['spans'][0]['text'].isupper() or any(keyword in block['lines'][0]['spans'][0]['font'] for keyword in font_keywords):
                    return True
        for line in block['lines']:
            if 'spans' in line and len(line['spans']) > 0:
                for span in line['spans']:
                    # 检测"keyword"是否在span中，忽略大小写
                    if "keyword" in span['text'].lower():
                        return True
    return False

def parse_footnotes_by_rule(remain_text_blocks, page_height, page_id, main_text_font):
    """
    根据给定的文本块、页高和页码，解析出符合规则的脚注文本块，并返回其边界框。

    Args:
        remain_text_blocks (list): 包含所有待处理的文本块的列表。
        page_height (float): 页面的高度。
        page_id (int): 页面的ID。

    Returns:
        list: 符合规则的脚注文本块的边界框列表。

    """
    # if page_id > 20:
    if page_id > 2:  # 为保证精确度，先只筛选前3页
        return []
    else:
        # 存储每一行的文本块大小的列表
        line_sizes = []
        # 存储每个文本块的平均行大小
        block_sizes = []
        # 存储每一行的字体信息
        # font_names = []
        font_names = Counter()
        if len(remain_text_blocks) > 0:
            for block in remain_text_blocks:
                block_line_sizes = []
                # block_fonts = []
                block_fonts = Counter()
                for line in block['lines']:
                    # 提取每个span的size属性，并计算行大小
                    span_sizes = [span['size'] for span in line['spans'] if 'size' in span]
                    if span_sizes:
                        line_size = sum(span_sizes) / len(span_sizes)
                        line_sizes.append(line_size)
                        block_line_sizes.append(line_size)
                    span_font = [(span['font'], len(span['text'])) for span in line['spans'] if 'font' in span and len(span['text']) > 0]
                    if span_font:
                        #  main_text_font应该用基于字数最多的字体而不是span级别的统计
                        # font_names.append(font_name for font_name in span_font)
                        # block_fonts.append(font_name for font_name in span_font)
                        for font, count in span_font:
                            # font_names.extend([font] * count)
                            # block_fonts.extend([font] * count)
                            font_names[font] += count
                            block_fonts[font] += count
                if block_line_sizes:
                    # 计算文本块的平均行大小
                    block_size = sum(block_line_sizes) / len(block_line_sizes)
                    # block_font = collections.Counter(block_fonts).most_common(1)[0][0]
                    block_font = block_fonts.most_common(1)[0][0]
                    block_sizes.append((block, block_size, block_font))

            # 计算main_text_size
            main_text_size = Counter(line_sizes).most_common(1)[0][0]
            # 计算main_text_font
            # main_text_font = collections.Counter(font_names).most_common(1)[0][0]
            # main_text_font = font_names.most_common(1)[0][0]
            # 删除一些可能被误识别为脚注的文本块
            block_sizes = [(block, block_size, block_font) for block, block_size, block_font in block_sizes if not need_remove(block)]

            # 检测footnote_block 并返回 footnote_bboxes
            # footnote_bboxes = [block['bbox'] for block, block_size, block_font in block_sizes if
            #                    block['bbox'][1] > page_height * 0.6 and block_size < main_text_size
            #                    and (len(block['lines']) < 5 or block_font != main_text_font)]
                               # and len(block['lines']) < 5]
            footnote_bboxes = [block['bbox'] for block, block_size, block_font in block_sizes if
                               block['bbox'][1] > page_height * 0.6 and
                               #  较为严格的规则
                               block_size < main_text_size and
                               (len(block['lines']) < 5 or
                                block_font != main_text_font)]

                               #  较为宽松的规则
                               # sum([block_size < main_text_size,
                               #      len(block['lines']) < 5,
                               #      block_font != main_text_font])
                               # >= 2]


            return footnote_bboxes
        else:
            return []



