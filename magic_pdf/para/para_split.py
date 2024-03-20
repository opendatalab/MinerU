from sklearn.cluster import DBSCAN
import numpy as np
from loguru import logger

from magic_pdf.libs.boxbase import _is_in
from magic_pdf.libs.ocr_content_type import ContentType


LINE_STOP_FLAG = ['.', '!', '?', '。', '！', '？',"：", ":", ")", "）", ";"]
INLINE_EQUATION = ContentType.InlineEquation
INTERLINE_EQUATION = ContentType.InterlineEquation
TEXT = "text"


def __get_span_text(span):
    c = span.get('content', '')
    if len(c)==0:
        c = span.get('image_path', '')
        
    return c
    
    
def __add_line_period(blocks, layout_bboxes):
    """
    为每行添加句号
    如果这个行
    1. 以行内公式结尾，但没有任何标点符号,此时加个句号，认为他就是段落结尾。
    """
    for block in blocks:
        for line in block['lines']:
            last_span = line['spans'][-1]
            span_type = last_span['type']
            if span_type in [INLINE_EQUATION]:
                span_content = last_span['content'].strip()
                if span_type==INLINE_EQUATION and span_content[-1] not in LINE_STOP_FLAG:
                    if span_type in [INLINE_EQUATION, INTERLINE_EQUATION]:
                        last_span['content'] = span_content + '.'



def __valign_lines(blocks, layout_bboxes):
    """
    在一个layoutbox内对齐行的左侧和右侧。
    扫描行的左侧和右侧，如果x0, x1差距不超过一个阈值，就强行对齐到所处layout的左右两侧（和layout有一段距离）。
    3是个经验值，TODO，计算得来，可以设置为1.5个正文字符。
    """
    
    min_distance = 3
    min_sample = 2
    new_layout_bboxes = []
    
    for layout_box in layout_bboxes:
        blocks_in_layoutbox = [b for b in blocks if _is_in(b['bbox'], layout_box['layout_bbox'])]
        if len(blocks_in_layoutbox)==0:
            continue
        
        x0_lst = np.array([[line['bbox'][0], 0] for block in blocks_in_layoutbox for line in block['lines']])
        x1_lst = np.array([[line['bbox'][2], 0] for block in blocks_in_layoutbox for line in block['lines']])
        x0_clusters = DBSCAN(eps=min_distance, min_samples=min_sample).fit(x0_lst)
        x1_clusters = DBSCAN(eps=min_distance, min_samples=min_sample).fit(x1_lst)
        x0_uniq_label = np.unique(x0_clusters.labels_)
        x1_uniq_label = np.unique(x1_clusters.labels_)
        
        x0_2_new_val = {} # 存储旧值对应的新值映射
        x1_2_new_val = {}
        for label in x0_uniq_label:
            if label==-1:
                continue
            x0_index_of_label = np.where(x0_clusters.labels_==label)
            x0_raw_val = x0_lst[x0_index_of_label][:,0]
            x0_new_val = np.min(x0_lst[x0_index_of_label][:,0])
            x0_2_new_val.update({idx: x0_new_val for idx in x0_raw_val})
        for label in x1_uniq_label:
            if label==-1:
                continue
            x1_index_of_label = np.where(x1_clusters.labels_==label)
            x1_raw_val = x1_lst[x1_index_of_label][:,0]
            x1_new_val = np.max(x1_lst[x1_index_of_label][:,0])
            x1_2_new_val.update({idx: x1_new_val for idx in x1_raw_val})
        
        for block in blocks_in_layoutbox:
            for line in block['lines']:
                x0, x1 = line['bbox'][0], line['bbox'][2]
                if x0 in x0_2_new_val:
                    line['bbox'][0] = int(x0_2_new_val[x0])

                if x1 in x1_2_new_val:
                    line['bbox'][2] = int(x1_2_new_val[x1])
            # 其余对不齐的保持不动
            
        # 由于修改了block里的line长度，现在需要重新计算block的bbox
        for block in blocks_in_layoutbox:
            block['bbox'] = [min([line['bbox'][0] for line in block['lines']]), 
                            min([line['bbox'][1] for line in block['lines']]), 
                            max([line['bbox'][2] for line in block['lines']]), 
                            max([line['bbox'][3] for line in block['lines']])]
            
        """新计算layout的bbox，因为block的bbox变了。"""
        layout_x0 = min([block['bbox'][0] for block in blocks_in_layoutbox])
        layout_y0 = min([block['bbox'][1] for block in blocks_in_layoutbox])
        layout_x1 = max([block['bbox'][2] for block in blocks_in_layoutbox])
        layout_y1 = max([block['bbox'][3] for block in blocks_in_layoutbox])
        new_layout_bboxes.append([layout_x0, layout_y0, layout_x1, layout_y1])
            
    return new_layout_bboxes


def __common_pre_proc(blocks, layout_bboxes):
    """
    不分语言的，对文本进行预处理
    """
    #__add_line_period(blocks, layout_bboxes)
    aligned_layout_bboxes = __valign_lines(blocks, layout_bboxes)
    
    return aligned_layout_bboxes

def __pre_proc_zh_blocks(blocks, layout_bboxes):
    """
    对中文文本进行分段预处理
    """
    pass


def __pre_proc_en_blocks(blocks, layout_bboxes):
    """
    对英文文本进行分段预处理
    """
    pass


def __group_line_by_layout(blocks, layout_bboxes, lang="en"):
    """
    每个layout内的行进行聚合
    """
    # 因为只是一个block一行目前, 一个block就是一个段落
    lines_group = []
    
    for lyout in layout_bboxes:
        lines = [line for block in blocks if _is_in(block['bbox'], lyout['layout_bbox']) for line in block['lines']]
        lines_group.append(lines)

    return lines_group
    

def __split_para_in_layoutbox(lines_group, new_layout_bbox, lang="en", char_avg_len=10):
    """
    lines_group 进行行分段——layout内部进行分段。lines_group内每个元素是一个Layoutbox内的所有行。
    1. 先计算每个group的左右边界。
    2. 然后根据行末尾特征进行分段。
        末尾特征：以句号等结束符结尾。并且距离右侧边界有一定距离。
        且下一行开头不留空白。
    
    """
    paras = []
    right_tail_distance = 1.5 * char_avg_len
    for lines in lines_group:
        total_lines = len(lines)
        if total_lines<=1: # 0行无需处理。1行无法分段。
            continue
        #layout_right = max([line['bbox'][2] for line in lines])
        layout_right = __find_layout_bbox_by_line(lines[0]['bbox'], new_layout_bbox)[2]
        para = [] # 元素是line
        
        for i, line in enumerate(lines):
            # 如果i有下一行，那么就要根据下一行位置综合判断是否要分段。如果i之后没有行，那么只需要判断一下行结尾特征。
            
            cur_line_type = line['spans'][-1]['type']
            #cur_line_last_char = line['spans'][-1]['content'][-1]
            next_line = lines[i+1] if i<total_lines-1 else None
            
            if cur_line_type in [TEXT, INLINE_EQUATION]:
                if line['bbox'][2] < layout_right - right_tail_distance:
                    para.append(line)
                    paras.append(para)
                    para = []
                elif line['bbox'][2] >= layout_right - right_tail_distance and next_line and next_line['bbox'][0] == layout_right: # 现在这行到了行尾沾满，下一行存在且顶格。
                    para.append(line)
                else: 
                    para.append(line)
                    paras.append(para)
                    para = []
            else: # 其他，图片、表格、行间公式，各自占一段
                if len(para)>0:  # 先把之前的段落加入到结果中
                    paras.append(para)
                    para = []
                paras.append([line]) # 再把当前行加入到结果中。当前行为行间公式、图、表等。
                para = []
        if len(para)>0:
            paras.append(para)
            para = []
                    
    return paras


def __find_layout_bbox_by_line(line_bbox, layout_bboxes):
    """
    根据line找到所在的layout
    """
    for layout in layout_bboxes:
        if _is_in(line_bbox, layout):
            return layout
    return None


def __connect_para_inter_layoutbox(layout_paras, new_layout_bbox, lang="en"):
    """
    layout之间进行分段。
    主要是计算前一个layOut的最后一行和后一个layout的第一行是否可以连接。
    连接的条件需要同时满足：
    1. 上一个layout的最后一行沾满整个行。并且没有结尾符号。
    2. 下一行开头不留空白。

    """
    connected_layout_paras = []
    for i, para in enumerate(layout_paras):
        if i==0:
            connected_layout_paras.append(para)
            continue
        pre_last_line = layout_paras[i-1][-1]
        next_first_line = layout_paras[i][0]
        pre_last_line_text = ''.join([__get_span_text(span) for span in pre_last_line['spans']])
        pre_last_line_type = pre_last_line['spans'][-1]['type']
        next_first_line_text = ''.join([__get_span_text(span) for span in next_first_line['spans']])
        next_first_line_type = next_first_line['spans'][0]['type']
        if pre_last_line_type not in [TEXT, INLINE_EQUATION] or next_first_line_type not in [TEXT, INLINE_EQUATION]: # TODO，真的要做好，要考虑跨table, image, 行间的情况
            connected_layout_paras.append(para)
            continue
        
        
        pre_x2_max = __find_layout_bbox_by_line(pre_last_line['bbox'], new_layout_bbox)[2]
        next_x0_min = __find_layout_bbox_by_line(next_first_line['bbox'], new_layout_bbox)[0]
        
        pre_last_line_text = pre_last_line_text.strip()
        next_first_line_text = next_first_line_text.strip()
        if pre_last_line['bbox'][2] == pre_x2_max and pre_last_line_text[-1] not in LINE_STOP_FLAG and next_first_line['bbox'][0]==next_x0_min: # 前面一行沾满了整个行，并且没有结尾符号.下一行没有空白开头。
            """连接段落条件成立，将前一个layout的段落和后一个layout的段落连接。"""
            connected_layout_paras[-1].extend(para)
        else:
            """连接段落条件不成立，将前一个layout的段落加入到结果中。"""
            connected_layout_paras.append(para)
    
    return connected_layout_paras


def __connect_para_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, lang):
    """
    连接起来相邻两个页面的段落——前一个页面最后一个段落和后一个页面的第一个段落。
    是否可以连接的条件：
    1. 前一个页面的最后一个段落最后一行沾满整个行。并且没有结尾符号。
    2. 后一个页面的第一个段落第一行没有空白开头。
    """
    pre_last_para = pre_page_paras[-1]
    next_first_para = next_page_paras[0]
    pre_last_line = pre_last_para[-1]
    next_first_line = next_first_para[0]
    pre_last_line_text = ''.join([__get_span_text(span) for span in pre_last_line['spans']])
    pre_last_line_type = pre_last_line['spans'][-1]['type']
    next_first_line_text = ''.join([__get_span_text(span) for span in next_first_line['spans']])
    next_first_line_type = next_first_line['spans'][0]['type']
    
    if pre_last_line_type not in [TEXT, INLINE_EQUATION] or next_first_line_type not in [TEXT, INLINE_EQUATION]: # TODO，真的要做好，要考虑跨table, image, 行间的情况
        # 不是文本，不连接
        return False
    
    pre_x2_max = __find_layout_bbox_by_line(pre_last_line['bbox'], pre_page_layout_bbox)[2]
    next_x0_min = __find_layout_bbox_by_line(next_first_line['bbox'], next_page_layout_bbox)[0]
    
    pre_last_line_text = pre_last_line_text.strip()
    next_first_line_text = next_first_line_text.strip()
    if pre_last_line['bbox'][2] == pre_x2_max and pre_last_line_text[-1] not in LINE_STOP_FLAG and next_first_line['bbox'][0]==next_x0_min: # 前面一行沾满了整个行，并且没有结尾符号.下一行没有空白开头。
        """连接段落条件成立，将前一个layout的段落和后一个layout的段落连接。"""
        pre_page_paras[-1].extend(next_first_para)
        next_page_paras.pop(0) # 删除后一个页面的第一个段落， 因为他已经被合并到前一个页面的最后一个段落了。
        return True
    else:
        return False


def __do_split(blocks, layout_bboxes, new_layout_bbox, lang="en"):
    """
    根据line和layout情况进行分段
    先实现一个根据行末尾特征分段的简单方法。
    """
    """
    算法思路：
    1. 扫描layout里每一行，找出来行尾距离layout有边界有一定距离的行。
    2. 从上述行中找到末尾是句号等可作为断行标志的行。
    3. 参照上述行尾特征进行分段。
    4. 图、表，目前独占一行，不考虑分段。
    """
    lines_group = __group_line_by_layout(blocks, layout_bboxes, lang) # block内分段
    layout_paras = __split_para_in_layoutbox(lines_group, new_layout_bbox, lang) # layout内分段
    connected_layout_paras = __connect_para_inter_layoutbox(layout_paras, new_layout_bbox, lang) # layout间链接段落
    return connected_layout_paras
    
    
def para_split(pdf_info_dict, lang="en"):
    """
    根据line和layout情况进行分段
    """
    new_layout_of_pages = [] # 数组的数组，每个元素是一个页面的layoutS
    for _, page in pdf_info_dict.items():
        blocks = page['preproc_blocks']
        layout_bboxes = page['layout_bboxes']
        new_layout_bbox = __common_pre_proc(blocks, layout_bboxes)
        new_layout_of_pages.append(new_layout_bbox)
        splited_blocks = __do_split(blocks, layout_bboxes, new_layout_bbox, lang)
        page['para_blocks'] = splited_blocks
        
    """连接页面与页面之间的可能合并的段落"""
    pdf_infos = list(pdf_info_dict.values())
    for i, page in enumerate(pdf_info_dict.values()):
        if i==0:
            continue
        pre_page_paras = pdf_infos[i-1]['para_blocks']
        next_page_paras = pdf_infos[i]['para_blocks']
        pre_page_layout_bbox = new_layout_of_pages[i-1]
        next_page_layout_bbox = new_layout_of_pages[i]
        
        is_conn= __connect_para_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, lang) 
        if is_conn:
            logger.info(f"连接了第{i-1}页和第{i}页的段落")
