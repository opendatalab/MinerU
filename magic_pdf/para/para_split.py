from sklearn.cluster import DBSCAN
import numpy as np
from loguru import logger

from magic_pdf.libs.boxbase import _is_in_or_part_overlap_with_area_ratio as is_in_layout
from magic_pdf.libs.ocr_content_type import ContentType


LINE_STOP_FLAG = ['.', '!', '?', '。', '！', '？',"：", ":", ")", "）", ";"]
INLINE_EQUATION = ContentType.InlineEquation
INTERLINE_EQUATION = ContentType.InterlineEquation
TEXT = ContentType.Text


def __get_span_text(span):
    c = span.get('content', '')
    if len(c)==0:
        c = span.get('image_path', '')
        
    return c
    

def __detect_list_lines(lines, new_layout_bboxes, lang):
    """
    探测是否包含了列表，并且把列表的行分开.
    这样的段落特点是，顶格字母大写/数字，紧跟着几行缩进的。缩进的行首字母含小写的。
    """
    def find_repeating_patterns(lst):
        indices = []
        ones_indices = []
        i = 0
        while i < len(lst) - 1:  # 确保余下元素至少有2个
            if lst[i] == 1 and lst[i+1] in [2, 3]:  # 额外检查以防止连续出现的1
                start = i
                ones_in_this_interval = [i]
                i += 1
                while i < len(lst) and lst[i] in [2, 3]:
                    i += 1
                # 验证下一个序列是否符合条件
                if i < len(lst) - 1 and lst[i] == 1 and lst[i+1] in [2, 3] and lst[i-1] in [2, 3]:
                    while i < len(lst) and lst[i] in [1, 2, 3]:
                        if lst[i] == 1:
                            ones_in_this_interval.append(i)
                        i += 1
                    indices.append((start, i - 1))
                    ones_indices.append(ones_in_this_interval)
                else:
                    i += 1
            else:
                i += 1
        return indices, ones_indices
    """===================="""
    def split_indices(slen, index_array):
        result = []
        last_end = 0
        
        for start, end in sorted(index_array):
            if start > last_end:
                # 前一个区间结束到下一个区间开始之间的部分标记为"text"
                result.append(('text', last_end, start - 1))
            # 区间内标记为"list"
            result.append(('list', start, end))
            last_end = end + 1

        if last_end < slen:
            # 如果最后一个区间结束后还有剩余的字符串，将其标记为"text"
            result.append(('text', last_end, slen - 1))

        return result
    """===================="""

    if lang!='en':
        return lines, None
    else:
        total_lines = len(lines)
        line_fea_encode = []
        """
        对每一行进行特征编码，编码规则如下：
        1. 如果行顶格，且大写字母开头或者数字开头，编码为1
        2. 如果顶格，其他非大写开头编码为4
        3. 如果非顶格，首字符大写，编码为2
        4. 如果非顶格，首字符非大写编码为3
        """
        for l in lines:
            first_char = __get_span_text(l['spans'][0])[0]
            layout_left = __find_layout_bbox_by_line(l['bbox'], new_layout_bboxes)[0]
            if l['bbox'][0] == layout_left:
                if first_char.isupper() or first_char.isdigit():
                    line_fea_encode.append(1)
                else:
                    line_fea_encode.append(4)
            else:
                if first_char.isupper():
                    line_fea_encode.append(2)
                else:
                    line_fea_encode.append(3)
                    
        # 然后根据编码进行分段, 选出来 1,2,3连续出现至少2次的行，认为是列表。
        
        list_indice, list_start_idx  = find_repeating_patterns(line_fea_encode)
        if len(list_indice)>0:
            logger.info(f"发现了列表，列表行数：{list_indice}， {list_start_idx}")
        
        # TODO check一下这个特列表里缩进的行左侧是不是对齐的。
        segments = []
        for start, end in list_indice:
            for i in range(start, end+1):
                if i>0:
                    if line_fea_encode[i] == 4:
                        logger.info(f"列表行的第{i}行不是顶格的")
                        break
            else:
                logger.info(f"列表行的第{start}到第{end}行是列表")
        
        return split_indices(total_lines, list_indice), list_start_idx
        
            

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
        blocks_in_layoutbox = [b for b in blocks if is_in_layout(b['bbox'], layout_box['layout_bbox'])]
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


def __align_text_in_layout(blocks, layout_bboxes):
    """
    由于ocr出来的line，有时候会在前后有一段空白，这个时候需要对文本进行对齐，超出的部分被layout左右侧截断。
    """
    for layout in layout_bboxes:
        lb = layout['layout_bbox']
        blocks_in_layoutbox = [b for b in blocks if is_in_layout(b['bbox'], lb)]
        if len(blocks_in_layoutbox)==0:
            continue
        
        for block in blocks_in_layoutbox:
            for line in block['lines']:
                x0, x1 = line['bbox'][0], line['bbox'][2]
                if x0 < lb[0]:
                    line['bbox'][0] = lb[0]
                if x1 > lb[2]:
                    line['bbox'][2] = lb[2]
    
 
def __common_pre_proc(blocks, layout_bboxes):
    """
    不分语言的，对文本进行预处理
    """
    #__add_line_period(blocks, layout_bboxes)
    __align_text_in_layout(blocks, layout_bboxes)
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
        lines = [line for block in blocks if is_in_layout(block['bbox'], lyout['layout_bbox']) for line in block['lines']]
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
    list_info = [] # 这个layout最后是不是列表,记录每一个layout里是不是列表开头，列表结尾
    layout_paras = []
    right_tail_distance = 1.5 * char_avg_len
    
    for lines in lines_group:
        paras = []
        total_lines = len(lines)
        if total_lines==0:
            continue # 0行无需处理
        if total_lines==1: # 1行无法分段。
            layout_paras.append([lines])
            list_info.append([False, False])
            continue
        
        """在进入到真正的分段之前，要对文字块从统计维度进行对齐方式的探测，
            对齐方式分为以下：
            1. 左对齐的文本块(特点是左侧顶格，或者左侧不顶格但是右侧顶格的行数大于非顶格的行数，顶格的首字母有大写也有小写)
                1) 右侧对齐的行，单独成一段
                2) 中间对齐的行，按照字体/行高聚合成一段
            2. 左对齐的列表块（其特点是左侧顶格的行数小于等于非顶格的行数，非定格首字母会有小写，顶格90%是大写。并且左侧顶格行数大于1，大于1是为了这种模式连续出现才能称之为列表）
                这样的文本块，顶格的为一个段落开头，紧随其后非顶格的行属于这个段落。
        """
        
        text_segments, list_start_line = __detect_list_lines(lines, new_layout_bbox, lang)
        """根据list_range，把lines分成几个部分
        
        """
        
        layout_right = __find_layout_bbox_by_line(lines[0]['bbox'], new_layout_bbox)[2]
        layout_left = __find_layout_bbox_by_line(lines[0]['bbox'], new_layout_bbox)[0]
        para = [] # 元素是line
        layout_list_info = [False, False] # 这个layout最后是不是列表,记录每一个layout里是不是列表开头，列表结尾
        for content_type, start, end in text_segments:
            if content_type == 'list':
                for i, line in enumerate(lines[start:end+1]):
                    line_x0 = line['bbox'][0]
                    if line_x0 == layout_left: # 列表开头
                        if len(para)>0:
                            paras.append(para)
                            para = []
                        para.append(line)
                    else:
                        para.append(line)
                if len(para)>0:
                    paras.append(para)
                    para = []
                if start==0:
                    layout_list_info[0] = True
                if end==total_lines-1:
                    layout_list_info[1] = True
            else: # 是普通文本
                for i, line in enumerate(lines[start:end+1]):
                    # 如果i有下一行，那么就要根据下一行位置综合判断是否要分段。如果i之后没有行，那么只需要判断i行自己的结尾特征。
                    cur_line_type = line['spans'][-1]['type']
                    next_line = lines[i+1] if i<total_lines-1 else None
                    
                    if cur_line_type in [TEXT, INLINE_EQUATION]:
                        if line['bbox'][2] < layout_right - right_tail_distance:
                            para.append(line)
                            paras.append(para)
                            para = []
                        elif line['bbox'][2] >= layout_right - right_tail_distance and next_line and next_line['bbox'][0] == layout_left: # 现在这行到了行尾沾满，下一行存在且顶格。
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
                
        list_info.append(layout_list_info)
        layout_paras.append(paras)
        paras = []
                
                    
    return layout_paras, list_info

def __connect_list_inter_layout(layout_paras, new_layout_bbox, layout_list_info, page_num, lang):
    """
    如果上个layout的最后一个段落是列表，下一个layout的第一个段落也是列表，那么将他们连接起来。 TODO 因为没有区分列表和段落，所以这个方法暂时不实现。
    根据layout_list_info判断是不是列表。，下个layout的第一个段如果不是列表，那么看他们是否有几行都有相同的缩进。
    """
    if len(layout_paras)==0 or len(layout_list_info)==0: # 0的时候最后的return 会出错
        return layout_paras, [False, False]
        
    for i in range(1, len(layout_paras)):
        pre_layout_list_info = layout_list_info[i-1]
        next_layout_list_info = layout_list_info[i]
        pre_last_para = layout_paras[i-1][-1]
        next_paras = layout_paras[i]
        next_first_para = next_paras[0]
        
        if pre_layout_list_info[1] and not next_layout_list_info[0]: # 前一个是列表结尾，后一个是非列表开头，此时检测是否有相同的缩进
            logger.info(f"连接page {page_num} 内的list")
            # 向layout_paras[i] 寻找开头具有相同缩进的连续的行
            may_list_lines = []
            for j in range(len(next_paras)):
                line = next_paras[j]
                if len(line)==1: # 只可能是一行，多行情况再需要分析了
                    if line[0]['bbox'][0] > __find_layout_bbox_by_line(line[0]['bbox'], new_layout_bbox)[0]:
                        may_list_lines.append(line[0])
                    else:
                        break
                else:
                    break
            # 如果这些行的缩进是相等的，那么连到上一个layout的最后一个段落上。
            if len(may_list_lines)>0 and len(set([x['bbox'][0] for x in may_list_lines]))==1:
                pre_last_para.extend(may_list_lines)
                layout_paras[i] = layout_paras[i][len(may_list_lines):]
                           
    return layout_paras, [layout_list_info[0][0], layout_list_info[-1][1]] # 同时还返回了这个页面级别的开头、结尾是不是列表的信息


def __connect_list_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox,  pre_page_list_info, next_page_list_info, page_num, lang):
    """
    如果上个layout的最后一个段落是列表，下一个layout的第一个段落也是列表，那么将他们连接起来。 TODO 因为没有区分列表和段落，所以这个方法暂时不实现。
    根据layout_list_info判断是不是列表。，下个layout的第一个段如果不是列表，那么看他们是否有几行都有相同的缩进。
    """
    if len(pre_page_paras)==0 or len(next_page_paras)==0: # 0的时候最后的return 会出错
        return False
    
    if pre_page_list_info[1] and not next_page_list_info[0]: # 前一个是列表结尾，后一个是非列表开头，此时检测是否有相同的缩进
        logger.info(f"连接page {page_num} 内的list")
        # 向layout_paras[i] 寻找开头具有相同缩进的连续的行
        may_list_lines = []
        for j in range(len(next_page_paras[0])):
            line = next_page_paras[0][j]
            if len(line)==1: # 只可能是一行，多行情况再需要分析了
                if line[0]['bbox'][0] > __find_layout_bbox_by_line(line[0]['bbox'], next_page_layout_bbox)[0]:
                    may_list_lines.append(line[0])
                else:
                    break
            else:
                break
        # 如果这些行的缩进是相等的，那么连到上一个layout的最后一个段落上。
        if len(may_list_lines)>0 and len(set([x['bbox'][0] for x in may_list_lines]))==1:
            pre_page_paras[-1].append(may_list_lines)
            next_page_paras[0] = next_page_paras[0][len(may_list_lines):]
            return True
                       
    return False


def __find_layout_bbox_by_line(line_bbox, layout_bboxes):
    """
    根据line找到所在的layout
    """
    for layout in layout_bboxes:
        if is_in_layout(line_bbox, layout):
            return layout
    return None


def __connect_para_inter_layoutbox(layout_paras, new_layout_bbox, lang):
    """
    layout之间进行分段。
    主要是计算前一个layOut的最后一行和后一个layout的第一行是否可以连接。
    连接的条件需要同时满足：
    1. 上一个layout的最后一行沾满整个行。并且没有结尾符号。
    2. 下一行开头不留空白。

    """
    connected_layout_paras = []
    if len(layout_paras)==0:
        return connected_layout_paras
    
    connected_layout_paras.append(layout_paras[0])
    for i in range(1, len(layout_paras)):
        try:
            if len(layout_paras[i])==0 or len(layout_paras[i-1])==0: #  TODO 考虑连接问题，
                continue
            pre_last_line = layout_paras[i-1][-1][-1]
            next_first_line = layout_paras[i][0][0]
        except Exception as e:
            logger.error(f"page layout {i} has no line")
            continue
        pre_last_line_text = ''.join([__get_span_text(span) for span in pre_last_line['spans']])
        pre_last_line_type = pre_last_line['spans'][-1]['type']
        next_first_line_text = ''.join([__get_span_text(span) for span in next_first_line['spans']])
        next_first_line_type = next_first_line['spans'][0]['type']
        if pre_last_line_type not in [TEXT, INLINE_EQUATION] or next_first_line_type not in [TEXT, INLINE_EQUATION]:
            connected_layout_paras.append(layout_paras[i])
            continue
        
        pre_x2_max = __find_layout_bbox_by_line(pre_last_line['bbox'], new_layout_bbox)[2]
        next_x0_min = __find_layout_bbox_by_line(next_first_line['bbox'], new_layout_bbox)[0]
        
        pre_last_line_text = pre_last_line_text.strip()
        next_first_line_text = next_first_line_text.strip()
        if pre_last_line['bbox'][2] == pre_x2_max and pre_last_line_text[-1] not in LINE_STOP_FLAG and next_first_line['bbox'][0]==next_x0_min: # 前面一行沾满了整个行，并且没有结尾符号.下一行没有空白开头。
            """连接段落条件成立，将前一个layout的段落和后一个layout的段落连接。"""
            connected_layout_paras[-1][-1].extend(layout_paras[i][0])
            layout_paras[i].pop(0) # 删除后一个layout的第一个段落， 因为他已经被合并到前一个layout的最后一个段落了。
            if len(layout_paras[i])==0:
                layout_paras.pop(i)
            else:
                connected_layout_paras.append(layout_paras[i])
        else:                            
            """连接段落条件不成立，将前一个layout的段落加入到结果中。"""
            connected_layout_paras.append(layout_paras[i])
    
    return connected_layout_paras


def __connect_para_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, page_num, lang):
    """
    连接起来相邻两个页面的段落——前一个页面最后一个段落和后一个页面的第一个段落。
    是否可以连接的条件：
    1. 前一个页面的最后一个段落最后一行沾满整个行。并且没有结尾符号。
    2. 后一个页面的第一个段落第一行没有空白开头。
    """
    # 有的页面可能压根没有文字
    if len(pre_page_paras)==0 or len(next_page_paras)==0 or len(pre_page_paras[0])==0 or len(next_page_paras[0])==0: # TODO [[]]为什么出现在pre_page_paras里？
        return False
    pre_last_para = pre_page_paras[-1][-1]
    next_first_para = next_page_paras[0][0]
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
        pre_last_para.extend(next_first_para)
        next_page_paras[0].pop(0) # 删除后一个页面的第一个段落， 因为他已经被合并到前一个页面的最后一个段落了。
        return True
    else:
        return False

def find_consecutive_true_regions(input_array):
    start_index = None  # 连续True区域的起始索引
    regions = []  # 用于保存所有连续True区域的起始和结束索引

    for i in range(len(input_array)):
        # 如果我们找到了一个True值，并且当前并没有在连续True区域中
        if input_array[i] and start_index is None:
            start_index = i  # 记录连续True区域的起始索引

        # 如果我们找到了一个False值，并且当前在连续True区域中
        elif not input_array[i] and start_index is not None:
            # 如果连续True区域长度大于1，那么将其添加到结果列表中
            if i - start_index > 1: 
                regions.append((start_index, i-1)) 
            start_index = None  # 重置起始索引

    # 如果最后一个元素是True，那么需要将最后一个连续True区域加入到结果列表中
    if start_index is not None and len(input_array) - start_index > 1:
        regions.append((start_index, len(input_array)-1))

    return regions


def __connect_middle_align_text(page_paras, new_layout_bbox, page_num, lang, debug_mode):
    """
    找出来中间对齐的连续单行文本，如果连续行高度相同，那么合并为一个段落。
    一个line居中的条件是：
    1. 水平中心点跨越layout的中心点。
    2. 左右两侧都有空白
    """
    
    for layout_i, layout_para in enumerate(page_paras):
        layout_box = new_layout_bbox[layout_i]
        single_line_paras_tag = []
        for i in range(len(layout_para)):
            single_line_paras_tag.append(len(layout_para[i])==1 and layout_para[i][0]['spans'][0]['type']==TEXT)
            
        """找出来连续的单行文本，如果连续行高度相同，那么合并为一个段落。"""
        consecutive_single_line_indices = find_consecutive_true_regions(single_line_paras_tag)
        if len(consecutive_single_line_indices)>0:
            index_offset = 0
            """检查这些行是否是高度相同的，居中的"""
            for start, end in consecutive_single_line_indices:
                start += index_offset
                end += index_offset
                line_hi = np.array([line[0]['bbox'][3]-line[0]['bbox'][1] for line in layout_para[start:end+1]])
                first_line_text = ''.join([__get_span_text(span) for span in layout_para[start][0]['spans']])
                if "Table" in first_line_text or "Figure" in first_line_text:
                    pass
                if debug_mode:
                    logger.debug(line_hi.std())
                
                if line_hi.std()<2:
                    """行高度相同，那么判断是否居中"""
                    all_left_x0 = [line[0]['bbox'][0] for line in layout_para[start:end+1]]
                    all_right_x1 = [line[0]['bbox'][2] for line in layout_para[start:end+1]]
                    layout_center = (layout_box[0] + layout_box[2]) / 2
                    if all([x0 < layout_center < x1 for x0, x1 in zip(all_left_x0, all_right_x1)]) \
                    and not all([x0==layout_box[0] for x0 in all_left_x0]) \
                    and not all([x1==layout_box[2] for x1 in all_right_x1]):
                        merge_para = [l[0] for l in layout_para[start:end+1]]
                        para_text = ''.join([__get_span_text(span) for line in merge_para for span in line['spans']])
                        if debug_mode:
                            logger.debug(para_text)
                        layout_para[start:end+1] = [merge_para]
                        index_offset -= end-start
                        
    return
            

def __merge_signle_list_text(page_paras, new_layout_bbox, page_num, lang):
    """
    找出来连续的单行文本，如果首行顶格，接下来的几个单行段落缩进对齐，那么合并为一个段落。
    """
    
    pass


def __do_split_page(blocks, layout_bboxes, new_layout_bbox, page_num, lang):
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
    if page_num==343:
        pass
    lines_group = __group_line_by_layout(blocks, layout_bboxes, lang) # block内分段
    layout_paras, layout_list_info = __split_para_in_layoutbox(lines_group, new_layout_bbox, lang) # layout内分段
    layout_paras2, page_list_info = __connect_list_inter_layout(layout_paras, new_layout_bbox, layout_list_info, page_num, lang) # layout之间连接列表段落
    connected_layout_paras = __connect_para_inter_layoutbox(layout_paras2, new_layout_bbox, lang) # layout间链接段落
    
    
    return connected_layout_paras, page_list_info
       
    
def para_split(pdf_info_dict, debug_mode, lang="en"):
    """
    根据line和layout情况进行分段
    """
    new_layout_of_pages = [] # 数组的数组，每个元素是一个页面的layoutS
    all_page_list_info = [] # 保存每个页面开头和结尾是否是列表
    for page_num, page in pdf_info_dict.items():
        blocks = page['preproc_blocks']
        layout_bboxes = page['layout_bboxes']
        new_layout_bbox = __common_pre_proc(blocks, layout_bboxes)
        new_layout_of_pages.append(new_layout_bbox)
        splited_blocks, page_list_info = __do_split_page(blocks, layout_bboxes, new_layout_bbox, page_num, lang)
        all_page_list_info.append(page_list_info)
        page['para_blocks'] = splited_blocks
        
    """连接页面与页面之间的可能合并的段落"""
    pdf_infos = list(pdf_info_dict.values())
    for page_num, page in enumerate(pdf_info_dict.values()):
        if page_num==0:
            continue
        pre_page_paras = pdf_infos[page_num-1]['para_blocks']
        next_page_paras = pdf_infos[page_num]['para_blocks']
        pre_page_layout_bbox = new_layout_of_pages[page_num-1]
        next_page_layout_bbox = new_layout_of_pages[page_num]
        
        is_conn = __connect_para_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, page_num, lang)
        if debug_mode:
            if is_conn:
                logger.info(f"连接了第{page_num-1}页和第{page_num}页的段落")
            
        is_list_conn = __connect_list_inter_page(pre_page_paras, next_page_paras, pre_page_layout_bbox, next_page_layout_bbox, all_page_list_info[page_num-1], all_page_list_info[page_num], page_num, lang)
        if debug_mode:
            if is_list_conn:
                logger.info(f"连接了第{page_num-1}页和第{page_num}页的列表段落")
            
    """接下来可能会漏掉一些特别的一些可以合并的内容，对他们进行段落连接
    1. 正文中有时出现一个行顶格，接下来几行缩进的情况。
    2. 居中的一些连续单行，如果高度相同，那么可能是一个段落。
    """
    for page_num, page in enumerate(pdf_info_dict.values()):
        page_paras = page['para_blocks']
        new_layout_bbox = new_layout_of_pages[page_num]
        __connect_middle_align_text(page_paras, new_layout_bbox, page_num, lang, debug_mode=debug_mode)
        __merge_signle_list_text(page_paras, new_layout_bbox, page_num, lang)
