"""
对pdf上的box进行layout识别，并对内部组成的box进行排序
"""

from loguru import logger
from magic_pdf.layout.bbox_sort import CONTENT_IDX, CONTENT_TYPE_IDX, X0_EXT_IDX, X0_IDX, X1_EXT_IDX, X1_IDX, Y0_EXT_IDX, Y0_IDX, Y1_EXT_IDX, Y1_IDX, paper_bbox_sort
from magic_pdf.layout.layout_det_utils import find_all_left_bbox_direct, find_all_right_bbox_direct, find_bottom_bbox_direct_from_left_edge, find_bottom_bbox_direct_from_right_edge, find_top_bbox_direct_from_left_edge, find_top_bbox_direct_from_right_edge, find_all_top_bbox_direct, find_all_bottom_bbox_direct, get_left_edge_bboxes, get_right_edge_bboxes
from magic_pdf.libs.boxbase import get_bbox_in_boundry


LAYOUT_V = "V"
LAYOUT_H = "H"
LAYOUT_UNPROC = "U"
LAYOUT_BAD = "B"

def _is_single_line_text(bbox):
    """
    检查bbox里面的文字是否只有一行
    """
    return True # TODO 
    box_type = bbox[CONTENT_TYPE_IDX]
    if box_type != 'text':
        return False
    paras = bbox[CONTENT_IDX]["paras"]
    text_content = ""
    for para_id, para in paras.items():  # 拼装内部的段落文本
        is_title = para['is_title']
        if is_title!=0:
            text_content += f"## {para['text']}"
        else:
            text_content += para["text"]
        text_content += "\n\n"
                
    return bbox[CONTENT_TYPE_IDX] == 'text' and len(text_content.split("\n\n")) <= 1


def _horizontal_split(bboxes:list, boundry:tuple, avg_font_size=20)-> list:
    """
    对bboxes进行水平切割
    方法是：找到左侧和右侧都没有被直接遮挡的box，然后进行扩展，之后进行切割
    return:
        返回几个大的Layout区域 [[x0, y0, x1, y1, "h|u|v"], ], h代表水平，u代表未探测的，v代表垂直布局
    """
    sorted_layout_blocks = [] # 这是要最终返回的值
    
    bound_x0, bound_y0, bound_x1, bound_y1 = boundry
    all_bboxes = get_bbox_in_boundry(bboxes, boundry)
    #all_bboxes = paper_bbox_sort(all_bboxes, abs(bound_x1-bound_x0), abs(bound_y1-bound_x0)) # 大致拍下序, 这个是基于直接遮挡的。
    """
    首先在水平方向上扩展独占一行的bbox
    
    """
    last_h_split_line_y1 = bound_y0 #记录下上次的水平分割线
    for i, bbox in enumerate(all_bboxes):
        left_nearest_bbox = find_all_left_bbox_direct(bbox, all_bboxes) # 非扩展线
        right_nearest_bbox = find_all_right_bbox_direct(bbox, all_bboxes)
        if left_nearest_bbox is None and right_nearest_bbox is None: # 独占一行
            """
            然而，如果只是孤立的一行文字，那么就还要满足以下几个条件才可以：
            1. bbox和中心线相交。或者
            2. 上方或者下方也存在同类水平的独占一行的bbox。 或者
            3. TODO 加强条件：这个bbox上方和下方是同一列column，那么就不能算作独占一行
            """
            # 先检查这个bbox里是否只包含一行文字
            is_single_line =  _is_single_line_text(bbox)
            """
            这里有个点需要注意，当页面内容不是居中的时候，第一次调用传递的是page的boundry，这个时候mid_x就不是中心线了.
            所以这里计算出最紧致的boundry，然后再计算mid_x
            """
            boundry_real_x0, boundry_real_x1 = min([bbox[X0_IDX] for bbox in all_bboxes]), max([bbox[X1_IDX] for bbox in all_bboxes])
            mid_x = (boundry_real_x0+boundry_real_x1)/2  
            # 检查这个box是否内容在中心线有交
            # 必须跨过去2个字符的宽度
            is_cross_boundry_mid_line = min(mid_x-bbox[X0_IDX], bbox[X1_IDX]-mid_x) > avg_font_size*2
            """
            检查条件2
            """
            is_belong_to_col = False
            """
            检查是否能被上方col吸收，方法是：
            1. 上方非空且不是独占一行的，并且
            2. 从上个水平分割的最大y=y1开始到当前bbox,最左侧的bbox的[min_x0, max_x1],能够覆盖当前box的[x0, x1]
            """
            """
            以迭代的方式向上找，查找范围是[bound_x0, last_h_sp, bound_x1, bbox[Y0_IDX]]
            """
            #先确定上方的y0, y0
            b_y0, b_y1 = last_h_split_line_y1, bbox[Y0_IDX]
            #然后从box开始逐个向上找到所有与box在x上有交集的box
            box_to_check = [bound_x0, b_y0, bound_x1, b_y1]
            bbox_in_bound_check = get_bbox_in_boundry(all_bboxes, box_to_check)
            
            bboxes_on_top = []
            virtual_box = bbox
            while True:
                b_on_top = find_all_top_bbox_direct(virtual_box, bbox_in_bound_check)
                if b_on_top is not None:
                    bboxes_on_top.append(b_on_top)
                    virtual_box = [min([virtual_box[X0_IDX], b_on_top[X0_IDX]]), min(virtual_box[Y0_IDX], b_on_top[Y0_IDX]), max([virtual_box[X1_IDX], b_on_top[X1_IDX]]), b_y1]
                else:
                    break

            # 随后确定这些box的最小x0, 最大x1
            if len(bboxes_on_top)>0 and len(bboxes_on_top) != len(bbox_in_bound_check):# virtual_box可能会膨胀到占满整个区域，这实际上就不能属于一个col了。
                min_x0, max_x1 = virtual_box[X0_IDX], virtual_box[X1_IDX]
                # 然后采用一种比较粗糙的方法，看min_x0，max_x1是否与位于[bound_x0, last_h_sp, bound_x1, bbox[Y0_IDX]]之间的box有相交
                
                if not any([b[X0_IDX] <= min_x0-1 <= b[X1_IDX] or b[X0_IDX] <= max_x1+1 <= b[X1_IDX] for b in bbox_in_bound_check]):
                    # 其上，下都不能被扩展成行，暂时只检查一下上方 TODO
                    top_nearest_bbox = find_all_top_bbox_direct(bbox, bboxes)
                    bottom_nearest_bbox = find_all_bottom_bbox_direct(bbox, bboxes)
                    if not any([
                        top_nearest_bbox is not None and (find_all_left_bbox_direct(top_nearest_bbox, bboxes) is  None and  find_all_right_bbox_direct(top_nearest_bbox, bboxes) is None),
                        bottom_nearest_bbox is not None and (find_all_left_bbox_direct(bottom_nearest_bbox, bboxes) is  None and  find_all_right_bbox_direct(bottom_nearest_bbox, bboxes) is None),
                        top_nearest_bbox is None or bottom_nearest_bbox is None
                        ]):
                            is_belong_to_col = True
                
            # 检查是否能被下方col吸收 TODO
            
            """
            这里为什么没有is_cross_boundry_mid_line的条件呢？
            确实有些杂志左右两栏宽度不是对称的。
            """
            if not is_belong_to_col or is_cross_boundry_mid_line:
                bbox[X0_EXT_IDX] = bound_x0
                bbox[Y0_EXT_IDX] = bbox[Y0_IDX]
                bbox[X1_EXT_IDX] = bound_x1
                bbox[Y1_EXT_IDX] = bbox[Y1_IDX]
                last_h_split_line_y1 = bbox[Y1_IDX] # 更新这条线
            else:
                continue
    """
    此时独占一行的被成功扩展到指定的边界上，这个时候利用边界条件合并连续的bbox，成为一个group
    然后合并所有连续水平方向的bbox.
    """
    all_bboxes.sort(key=lambda x: x[Y0_IDX])
    h_bboxes = []
    h_bbox_group = []

    for bbox in all_bboxes:
        if bbox[X0_EXT_IDX] == bound_x0 and bbox[X1_EXT_IDX] == bound_x1:
            h_bbox_group.append(bbox)
        else:
            if len(h_bbox_group)>0:
                h_bboxes.append(h_bbox_group) 
                h_bbox_group = []
    # 最后一个group
    if len(h_bbox_group)>0:
        h_bboxes.append(h_bbox_group)

    """
    现在h_bboxes里面是所有的group了，每个group都是一个list
    对h_bboxes里的每个group进行计算放回到sorted_layouts里
    """
    h_layouts = []
    for gp in h_bboxes:
        gp.sort(key=lambda x: x[Y0_IDX])
        # 然后计算这个group的layout_bbox，也就是最小的x0,y0, 最大的x1,y1
        x0, y0, x1, y1 = gp[0][X0_EXT_IDX], gp[0][Y0_EXT_IDX], gp[-1][X1_EXT_IDX], gp[-1][Y1_EXT_IDX]
        h_layouts.append([x0, y0, x1, y1, LAYOUT_H]) # 水平的布局
        
    """
    接下来利用这些连续的水平bbox的layout_bbox的y0, y1，从水平上切分开其余的为几个部分
    """
    h_split_lines = [bound_y0]
    for gp in h_bboxes: # gp是一个list[bbox_list]
        y0, y1 = gp[0][1], gp[-1][3]
        h_split_lines.append(y0)
        h_split_lines.append(y1)
    h_split_lines.append(bound_y1)
    
    unsplited_bboxes = []
    for i in range(0, len(h_split_lines), 2):
        start_y0, start_y1 = h_split_lines[i:i+2]
        # 然后找出[start_y0, start_y1]之间的其他bbox，这些组成一个未分割板块
        bboxes_in_block = [bbox for bbox in all_bboxes if bbox[Y0_IDX]>=start_y0 and bbox[Y1_IDX]<=start_y1]
        unsplited_bboxes.append(bboxes_in_block)
    # 接着把未处理的加入到h_layouts里
    for bboxes_in_block in unsplited_bboxes:
        if len(bboxes_in_block) == 0:
            continue
        x0, y0, x1, y1 = bound_x0, min([bbox[Y0_IDX] for bbox in bboxes_in_block]), bound_x1, max([bbox[Y1_IDX] for bbox in bboxes_in_block])
        h_layouts.append([x0, y0, x1, y1, LAYOUT_UNPROC])
        
    h_layouts.sort(key=lambda x: x[1]) # 按照y0排序, 也就是从上到下的顺序
    
    """
    转换成如下格式返回
    """
    for layout in h_layouts:
        sorted_layout_blocks.append({
            "layout_bbox": layout[:4],
            "layout_label":layout[4],
            "sub_layout":[],
        })
    return sorted_layout_blocks
   
###############################################################################################
#
#  垂直方向的处理
#
#
############################################################################################### 
def _vertical_align_split_v1(bboxes:list, boundry:tuple)-> list:
    """
    计算垂直方向上的对齐， 并分割bboxes成layout。负责对一列多行的进行列维度分割。
    如果不能完全分割，剩余部分作为layout_lable为u的layout返回
    -----------------------
    |     |           |
    |     |           |
    |     |           |
    |     |           |
    -------------------------
    此函数会将：以上布局将会切分出来2列
    """
    sorted_layout_blocks = [] # 这是要最终返回的值
    new_boundry = [boundry[0], boundry[1], boundry[2], boundry[3]]
    
    v_blocks = []
    """
    先从左到右切分
    """
    while True: 
        all_bboxes = get_bbox_in_boundry(bboxes, new_boundry)
        left_edge_bboxes = get_left_edge_bboxes(all_bboxes)
        if len(left_edge_bboxes) == 0:
            break
        right_split_line_x1 = max([bbox[X1_IDX] for bbox in left_edge_bboxes])+1
        # 然后检查这条线能不与其他bbox的左边界相交或者重合
        if any([bbox[X0_IDX] <= right_split_line_x1 <= bbox[X1_IDX] for bbox in all_bboxes]):
            # 垂直切分线与某些box发生相交，说明无法完全垂直方向切分。
            break
        else: # 说明成功分割出一列
            # 找到左侧边界最靠左的bbox作为layout的x0
            layout_x0 = min([bbox[X0_IDX] for bbox in left_edge_bboxes]) # 这里主要是为了画出来有一定间距
            v_blocks.append([layout_x0, new_boundry[1], right_split_line_x1, new_boundry[3], LAYOUT_V])
            new_boundry[0] = right_split_line_x1 # 更新边界
            
    """
    再从右到左切， 此时如果还是无法完全切分，那么剩余部分作为layout_lable为u的layout返回
    """
    unsplited_block = []
    while True:
        all_bboxes = get_bbox_in_boundry(bboxes, new_boundry)
        right_edge_bboxes = get_right_edge_bboxes(all_bboxes)
        if len(right_edge_bboxes) == 0:
            break
        left_split_line_x0 = min([bbox[X0_IDX] for bbox in right_edge_bboxes])-1
        # 然后检查这条线能不与其他bbox的左边界相交或者重合
        if any([bbox[X0_IDX] <= left_split_line_x0 <= bbox[X1_IDX] for bbox in all_bboxes]):
            # 这里是余下的
            unsplited_block.append([new_boundry[0], new_boundry[1], new_boundry[2], new_boundry[3], LAYOUT_UNPROC])
            break
        else:
            # 找到右侧边界最靠右的bbox作为layout的x1
            layout_x1 = max([bbox[X1_IDX] for bbox in right_edge_bboxes])
            v_blocks.append([left_split_line_x0, new_boundry[1], layout_x1, new_boundry[3], LAYOUT_V])
            new_boundry[2] = left_split_line_x0 # 更新右边界
            
    """
    最后拼装成layout格式返回
    """
    for block in v_blocks:
        sorted_layout_blocks.append({
            "layout_bbox": block[:4],
            "layout_label":block[4],
            "sub_layout":[],
        })
    for block in unsplited_block:
        sorted_layout_blocks.append({
            "layout_bbox": block[:4],
            "layout_label":block[4],
            "sub_layout":[],
        })
    
    # 按照x0排序
    sorted_layout_blocks.sort(key=lambda x: x['layout_bbox'][0])
    return sorted_layout_blocks
            
def _vertical_align_split_v2(bboxes:list, boundry:tuple)-> list:
    """
    改进的 _vertical_align_split算法，原算法会因为第二列的box由于左侧没有遮挡被认为是左侧的一部分，导致整个layout多列被识别为一列。
    利用从左上角的box开始向下看的方法，不断扩展w_x0, w_x1，直到不能继续向下扩展，或者到达边界下边界。
    """
    sorted_layout_blocks = [] # 这是要最终返回的值
    new_boundry = [boundry[0], boundry[1], boundry[2], boundry[3]]
    bad_boxes = [] # 被割中的box
    v_blocks = []
    while True:
        all_bboxes = get_bbox_in_boundry(bboxes, new_boundry)
        if len(all_bboxes) == 0:
            break
        left_top_box = min(all_bboxes, key=lambda x: (x[X0_IDX],x[Y0_IDX]))# 这里应该加强，检查一下必须是在第一列的 TODO
        start_box = [left_top_box[X0_IDX], left_top_box[Y0_IDX], left_top_box[X1_IDX], left_top_box[Y1_IDX]]
        w_x0, w_x1 = left_top_box[X0_IDX], left_top_box[X1_IDX]
        """
        然后沿着这个box线向下找最近的那个box, 然后扩展w_x0, w_x1
        扩展之后，宽度会增加，随后用x=w_x1来检测在边界内是否有box与相交，如果相交，那么就说明不能再扩展了。
        当不能扩展的时候就要看是否到达下边界：
        1. 达到，那么更新左边界继续分下一个列
        2. 没有达到，那么此时开始从右侧切分进入下面的循环里
        """
        while left_top_box is not None: # 向下去找
            virtual_box = [w_x0, left_top_box[Y0_IDX], w_x1, left_top_box[Y1_IDX]]
            left_top_box = find_bottom_bbox_direct_from_left_edge(virtual_box, all_bboxes)
            if left_top_box:
                w_x0, w_x1 = min(virtual_box[X0_IDX], left_top_box[X0_IDX]), max([virtual_box[X1_IDX], left_top_box[X1_IDX]])
        # 万一这个初始的box在column中间，那么还要向上看
        start_box = [w_x0, start_box[Y0_IDX], w_x1, start_box[Y1_IDX]] # 扩展一下宽度更鲁棒
        left_top_box = find_top_bbox_direct_from_left_edge(start_box, all_bboxes)
        while left_top_box is not None: # 向上去找
            virtual_box = [w_x0, left_top_box[Y0_IDX], w_x1, left_top_box[Y1_IDX]]
            left_top_box = find_top_bbox_direct_from_left_edge(virtual_box, all_bboxes)
            if left_top_box:
                w_x0, w_x1 = min(virtual_box[X0_IDX], left_top_box[X0_IDX]), max([virtual_box[X1_IDX], left_top_box[X1_IDX]])
        
        # 检查相交  
        if any([bbox[X0_IDX] <= w_x1+1 <= bbox[X1_IDX] for bbox in all_bboxes]):
            for b in all_bboxes:
                if b[X0_IDX] <= w_x1+1 <= b[X1_IDX]:
                    bad_boxes.append([b[X0_IDX], b[Y0_IDX], b[X1_IDX], b[Y1_IDX]])
            break
        else: # 说明成功分割出一列
            v_blocks.append([w_x0, new_boundry[1], w_x1, new_boundry[3], LAYOUT_V])
            new_boundry[0] = w_x1 # 更新边界
    
    """
    接着开始从右上角的box扫描
    """
    w_x0 , w_x1 = 0, 0
    unsplited_block = []
    while True:
        all_bboxes = get_bbox_in_boundry(bboxes, new_boundry)
        if len(all_bboxes) == 0:
            break
        # 先找到X1最大的
        bbox_list_sorted = sorted(all_bboxes, key=lambda bbox: bbox[X1_IDX], reverse=True)
        # Then, find the boxes with the smallest Y0 value
        bigest_x1 = bbox_list_sorted[0][X1_IDX]
        boxes_with_bigest_x1 = [bbox for bbox in bbox_list_sorted if bbox[X1_IDX] == bigest_x1] # 也就是最靠右的那些
        right_top_box = min(boxes_with_bigest_x1, key=lambda bbox: bbox[Y0_IDX]) # y0最小的那个
        start_box = [right_top_box[X0_IDX], right_top_box[Y0_IDX], right_top_box[X1_IDX], right_top_box[Y1_IDX]]
        w_x0, w_x1 = right_top_box[X0_IDX], right_top_box[X1_IDX]
        
        while right_top_box is not None:
            virtual_box = [w_x0, right_top_box[Y0_IDX], w_x1, right_top_box[Y1_IDX]]
            right_top_box = find_bottom_bbox_direct_from_right_edge(virtual_box, all_bboxes)
            if right_top_box:
                w_x0, w_x1 = min([w_x0, right_top_box[X0_IDX]]), max([w_x1, right_top_box[X1_IDX]])
        # 在向上扫描
        start_box = [w_x0, start_box[Y0_IDX], w_x1, start_box[Y1_IDX]] # 扩展一下宽度更鲁棒
        right_top_box = find_top_bbox_direct_from_right_edge(start_box, all_bboxes)
        while right_top_box is not None:
            virtual_box = [w_x0, right_top_box[Y0_IDX], w_x1, right_top_box[Y1_IDX]]
            right_top_box = find_top_bbox_direct_from_right_edge(virtual_box, all_bboxes)
            if right_top_box:
                w_x0, w_x1 = min([w_x0, right_top_box[X0_IDX]]), max([w_x1, right_top_box[X1_IDX]])
                
        # 检查是否与其他box相交， 垂直切分线与某些box发生相交，说明无法完全垂直方向切分。
        if any([bbox[X0_IDX] <= w_x0-1 <= bbox[X1_IDX] for bbox in all_bboxes]):
            unsplited_block.append([new_boundry[0], new_boundry[1], new_boundry[2], new_boundry[3], LAYOUT_UNPROC])
            for b in all_bboxes:
                if b[X0_IDX] <= w_x0-1 <= b[X1_IDX]:
                    bad_boxes.append([b[X0_IDX], b[Y0_IDX], b[X1_IDX], b[Y1_IDX]])
            break
        else: # 说明成功分割出一列
            v_blocks.append([w_x0, new_boundry[1], w_x1, new_boundry[3], LAYOUT_V])
            new_boundry[2] = w_x0
    
    """转换数据结构"""
    for block in v_blocks:
        sorted_layout_blocks.append({
            "layout_bbox": block[:4],
            "layout_label":block[4],
            "sub_layout":[],
        })
        
    for block in unsplited_block:
        sorted_layout_blocks.append({
            "layout_bbox": block[:4],
            "layout_label":block[4],
            "sub_layout":[],
            "bad_boxes": bad_boxes # 记录下来，这个box是被割中的
        })
        
        
    # 按照x0排序
    sorted_layout_blocks.sort(key=lambda x: x['layout_bbox'][0])
    return sorted_layout_blocks
                
    


def _try_horizontal_mult_column_split(bboxes:list, boundry:tuple)-> list:
    """
    尝试水平切分，如果切分不动，那就当一个BAD_LAYOUT返回
    ------------------
    |        |       |
    ------------------
    |    |       |   |   <-  这里是此函数要切分的场景
    ------------------
    |        |       |
    |        |       |
    """
    pass




def _vertical_split(bboxes:list, boundry:tuple)-> list:
    """
    从垂直方向进行切割，分block
    这个版本里，如果垂直切分不动，那就当一个BAD_LAYOUT返回
    
                                --------------------------
                                    |        |       |
                                    |        |       |
                                | |
    这种列是此函数要切分的  ->    | |    
                                | |
                                    |        |       |
                                    |        |       |
                                -------------------------
    """
    sorted_layout_blocks = [] # 这是要最终返回的值
    
    bound_x0, bound_y0, bound_x1, bound_y1 = boundry
    all_bboxes = get_bbox_in_boundry(bboxes, boundry)
    """
    all_bboxes = fix_vertical_bbox_pos(all_bboxes) # 垂直方向解覆盖
    all_bboxes = fix_hor_bbox_pos(all_bboxes)  # 水平解覆盖
    
    这两行代码目前先不执行，因为公式检测，表格检测还不是很成熟，导致非常多的textblock参与了运算，时间消耗太大。
    这两行代码的作用是：
    如果遇到互相重叠的bbox, 那么会把面积较小的box进行压缩，从而避免重叠。对布局切分来说带来正反馈。
    """
    
    #all_bboxes = paper_bbox_sort(all_bboxes, abs(bound_x1-bound_x0), abs(bound_y1-bound_x0)) # 大致拍下序, 这个是基于直接遮挡的。
    """
    首先在垂直方向上扩展独占一行的bbox
    
    """
    for bbox in all_bboxes:
        top_nearest_bbox = find_all_top_bbox_direct(bbox, all_bboxes) # 非扩展线
        bottom_nearest_bbox = find_all_bottom_bbox_direct(bbox, all_bboxes)
        if top_nearest_bbox is None and bottom_nearest_bbox is None  and not any([b[X0_IDX]<bbox[X1_IDX]<b[X1_IDX] or b[X0_IDX]<bbox[X0_IDX]<b[X1_IDX] for b in all_bboxes]): # 独占一列, 且不和其他重叠
            bbox[X0_EXT_IDX] = bbox[X0_IDX]
            bbox[Y0_EXT_IDX] = bound_y0
            bbox[X1_EXT_IDX] = bbox[X1_IDX]
            bbox[Y1_EXT_IDX] = bound_y1
            
    """
    此时独占一列的被成功扩展到指定的边界上，这个时候利用边界条件合并连续的bbox，成为一个group
    然后合并所有连续垂直方向的bbox.
    """
    all_bboxes.sort(key=lambda x: x[X0_IDX])
    # fix: 这里水平方向的列不要合并成一个行，因为需要保证返回给下游的最小block，总是可以无脑从上到下阅读文字。
    v_bboxes = []
    for box in all_bboxes:
        if box[Y0_EXT_IDX] == bound_y0 and box[Y1_EXT_IDX] == bound_y1: 
            v_bboxes.append(box)
    
    """
    现在v_bboxes里面是所有的group了，每个group都是一个list
    对v_bboxes里的每个group进行计算放回到sorted_layouts里
    """
    v_layouts = []
    for vbox in v_bboxes:
        #gp.sort(key=lambda x: x[X0_IDX])
        # 然后计算这个group的layout_bbox，也就是最小的x0,y0, 最大的x1,y1
        x0, y0, x1, y1 = vbox[X0_EXT_IDX], vbox[Y0_EXT_IDX], vbox[X1_EXT_IDX], vbox[Y1_EXT_IDX]
        v_layouts.append([x0, y0, x1, y1, LAYOUT_V]) # 垂直的布局
        
    """
    接下来利用这些连续的垂直bbox的layout_bbox的x0, x1，从垂直上切分开其余的为几个部分
    """
    v_split_lines = [bound_x0]
    for gp in v_bboxes:
        x0, x1 = gp[X0_IDX], gp[X1_IDX]
        v_split_lines.append(x0)
        v_split_lines.append(x1)
    v_split_lines.append(bound_x1)
    
    unsplited_bboxes = []
    for i in range(0, len(v_split_lines), 2):
        start_x0, start_x1 = v_split_lines[i:i+2]
        # 然后找出[start_x0, start_x1]之间的其他bbox，这些组成一个未分割板块
        bboxes_in_block = [bbox for bbox in all_bboxes if bbox[X0_IDX]>=start_x0 and bbox[X1_IDX]<=start_x1]
        unsplited_bboxes.append(bboxes_in_block)
    # 接着把未处理的加入到v_layouts里
    for bboxes_in_block in unsplited_bboxes:
        if len(bboxes_in_block) == 0:
            continue
        x0, y0, x1, y1 = min([bbox[X0_IDX] for bbox in bboxes_in_block]), bound_y0, max([bbox[X1_IDX] for bbox in bboxes_in_block]), bound_y1
        v_layouts.append([x0, y0, x1, y1, LAYOUT_UNPROC]) # 说明这篇区域未能够分析出可靠的版面
        
    v_layouts.sort(key=lambda x: x[0]) # 按照x0排序, 也就是从左到右的顺序
    
    for layout in v_layouts:
        sorted_layout_blocks.append({
            "layout_bbox": layout[:4],
            "layout_label":layout[4],
            "sub_layout":[],
        })
        
    """
    至此，垂直方向切成了2种类型，其一是独占一列的，其二是未处理的。
    下面对这些未处理的进行垂直方向切分，这个切分要切出来类似“吕”这种类型的垂直方向的布局
    """
    for i, layout in enumerate(sorted_layout_blocks):
        if layout['layout_label'] == LAYOUT_UNPROC:
            x0, y0, x1, y1 = layout['layout_bbox']
            v_split_layouts = _vertical_align_split_v2(bboxes, [x0, y0, x1, y1])
            sorted_layout_blocks[i] = {
                "layout_bbox": [x0, y0, x1, y1],
                "layout_label": LAYOUT_H,
                "sub_layout": v_split_layouts
            }
            layout['layout_label'] = LAYOUT_H # 被垂线切分成了水平布局
    
    return sorted_layout_blocks
    

def split_layout(bboxes:list, boundry:tuple, page_num:int)-> list:
    """
    把bboxes切割成layout
    return:
    [
        {
            "layout_bbox": [x0, y0, x1, y1],
            "layout_label":"u|v|h|b", 未处理|垂直|水平|BAD_LAYOUT
            "sub_layout": [] #每个元素都是[x0, y0, x1, y1, block_content, idx_x, idx_y, content_type, ext_x0, ext_y0, ext_x1, ext_y1], 并且顺序就是阅读顺序
        }
    ]
    example:
    [
        {
            "layout_bbox": [0, 0, 100, 100],
            "layout_label":"u|v|h|b",
            "sub_layout":[
                
            ]
        },
        {
            "layout_bbox": [0, 0, 100, 100],
            "layout_label":"u|v|h|b",
            "sub_layout":[
                {
                    "layout_bbox": [0, 0, 100, 100],
                    "layout_label":"u|v|h|b",
                    "content_bboxes":[
                        [],
                        [],
                        []
                    ]
                },
                {
                    "layout_bbox": [0, 0, 100, 100],
                    "layout_label":"u|v|h|b",
                    "sub_layout":[
                        
                    ]
                }
        }
    ]  
    """
    sorted_layouts = [] # 最终返回的结果
    
    boundry_x0, boundry_y0, boundry_x1, boundry_y1 = boundry
    if len(bboxes) <=1:
        return [
            {
                "layout_bbox": [boundry_x0, boundry_y0, boundry_x1, boundry_y1],
                "layout_label": LAYOUT_V,
                "sub_layout":[]
            }
        ]
        
    """
    接下来按照先水平后垂直的顺序进行切分
    """
    bboxes = paper_bbox_sort(bboxes, boundry_x1-boundry_x0, boundry_y1-boundry_y0)
    sorted_layouts = _horizontal_split(bboxes, boundry) # 通过水平分割出来的layout
    for i, layout in enumerate(sorted_layouts):
        x0, y0, x1, y1 = layout['layout_bbox']
        layout_type = layout['layout_label']
        if layout_type == LAYOUT_UNPROC: # 说明是非独占单行的，这些需要垂直切分
            v_split_layouts = _vertical_split(bboxes, [x0, y0, x1, y1])
            
            """
            最后这里有个逻辑问题：如果这个函数只分离出来了一个column layout，那么这个layout分割肯定超出了算法能力范围。因为我们假定的是传进来的
            box已经把行全部剥离了，所以这里必须十多个列才可以。如果只剥离出来一个layout，并且是多个box，那么就说明这个layout是无法分割的，标记为LAYOUT_UNPROC
            """
            layout_label = LAYOUT_V
            if len(v_split_layouts) == 1:
                if len(v_split_layouts[0]['sub_layout']) == 0:
                    layout_label = LAYOUT_UNPROC
                    #logger.warning(f"WARNING: pageno={page_num}, 无法分割的layout: ", v_split_layouts)
            
            """
            组合起来最终的layout
            """
            sorted_layouts[i] = {
                "layout_bbox": [x0, y0, x1, y1],
                "layout_label": layout_label,
                "sub_layout": v_split_layouts
            }
            layout['layout_label'] = LAYOUT_H
        
    """
    水平和垂直方向都切分完毕了。此时还有一些未处理的，这些未处理的可能是因为水平和垂直方向都无法切分。
    这些最后调用_try_horizontal_mult_block_split做一次水平多个block的联合切分，如果也不能切分最终就当做BAD_LAYOUT返回
    """
    # TODO
    
    return sorted_layouts


def get_bboxes_layout(all_boxes:list, boundry:tuple, page_id:int):
    """
    对利用layout排序之后的box，进行排序
    return:
    [
        {
            "layout_bbox": [x0, y0, x1, y1],
            "layout_label":"u|v|h|b", 未处理|垂直|水平|BAD_LAYOUT
        }，
    ]
    """
    def _preorder_traversal(layout):
        """
        对sorted_layouts的叶子节点，也就是len(sub_layout)==0的节点进行排序。排序按照前序遍历的顺序，也就是从上到下，从左到右的顺序
        """
        sorted_layout_blocks = []
        for layout in layout:
            sub_layout = layout['sub_layout']
            if len(sub_layout) == 0:
                sorted_layout_blocks.append(layout)
            else:
                s = _preorder_traversal(sub_layout)
                sorted_layout_blocks.extend(s)
        return sorted_layout_blocks
    # -------------------------------------------------------------------------------------------------------------------------
    sorted_layouts = split_layout(all_boxes, boundry, page_id)# 先切分成layout，得到一个Tree
    total_sorted_layout_blocks  = _preorder_traversal(sorted_layouts)
    return total_sorted_layout_blocks, sorted_layouts


def get_columns_cnt_of_layout(layout_tree):
    """
    获取一个layout的宽度
    """
    max_width_list = [0] # 初始化一个元素，防止max,min函数报错
    
    for items in layout_tree: # 针对每一层（横切）计算列数，横着的算一列
        layout_type = items['layout_label']
        sub_layouts = items['sub_layout']
        if len(sub_layouts)==0:
            max_width_list.append(1)
        else:
            if layout_type == LAYOUT_H:
                max_width_list.append(1)
            else:
                width = 0
                for l in sub_layouts:
                    if len(l['sub_layout']) == 0:
                        width += 1
                    else:
                        for lay in l['sub_layout']:
                            width += get_columns_cnt_of_layout([lay])
                max_width_list.append(width)
                
    return max(max_width_list)

                
    
def sort_with_layout(bboxes:list, page_width, page_height) -> (list,list):
    """
    输入是一个bbox的list.
    获取到输入之后，先进行layout切分，然后对这些bbox进行排序。返回排序后的bboxes
    """

    new_bboxes = []
    for box in bboxes:
        # new_bboxes.append([box[0], box[1], box[2], box[3], None, None, None, 'text', None, None, None, None])
        new_bboxes.append([box[0], box[1], box[2], box[3], None, None, None, 'text', None, None, None, None, box[4]])
    
    layout_bboxes, _ = get_bboxes_layout(new_bboxes, [0, 0, page_width, page_height], 0)
    if any([lay['layout_label']==LAYOUT_UNPROC for lay in layout_bboxes]):
            logger.warning(f"drop this pdf, reason: 复杂版面")
            return None,None
        
    sorted_bboxes = []    
    # 利用layout bbox每次框定一些box，然后排序
    for layout in layout_bboxes:
        lbox = layout['layout_bbox']
        bbox_in_layout = get_bbox_in_boundry(new_bboxes, lbox)
        sorted_bbox = paper_bbox_sort(bbox_in_layout, lbox[2]-lbox[0], lbox[3]-lbox[1])
        sorted_bboxes.extend(sorted_bbox)
        
    return sorted_bboxes, layout_bboxes


def sort_text_block(text_block, layout_bboxes):
    """
    对一页的text_block进行排序
    """
    sorted_text_bbox = []
    all_text_bbox = []
    # 做一个box=>text的映射
    box_to_text = {}
    for blk in text_block:
        box = blk['bbox']
        box_to_text[(box[0], box[1], box[2], box[3])] = blk
        all_text_bbox.append(box)
        
    # text_blocks_to_sort = []
    # for box in box_to_text.keys():
    #     text_blocks_to_sort.append([box[0], box[1], box[2], box[3], None, None, None, 'text', None, None, None, None])
    
    # 按照layout_bboxes的顺序，对text_block进行排序
    for layout in layout_bboxes:
        layout_box = layout['layout_bbox']
        text_bbox_in_layout = get_bbox_in_boundry(all_text_bbox, [layout_box[0]-1, layout_box[1]-1, layout_box[2]+1, layout_box[3]+1])
        #sorted_bbox = paper_bbox_sort(text_bbox_in_layout, layout_box[2]-layout_box[0], layout_box[3]-layout_box[1])
        text_bbox_in_layout.sort(key = lambda x: x[1]) # 一个layout内部的box，按照y0自上而下排序
        #sorted_bbox = [[b] for b in text_blocks_to_sort]
        for sb in text_bbox_in_layout:
            sorted_text_bbox.append(box_to_text[(sb[0], sb[1], sb[2], sb[3])])
        
    return sorted_text_bbox
