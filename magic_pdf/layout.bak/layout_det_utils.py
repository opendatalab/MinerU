from magic_pdf.layout.bbox_sort import X0_EXT_IDX, X0_IDX, X1_EXT_IDX, X1_IDX, Y0_IDX, Y1_EXT_IDX, Y1_IDX
from magic_pdf.libs.boxbase import _is_bottom_full_overlap, _left_intersect, _right_intersect


def find_all_left_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    在all_bboxes里找到所有右侧垂直方向上和this_bbox有重叠的bbox， 不用延长线
    并且要考虑两个box左右相交的情况，如果相交了，那么右侧的box就不算最左侧。
    """
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX] 
         and any([
         box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
         this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
         box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]]) or _left_intersect(box[:4], this_bbox[:4])]
        
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个——x1最大的那个
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x[X1_EXT_IDX] if x[X1_EXT_IDX] else x[X1_IDX], reverse=True)
        left_boxes = left_boxes[0]
    else:
        left_boxes = None
    return left_boxes

def find_all_right_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox右侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    right_bboxes = [box for box in all_bboxes if box[X0_IDX] >= this_bbox[X1_IDX] 
        and any([
        this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
        box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
        box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]]) or _right_intersect(this_bbox[:4], box[:4])]
    
    if len(right_bboxes)>0:
        right_bboxes.sort(key=lambda x: x[X0_EXT_IDX] if x[X0_EXT_IDX] else x[X0_IDX])
        right_bboxes = right_bboxes[0]
    else:
        right_bboxes = None
    return right_bboxes

def find_all_top_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox上侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    top_bboxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(top_bboxes)>0:
        top_bboxes.sort(key=lambda x: x[Y1_EXT_IDX] if x[Y1_EXT_IDX] else x[Y1_IDX], reverse=True)
        top_bboxes = top_bboxes[0]
    else:
        top_bboxes = None
    return top_bboxes

def find_all_bottom_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox下侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    bottom_bboxes = [box for box in all_bboxes if box[Y0_IDX] >= this_bbox[Y1_IDX] and any([
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(bottom_bboxes)>0:
        bottom_bboxes.sort(key=lambda x:  x[Y0_IDX])
        bottom_bboxes = bottom_bboxes[0]
    else:
        bottom_bboxes = None
    return bottom_bboxes

# ===================================================================================================================
def find_bottom_bbox_direct_from_right_edge(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox下侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    bottom_bboxes = [box for box in all_bboxes if box[Y0_IDX] >= this_bbox[Y1_IDX] and any([
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(bottom_bboxes)>0:
        # y0最小， X1最大的那个,也就是box上边缘最靠近this_bbox的那个,并且还最靠右
        bottom_bboxes.sort(key=lambda x: x[Y0_IDX])
        bottom_bboxes = [box for box in bottom_bboxes if box[Y0_IDX]==bottom_bboxes[0][Y0_IDX]]
        # 然后再y1相同的情况下，找到x1最大的那个
        bottom_bboxes.sort(key=lambda x: x[X1_IDX], reverse=True)
        bottom_bboxes = bottom_bboxes[0]
    else:
        bottom_bboxes = None
    return bottom_bboxes

def find_bottom_bbox_direct_from_left_edge(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox下侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    bottom_bboxes = [box for box in all_bboxes if box[Y0_IDX] >= this_bbox[Y1_IDX] and any([
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(bottom_bboxes)>0:
        # y0最小， X0最小的那个
        bottom_bboxes.sort(key=lambda x: x[Y0_IDX])
        bottom_bboxes = [box for box in bottom_bboxes if box[Y0_IDX]==bottom_bboxes[0][Y0_IDX]]
        # 然后再y0相同的情况下，找到x0最小的那个
        bottom_bboxes.sort(key=lambda x: x[X0_IDX])
        bottom_bboxes = bottom_bboxes[0]
    else:
        bottom_bboxes = None
    return bottom_bboxes

def find_top_bbox_direct_from_left_edge(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox上侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    top_bboxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(top_bboxes)>0:
        # y1最大， X0最小的那个
        top_bboxes.sort(key=lambda x: x[Y1_IDX], reverse=True)
        top_bboxes = [box for box in top_bboxes if box[Y1_IDX]==top_bboxes[0][Y1_IDX]]
        # 然后再y1相同的情况下，找到x0最小的那个
        top_bboxes.sort(key=lambda x: x[X0_IDX])
        top_bboxes = top_bboxes[0]
    else:
        top_bboxes = None
    return top_bboxes

def find_top_bbox_direct_from_right_edge(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox上侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    top_bboxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
        this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    
    if len(top_bboxes)>0:
        # y1最大， X1最大的那个
        top_bboxes.sort(key=lambda x: x[Y1_IDX], reverse=True)
        top_bboxes = [box for box in top_bboxes if box[Y1_IDX]==top_bboxes[0][Y1_IDX]]
        # 然后再y1相同的情况下，找到x1最大的那个
        top_bboxes.sort(key=lambda x: x[X1_IDX], reverse=True)
        top_bboxes = top_bboxes[0]
    else:
        top_bboxes = None
    return top_bboxes
    
# ===================================================================================================================

def get_left_edge_bboxes(all_bboxes) -> list:
    """
    返回最左边的bbox
    """
    left_bboxes = [box for box in all_bboxes if find_all_left_bbox_direct(box, all_bboxes) is None]
    return left_bboxes
    
def get_right_edge_bboxes(all_bboxes) -> list:
    """
    返回最右边的bbox
    """
    right_bboxes = [box for box in all_bboxes if find_all_right_bbox_direct(box, all_bboxes) is None]
    return right_bboxes

def fix_vertical_bbox_pos(bboxes:list):
    """
    检查这批bbox在垂直方向是否有轻微的重叠，如果重叠了，就把重叠的bbox往下移动一点
    在x方向上必须一个包含或者被包含，或者完全重叠，不能只有部分重叠
    """
    bboxes.sort(key=lambda x: x[Y0_IDX]) # 从上向下排列
    for i in range(0, len(bboxes)):
        for j in range(i+1, len(bboxes)):
            if _is_bottom_full_overlap(bboxes[i][:4], bboxes[j][:4]):
                # 如果两个bbox有部分重叠，那么就把下面的bbox往下移动一点
                bboxes[j][Y0_IDX] = bboxes[i][Y1_IDX] + 2 # 2是个经验值
                break
    return bboxes
