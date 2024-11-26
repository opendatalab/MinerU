# 定义这里的bbox是一个list [x0, y0, x1, y1, block_content, idx_x, idx_y, content_type, ext_x0, ext_y0, ext_x1, ext_y1], 初始时候idx_x, idx_y都是None
# 其中x0, y0代表左上角坐标，x1, y1代表右下角坐标，坐标原点在左上角。



from magic_pdf.layout.layout_spiler_recog import get_spilter_of_page
from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap, _is_vertical_full_overlap
from magic_pdf.libs.commons import mymax

X0_IDX = 0
Y0_IDX = 1
X1_IDX = 2
Y1_IDX = 3
CONTENT_IDX = 4
IDX_X = 5
IDX_Y = 6
CONTENT_TYPE_IDX = 7

X0_EXT_IDX = 8
Y0_EXT_IDX = 9
X1_EXT_IDX = 10
Y1_EXT_IDX = 11


def prepare_bboxes_for_layout_split(image_info, image_backup_info, table_info, inline_eq_info, interline_eq_info, text_raw_blocks: dict, page_boundry, page):
    """
    text_raw_blocks:结构参考test/assets/papre/pymu_textblocks.json
    把bbox重新组装成一个list，每个元素[x0, y0, x1, y1, block_content, idx_x, idx_y, content_type, ext_x0, ext_y0, ext_x1, ext_y1], 初始时候idx_x, idx_y都是None. 对于图片、公式来说，block_content是图片的地址， 对于段落来说，block_content是pymupdf里的block结构
    """
    all_bboxes = []
    
    for image in image_info:
        box = image['bbox']
        # 由于没有实现横向的栏切分，因此在这里先过滤掉一些小的图片。这些图片有可能影响layout，造成没有横向栏切分的情况下，layout切分不准确。例如 scihub_76500000/libgen.scimag76570000-76570999.zip_10.1186/s13287-019-1355-1
        # 把长宽都小于50的去掉
        if abs(box[0]-box[2]) < 50 and abs(box[1]-box[3]) < 50:
            continue
        all_bboxes.append([box[0], box[1], box[2], box[3], None, None, None, 'image', None, None, None, None])
        
    for table in table_info:
        box = table['bbox']
        all_bboxes.append([box[0], box[1], box[2], box[3], None, None, None, 'table', None, None, None, None])
    
    """由于公式与段落混合，因此公式不再参与layout划分，无需加入all_bboxes"""
    # 加入文本block
    text_block_temp = []
    for block in text_raw_blocks:
        bbox = block['bbox']
        text_block_temp.append([bbox[0], bbox[1], bbox[2], bbox[3], None, None, None, 'text', None, None, None, None])
        
    text_block_new = resolve_bbox_overlap_for_layout_det(text_block_temp)   
    text_block_new = filter_lines_bbox(text_block_new) # 去掉线条bbox，有可能让layout探测陷入无限循环
    
        
    """找出会影响layout的色块、横向分割线"""
    spilter_bboxes = get_spilter_of_page(page, [b['bbox'] for b in image_info]+[b['bbox'] for b in image_backup_info], [b['bbox'] for b in table_info], )
    # 还要去掉存在于spilter_bboxes里的text_block
    if len(spilter_bboxes) > 0:
        text_block_new = [box for box in text_block_new if not any([_is_in_or_part_overlap(box[:4], spilter_bbox) for spilter_bbox in spilter_bboxes])]
        
    for bbox in text_block_new:
        all_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], None, None, None, 'text', None, None, None, None]) 
        
    for bbox in spilter_bboxes:
        all_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], None, None, None, 'spilter', None, None, None, None])
    
     
    return all_bboxes

def resolve_bbox_overlap_for_layout_det(bboxes:list):
    """
    1. 去掉bbox互相包含的，去掉被包含的
    2. 上下方向上如果有重叠，就扩大大box范围，直到覆盖小box
    """
    def _is_in_other_bbox(i:int):
        """
        判断i个box是否被其他box有所包含
        """
        for j in range(0, len(bboxes)):
            if j!=i and _is_in(bboxes[i][:4], bboxes[j][:4]):
                return True
            # elif j!=i and _is_bottom_full_overlap(bboxes[i][:4], bboxes[j][:4]):
            #     return True
            
        return False
    
    # 首先去掉被包含的bbox
    new_bbox_1 = []
    for i in range(0, len(bboxes)):
        if not _is_in_other_bbox(i):
            new_bbox_1.append(bboxes[i])
            
    # 其次扩展大的box
    new_box = []
    new_bbox_2 = []
    len_1 = len(new_bbox_2)
    while True:
        merged_idx = []
        for i in range(0, len(new_bbox_1)):
            if i in merged_idx:
                continue
            for j in range(i+1, len(new_bbox_1)):
                if j in merged_idx:
                    continue
                bx1 = new_bbox_1[i]
                bx2 = new_bbox_1[j]
                if i!=j and _is_vertical_full_overlap(bx1[:4], bx2[:4]):
                    merged_box = min([bx1[0], bx2[0]]), min([bx1[1], bx2[1]]), max([bx1[2], bx2[2]]), max([bx1[3], bx2[3]])
                    new_bbox_2.append(merged_box)
                    merged_idx.append(i)
                    merged_idx.append(j)
                    
        for i in range(0, len(new_bbox_1)): # 没有合并的加入进来
            if i not in merged_idx:
                new_bbox_2.append(new_bbox_1[i])        

        if len(new_bbox_2)==0 or len_1==len(new_bbox_2):
            break
        else:
            len_1 = len(new_bbox_2)
            new_box = new_bbox_2
            new_bbox_1, new_bbox_2 = new_bbox_2, []
                        
    return new_box


def filter_lines_bbox(bboxes: list):
    """
    过滤掉bbox为空的行
    """
    new_box = []
    for box in bboxes:
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        if abs(x0-x1)<=1 or abs(y0-y1)<=1:
            continue
        else:
            new_box.append(box)
    return new_box


################################################################################
# 第一种排序算法
# 以下是基于延长线遮挡做的一个算法
#
################################################################################
def find_all_left_bbox(this_bbox, all_bboxes) -> list:
    """
    寻找this_bbox左边的所有bbox
    """
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX]]
    return left_boxes


def find_all_top_bbox(this_bbox, all_bboxes) -> list:
    """
    寻找this_bbox上面的所有bbox
    """
    top_boxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX]]
    return top_boxes


def get_and_set_idx_x(this_bbox, all_bboxes) -> int:
    """
    寻找this_bbox在all_bboxes中的遮挡深度 idx_x
    """
    if this_bbox[IDX_X] is not None:
        return this_bbox[IDX_X]
    else:
        all_left_bboxes = find_all_left_bbox(this_bbox, all_bboxes)
        if len(all_left_bboxes) == 0:
            this_bbox[IDX_X] = 0
        else:
            all_left_bboxes_idx = [get_and_set_idx_x(bbox, all_bboxes) for bbox in all_left_bboxes]
            max_idx_x = mymax(all_left_bboxes_idx)
            this_bbox[IDX_X] = max_idx_x + 1
        return this_bbox[IDX_X]


def get_and_set_idx_y(this_bbox, all_bboxes) -> int:
    """
    寻找this_bbox在all_bboxes中y方向的遮挡深度 idx_y
    """
    if this_bbox[IDX_Y] is not None:
        return this_bbox[IDX_Y]
    else:
        all_top_bboxes = find_all_top_bbox(this_bbox, all_bboxes)
        if len(all_top_bboxes) == 0:
            this_bbox[IDX_Y] = 0
        else:
            all_top_bboxes_idx = [get_and_set_idx_y(bbox, all_bboxes) for bbox in all_top_bboxes]
            max_idx_y = mymax(all_top_bboxes_idx)
            this_bbox[IDX_Y] = max_idx_y + 1
        return this_bbox[IDX_Y]


def bbox_sort(all_bboxes: list):
    """
    排序
    """
    all_bboxes_idx_x = [get_and_set_idx_x(bbox, all_bboxes) for bbox in all_bboxes]
    all_bboxes_idx_y = [get_and_set_idx_y(bbox, all_bboxes) for bbox in all_bboxes]
    all_bboxes_idx = [(idx_x, idx_y) for idx_x, idx_y in zip(all_bboxes_idx_x, all_bboxes_idx_y)]

    all_bboxes_idx = [idx_x_y[0] * 100000 + idx_x_y[1] for idx_x_y in all_bboxes_idx]  # 变换成一个点，保证能够先X，X相同时按Y排序
    all_bboxes_idx = list(zip(all_bboxes_idx, all_bboxes))
    all_bboxes_idx.sort(key=lambda x: x[0])
    sorted_bboxes = [bbox for idx, bbox in all_bboxes_idx]
    return sorted_bboxes


################################################################################
# 第二种排序算法
# 下面的算法在计算idx_x和idx_y的时候不考虑延长线，而只考虑实际的长或者宽被遮挡的情况
#
################################################################################

def find_left_nearest_bbox(this_bbox, all_bboxes) -> list:
    """
    在all_bboxes里找到所有右侧高度和this_bbox有重叠的bbox
    """
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX] and any([
         box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
         this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
         box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]])]
        
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x[X1_IDX], reverse=True)
        left_boxes = [left_boxes[0]]
    else:
        left_boxes = []
    return left_boxes


def get_and_set_idx_x_2(this_bbox, all_bboxes):
    """
    寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_x
    这个遮挡深度不考虑延长线，而是被实际的长或者宽遮挡的情况
    """
    if this_bbox[IDX_X] is not None:
        return this_bbox[IDX_X]
    else:
        left_nearest_bbox = find_left_nearest_bbox(this_bbox, all_bboxes)
        if len(left_nearest_bbox) == 0:
            this_bbox[IDX_X] = 0
        else:
            left_idx_x = get_and_set_idx_x_2(left_nearest_bbox[0], all_bboxes)
            this_bbox[IDX_X] = left_idx_x + 1
        return this_bbox[IDX_X]


def find_top_nearest_bbox(this_bbox, all_bboxes) -> list:
    """
    在all_bboxes里找到所有下侧宽度和this_bbox有重叠的bbox
    """
    top_boxes = [box for box in all_bboxes if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
         this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个
    if len(top_boxes) > 0:
        top_boxes.sort(key=lambda x: x[Y1_IDX], reverse=True)
        top_boxes = [top_boxes[0]]
    else:
        top_boxes = []
    return top_boxes


def get_and_set_idx_y_2(this_bbox, all_bboxes):
    """
    寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_y
    这个遮挡深度不考虑延长线，而是被实际的长或者宽遮挡的情况
    """
    if this_bbox[IDX_Y] is not None:
        return this_bbox[IDX_Y]
    else:
        top_nearest_bbox = find_top_nearest_bbox(this_bbox, all_bboxes)
        if len(top_nearest_bbox) == 0:
            this_bbox[IDX_Y] = 0
        else:
            top_idx_y = get_and_set_idx_y_2(top_nearest_bbox[0], all_bboxes)
            this_bbox[IDX_Y] = top_idx_y + 1
        return this_bbox[IDX_Y]


def paper_bbox_sort(all_bboxes: list, page_width, page_height):
    all_bboxes_idx_x = [get_and_set_idx_x_2(bbox, all_bboxes) for bbox in all_bboxes]
    all_bboxes_idx_y = [get_and_set_idx_y_2(bbox, all_bboxes) for bbox in all_bboxes]
    all_bboxes_idx = [(idx_x, idx_y) for idx_x, idx_y in zip(all_bboxes_idx_x, all_bboxes_idx_y)]

    all_bboxes_idx = [idx_x_y[0] * 100000 + idx_x_y[1] for idx_x_y in all_bboxes_idx]  # 变换成一个点，保证能够先X，X相同时按Y排序
    all_bboxes_idx = list(zip(all_bboxes_idx, all_bboxes))
    all_bboxes_idx.sort(key=lambda x: x[0])
    sorted_bboxes = [bbox for idx, bbox in all_bboxes_idx]
    return sorted_bboxes

################################################################################
"""
第三种排序算法, 假设page的最左侧为X0，最右侧为X1，最上侧为Y0，最下侧为Y1
这个排序算法在第二种算法基础上增加对bbox的预处理步骤。预处理思路如下：
1. 首先在水平方向上对bbox进行扩展。扩展方法是：
    - 对每个bbox，找到其左边最近的bbox（也就是y方向有重叠），然后将其左边界扩展到左边最近bbox的右边界(x1+1),这里加1是为了避免重叠。如果没有左边的bbox，那么就将其左边界扩展到page的最左侧X0。
    - 对每个bbox，找到其右边最近的bbox（也就是y方向有重叠），然后将其右边界扩展到右边最近bbox的左边界(x0-1),这里减1是为了避免重叠。如果没有右边的bbox，那么就将其右边界扩展到page的最右侧X1。
    - 经过上面2个步骤，bbox扩展到了水平方向的最大范围。[左最近bbox.x1+1, 右最近bbox.x0-1]
    
2. 合并所有的连续水平方向的bbox, 合并方法是：
    - 对bbox进行y方向排序，然后从上到下遍历所有bbox，如果当前bbox和下一个bbox的x0, x1等于X0, X1，那么就合并这两个bbox。
    
3. 然后在垂直方向上对bbox进行扩展。扩展方法是：
    - 首先从page上切割掉合并后的水平bbox, 得到几个新的block
    针对每个block
    - x0: 扎到位于左侧x=x0延长线的左侧所有的bboxes, 找到最大的x1,让x0=x1+1。如果没有，则x0=X0
    - x1: 找到位于右侧x=x1延长线右侧所有的bboxes， 找到最小的x0, 让x1=x0-1。如果没有，则x1=X1
    随后在垂直方向上合并所有的连续的block，方法如下：
    - 对block进行x方向排序，然后从左到右遍历所有block，如果当前block和下一个block的x0, x1相等，那么就合并这两个block。
    如果垂直切分后所有小bbox都被分配到了一个block, 那么分割就完成了。这些合并后的block打上标签'GOOD_LAYOUT’
    如果在某个垂直方向上无法被完全分割到一个block，那么就将这个block打上标签'BAD_LAYOUT'。
    至此完成，一个页面的预处理，天然的block要么属于'GOOD_LAYOUT'，要么属于'BAD_LAYOUT'。针对含有'BAD_LAYOUT'的页面，可以先按照自上而下，自左到右进行天然排序，也可以先过滤掉这种书籍。
    (完成条件下次加强：进行水平方向切分，把混乱的layout部分尽可能切割出去)
"""
################################################################################
def find_left_neighbor_bboxes(this_bbox, all_bboxes) -> list:
    """
    在all_bboxes里找到所有右侧高度和this_bbox有重叠的bbox
    这里使用扩展之后的bbox
    """
    left_boxes = [box for box in all_bboxes if box[X1_EXT_IDX] <= this_bbox[X0_EXT_IDX] and any([
         box[Y0_EXT_IDX] < this_bbox[Y0_EXT_IDX] < box[Y1_EXT_IDX], box[Y0_EXT_IDX] < this_bbox[Y1_EXT_IDX] < box[Y1_EXT_IDX],
         this_bbox[Y0_EXT_IDX] < box[Y0_EXT_IDX] < this_bbox[Y1_EXT_IDX], this_bbox[Y0_EXT_IDX] < box[Y1_EXT_IDX] < this_bbox[Y1_EXT_IDX],
         box[Y0_EXT_IDX]==this_bbox[Y0_EXT_IDX] and box[Y1_EXT_IDX]==this_bbox[Y1_EXT_IDX]])]
        
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x[X1_EXT_IDX], reverse=True)
        left_boxes = left_boxes
    else:
        left_boxes = []
    return left_boxes

def find_top_neighbor_bboxes(this_bbox, all_bboxes) -> list:
    """
    在all_bboxes里找到所有下侧宽度和this_bbox有重叠的bbox
    这里使用扩展之后的bbox
    """
    top_boxes = [box for box in all_bboxes if box[Y1_EXT_IDX] <= this_bbox[Y0_EXT_IDX] and any([
        box[X0_EXT_IDX] < this_bbox[X0_EXT_IDX] < box[X1_EXT_IDX], box[X0_EXT_IDX] < this_bbox[X1_EXT_IDX] < box[X1_EXT_IDX],
         this_bbox[X0_EXT_IDX] < box[X0_EXT_IDX] < this_bbox[X1_EXT_IDX], this_bbox[X0_EXT_IDX] < box[X1_EXT_IDX] < this_bbox[X1_EXT_IDX],
        box[X0_EXT_IDX]==this_bbox[X0_EXT_IDX] and box[X1_EXT_IDX]==this_bbox[X1_EXT_IDX]])]
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个
    if len(top_boxes) > 0:
        top_boxes.sort(key=lambda x: x[Y1_EXT_IDX], reverse=True)
        top_boxes = top_boxes
    else:
        top_boxes = []
    return top_boxes

def get_and_set_idx_x_2_ext(this_bbox, all_bboxes):
    """
    寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_x
    这个遮挡深度不考虑延长线，而是被实际的长或者宽遮挡的情况
    """
    if this_bbox[IDX_X] is not None:
        return this_bbox[IDX_X]
    else:
        left_nearest_bbox = find_left_neighbor_bboxes(this_bbox, all_bboxes)
        if len(left_nearest_bbox) == 0:
            this_bbox[IDX_X] = 0
        else:
            left_idx_x = [get_and_set_idx_x_2(b, all_bboxes) for b in left_nearest_bbox]
            this_bbox[IDX_X] = mymax(left_idx_x) + 1
        return this_bbox[IDX_X]
   
def get_and_set_idx_y_2_ext(this_bbox, all_bboxes):
    """
    寻找this_bbox在all_bboxes中的被直接遮挡的深度 idx_y
    这个遮挡深度不考虑延长线，而是被实际的长或者宽遮挡的情况
    """
    if this_bbox[IDX_Y] is not None:
        return this_bbox[IDX_Y]
    else:
        top_nearest_bbox = find_top_neighbor_bboxes(this_bbox, all_bboxes)
        if len(top_nearest_bbox) == 0:
            this_bbox[IDX_Y] = 0
        else:
            top_idx_y = [get_and_set_idx_y_2_ext(b, all_bboxes) for b in top_nearest_bbox]
            this_bbox[IDX_Y] = mymax(top_idx_y) + 1
        return this_bbox[IDX_Y]
 
def _paper_bbox_sort_ext(all_bboxes: list):
    all_bboxes_idx_x = [get_and_set_idx_x_2_ext(bbox, all_bboxes) for bbox in all_bboxes]
    all_bboxes_idx_y = [get_and_set_idx_y_2_ext(bbox, all_bboxes) for bbox in all_bboxes]
    all_bboxes_idx = [(idx_x, idx_y) for idx_x, idx_y in zip(all_bboxes_idx_x, all_bboxes_idx_y)]

    all_bboxes_idx = [idx_x_y[0] * 100000 + idx_x_y[1] for idx_x_y in all_bboxes_idx]  # 变换成一个点，保证能够先X，X相同时按Y排序
    all_bboxes_idx = list(zip(all_bboxes_idx, all_bboxes))
    all_bboxes_idx.sort(key=lambda x: x[0])
    sorted_bboxes = [bbox for idx, bbox in all_bboxes_idx]
    return sorted_bboxes

# ===============================================================================================
def find_left_bbox_ext_line(this_bbox, all_bboxes) -> list:
    """
    寻找this_bbox左边的所有bbox, 使用延长线
    """
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX]]
    if len(left_boxes):
        left_boxes.sort(key=lambda x: x[X1_IDX], reverse=True)
        left_boxes = left_boxes[0]
    else:
        left_boxes = None
    
    return left_boxes

def find_right_bbox_ext_line(this_bbox, all_bboxes) -> list:
    """
    寻找this_bbox右边的所有bbox, 使用延长线
    """
    right_boxes = [box for box in all_bboxes if box[X0_IDX] >= this_bbox[X1_IDX]]
    if len(right_boxes):
        right_boxes.sort(key=lambda x: x[X0_IDX])
        right_boxes = right_boxes[0]
    else:
        right_boxes = None
    return right_boxes

# =============================================================================================

def find_left_nearest_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    在all_bboxes里找到所有右侧高度和this_bbox有重叠的bbox， 不用延长线并且不能像
    """
    left_boxes = [box for box in all_bboxes if box[X1_IDX] <= this_bbox[X0_IDX] and any([
         box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
         this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
         box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]])]
        
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个——x1最大的那个
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x[X1_EXT_IDX] if x[X1_EXT_IDX] else x[X1_IDX], reverse=True)
        left_boxes = left_boxes[0]
    else:
        left_boxes = None
    return left_boxes

def find_right_nearst_bbox_direct(this_bbox, all_bboxes) -> list:
    """
    找到在this_bbox右侧且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    right_bboxes = [box for box in all_bboxes if box[X0_IDX] >= this_bbox[X1_IDX] and any([
        this_bbox[Y0_IDX] < box[Y0_IDX] < this_bbox[Y1_IDX], this_bbox[Y0_IDX] < box[Y1_IDX] < this_bbox[Y1_IDX],
        box[Y0_IDX] < this_bbox[Y0_IDX] < box[Y1_IDX], box[Y0_IDX] < this_bbox[Y1_IDX] < box[Y1_IDX],
        box[Y0_IDX]==this_bbox[Y0_IDX] and box[Y1_IDX]==this_bbox[Y1_IDX]])]
    
    if len(right_bboxes)>0:
        right_bboxes.sort(key=lambda x: x[X0_EXT_IDX] if x[X0_EXT_IDX] else x[X0_IDX])
        right_bboxes = right_bboxes[0]
    else:
        right_bboxes = None
    return right_bboxes

def reset_idx_x_y(all_boxes:list)->list:
    for box in all_boxes:
        box[IDX_X] = None
        box[IDX_Y] = None
        
    return all_boxes

# ===================================================================================================
def find_top_nearest_bbox_direct(this_bbox, bboxes_collection) -> list:
    """
    找到在this_bbox上方且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    top_bboxes = [box for box in bboxes_collection if box[Y1_IDX] <= this_bbox[Y0_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
         this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    # 然后再过滤一下，找到上方距离this_bbox最近的那个
    if len(top_bboxes) > 0:
        top_bboxes.sort(key=lambda x: x[Y1_IDX], reverse=True)
        top_bboxes = top_bboxes[0]
    else:
        top_bboxes = None
    return top_bboxes

def find_bottom_nearest_bbox_direct(this_bbox, bboxes_collection) -> list:
    """
    找到在this_bbox下方且距离this_bbox距离最近的bbox.必须是直接遮挡的那种
    """
    bottom_bboxes = [box for box in bboxes_collection if box[Y0_IDX] >= this_bbox[Y1_IDX] and any([
        box[X0_IDX] < this_bbox[X0_IDX] < box[X1_IDX], box[X0_IDX] < this_bbox[X1_IDX] < box[X1_IDX],
         this_bbox[X0_IDX] < box[X0_IDX] < this_bbox[X1_IDX], this_bbox[X0_IDX] < box[X1_IDX] < this_bbox[X1_IDX],
        box[X0_IDX]==this_bbox[X0_IDX] and box[X1_IDX]==this_bbox[X1_IDX]])]
    # 然后再过滤一下，找到水平上距离this_bbox最近的那个
    if len(bottom_bboxes) > 0:
        bottom_bboxes.sort(key=lambda x: x[Y0_IDX])
        bottom_bboxes = bottom_bboxes[0]
    else:
        bottom_bboxes = None
    return bottom_bboxes

def find_boundry_bboxes(bboxes:list) -> tuple:
    """
    找到bboxes的边界——找到所有bbox里最小的(x0, y0), 最大的(x1, y1)
    """
    x0, y0, x1, y1 = bboxes[0][X0_IDX], bboxes[0][Y0_IDX], bboxes[0][X1_IDX], bboxes[0][Y1_IDX]
    for box in bboxes:
        x0 = min(box[X0_IDX], x0)
        y0 = min(box[Y0_IDX], y0)
        x1 = max(box[X1_IDX], x1)
        y1 = max(box[Y1_IDX], y1)
        
    return x0, y0, x1, y1
    

def extend_bbox_vertical(bboxes:list, boundry_x0, boundry_y0, boundry_x1, boundry_y1) -> list:
    """
    在垂直方向上扩展能够直接垂直打通的bbox,也就是那些上下都没有其他box的bbox
    """
    for box in bboxes:
        top_nearest_bbox = find_top_nearest_bbox_direct(box, bboxes)
        bottom_nearest_bbox = find_bottom_nearest_bbox_direct(box, bboxes)
        if top_nearest_bbox is None and bottom_nearest_bbox is None: # 独占一列
            box[X0_EXT_IDX] = box[X0_IDX]
            box[Y0_EXT_IDX] = boundry_y0
            box[X1_EXT_IDX] = box[X1_IDX]
            box[Y1_EXT_IDX] = boundry_y1
        # else:
        #     if top_nearest_bbox is None:
        #         box[Y0_EXT_IDX] = boundry_y0
        #     else:
        #         box[Y0_EXT_IDX] = top_nearest_bbox[Y1_IDX] + 1
        #     if bottom_nearest_bbox is None:
        #         box[Y1_EXT_IDX] = boundry_y1
        #     else:
        #         box[Y1_EXT_IDX] = bottom_nearest_bbox[Y0_IDX] - 1
        #     box[X0_EXT_IDX] = box[X0_IDX]
        #     box[X1_EXT_IDX] = box[X1_IDX]
    return bboxes
    

# ===================================================================================================

def paper_bbox_sort_v2(all_bboxes: list, page_width:int, page_height:int):
    """
    增加预处理行为的排序:
    return:
    [
        {
            "layout_bbox": [x0, y0, x1, y1],
            "layout_label":"GOOD_LAYOUT/BAD_LAYOUT",
            "content_bboxes": [] #每个元素都是[x0, y0, x1, y1, block_content, idx_x, idx_y, content_type, ext_x0, ext_y0, ext_x1, ext_y1], 并且顺序就是阅读顺序
        }
    ]
    """
    sorted_layouts = [] # 最后的返回结果
    page_x0, page_y0, page_x1, page_y1 = 1, 1, page_width-1, page_height-1
    
    all_bboxes = paper_bbox_sort(all_bboxes) # 大致拍下序
    # 首先在水平方向上扩展独占一行的bbox
    for bbox in all_bboxes:
        left_nearest_bbox = find_left_nearest_bbox_direct(bbox, all_bboxes) # 非扩展线
        right_nearest_bbox = find_right_nearst_bbox_direct(bbox, all_bboxes)
        if left_nearest_bbox is None and right_nearest_bbox is None: # 独占一行
            bbox[X0_EXT_IDX] = page_x0
            bbox[Y0_EXT_IDX] = bbox[Y0_IDX]
            bbox[X1_EXT_IDX] = page_x1
            bbox[Y1_EXT_IDX] = bbox[Y1_IDX]
            
    # 此时独占一行的被成功扩展到指定的边界上，这个时候利用边界条件合并连续的bbox，成为一个group
    if len(all_bboxes)==1:
        return [{"layout_bbox": [page_x0, page_y0, page_x1, page_y1], "layout_label":"GOOD_LAYOUT", "content_bboxes": all_bboxes}]
    if len(all_bboxes)==0:
        return []
    
    """
    然后合并所有连续水平方向的bbox.
    
    """
    all_bboxes.sort(key=lambda x: x[Y0_IDX])
    h_bboxes = []
    h_bbox_group = []
    v_boxes = []

    for bbox in all_bboxes:
        if bbox[X0_IDX] == page_x0 and bbox[X1_IDX] == page_x1:
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
    for gp in h_bboxes:
        gp.sort(key=lambda x: x[Y0_IDX])
        block_info = {"layout_label":"GOOD_LAYOUT", "content_bboxes": gp}
        # 然后计算这个group的layout_bbox，也就是最小的x0,y0, 最大的x1,y1
        x0, y0, x1, y1 = gp[0][X0_EXT_IDX], gp[0][Y0_EXT_IDX], gp[-1][X1_EXT_IDX], gp[-1][Y1_EXT_IDX]
        block_info["layout_bbox"] = [x0, y0, x1, y1]
        sorted_layouts.append(block_info)
        
    # 接下来利用这些连续的水平bbox的layout_bbox的y0, y1，从水平上切分开其余的为几个部分
    h_split_lines = [page_y0]
    for gp in h_bboxes:
        layout_bbox = gp['layout_bbox']
        y0, y1 = layout_bbox[1], layout_bbox[3]
        h_split_lines.append(y0)
        h_split_lines.append(y1)
    h_split_lines.append(page_y1)
    
    unsplited_bboxes = []
    for i in range(0, len(h_split_lines), 2):
        start_y0, start_y1 = h_split_lines[i:i+2]
        # 然后找出[start_y0, start_y1]之间的其他bbox，这些组成一个未分割板块
        bboxes_in_block = [bbox for bbox in all_bboxes if bbox[Y0_IDX]>=start_y0 and bbox[Y1_IDX]<=start_y1]
        unsplited_bboxes.append(bboxes_in_block)
    # ================== 至此，水平方向的 已经切分排序完毕====================================
    """
    接下来针对每个非水平的部分切分垂直方向的
    此时，只剩下了无法被完全水平打通的bbox了。对这些box，优先进行垂直扩展，然后进行垂直切分.
    分3步：
    1. 先把能完全垂直打通的隔离出去当做一个layout
    2. 其余的先垂直切分
    3. 垂直切分之后的部分再尝试水平切分
    4. 剩下的不能被切分的各个部分当成一个layout
    """
    # 对每部分进行垂直切分
    for bboxes_in_block in unsplited_bboxes:
        # 首先对这个block的bbox进行垂直方向上的扩展
        boundry_x0, boundry_y0, boundry_x1, boundry_y1 = find_boundry_bboxes(bboxes_in_block) 
        # 进行垂直方向上的扩展
        extended_vertical_bboxes = extend_bbox_vertical(bboxes_in_block, boundry_x0, boundry_y0, boundry_x1, boundry_y1)
        # 然后对这个block进行垂直方向上的切分
        extend_bbox_vertical.sort(key=lambda x: x[X0_IDX]) # x方向上从小到大，代表了从左到右读取
        v_boxes_group = []
        for bbox in extended_vertical_bboxes:
            if bbox[Y0_IDX]==boundry_y0 and bbox[Y1_IDX]==boundry_y1:
                v_boxes_group.append(bbox)
            else:
                if len(v_boxes_group)>0:
                    v_boxes.append(v_boxes_group)
                    v_boxes_group = []
                    
        if len(v_boxes_group)>0:
            
            v_boxes.append(v_boxes_group)
            
        # 把连续的垂直部分加入到sorted_layouts里。注意这个时候已经是连续的垂直部分了，因为上面已经做了
        for gp in v_boxes:
            gp.sort(key=lambda x: x[X0_IDX])
            block_info = {"layout_label":"GOOD_LAYOUT", "content_bboxes": gp}
            # 然后计算这个group的layout_bbox，也就是最小的x0,y0, 最大的x1,y1
            x0, y0, x1, y1 = gp[0][X0_EXT_IDX], gp[0][Y0_EXT_IDX], gp[-1][X1_EXT_IDX], gp[-1][Y1_EXT_IDX]
            block_info["layout_bbox"] = [x0, y0, x1, y1]
            sorted_layouts.append(block_info)
            
        # 在垂直方向上，划分子块，也就是用贯通的垂直线进行切分。这些被切分出来的块，极大可能是可被垂直切分的，如果不能完全的垂直切分，那么尝试水平切分。都不能的则当成一个layout
        v_split_lines = [boundry_x0]
        for gp in v_boxes:
            layout_bbox = gp['layout_bbox']
            x0, x1 = layout_bbox[0], layout_bbox[2]
            v_split_lines.append(x0)
            v_split_lines.append(x1)
        v_split_lines.append(boundry_x1)
        
    reset_idx_x_y(all_bboxes)
    all_boxes = _paper_bbox_sort_ext(all_bboxes)
    return all_boxes
            
    
    
    
    



