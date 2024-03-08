

from loguru import logger


def _is_in_or_part_overlap(box1, box2) -> bool:
    """
    两个bbox是否有部分重叠或者包含
    """
    if box1 is None or box2 is None:
        return False
    
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return not (x1_1 < x0_2 or  # box1在box2的左边
                x0_1 > x1_2 or  # box1在box2的右边
                y1_1 < y0_2 or  # box1在box2的上边
                y0_1 > y1_2)    # box1在box2的下边

def _is_in(box1, box2) -> bool:
    """
    box1是否完全在box2里面
    """
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return (x0_1 >= x0_2 and  # box1的左边界不在box2的左边外
            y0_1 >= y0_2 and  # box1的上边界不在box2的上边外
            x1_1 <= x1_2 and  # box1的右边界不在box2的右边外
            y1_1 <= y1_2)     # box1的下边界不在box2的下边外
    
def _is_part_overlap(box1, box2) -> bool:
    """
    两个bbox是否有部分重叠，但不完全包含
    """
    if box1 is None or box2 is None:
        return False
    
    return _is_in_or_part_overlap(box1, box2) and not _is_in(box1, box2)

def _left_intersect(left_box, right_box):
    "检查两个box的左边界是否有交集，也就是left_box的右边界是否在right_box的左边界内"
    if left_box is None or right_box is None:
        return False
    
    x0_1, y0_1, x1_1, y1_1 = left_box
    x0_2, y0_2, x1_2, y1_2 = right_box
    
    return x1_1>x0_2 and x0_1<x0_2 and (y0_1<=y0_2<=y1_1 or y0_1<=y1_2<=y1_1)

def _right_intersect(left_box, right_box):
    """
    检查box是否在右侧边界有交集，也就是left_box的左边界是否在right_box的右边界内
    """
    if left_box is None or right_box is None:
        return False
    
    x0_1, y0_1, x1_1, y1_1 = left_box
    x0_2, y0_2, x1_2, y1_2 = right_box
    
    return x0_1<x1_2 and x1_1>x1_2 and (y0_1<=y0_2<=y1_1 or y0_1<=y1_2<=y1_1)


def _is_vertical_full_overlap(box1, box2, x_torlence=2):
    """
    x方向上：要么box1包含box2, 要么box2包含box1。不能部分包含
    y方向上：box1和box2有重叠
    """
    # 解析box的坐标
    x11, y11, x12, y12 = box1  # 左上角和右下角的坐标 (x1, y1, x2, y2)
    x21, y21, x22, y22 = box2

    # 在x轴方向上，box1是否包含box2 或 box2包含box1
    contains_in_x = (x11-x_torlence <= x21 and x12+x_torlence >= x22) or (x21-x_torlence <= x11 and x22+x_torlence >= x12)

    # 在y轴方向上，box1和box2是否有重叠
    overlap_in_y = not (y12 < y21 or y11 > y22)

    return contains_in_x and overlap_in_y
    

def _is_bottom_full_overlap(box1, box2, y_tolerance=2):
    """
    检查box1下方和box2的上方有轻微的重叠，轻微程度收到y_tolerance的限制
    这个函数和_is_vertical-full_overlap的区别是，这个函数允许box1和box2在x方向上有轻微的重叠,允许一定的模糊度
    """
    if box1 is None or box2 is None:
        return False
    
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    tolerance_margin = 2
    is_xdir_full_overlap = ((x0_1-tolerance_margin<=x0_2<=x1_1+tolerance_margin and x0_1-tolerance_margin<=x1_2<=x1_1+tolerance_margin) or (x0_2-tolerance_margin<=x0_1<=x1_2+tolerance_margin and x0_2-tolerance_margin<=x1_1<=x1_2+tolerance_margin))
    
    return y0_2<y1_1 and 0<(y1_1-y0_2)<y_tolerance and is_xdir_full_overlap

def _is_left_overlap(box1, box2,):
    """
    检查box1的左侧是否和box2有重叠
    在Y方向上可以是部分重叠或者是完全重叠。不分box1和box2的上下关系，也就是无论box1在box2下方还是box2在box1下方，都可以检测到重叠。
    X方向上
    """
    def __overlap_y(Ay1, Ay2, By1, By2):
        return max(0, min(Ay2, By2) - max(Ay1, By1))
    
    if box1 is None or box2 is None:
        return False
    
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    y_overlap_len = __overlap_y(y0_1, y1_1, y0_2, y1_2)
    ratio_1 = 1.0 * y_overlap_len / (y1_1 - y0_1) if y1_1-y0_1!=0 else 0
    ratio_2 = 1.0 * y_overlap_len / (y1_2 - y0_2) if y1_2-y0_2!=0 else 0
    vertical_overlap_cond = ratio_1 >= 0.5 or ratio_2 >= 0.5
    
    #vertical_overlap_cond = y0_1<=y0_2<=y1_1 or y0_1<=y1_2<=y1_1 or y0_2<=y0_1<=y1_2 or y0_2<=y1_1<=y1_2
    return x0_1<=x0_2<=x1_1 and vertical_overlap_cond


def __is_overlaps_y_exceeds_threshold(bbox1, bbox2, overlap_ratio_threshold=0.8):
    """检查两个bbox在y轴上是否有重叠，并且该重叠区域的高度占两个bbox高度更低的那个超过80%"""
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2

    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    height1, height2 = y1_1 - y0_1, y1_2 - y0_2
    max_height = max(height1, height2)
    min_height = min(height1, height2)

    return (overlap / min_height) > overlap_ratio_threshold



def calculate_iou(bbox1, bbox2):
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both rectangles
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Compute the intersection over union by taking the intersection area 
    # and dividing it by the sum of both areas minus the intersection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


def calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2):
    """
    计算box1和box2的重叠面积占最小面积的box的比例
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    min_box_area = min([(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]), (bbox2[3]-bbox2[1])*(bbox2[2]-bbox2[0])])
    if min_box_area==0:
        return 0
    else:
        return intersection_area / min_box_area

def calculate_overlap_area_in_bbox1_area_ratio(bbox1, bbox2):
    """
    计算box1和box2的重叠面积占bbox1的比例
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    if bbox1_area == 0:
        return 0
    else:
        return intersection_area / bbox1_area


def get_minbox_if_overlap_by_ratio(bbox1, bbox2, ratio):
    """
    通过calculate_overlap_area_2_minbox_area_ratio计算两个bbox重叠的面积占最小面积的box的比例
    如果比例大于ratio，则返回小的那个bbox,
    否则返回None
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    overlap_ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
    if overlap_ratio > ratio and area1 < area2:
        return bbox1
    elif overlap_ratio > ratio and area2 < area1:
        return bbox2
    else:
        return None

def get_bbox_in_boundry(bboxes:list, boundry:tuple)-> list:
    x0, y0, x1, y1 = boundry
    new_boxes = [box for box in bboxes if box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1]
    return new_boxes


def is_vbox_on_side(bbox, width, height, side_threshold=0.2):
    """
    判断一个bbox是否在pdf页面的边缘
    """
    x0, x1 = bbox[0], bbox[2]
    if x1<=width*side_threshold or x0>=width*(1-side_threshold):
        return True
    return False

def find_top_nearest_text_bbox(pymu_blocks, obj_bbox):
    tolerance_margin = 4
    top_boxes = [box for box in pymu_blocks if obj_bbox[1]-box['bbox'][3] >=-tolerance_margin and not _is_in(box['bbox'], obj_bbox)]
    # 然后找到X方向上有互相重叠的
    top_boxes = [box for box in top_boxes if any([obj_bbox[0]-tolerance_margin <=box['bbox'][0]<=obj_bbox[2]+tolerance_margin, 
                                                  obj_bbox[0]-tolerance_margin <=box['bbox'][2]<=obj_bbox[2]+tolerance_margin,
                                                    box['bbox'][0]-tolerance_margin <=obj_bbox[0]<=box['bbox'][2]+tolerance_margin,
                                                    box['bbox'][0]-tolerance_margin <=obj_bbox[2]<=box['bbox'][2]+tolerance_margin
                                                  ])]
    
    # 然后找到y1最大的那个
    if len(top_boxes)>0:
        top_boxes.sort(key=lambda x: x['bbox'][3], reverse=True)
        return top_boxes[0]
    else:
        return None
    

def find_bottom_nearest_text_bbox(pymu_blocks, obj_bbox):
    bottom_boxes = [box for box in pymu_blocks if box['bbox'][1] - obj_bbox[3]>=-2 and not _is_in(box['bbox'], obj_bbox)]
    # 然后找到X方向上有互相重叠的
    bottom_boxes = [box for box in bottom_boxes if any([obj_bbox[0]-2 <=box['bbox'][0]<=obj_bbox[2]+2, 
                                                  obj_bbox[0]-2 <=box['bbox'][2]<=obj_bbox[2]+2,
                                                    box['bbox'][0]-2 <=obj_bbox[0]<=box['bbox'][2]+2,
                                                    box['bbox'][0]-2 <=obj_bbox[2]<=box['bbox'][2]+2
                                                  ])]
    
    # 然后找到y0最小的那个
    if len(bottom_boxes)>0:
        bottom_boxes.sort(key=lambda x: x['bbox'][1], reverse=False)
        return bottom_boxes[0]
    else:
        return None

def find_left_nearest_text_bbox(pymu_blocks, obj_bbox):
    """
    寻找左侧最近的文本block
    """
    left_boxes = [box for box in pymu_blocks if obj_bbox[0]-box['bbox'][2]>=-2 and not _is_in(box['bbox'], obj_bbox)]
    # 然后找到X方向上有互相重叠的
    left_boxes = [box for box in left_boxes if any([obj_bbox[1]-2 <=box['bbox'][1]<=obj_bbox[3]+2, 
                                                  obj_bbox[1]-2 <=box['bbox'][3]<=obj_bbox[3]+2,
                                                    box['bbox'][1]-2 <=obj_bbox[1]<=box['bbox'][3]+2,
                                                    box['bbox'][1]-2 <=obj_bbox[3]<=box['bbox'][3]+2
                                                  ])]
    
    # 然后找到x1最大的那个
    if len(left_boxes)>0:
        left_boxes.sort(key=lambda x: x['bbox'][2], reverse=True)
        return left_boxes[0]
    else:
        return None
    

def find_right_nearest_text_bbox(pymu_blocks, obj_bbox):
    """
    寻找右侧最近的文本block
    """
    right_boxes = [box for box in pymu_blocks if box['bbox'][0]-obj_bbox[2]>=-2 and not _is_in(box['bbox'], obj_bbox)]
    # 然后找到X方向上有互相重叠的
    right_boxes = [box for box in right_boxes if any([obj_bbox[1]-2 <=box['bbox'][1]<=obj_bbox[3]+2, 
                                                  obj_bbox[1]-2 <=box['bbox'][3]<=obj_bbox[3]+2,
                                                    box['bbox'][1]-2 <=obj_bbox[1]<=box['bbox'][3]+2,
                                                    box['bbox'][1]-2 <=obj_bbox[3]<=box['bbox'][3]+2
                                                  ])]
    
    # 然后找到x0最小的那个
    if len(right_boxes)>0:
        right_boxes.sort(key=lambda x: x['bbox'][0], reverse=False)
        return right_boxes[0]
    else:
        return None
