import math


def is_in(box1, box2) -> bool:
    """box1是否完全在box2里面."""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return (
        x0_1 >= x0_2  # box1的左边界不在box2的左边外
        and y0_1 >= y0_2  # box1的上边界不在box2的上边外
        and x1_1 <= x1_2  # box1的右边界不在box2的右边外
        and y1_1 <= y1_2
    )  # box1的下边界不在box2的下边外


def bbox_relative_pos(bbox1, bbox2):
    """判断两个矩形框的相对位置关系.

    Args:
        bbox1: 一个四元组，表示第一个矩形框的左上角和右下角的坐标，格式为(x1, y1, x1b, y1b)
        bbox2: 一个四元组，表示第二个矩形框的左上角和右下角的坐标，格式为(x2, y2, x2b, y2b)

    Returns:
        一个四元组，表示矩形框1相对于矩形框2的位置关系，格式为(left, right, bottom, top)
        其中，left表示矩形框1是否在矩形框2的左侧，right表示矩形框1是否在矩形框2的右侧，
        bottom表示矩形框1是否在矩形框2的下方，top表示矩形框1是否在矩形框2的上方
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    return left, right, bottom, top


def bbox_distance(bbox1, bbox2):
    """计算两个矩形框的距离。

    Args:
        bbox1 (tuple): 第一个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (tuple): 第二个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。

    Returns:
        float: 矩形框之间的距离。
    """

    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)

    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    return 0.0
