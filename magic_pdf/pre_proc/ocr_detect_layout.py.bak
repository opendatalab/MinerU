import fitz

from magic_pdf.layout.layout_sort import get_bboxes_layout
from magic_pdf.libs.boxbase import _is_part_overlap, _is_in
from magic_pdf.libs.coordinate_transform import get_scale_ratio


def get_center_point(bbox):
    """
    根据边界框坐标信息，计算出该边界框的中心点坐标。
    Args:
        bbox (list): 边界框坐标信息，包含四个元素，分别为左上角x坐标、左上角y坐标、右下角x坐标、右下角y坐标。
    Returns:
        list: 中心点坐标信息，包含两个元素，分别为x坐标和y坐标。
    """
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


def get_area(bbox):
    """
    根据边界框坐标信息，计算出该边界框的面积。
    Args:
        bbox (list): 边界框坐标信息，包含四个元素，分别为左上角x坐标、左上角y坐标、右下角x坐标、右下角y坐标。
    Returns:
        float: 该边界框的面积。
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def adjust_layouts(layout_bboxes, page_boundry, page_id):
    # 遍历所有布局框
    for i in range(len(layout_bboxes)):
        # 遍历当前布局框之后的布局框
        for j in range(i + 1, len(layout_bboxes)):
            # 判断两个布局框是否重叠
            if _is_part_overlap(layout_bboxes[i], layout_bboxes[j]):
                # 计算每个布局框的中心点坐标和面积
                area_i = get_area(layout_bboxes[i])
                area_j = get_area(layout_bboxes[j])

                # 较大布局框和较小布局框的赋值
                if area_i > area_j:
                    larger_layout, smaller_layout = layout_bboxes[i], layout_bboxes[j]
                else:
                    larger_layout, smaller_layout = layout_bboxes[j], layout_bboxes[i]

                center_large = get_center_point(larger_layout)
                center_small = get_center_point(smaller_layout)
                # 计算横向和纵向的距离差
                distance_x = center_large[0] - center_small[0]
                distance_y = center_large[1] - center_small[1]

                # 根据距离差判断重叠方向并修正边界
                if abs(distance_x) > abs(distance_y):  # 左右重叠
                    if distance_x > 0 and larger_layout[0] < smaller_layout[2]:
                        larger_layout[0] = smaller_layout[2]+1
                    if distance_x < 0 and larger_layout[2] > smaller_layout[0]:
                        larger_layout[2] = smaller_layout[0]-1
                else:  # 上下重叠
                    if distance_y > 0 and larger_layout[1] < smaller_layout[3]:
                        larger_layout[1] = smaller_layout[3]+1
                    if distance_y < 0 and larger_layout[3] > smaller_layout[1]:
                        larger_layout[3] = smaller_layout[1]-1
    # 排序调整布局边界框列表
    new_bboxes = []
    for layout_bbox in layout_bboxes:
        new_bboxes.append([layout_bbox[0], layout_bbox[1], layout_bbox[2], layout_bbox[3], None, None, None, None, None, None, None, None, None])

    layout_bboxes, layout_tree = get_bboxes_layout(new_bboxes, page_boundry, page_id)

    # 返回排序调整后的布局边界框列表
    return layout_bboxes, layout_tree


def layout_detect(layout_info, page: fitz.Page, ocr_page_info):
    """
    对输入的布局信息进行解析，提取出每个子布局的边界框，并对所有子布局进行排序调整。

    Args:
        layout_info (list): 包含子布局信息的列表，每个子布局信息为字典类型，包含'poly'字段，表示子布局的边界框坐标信息。

    Returns:
        list: 经过排序调整后的所有子布局边界框信息的列表，每个边界框信息为字典类型，包含'layout_bbox'字段，表示边界框的坐标信息。

    """
    page_id = ocr_page_info['page_info']['page_no']-1
    horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(ocr_page_info, page)
    # 初始化布局边界框列表
    layout_bboxes = []
    # 遍历每个子布局
    for sub_layout in layout_info:
        # 提取子布局的边界框坐标信息
        x0, y0, _, _, x1, y1, _, _ = sub_layout['poly']
        bbox = [int(x0 / horizontal_scale_ratio), int(y0 / vertical_scale_ratio),
                int(x1 / horizontal_scale_ratio), int(y1 / vertical_scale_ratio)]

        # 将子布局的边界框添加到列表中
        layout_bboxes.append(bbox)

    # 初始化新的布局边界框列表
    new_layout_bboxes = []
    # 遍历每个布局边界框
    for i in range(len(layout_bboxes)):
        # 初始化标记变量，用于判断当前边界框是否需要保留
        keep = True
        # 获取当前边界框的坐标信息
        box_i = layout_bboxes[i]

        # 遍历其他边界框
        for j in range(len(layout_bboxes)):
            # 排除当前边界框自身
            if i != j:
                # 获取其他边界框的坐标信息
                box_j = layout_bboxes[j]
                # 检测box_i是否被box_j包含
                if _is_in(box_i, box_j):
                    # 如果当前边界框被其他边界框包含，则标记为不需要保留
                    keep = False
                    # 跳出内层循环
                    break

        # 如果当前边界框需要保留，则添加到新的布局边界框列表中
        if keep:
            new_layout_bboxes.append(layout_bboxes[i])

    # 对新的布局边界框列表进行排序调整
    page_width = page.rect.width
    page_height = page.rect.height
    page_boundry = [0, 0, page_width, page_height]
    layout_bboxes, layout_tree = adjust_layouts(new_layout_bboxes, page_boundry, page_id)

    # 返回排序调整后的布局边界框列表
    return layout_bboxes, layout_tree
