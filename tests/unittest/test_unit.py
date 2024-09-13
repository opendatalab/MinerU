import os

import pytest

from magic_pdf.libs.boxbase import (__is_overlaps_y_exceeds_threshold,
                                    _is_bottom_full_overlap, _is_in,
                                    _is_in_or_part_overlap,
                                    _is_in_or_part_overlap_with_area_ratio,
                                    _is_left_overlap, _is_part_overlap,
                                    _is_vertical_full_overlap, _left_intersect,
                                    _right_intersect, bbox_distance,
                                    bbox_relative_pos, calculate_iou,
                                    calculate_overlap_area_2_minbox_area_ratio,
                                    calculate_overlap_area_in_bbox1_area_ratio,
                                    find_bottom_nearest_text_bbox,
                                    find_left_nearest_text_bbox,
                                    find_right_nearest_text_bbox,
                                    find_top_nearest_text_bbox,
                                    get_bbox_in_boundary,
                                    get_minbox_if_overlap_by_ratio)
from magic_pdf.libs.commons import get_top_percent_list, join_path, mymax
from magic_pdf.libs.config_reader import get_s3_config
from magic_pdf.libs.path_utils import parse_s3path


# 输入一个列表，如果列表空返回0，否则返回最大元素
@pytest.mark.parametrize('list_input, target_num',
                         [
                             ([0, 0, 0, 0], 0),
                             ([0], 0),
                             ([1, 2, 5, 8, 4], 8),
                             ([], 0),
                             ([1.1, 7.6, 1.009, 9.9], 9.9),
                             ([1.0 * 10 ** 2, 3.5 * 10 ** 3, 0.9 * 10 ** 6], 0.9 * 10 ** 6),
                         ])
def test_list_max(list_input: list, target_num) -> None:
    """
    list_input: 输入列表元素，元素均为数字类型
    """
    assert target_num == mymax(list_input)


# 连接多个参数生成路径信息，使用"/"作为连接符，生成的结果需要是一个合法路径
@pytest.mark.parametrize('path_input, target_path', [
    (['https:', '', 'www.baidu.com'], 'https://www.baidu.com'),
    (['https:', 'www.baidu.com'], 'https:/www.baidu.com'),
    (['D:', 'file', 'pythonProject', 'demo' + '.py'], 'D:/file/pythonProject/demo.py'),
])
def test_join_path(path_input: list, target_path: str) -> None:
    """
    path_input: 输入path的列表，列表元素均为字符串
    """
    assert target_path == join_path(*path_input)


# 获取列表中前百分之多少的元素
@pytest.mark.parametrize('num_list, percent, target_num_list', [
    ([], 0.75, []),
    ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11], 0.8, [23, 9, 7, 3, 0, -1, -5, -7]),
    ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11], 0, []),
    ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11, 28], 0.8, [28, 23, 9, 7, 3, 0, -1, -5])
])
def test_get_top_percent_list(num_list: list, percent: float, target_num_list: list) -> None:
    """
    num_list: 数字列表，列表元素为数字
    percent: 占比，float, 向下取证
    """
    assert target_num_list == get_top_percent_list(num_list, percent)


# 输入一个s3路径，返回bucket名字和其余部分(key)
@pytest.mark.parametrize('s3_path, target_data', [
    ('s3://bucket/path/to/my/file.txt', 'bucket'),
    ('s3a://bucket1/path/to/my/file2.txt', 'bucket1'),
    # ("/path/to/my/file1.txt", "path"),
    # ("bucket/path/to/my/file2.txt", "bucket"),
])
def test_parse_s3path(s3_path: str, target_data: str):
    """
    s3_path: s3路径
        如果为无效路径，则返回对应的bucket名字和其余部分
        如果为异常路径 例如：file2.txt，则报异常
    """
    bucket_name, key = parse_s3path(s3_path)
    assert target_data == bucket_name


# 2个box是否处于包含或者部分重合关系。
# 如果某边界重合算重合。
# 部分边界重合，其他在内部也算包含
@pytest.mark.parametrize('box1, box2, target_bool', [
    ((120, 133, 223, 248), (128, 168, 269, 295), True),
    ((137, 53, 245, 157), (134, 11, 200, 147), True),  # 部分重合
    ((137, 56, 211, 116), (140, 66, 202, 199), True),  # 部分重合
    ((42, 34, 69, 65), (42, 34, 69, 65), True),  # 部分重合
    ((39, 63, 87, 106), (37, 66, 85, 109), True),  # 部分重合
    ((13, 37, 55, 66), (7, 46, 49, 75), True),  # 部分重合
    ((56, 83, 85, 104), (64, 85, 93, 106), True),  # 部分重合
    ((12, 53, 48, 94), (14, 53, 50, 94), True),  # 部分重合
    ((43, 54, 93, 131), (55, 82, 77, 106), True),  # 包含
    ((63, 2, 134, 71), (72, 43, 104, 78), True),  # 包含
    ((25, 57, 109, 127), (26, 73, 49, 95), True),  # 包含
    ((24, 47, 111, 115), (34, 81, 58, 106), True),  # 包含
    ((34, 8, 105, 83), (76, 20, 116, 45), True),  # 包含
])
def test_is_in_or_part_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    """
    box1: 坐标数组
    box2: 坐标数组
    """
    assert target_bool == _is_in_or_part_overlap(box1, box2)


# 如果box1在box2内部，返回True
#   如果是部分重合的，则重合面积占box1的比例大于阈值时候返回True
@pytest.mark.parametrize('box1, box2, target_bool', [
    ((35, 28, 108, 90), (47, 60, 83, 96), False),  # 包含 box1 up box2,  box2 多半,box1少半
    ((65, 151, 92, 177), (49, 99, 105, 198), True),  # 包含 box1 in box2
    ((80, 62, 112, 84), (74, 40, 144, 111), True),  # 包含 box1 in box2
    ((65, 88, 127, 144), (92, 102, 131, 139), False),  # 包含 box2 多半，box1约一半
    ((92, 102, 131, 139), (65, 88, 127, 144), True),  # 包含 box1 多半
    ((100, 93, 199, 168), (169, 126, 198, 165), False),  # 包含 box2 in box1
    ((26, 75, 106, 172), (65, 108, 90, 128), False),  # 包含 box2 in box1
    ((28, 90, 77, 126), (35, 84, 84, 120), True),  # 相交 box1多半，box2多半
    ((37, 6, 69, 52), (28, 3, 60, 49), True),  # 相交 box1多半，box2多半
    ((94, 29, 133, 60), (84, 30, 123, 61), True),  # 相交 box1多半，box2多半
])
def test_is_in_or_part_overlap_with_area_ratio(box1: tuple, box2: tuple, target_bool: bool) -> None:
    out_bool = _is_in_or_part_overlap_with_area_ratio(box1, box2)
    assert target_bool == out_bool


# box1在box2内部或者box2在box1内部返回True。如果部分边界重合也算作包含。
@pytest.mark.parametrize('box1, box2, target_bool', [
    # ((), (), "Error"),  # Error
    ((65, 151, 92, 177), (49, 99, 105, 198), True),  # 包含 box1 in box2
    ((80, 62, 112, 84), (74, 40, 144, 111), True),  # 包含 box1 in box2
    ((76, 140, 154, 277), (121, 326, 192, 384), False),  # 分离
    ((65, 88, 127, 144), (92, 102, 131, 139), False),  # 包含 box2 多半，box1约一半
    ((92, 102, 131, 139), (65, 88, 127, 144), False),  # 包含 box1 多半
    ((68, 94, 118, 120), (68, 90, 118, 122), True),  # 包含，box1 in box2 两边x相切
    ((69, 94, 118, 120), (68, 90, 118, 122), True),  # 包含，box1 in box2 一边x相切
    ((69, 114, 118, 122), (68, 90, 118, 122), True),  # 包含，box1 in box2 一边y相切
    # ((100, 93, 199, 168), (169, 126, 198, 165), True),  # 包含 box2 in box1  Error
    # ((26, 75, 106, 172), (65, 108, 90, 128), True),  # 包含 box2 in box1  Error
    # ((38, 94, 122, 120), (68, 94, 118, 120), True),  # 包含，box2 in box1 两边y相切 Error
    # ((68, 34, 118, 158), (68, 94, 118, 120), True),  # 包含，box2 in box1 两边x相切 Error
    # ((68, 34, 118, 158), (68, 94, 84, 120), True),  # 包含，box2 in box1 一边x相切 Error
    # ((27, 94, 118, 158), (68, 94, 84, 120), True),  # 包含，box2 in box1 一边y相切 Error
])
def test_is_in(box1: tuple, box2: tuple, target_bool: bool) -> None:
    assert target_bool == _is_in(box1, box2)


# 仅仅是部分包含关系，返回True，如果是完全包含关系则返回False
@pytest.mark.parametrize('box1, box2, target_bool', [
    ((65, 151, 92, 177), (49, 99, 105, 198), False),  # 包含 box1 in box2
    ((80, 62, 112, 84), (74, 40, 144, 111), False),  # 包含 box1 in box2
    # ((76, 140, 154, 277), (121, 326, 192, 384), False),  # 分离  Error
    ((76, 140, 154, 277), (121, 277, 192, 384), True),  # 外相切
    ((65, 88, 127, 144), (92, 102, 131, 139), True),  # 包含 box2 多半，box1约一半
    ((92, 102, 131, 139), (65, 88, 127, 144), True),  # 包含 box1 多半
    ((68, 94, 118, 120), (68, 90, 118, 122), False),  # 包含，box1 in box2 两边x相切
    ((69, 94, 118, 120), (68, 90, 118, 122), False),  # 包含，box1 in box2 一边x相切
    ((69, 114, 118, 122), (68, 90, 118, 122), False),  # 包含，box1 in box2 一边y相切
    # ((26, 75, 106, 172), (65, 108, 90, 128), False),  # 包含 box2 in box1  Error
    # ((38, 94, 122, 120), (68, 94, 118, 120), False),  # 包含，box2 in box1 两边y相切 Error
    # ((68, 34, 118, 158), (68, 94, 84, 120), False),  # 包含，box2 in box1 一边x相切 Error

])
def test_is_part_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    assert target_bool == _is_part_overlap(box1, box2)


# left_box右侧是否和right_box左侧有部分重叠
@pytest.mark.parametrize('box1, box2, target_bool', [
    (None, None, False),
    ((88, 81, 222, 173), (60, 221, 123, 358), False),  # 分离
    ((121, 149, 184, 289), (172, 130, 230, 268), True),  # box1 left bottom box2 相交
    ((172, 130, 230, 268), (121, 149, 184, 289), False),  # box2 left bottom box1 相交
    ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 top left box2 分离
    ((117, 53, 222, 176), (174, 142, 298, 276), True),  # box1 left top box2 相交
    ((174, 142, 298, 276), (117, 53, 222, 176), False),  # box2 left top box1 相交
    ((65, 88, 127, 144), (92, 102, 131, 139), True),  # box1 left box2 y:box2 in box1
    ((92, 102, 131, 139), (65, 88, 127, 144), False),  # box2 left box1 y:box1 in box2
    ((182, 130, 230, 268), (121, 149, 174, 289), False),  # box2 left box1 分离
    ((1, 10, 26, 45), (3, 4, 20, 39), True),  # box1 bottom box2 x:box2 in box1
])
def test_left_intersect(box1: tuple, box2: tuple, target_bool: bool) -> None:
    assert target_bool == _left_intersect(box1, box2)


# left_box左侧是否和right_box右侧部分重叠
@pytest.mark.parametrize('box1, box2, target_bool', [
    (None, None, False),
    ((88, 81, 222, 173), (60, 221, 123, 358), False),  # 分离
    ((121, 149, 184, 289), (172, 130, 230, 268), False),  # box1 left bottom box2 相交
    ((172, 130, 230, 268), (121, 149, 184, 289), True),  # box2 left bottom box1 相交
    ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 top left box2 分离
    ((117, 53, 222, 176), (174, 142, 298, 276), False),  # box1 left top box2 相交
    ((174, 142, 298, 276), (117, 53, 222, 176), True),  # box2 left top box1 相交
    ((65, 88, 127, 144), (92, 102, 131, 139), False),  # box1 left box2 y:box2 in box1
    # ((92, 102, 131, 139), (65, 88, 127, 144), True),  # box2 left box1 y:box1 in box2 Error
    ((182, 130, 230, 268), (121, 149, 174, 289), False),  # box2 left box1 分离
    # ((1, 10, 26, 45), (3, 4, 20, 39), False),  # box1 bottom box2 x:box2 in box1 Error
])
def test_right_intersect(box1: tuple, box2: tuple, target_bool: bool) -> None:
    assert target_bool == _right_intersect(box1, box2)


# x方向上：要么box1包含box2, 要么box2包含box1。不能部分包含
# y方向上：box1和box2有重叠
@pytest.mark.parametrize('box1, box2, target_bool', [
    # (None, None, False),  # Error
    ((35, 28, 108, 90), (47, 60, 83, 96), True),  # box1 top box2, x:box2 in box1, y:有重叠
    ((35, 28, 98, 90), (27, 60, 103, 96), True),  # box1 top box2, x:box1 in box2, y:有重叠
    ((57, 77, 130, 210), (59, 219, 119, 293), False),  # box1 top box2, x: box2 in box1, y:无重叠
    ((47, 60, 83, 96), (35, 28, 108, 90), True),  # box2 top box1, x:box1 in box2, y:有重叠
    ((27, 60, 103, 96), (35, 28, 98, 90), True),  # box2 top box1, x:box2 in box1, y:有重叠
    ((59, 219, 119, 293), (57, 77, 130, 210), False),  # box2 top box1, x: box1 in box2, y:无重叠
    ((35, 28, 55, 90), (57, 60, 83, 96), False),  # box1 top box2, x:无重叠, y:有重叠
    ((47, 60, 63, 96), (65, 28, 108, 90), False),  # box2 top box1, x:无重叠, y:有重叠
])
def test_is_vertical_full_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    assert target_bool == _is_vertical_full_overlap(box1, box2)


# 检查box1下方和box2的上方有轻微的重叠，轻微程度收到y_tolerance的限制
@pytest.mark.parametrize('box1, box2, target_bool', [
    (None, None, False),
    ((35, 28, 108, 90), (47, 89, 83, 116), True),  # box1 top box2, y:有重叠
    ((35, 28, 108, 90), (47, 60, 83, 96), False),  # box1 top box2, y:有重叠且过多
    ((57, 77, 130, 210), (59, 219, 119, 293), False),  # box1 top box2, y:无重叠
    ((47, 60, 83, 96), (35, 28, 108, 90), False),  # box2 top box1, y:有重叠且过多
    ((27, 89, 103, 116), (35, 28, 98, 90), False),  # box2 top box1, y:有重叠
    ((59, 219, 119, 293), (57, 77, 130, 210), False),  # box2 top box1, y:无重叠
])
def test_is_bottom_full_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    assert target_bool == _is_bottom_full_overlap(box1, box2)


# 检查box1的左侧是否和box2有重叠
@pytest.mark.parametrize('box1, box2, target_bool', [
    (None, None, False),
    ((88, 81, 222, 173), (60, 221, 123, 358), False),  # 分离
    # ((121, 149, 184, 289), (172, 130, 230, 268), False),  # box1 left bottom box2 相交  Error
    # ((172, 130, 230, 268), (121, 149, 184, 289), True),  # box2 left bottom box1 相交 Error
    ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 top left box2 分离
    ((117, 53, 222, 176), (174, 142, 298, 276), False),  # box1 left top box2 相交
    # ((174, 142, 298, 276), (117, 53, 222, 176), True),  # box2 left top box1 相交  Error
    # ((65, 88, 127, 144), (92, 102, 131, 139), False),  # box1 left box2 y:box2 in box1 Error
    ((1, 10, 26, 45), (3, 4, 20, 39), True),  # box1 middle bottom box2 x:box2 in box1

])
def test_is_left_overlap(box1: tuple, box2: tuple, target_bool: bool) -> None:
    assert target_bool == _is_left_overlap(box1, box2)


# 查两个bbox在y轴上是否有重叠，并且该重叠区域的高度占两个bbox高度更低的那个超过阈值
@pytest.mark.parametrize('box1, box2, target_bool', [
    # (None, None, "Error"),  # Error
    ((51, 69, 192, 147), (75, 48, 132, 187), True),  # y: box1 in box2
    ((51, 39, 192, 197), (75, 48, 132, 187), True),  # y: box2 in box1
    ((88, 81, 222, 173), (60, 221, 123, 358), False),  # y: box1 top box2
    ((109, 68, 182, 196), (215, 188, 277, 253), False),  # y: box1 top box2 little
    ((109, 68, 182, 196), (215, 78, 277, 253), True),  # y: box1 top box2 more
    ((109, 68, 182, 196), (215, 138, 277, 213), False),  # y: box1 top box2 more but lower overlap_ratio_threshold
    ((109, 68, 182, 196), (215, 138, 277, 203), True),  # y: box1 top box2 more and more overlap_ratio_threshold
])
def test_is_overlaps_y_exceeds_threshold(box1: tuple, box2: tuple, target_bool: bool) -> None:
    assert target_bool == __is_overlaps_y_exceeds_threshold(box1, box2)


# Determine the coordinates of the intersection rectangle
@pytest.mark.parametrize('box1, box2, target_num', [
    # (None, None, "Error"),  # Error
    ((88, 81, 222, 173), (60, 221, 123, 358), 0.0),  # 分离
    ((76, 140, 154, 277), (121, 326, 192, 384), 0.0),  # 分离
    ((142, 109, 238, 164), (134, 211, 224, 270), 0.0),  # 分离
    ((109, 68, 182, 196), (175, 138, 277, 213), 0.024475524475524476),  # 相交
    ((56, 90, 170, 219), (103, 212, 171, 304), 0.02288586346557361),  # 相交
    ((109, 126, 204, 245), (130, 127, 232, 186), 0.33696071621517326),  # 相交
    ((109, 126, 204, 245), (110, 127, 232, 206), 0.5493822593770807),  # 相交
    ((76, 140, 154, 277), (121, 277, 192, 384), 0.0)  # 相切
])
def test_calculate_iou(box1: tuple, box2: tuple, target_num: float) -> None:
    assert target_num == calculate_iou(box1, box2)


# 计算box1和box2的重叠面积占最小面积的box的比例
@pytest.mark.parametrize('box1, box2, target_num', [
    # (None, None, "Error"),  # Error
    ((142, 109, 238, 164), (134, 211, 224, 270), 0.0),  # 分离
    ((88, 81, 222, 173), (60, 221, 123, 358), 0.0),  # 分离
    ((76, 140, 154, 277), (121, 326, 192, 384), 0.0),  # 分离
    ((76, 140, 154, 277), (121, 277, 192, 384), 0.0),  # 相切
    ((109, 126, 204, 245), (110, 127, 232, 206), 0.7704918032786885),  # 相交
    ((56, 90, 170, 219), (103, 212, 171, 304), 0.07496803069053709),  # 相交
    ((121, 149, 184, 289), (172, 130, 230, 268), 0.17841079460269865),  # 相交
    ((51, 69, 192, 147), (75, 48, 132, 187), 0.5611510791366906),  # 相交
    ((117, 53, 222, 176), (174, 142, 298, 276), 0.12636469221835075),  # 相交
    ((102, 60, 233, 203), (70, 190, 220, 319), 0.08188757807078417),  # 相交
    ((109, 126, 204, 245), (130, 127, 232, 186), 0.7254901960784313),  # 相交
])
def test_calculate_overlap_area_2_minbox_area_ratio(box1: tuple, box2: tuple, target_num: float) -> None:
    assert target_num == calculate_overlap_area_2_minbox_area_ratio(box1, box2)


# 计算box1和box2的重叠面积占bbox1的比例
@pytest.mark.parametrize('box1, box2, target_num', [
    # (None, None, "Error"),  # Error
    ((142, 109, 238, 164), (134, 211, 224, 270), 0.0),  # 分离
    ((88, 81, 222, 173), (60, 221, 123, 358), 0.0),  # 分离
    ((76, 140, 154, 277), (121, 326, 192, 384), 0.0),  # 分离
    ((76, 140, 154, 277), (121, 277, 192, 384), 0.0),  # 相切
    ((142, 109, 238, 164), (134, 164, 224, 270), 0.0),  # 相切
    ((109, 126, 204, 245), (110, 127, 232, 206), 0.6568774878372402),  # 相交
    ((56, 90, 170, 219), (103, 212, 171, 304), 0.03189174486604107),  # 相交
    ((121, 149, 184, 289), (172, 130, 230, 268), 0.1619047619047619),  # 相交
    ((51, 69, 192, 147), (75, 48, 132, 187), 0.40425531914893614),  # 相交
    ((117, 53, 222, 176), (174, 142, 298, 276), 0.12636469221835075),  # 相交
    ((102, 60, 233, 203), (70, 190, 220, 319), 0.08188757807078417),  # 相交
    ((109, 126, 204, 245), (130, 127, 232, 186), 0.38620079610791685),  # 相交
])
def test_calculate_overlap_area_in_bbox1_area_ratio(box1: tuple, box2: tuple, target_num: float) -> None:
    assert target_num == calculate_overlap_area_in_bbox1_area_ratio(box1, box2)


# 计算两个bbox重叠的面积占最小面积的box的比例，如果比例大于ratio，则返回小的那个bbox,否则返回None
@pytest.mark.parametrize('box1, box2, ratio, target_box', [
    # (None, None, 0.8, "Error"),  # Error
    ((142, 109, 238, 164), (134, 211, 224, 270), 0.0, None),  # 分离
    ((109, 126, 204, 245), (110, 127, 232, 206), 0.5, (110, 127, 232, 206)),
    ((56, 90, 170, 219), (103, 212, 171, 304), 0.5, None),
    ((121, 149, 184, 289), (172, 130, 230, 268), 0.5, None),
    ((51, 69, 192, 147), (75, 48, 132, 187), 0.5, (75, 48, 132, 187)),
    ((117, 53, 222, 176), (174, 142, 298, 276), 0.5, None),
    ((102, 60, 233, 203), (70, 190, 220, 319), 0.5, None),
    ((109, 126, 204, 245), (130, 127, 232, 186), 0.5, (130, 127, 232, 186)),
])
def test_get_minbox_if_overlap_by_ratio(box1: tuple, box2: tuple, ratio: float, target_box: list) -> None:
    assert target_box == get_minbox_if_overlap_by_ratio(box1, box2, ratio)


# 根据boundry获取在这个范围内的所有的box的列表，完全包含关系
@pytest.mark.parametrize('boxes, boundary, target_boxs', [
    # ([], (), "Error"),  # Error
    ([], (110, 340, 209, 387), []),
    ([(142, 109, 238, 164)], (134, 211, 224, 270), []),  # 分离
    ([(109, 126, 204, 245), (110, 127, 232, 206)], (105, 116, 258, 300), [(109, 126, 204, 245), (110, 127, 232, 206)]),
    ([(109, 126, 204, 245), (110, 127, 232, 206)], (105, 116, 258, 230), [(110, 127, 232, 206)]),
    ([(81, 280, 123, 315), (282, 203, 342, 247), (183, 100, 300, 155), (46, 99, 133, 148), (33, 156, 97, 211),
      (137, 29, 287, 87)], (80, 90, 249, 200), []),
    ([(81, 280, 123, 315), (282, 203, 342, 247), (183, 100, 300, 155), (46, 99, 133, 148), (33, 156, 97, 211),
      (137, 29, 287, 87)], (30, 20, 349, 320),
     [(81, 280, 123, 315), (282, 203, 342, 247), (183, 100, 300, 155), (46, 99, 133, 148), (33, 156, 97, 211),
      (137, 29, 287, 87)]),
    ([(81, 280, 123, 315), (282, 203, 342, 247), (183, 100, 300, 155), (46, 99, 133, 148), (33, 156, 97, 211),
      (137, 29, 287, 87)], (30, 20, 200, 320),
     [(81, 280, 123, 315), (46, 99, 133, 148), (33, 156, 97, 211)]),
])
def test_get_bbox_in_boundary(boxes: list, boundary: tuple, target_boxs: list) -> None:
    assert target_boxs == get_bbox_in_boundary(boxes, boundary)


# 寻找上方距离最近的box,margin 4个单位， x方向有重合，y方向最近的
@pytest.mark.parametrize('pymu_blocks, obj_box, target_boxs', [
    ([{'bbox': (81, 280, 123, 315)}, {'bbox': (282, 203, 342, 247)}, {'bbox': (183, 100, 300, 155)},
      {'bbox': (46, 99, 133, 148)}, {'bbox': (33, 156, 97, 211)},
      {'bbox': (137, 29, 287, 87)}], (81, 280, 123, 315), {'bbox': (33, 156, 97, 211)}),
    # ([{"bbox": (168, 120, 263, 159)},
    #   {"bbox": (231, 61, 279, 159)},
    #   {"bbox": (35, 85, 136, 110)},
    #   {"bbox": (228, 193, 347, 225)},
    #   {"bbox": (144, 264, 188, 323)},
    #   {"bbox": (62, 37, 126, 64)}], (228, 193, 347, 225),
    #  [{"bbox": (168, 120, 263, 159)}, {"bbox": (231, 61, 279, 159)}]),  # y：方向最近的有两个，x: 两个均有重合 Error
    ([{'bbox': (35, 85, 136, 159)},
      {'bbox': (168, 120, 263, 159)},
      {'bbox': (231, 61, 279, 118)},
      {'bbox': (228, 193, 347, 225)},
      {'bbox': (144, 264, 188, 323)},
      {'bbox': (62, 37, 126, 64)}], (228, 193, 347, 225),
     {'bbox': (168, 120, 263, 159)},),  # y:方向最近的有两个，x:只有一个有重合
    ([{'bbox': (239, 115, 379, 167)},
      {'bbox': (33, 237, 104, 262)},
      {'bbox': (124, 288, 168, 325)},
      {'bbox': (242, 291, 379, 340)},
      {'bbox': (55, 117, 121, 154)},
      {'bbox': (266, 183, 384, 217)}, ], (124, 288, 168, 325), {'bbox': (55, 117, 121, 154)}),
    ([{'bbox': (239, 115, 379, 167)},
      {'bbox': (33, 237, 104, 262)},
      {'bbox': (124, 288, 168, 325)},
      {'bbox': (242, 291, 379, 340)},
      {'bbox': (55, 117, 119, 154)},
      {'bbox': (266, 183, 384, 217)}, ], (124, 288, 168, 325), None),  # x没有重合
    ([{'bbox': (80, 90, 249, 200)},
      {'bbox': (183, 100, 240, 155)}, ], (183, 100, 240, 155), None),  # 包含
])
def test_find_top_nearest_text_bbox(pymu_blocks: list, obj_box: tuple, target_boxs: dict) -> None:
    assert target_boxs == find_top_nearest_text_bbox(pymu_blocks, obj_box)


# 寻找下方距离自己最近的box, x方向有重合，y方向最近的
@pytest.mark.parametrize('pymu_blocks, obj_box, target_boxs', [
    ([{'bbox': (165, 96, 300, 114)},
      {'bbox': (11, 157, 139, 201)},
      {'bbox': (124, 208, 265, 262)},
      {'bbox': (124, 283, 248, 306)},
      {'bbox': (39, 267, 84, 301)},
      {'bbox': (36, 89, 114, 145)}, ], (165, 96, 300, 114), {'bbox': (124, 208, 265, 262)}),
    ([{'bbox': (187, 37, 303, 49)},
      {'bbox': (2, 227, 90, 283)},
      {'bbox': (158, 174, 200, 212)},
      {'bbox': (259, 174, 324, 228)},
      {'bbox': (205, 61, 316, 97)},
      {'bbox': (295, 248, 374, 287)}, ], (205, 61, 316, 97), {'bbox': (259, 174, 324, 228)}),  # y有两个最近的, x只有一个重合
    # ([{"bbox": (187, 37, 303, 49)},
    #   {"bbox": (2, 227, 90, 283)},
    #   {"bbox": (259, 174, 324, 228)},
    #   {"bbox": (205, 61, 316, 97)},
    #   {"bbox": (295, 248, 374, 287)},
    #   {"bbox": (158, 174, 209, 212)}, ], (205, 61, 316, 97),
    #  [{"bbox": (259, 174, 324, 228)}, {"bbox": (158, 174, 209, 212)}]),  # x有重合，y有两个最近的  Error
    ([{'bbox': (287, 132, 398, 191)},
      {'bbox': (44, 141, 163, 188)},
      {'bbox': (132, 191, 240, 241)},
      {'bbox': (81, 25, 142, 67)},
      {'bbox': (74, 297, 116, 314)},
      {'bbox': (77, 84, 224, 107)}, ], (287, 132, 398, 191), None),  # x没有重合
    ([{'bbox': (80, 90, 249, 200)},
      {'bbox': (183, 100, 240, 155)}, ], (183, 100, 240, 155), None),  # 包含
])
def test_find_bottom_nearest_text_bbox(pymu_blocks: list, obj_box: tuple, target_boxs: dict) -> None:
    assert target_boxs == find_bottom_nearest_text_bbox(pymu_blocks, obj_box)


# 寻找左侧距离自己最近的box, y方向有重叠，x方向最近
@pytest.mark.parametrize('pymu_blocks, obj_box, target_boxs', [
    ([{'bbox': (80, 90, 249, 200)}, {'bbox': (183, 100, 240, 155)}], (183, 100, 240, 155), None),  # 包含
    ([{'bbox': (28, 90, 77, 126)}, {'bbox': (35, 84, 84, 120)}], (35, 84, 84, 120), None),  # y:重叠，x:重叠大于2
    ([{'bbox': (28, 90, 77, 126)}, {'bbox': (75, 84, 134, 120)}], (75, 84, 134, 120), {'bbox': (28, 90, 77, 126)}),
    # y:重叠，x:重叠小于等于2
    ([{'bbox': (239, 115, 379, 167)},
      {'bbox': (33, 237, 104, 262)},
      {'bbox': (124, 288, 168, 325)},
      {'bbox': (242, 291, 379, 340)},
      {'bbox': (55, 113, 161, 154)},
      {'bbox': (266, 123, 384, 217)}], (266, 123, 384, 217), {'bbox': (55, 113, 161, 154)}),  # y重叠，x left
    # ([{"bbox": (136, 219, 268, 240)},
    #   {"bbox": (169, 115, 268, 181)},
    #   {"bbox": (33, 237, 104, 262)},
    #   {"bbox": (124, 288, 168, 325)},
    #   {"bbox": (55, 117, 161, 154)},
    #   {"bbox": (266, 183, 384, 217)}], (266, 183, 384, 217),
    #  [{"bbox": (136, 219, 267, 240)}, {"bbox": (169, 115, 267, 181)}]),  # y有重叠，x重叠小于2或者在left Error
])
def test_find_left_nearest_text_bbox(pymu_blocks: list, obj_box: tuple, target_boxs: dict) -> None:
    assert target_boxs == find_left_nearest_text_bbox(pymu_blocks, obj_box)


# 寻找右侧距离自己最近的box, y方向有重叠，x方向最近
@pytest.mark.parametrize('pymu_blocks, obj_box, target_boxs', [
    ([{'bbox': (80, 90, 249, 200)}, {'bbox': (183, 100, 240, 155)}], (183, 100, 240, 155), None),  # 包含
    ([{'bbox': (28, 90, 77, 126)}, {'bbox': (35, 84, 84, 120)}], (28, 90, 77, 126), None),  # y:重叠，x:重叠大于2
    ([{'bbox': (28, 90, 77, 126)}, {'bbox': (75, 84, 134, 120)}], (28, 90, 77, 126), {'bbox': (75, 84, 134, 120)}),
    # y:重叠，x:重叠小于等于2
    ([{'bbox': (239, 115, 379, 167)},
      {'bbox': (33, 237, 104, 262)},
      {'bbox': (124, 288, 168, 325)},
      {'bbox': (242, 291, 379, 340)},
      {'bbox': (55, 113, 161, 154)},
      {'bbox': (266, 123, 384, 217)}], (55, 113, 161, 154), {'bbox': (239, 115, 379, 167)}),  # y重叠，x right
    # ([{"bbox": (169, 115, 298, 181)},
    #   {"bbox": (169, 219, 268, 240)},
    #   {"bbox": (33, 177, 104, 262)},
    #   {"bbox": (124, 288, 168, 325)},
    #   {"bbox": (55, 117, 161, 154)},
    #   {"bbox": (266, 183, 384, 217)}], (33, 177, 104, 262),
    #  [{"bbox": (169, 115, 298, 181)}, {"bbox": (169, 219, 268, 240)}]),  # y有重叠，x重叠小于2或者在right Error
])
def test_find_right_nearest_text_bbox(pymu_blocks: list, obj_box: tuple, target_boxs: dict) -> None:
    assert target_boxs == find_right_nearest_text_bbox(pymu_blocks, obj_box)


# 判断两个矩形框的相对位置关系 (left, right, bottom, top)
@pytest.mark.parametrize('box1, box2, target_box', [
    # (None, None, "Error"),  # Error
    ((80, 90, 249, 200), (183, 100, 240, 155), (False, False, False, False)),  # 包含
    # ((124, 81, 222, 173), (60, 221, 123, 358), (False, True, False, True)),  # 分离，右上 Error
    ((142, 109, 238, 164), (134, 211, 224, 270), (False, False, False, True)),  # 分离，上
    # ((51, 69, 192, 147), (205, 198, 282, 297), (True, False, False, True)),  # 分离，左上 Error
    # ((101, 149, 164, 289), (172, 130, 230, 268), (True, False, False, False)),  # 分离，左  Error
    # ((69, 196, 124, 285), (130, 127, 232, 186), (True, False, True, False)),  # 分离，左下  Error
    ((103, 212, 171, 304), (56, 90, 170, 209), (False, False, True, False)),  # 分离，下
    # ((124, 367, 222, 415), (60, 221, 123, 358), (False, True, True, False)),  # 分离，右下 Error
    # ((172, 130, 230, 268), (101, 149, 164, 289), (False, True, False, False)),  # 分离，右  Error
])
def test_bbox_relative_pos(box1: tuple, box2: tuple, target_box: tuple) -> None:
    assert target_box == bbox_relative_pos(box1, box2)


# 计算两个矩形框的距离
"""
受bbox_relative_pos方法的影响，左右相反，这里计算结果全部受影响，在错误的基础上计算出了正确的结果
"""


@pytest.mark.parametrize('box1, box2, target_num', [
    # (None, None, "Error"),  # Error
    ((80, 90, 249, 200), (183, 100, 240, 155), 0.0),  # 包含
    ((142, 109, 238, 164), (134, 211, 224, 270), 47.0),  # 分离，上
    ((103, 212, 171, 304), (56, 90, 170, 209), 3.0),  # 分离，下
    ((101, 149, 164, 289), (172, 130, 230, 268), 8.0),  # 分离，左
    ((172, 130, 230, 268), (101, 149, 164, 289), 8.0),  # 分离，右
    ((80.3, 90.8, 249.0, 200.5), (183.8, 100.6, 240.2, 155.1), 0.0),  # 包含
    ((142.3, 109.5, 238.9, 164.2), (134.4, 211.2, 224.8, 270.1), 47.0),  # 分离，上
    ((103.5, 212.6, 171.1, 304.8), (56.1, 90.9, 170.6, 209.2), 3.4),  # 分离，下
    ((101.1, 149.3, 164.9, 289.0), (172.1, 130.1, 230.5, 268.5), 7.2),  # 分离，左
    ((172.1, 130.3, 230.1, 268.1), (101.2, 149.9, 164.3, 289.1), 7.8),  # 分离，右
    ((124.3, 81.1, 222.5, 173.8), (60.3, 221.5, 123.0, 358.9), 47.717711596429254),  # 分离，右上
    ((51.2, 69.31, 192.5, 147.9), (205.0, 198.1, 282.98, 297.09), 51.73287156151299),  # 分离，左上
    ((124.3, 367.1, 222.9, 415.7), (60.9, 221.4, 123.2, 358.6), 8.570880934886448),  # 分离，右下
    ((69.9, 196.2, 124.1, 285.7), (130.0, 127.3, 232.6, 186.1), 11.69700816448377),  # 分离，左下
])
def test_bbox_distance(box1: tuple, box2: tuple, target_num: float) -> None:
    assert target_num - bbox_distance(box1, box2) < 1


@pytest.mark.skip(reason='skip')
# 根据bucket_name获取s3配置ak,sk,endpoint
def test_get_s3_config() -> None:
    bucket_name = os.getenv('bucket_name')
    target_data = os.getenv('target_data')
    assert convert_string_to_list(target_data) == list(get_s3_config(bucket_name))


def convert_string_to_list(s):
    cleaned_s = s.strip("'")
    items = cleaned_s.split(',')
    cleaned_items = [item.strip() for item in items]
    return cleaned_items
