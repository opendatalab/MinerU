import pytest

from magic_pdf.libs.boxbase import _is_in_or_part_overlap, _is_in_or_part_overlap_with_area_ratio, _is_in, \
    _is_part_overlap, _left_intersect, _right_intersect, _is_vertical_full_overlap, _is_bottom_full_overlap, \
    _is_left_overlap
from magic_pdf.libs.commons import mymax, join_path, get_top_percent_list
from magic_pdf.libs.path_utils import parse_s3path


class Testpy():
    # 输入一个列表，如果列表空返回0，否则返回最大元素
    @pytest.mark.parametrize("list_input, target_num",
                             [
                                 # ([0, 0, 0, 0], 0),
                                 # ([0], 0),
                                 # ([1, 2, 5, 8, 4], 8),
                                 # ([], 0),
                                 # ([1.1, 7.6, 1.009, 9.9], 9.9),
                                 ([1.0 * 10 ** 2, 3.5 * 10 ** 3, 0.9 * 10 ** 6], 0.9 * 10 ** 6),
                             ])
    def test_list_max(self, list_input: list, target_num) -> None:
        """
        list_input: 输入列表元素，元素均为数字类型
        """
        assert target_num == mymax(list_input)

    # 连接多个参数生成路径信息，使用"/"作为连接符，生成的结果需要是一个合法路径
    @pytest.mark.parametrize("path_input, target_path", [
        # (['https:', '', 'www.baidu.com'], 'https://www.baidu.com'),
        # (['https:', 'www.baidu.com'], 'https:/www.baidu.com'),
        (['D:', 'file', 'pythonProject', 'demo' + '.py'], 'D:/file/pythonProject/demo.py'),
    ])
    def test_join_path(self, path_input: list, target_path: str) -> None:
        """
        path_input: 输入path的列表，列表元素均为字符串
        """
        assert target_path == join_path(*path_input)

    # 获取列表中前百分之多少的元素
    @pytest.mark.parametrize("num_list, percent, target_num_list", [
        # ([], 0.75, []),
        # ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11], 0.8, [23, 9, 7, 3, 0, -1, -5, -7]),
        # ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11], 0, []),
        ([-5, -10, 9, 3, 7, -7, 0, 23, -1, -11, 28], 0.8, [28, 23, 9, 7, 3, 0, -1, -5])
    ])
    def test_get_top_percent_list(self, num_list: list, percent: float, target_num_list: list) -> None:
        """
        num_list: 数字列表，列表元素为数字
        percent: 占比，float, 向下取证
        """
        assert target_num_list == get_top_percent_list(num_list, percent)

    # 输入一个s3路径，返回bucket名字和其余部分(key)
    @pytest.mark.parametrize("s3_path, target_data", [
        # ("s3://bucket/path/to/my/file.txt", "bucket"),
        # ("/path/to/my/file1.txt", "path"),
        ("bucket/path/to/my/file2.txt", "bucket"),
        # ("file2.txt", "False")
    ])
    def test_parse_s3path(self, s3_path: str, target_data: str):
        """
        s3_path: s3路径
            如果为无效路径，则返回对应的bucket名字和其余部分
            如果为异常路径 例如：file2.txt，则报异常
        """
        out_keys = parse_s3path(s3_path)
        assert target_data == out_keys[0]

    # 2个box是否处于包含或者部分重合关系。
    # 如果某边界重合算重合。
    # 部分边界重合，其他在内部也算包含
    @pytest.mark.parametrize("box1, box2, target_bool", [
        ((120, 133, 223, 248), (128, 168, 269, 295), True),
        # ((137, 53, 245, 157), (134, 11, 200, 147), True),  # 部分重合
        # ((137, 56, 211, 116), (140, 66, 202, 199), True),  # 部分重合
        # ((42, 34, 69, 65), (42, 34, 69, 65), True),  # 部分重合
        # ((39, 63, 87, 106), (37, 66, 85, 109), True),  # 部分重合
        # ((13, 37, 55, 66), (7, 46, 49, 75), True),  # 部分重合
        # ((56, 83, 85, 104), (64, 85, 93, 106), True),  # 部分重合
        # ((12, 53, 48, 94), (14, 53, 50, 94), True),  # 部分重合
        # ((43, 54, 93, 131), (55, 82, 77, 106), True),  # 包含
        # ((63, 2, 134, 71), (72, 43, 104, 78), True),  # 包含
        # ((25, 57, 109, 127), (26, 73, 49, 95), True),  # 包含
        # ((24, 47, 111, 115), (34, 81, 58, 106), True),  # 包含
        # ((34, 8, 105, 83), (76, 20, 116, 45), True),  # 包含
    ])
    def test_is_in_or_part_overlap(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        """
        box1: 坐标数组
        box2: 坐标数组
        """
        assert target_bool == _is_in_or_part_overlap(box1, box2)

    # 如果box1在box2内部，返回True
    #   如果是部分重合的，则重合面积占box1的比例大于阈值时候返回True
    @pytest.mark.parametrize("box1, box2, target_bool", [
        # ((35, 28, 108, 90), (47, 60, 83, 96), True),  # 包含 box1 up box2,  box2 多半,box1少半
        # ((65, 151, 92, 177), (49, 99, 105, 198), True),  # 包含 box1 in box2
        # ((80, 62, 112, 84), (74, 40, 144, 111), True),  # 包含 box1 in box2
        # ((65, 88, 127, 144), (92, 102, 131, 139), False),  # 包含 box2 多半，box1约一半
        # ((92, 102, 131, 139), (65, 88, 127, 144), True),  # 包含 box1 多半
        # ((100, 93, 199, 168), (169, 126, 198, 165), False),  # 包含 box2 in box1
        # ((26, 75, 106, 172), (65, 108, 90, 128), False),  # 包含 box2 in box1
        # ((28, 90, 77, 126), (35, 84, 84, 120), True),  # 相交 box1多半，box2多半
        # ((37, 6, 69, 52), (28, 3, 60, 49), True),  # 相交 box1多半，box2多半
        ((94, 29, 133, 60), (84, 30, 123, 61), True),  # 相交 box1多半，box2多半
    ])
    def test_is_in_or_part_overlap_with_area_ratio(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        out_bool = _is_in_or_part_overlap_with_area_ratio(box1, box2)
        assert target_bool == out_bool

    # box1在box2内部或者box2在box1内部返回True。如果部分边界重合也算作包含。
    @pytest.mark.parametrize("box1, box2, target_bool", [
        # ((), (), False),
        # ((65, 151, 92, 177), (49, 99, 105, 198), True),  # 包含 box1 in box2
        # ((80, 62, 112, 84), (74, 40, 144, 111), True),  # 包含 box1 in box2
        # ((76, 140, 154, 277), (121, 326, 192, 384), False),  # 分离
        # ((65, 88, 127, 144), (92, 102, 131, 139), False),  # 包含 box2 多半，box1约一半
        # ((92, 102, 131, 139), (65, 88, 127, 144), False),  # 包含 box1 多半
        # ((68, 94, 118, 120), (68, 90, 118, 122), True),  # 包含，box1 in box2 两边x相切
        # ((69, 94, 118, 120), (68, 90, 118, 122), True),  # 包含，box1 in box2 一边x相切
        ((69, 114, 118, 122), (68, 90, 118, 122), True),  # 包含，box1 in box2 一边y相切
        # ((100, 93, 199, 168), (169, 126, 198, 165), True),  # 包含 box2 in box1  Error
        # ((26, 75, 106, 172), (65, 108, 90, 128), True),  # 包含 box2 in box1  Error
        # ((38, 94, 122, 120), (68, 94, 118, 120), True),  # 包含，box2 in box1 两边y相切 Error
        # ((68, 34, 118, 158), (68, 94, 118, 120), True),  # 包含，box2 in box1 两边x相切 Error
        # ((68, 34, 118, 158), (68, 94, 84, 120), True),  # 包含，box2 in box1 一边x相切 Error
        # ((27, 94, 118, 158), (68, 94, 84, 120), True),  # 包含，box2 in box1 一边y相切 Error
    ])
    def test_is_in(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        assert target_bool == _is_in(box1, box2)

    # 仅仅是部分包含关系，返回True，如果是完全包含关系则返回False
    @pytest.mark.parametrize("box1, box2, target_bool", [
        ((65, 151, 92, 177), (49, 99, 105, 198), False),  # 包含 box1 in box2
        # ((80, 62, 112, 84), (74, 40, 144, 111), False),  # 包含 box1 in box2
        # ((76, 140, 154, 277), (121, 326, 192, 384), False),  # 分离  Error
        # ((76, 140, 154, 277), (121, 277, 192, 384), True),   # 外相切
        # ((65, 88, 127, 144), (92, 102, 131, 139), True),  # 包含 box2 多半，box1约一半
        # ((92, 102, 131, 139), (65, 88, 127, 144), True),  # 包含 box1 多半
        # ((68, 94, 118, 120), (68, 90, 118, 122), False),  # 包含，box1 in box2 两边x相切
        # ((69, 94, 118, 120), (68, 90, 118, 122), False),  # 包含，box1 in box2 一边x相切
        # ((69, 114, 118, 122), (68, 90, 118, 122), False),  # 包含，box1 in box2 一边y相切
        # ((26, 75, 106, 172), (65, 108, 90, 128), False),  # 包含 box2 in box1  Error
        # ((38, 94, 122, 120), (68, 94, 118, 120), False),  # 包含，box2 in box1 两边y相切 Error
        # ((68, 34, 118, 158), (68, 94, 84, 120), False),  # 包含，box2 in box1 一边x相切 Error

    ])
    def test_is_part_overlap(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        assert target_bool == _is_part_overlap(box1, box2)

    # left_box右侧是否和right_box左侧有部分重叠
    @pytest.mark.parametrize("box1, box2, target_bool", [
        # (None, None, False),
        # ((88, 81, 222, 173), (60, 221, 123, 358), False),  # 分离
        # ((121, 149, 184, 289), (172, 130, 230, 268), True),  # box1 left bottom box2 相交
        # ((172, 130, 230, 268),(121, 149, 184, 289),  False),  # box2 left bottom box1 相交
        # ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 top left box2 分离
        # ((117, 53, 222, 176), (174, 142, 298, 276), True),  # box1 left top box2 相交
        # ((174, 142, 298, 276), (117, 53, 222, 176), False),  # box2 left top box1 相交
        # ((65, 88, 127, 144), (92, 102, 131, 139), True),  # box1 left box2 y:box2 in box1
        # ((92, 102, 131, 139), (65, 88, 127, 144), False),  # box2 left box1 y:box1 in box2
        # ((182, 130, 230, 268), (121, 149, 174, 289), False),  # box2 left box1 分离
        ((1, 10, 26, 45), (3, 4, 20, 39), True),  # box1 bottom box2 x:box2 in box1
    ])
    def test_left_intersect(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        assert target_bool == _left_intersect(box1, box2)

    # left_box左侧是否和right_box右侧部分重叠
    @pytest.mark.parametrize("box1, box2, target_bool", [
        # (None, None, False),
        # ((88, 81, 222, 173), (60, 221, 123, 358), False),  # 分离
        # ((121, 149, 184, 289), (172, 130, 230, 268), False),  # box1 left bottom box2 相交
        # ((172, 130, 230, 268), (121, 149, 184, 289), True),  # box2 left bottom box1 相交
        # ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 top left box2 分离
        # ((117, 53, 222, 176), (174, 142, 298, 276), False),  # box1 left top box2 相交
        # ((174, 142, 298, 276), (117, 53, 222, 176), True),  # box2 left top box1 相交
        # ((65, 88, 127, 144), (92, 102, 131, 139), False),  # box1 left box2 y:box2 in box1
        # ((92, 102, 131, 139), (65, 88, 127, 144), True),  # box2 left box1 y:box1 in box2 Error
        ((182, 130, 230, 268), (121, 149, 174, 289), False),  # box2 left box1 分离
        # ((1, 10, 26, 45), (3, 4, 20, 39), False),  # box1 bottom box2 x:box2 in box1 Error
    ])
    def test_right_intersect(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        assert target_bool == _right_intersect(box1, box2)

    # x方向上：要么box1包含box2, 要么box2包含box1。不能部分包含
    # y方向上：box1和box2有重叠
    @pytest.mark.parametrize("box1, box2, target_bool", [
        # (None, None, False),  # Error
        # ((35, 28, 108, 90), (47, 60, 83, 96), True),  # box1 top box2, x:box2 in box1, y:有重叠
        # ((35, 28, 98, 90), (27, 60, 103, 96), True),  # box1 top box2, x:box1 in box2, y:有重叠
        # ((57, 77, 130, 210), (59, 219, 119, 293), False),  # box1 top box2, x: box2 in box1, y:无重叠
        # ((47, 60, 83, 96),(35, 28, 108, 90),  True),  # box2 top box1, x:box1 in box2, y:有重叠
        # ((27, 60, 103, 96), (35, 28, 98, 90), True),  # box2 top box1, x:box2 in box1, y:有重叠
        # ((59, 219, 119, 293), (57, 77, 130, 210), False),  # box2 top box1, x: box1 in box2, y:无重叠
        # ((35, 28, 55, 90), (57, 60, 83, 96), False),  # box1 top box2, x:无重叠, y:有重叠
        ((47, 60, 63, 96), (65, 28, 108, 90), False),  # box2 top box1, x:无重叠, y:有重叠
    ])
    def test_is_vertical_full_overlap(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        assert target_bool == _is_vertical_full_overlap(box1, box2)

    # 检查box1下方和box2的上方有轻微的重叠，轻微程度收到y_tolerance的限制
    @pytest.mark.parametrize("box1, box2, target_bool", [
        # (None, None, False),
        # ((35, 28, 108, 90), (47, 89, 83, 116), True),  # box1 top box2, y:有重叠
        # ((35, 28, 108, 90), (47, 60, 83, 96), False),  # box1 top box2, y:有重叠且过多
        # ((57, 77, 130, 210), (59, 219, 119, 293), False),  # box1 top box2, y:无重叠
        # ((47, 60, 83, 96), (35, 28, 108, 90), False),  # box2 top box1, y:有重叠且过多
        # ((27, 89, 103, 116), (35, 28, 98, 90), False),  # box2 top box1, y:有重叠
        ((59, 219, 119, 293), (57, 77, 130, 210), False),  # box2 top box1, y:无重叠
    ])
    def test_is_bottom_full_overlap(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        assert target_bool == _is_bottom_full_overlap(box1, box2)

    # 检查box1的左侧是否和box2有重叠
    @pytest.mark.parametrize("box1, box2, target_bool", [
        # (None, None, False),
        # ((88, 81, 222, 173), (60, 221, 123, 358), False),  # 分离
        # ((121, 149, 184, 289), (172, 130, 230, 268), False),  # box1 left bottom box2 相交  Error
        # ((172, 130, 230, 268), (121, 149, 184, 289), True),  # box2 left bottom box1 相交 Error
        # ((109, 68, 182, 146), (215, 188, 277, 253), False),  # box1 top left box2 分离
        # ((117, 53, 222, 176), (174, 142, 298, 276), False),  # box1 left top box2 相交
        # ((174, 142, 298, 276), (117, 53, 222, 176), True),  # box2 left top box1 相交  Error
        # ((65, 88, 127, 144), (92, 102, 131, 139), False),  # box1 left box2 y:box2 in box1 Error
        ((1, 10, 26, 45), (3, 4, 20, 39), True),  # box1 middle bottom box2 x:box2 in box1

    ])
    def test_is_left_overlap(self, box1: tuple, box2: tuple, target_bool: bool) -> None:
        assert target_bool == _is_left_overlap(box1, box2)

# python magicpdf.py --pdf  C:\Users\renpengli\Desktop\test\testpdf.pdf
