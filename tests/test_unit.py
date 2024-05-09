import pytest

from magic_pdf.libs.boxbase import _is_in_or_part_overlap, _is_in_or_part_overlap_with_area_ratio
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
        # ("file2.txt", "ValueError")
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
    def test_is_in_or_part_overlap(self, box1: list, box2: list, target_bool: bool) -> None:
        """
        box1: 坐标数组
        box2: 坐标数组
        """
        assert target_bool == _is_in_or_part_overlap(box1, box2)

    @pytest.mark.parametrize("box1, box2, target_bool", [
        # ((35, 28, 108, 90), (47, 60, 83, 96), True),  # 包含 box1 up box2,  box2 多半,box1少半
        # ((65, 151, 92, 177), (49, 99, 105, 198), True),  # 包含 box1 in box2
        # ((80, 62, 112, 84), (74, 40, 144, 111), True),  # 包含 box1 in box2
        # ((65, 88, 127, 144), (92, 102, 131, 139), True),  # 包含 box2 多半，box1约一半 异常
        # ((92, 102, 131, 139), (65, 88, 127, 144), True),  # 包含 box1 多半
        # ((100, 93, 199, 168), (169, 126, 198, 165), True),  # 包含 box2 in box1  异常
        # ((26, 75, 106, 172), (65, 108, 90, 128), True),  # 包含 box2 in box1
        # ((28, 90, 77, 126), (35, 84, 84, 120), True),  # 相交 box1多半，box2多半
        # ((37, 6, 69, 52), (28, 3, 60, 49), True),  # 相交 box1多半，box2多半
        ((94, 29, 133, 60), (84, 30, 123, 61), True),  # 相交 box1多半，box2多半
    ])
    def test_is_in_or_part_overlap_with_area_ratio(self, box1: list, box2: list, target_bool: bool) -> None:
        out_bool = _is_in_or_part_overlap_with_area_ratio(box1, box2)
        assert target_bool == out_bool
