# Copyright (c) Opendatalab. All rights reserved.
"""表格内联对象与 OCR 结果处理工具函数的单元测试。"""
import numpy as np
import pytest

from mineru.utils.table_inline_utils import (
    encode_table_inline_image,
    extract_table_inline_objects,
    normalize_table_ocr_rec_text,
    sort_table_ocr_result,
)


class TestSortTableOcrResult:
    """测试表格 OCR 结果排序。"""

    def _make_ocr_result(self, coords):
        """根据 (x, y) 坐标列表构造 OCR 结果结构。

        ocr_result 中每个元素为 [box, text]，box 为 4 个角点坐标。
        排序只关注左上角 (box[0]) 的 (x, y)。
        """
        return [
            [[[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]], f"text_{i}"]
            for i, (x, y) in enumerate(coords)
        ]

    def test_sort_same_row_by_x(self):
        """同一行内按 x 升序排列。"""
        ocr_result = self._make_ocr_result([(100, 50), (20, 50), (60, 50)])
        sort_table_ocr_result(ocr_result)
        xs = [float(item[0][0][0]) for item in ocr_result]
        assert xs == [20.0, 60.0, 100.0]

    def test_sort_different_rows_by_y(self):
        """不同行按 y 分组并排序。"""
        ocr_result = self._make_ocr_result([(10, 100), (5, 10), (8, 50)])
        sort_table_ocr_result(ocr_result)
        ys = [int(item[0][0][1]) for item in ocr_result]
        assert ys == [10, 50, 100]

    def test_sort_mixed_rows_with_tolerance(self):
        """y 差值在 10px 容差内视为同一行，行内按 x 排序。"""
        ocr_result = self._make_ocr_result([(100, 10), (20, 15), (60, 12)])
        sort_table_ocr_result(ocr_result)
        xs = [float(item[0][0][0]) for item in ocr_result]
        assert xs == [20.0, 60.0, 100.0]

    def test_sort_empty_result(self):
        """空列表不应报错。"""
        ocr_result = []
        sort_table_ocr_result(ocr_result)
        assert ocr_result == []


class TestNormalizeTableOcrRecText:
    """测试表格 OCR 文本规范化。"""

    def test_single_char_replacement(self):
        assert normalize_table_ocr_rec_text("香") == "否"

    def test_regex_replacement(self):
        assert normalize_table_ocr_rec_text("5號") == "5"

    def test_no_change(self):
        assert normalize_table_ocr_rec_text("hello") == "hello"

    def test_non_string_input(self):
        assert normalize_table_ocr_rec_text(123) == 123


class TestExtractTableInlineObjects:
    """测试表格内联对象提取。"""

    @pytest.fixture
    def blank_image(self):
        """创建一张 200x200 的白色图片。"""
        return np.full((200, 200, 3), 255, dtype=np.uint8)

    def test_image_inside_table(self, blank_image, monkeypatch):
        """位于表格内部的图片应被提取为内联对象。"""
        monkeypatch.setattr(
            "mineru.utils.table_inline_utils.encode_table_inline_image",
            lambda _img, _bbox: "data:image/png;base64,xxx",
        )

        table = {"label": "table", "bbox": [10, 10, 150, 150], "score": 0.9}
        image = {"label": "image", "bbox": [30, 30, 80, 80], "score": 0.8}
        layout_res = [table, image]

        result = extract_table_inline_objects(layout_res, blank_image, formula_enable=False)

        assert len(result) == 1
        assert len(layout_res) == 1  # image 已从 layout_res 中移除
        assert layout_res[0]["label"] == "table"
        inline_items = list(result.values())[0]
        assert len(inline_items) == 1
        assert inline_items[0]["kind"] == "image"
        assert inline_items[0]["score"] == 0.8

    def test_image_outside_table(self, blank_image, monkeypatch):
        """位于表格外部的图片不应被提取。"""
        monkeypatch.setattr(
            "mineru.utils.table_inline_utils.encode_table_inline_image",
            lambda _img, _bbox: "data:image/png;base64,xxx",
        )

        table = {"label": "table", "bbox": [10, 10, 50, 50], "score": 0.9}
        image = {"label": "image", "bbox": [100, 100, 150, 150], "score": 0.8}
        layout_res = [table, image]

        result = extract_table_inline_objects(layout_res, blank_image, formula_enable=False)

        # 函数会对每个 table 返回空列表条目
        assert len(result) == 1
        assert list(result.values())[0] == []
        assert len(layout_res) == 2  # image 未被移除

    def test_formula_inside_table(self, blank_image):
        """位于表格内部的公式应被提取为内联对象。"""
        table = {"label": "table", "bbox": [10, 10, 150, 150], "score": 0.9}
        formula = {"label": "inline_formula", "bbox": [30, 30, 80, 80], "latex": "x^2", "score": 0.85}
        layout_res = [table, formula]

        result = extract_table_inline_objects(layout_res, blank_image, formula_enable=True)

        assert len(result) == 1
        inline_items = list(result.values())[0]
        assert len(inline_items) == 1
        assert inline_items[0]["kind"] == "formula"
        assert inline_items[0]["content"] == "<eq>x^2</eq>"

    def test_formula_disabled(self, blank_image):
        """formula_enable=False 时不应提取公式。"""
        table = {"label": "table", "bbox": [10, 10, 150, 150], "score": 0.9}
        formula = {"label": "inline_formula", "bbox": [30, 30, 80, 80], "latex": "x^2", "score": 0.85}
        layout_res = [table, formula]

        result = extract_table_inline_objects(layout_res, blank_image, formula_enable=False)

        assert len(result) == 1
        assert list(result.values())[0] == []
        assert len(layout_res) == 2

    def test_no_table(self, blank_image):
        """没有表格时应返回空字典。"""
        image = {"label": "image", "bbox": [10, 10, 50, 50], "score": 0.8}
        layout_res = [image]

        result = extract_table_inline_objects(layout_res, blank_image, formula_enable=False)

        assert result == {}
        assert len(layout_res) == 1


class TestEncodeTableInlineImage:
    """测试表格内联图片编码。"""

    def test_encode_valid_bbox(self):
        """有效 bbox 应返回 base64 图片字符串。"""
        np_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = encode_table_inline_image(np_img, [10, 10, 50, 50])
        assert result.startswith("data:image/jpg;base64,")
        assert len(result) > len("data:image/jpg;base64,")

    def test_encode_empty_crop(self):
        """无效 bbox 应返回空字符串。"""
        np_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert encode_table_inline_image(np_img, [10, 10, 10, 50]) == ""
        assert encode_table_inline_image(np_img, [200, 200, 300, 300]) == ""
