import unittest

from magic_pdf.post_proc.detect_para import (
    is_bbox_overlap,
    is_in_bbox,
    is_line_right_aligned_from_neighbors,
    is_line_left_aligned_from_neighbors,
)

# from ... pdf2text_recogPara import * # another way to import

"""
Execute the following command to run the test under directory code-clean:

    python -m tests.test_para.test_pdf2text_recogPara_Common
    
    or 
    
    pytest -v -s app/pdf_toolbox/tests/test_para/test_pdf2text_recogPara_Common.py
    
"""


class TestIsBboxOverlap(unittest.TestCase):
    def test_overlap(self):
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        result = is_bbox_overlap(bbox1, bbox2)
        self.assertTrue(result)

    def test_no_overlap(self):
        bbox1 = [0, 0, 10, 10]
        bbox2 = [11, 11, 15, 15]
        result = is_bbox_overlap(bbox1, bbox2)
        self.assertFalse(result)

    def test_partial_overlap(self):
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        result = is_bbox_overlap(bbox1, bbox2)
        self.assertTrue(result)

    def test_same_bbox(self):
        bbox1 = [0, 0, 10, 10]
        bbox2 = [0, 0, 10, 10]
        result = is_bbox_overlap(bbox1, bbox2)
        self.assertTrue(result)


# Test is_in_bbox function
class TestIsInBbox(unittest.TestCase):
    def test_bbox1_in_bbox2(self):
        bbox1 = [0, 0, 10, 10]
        bbox2 = [0, 0, 20, 20]
        result = is_in_bbox(bbox1, bbox2)
        self.assertTrue(result)

    def test_bbox1_not_in_bbox2(self):
        bbox1 = [0, 0, 30, 30]
        bbox2 = [0, 0, 20, 20]
        result = is_in_bbox(bbox1, bbox2)
        self.assertFalse(result)

    def test_bbox1_equal_to_bbox2(self):
        bbox1 = [0, 0, 20, 20]
        bbox2 = [0, 0, 20, 20]
        result = is_in_bbox(bbox1, bbox2)
        self.assertTrue(result)

    def test_bbox1_partially_in_bbox2(self):
        bbox1 = [10, 10, 30, 30]
        bbox2 = [0, 0, 20, 20]
        result = is_in_bbox(bbox1, bbox2)
        self.assertFalse(result)


# Test is_line_right_aligned_from_neighbors function
class TestIsLineRightAlignedFromNeighbors(unittest.TestCase):
    def test_right_aligned_with_prev_line(self):
        curr_line_bbox = [0, 0, 100, 100]
        prev_line_bbox = [0, 0, 90, 100]
        next_line_bbox = None
        avg_char_width = 10
        direction = 0
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_right_aligned_with_next_line(self):
        curr_line_bbox = [0, 0, 100, 100]
        prev_line_bbox = None
        next_line_bbox = [0, 0, 110, 100]
        avg_char_width = 10
        direction = 1
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_right_aligned_with_both_lines(self):
        curr_line_bbox = [0, 0, 100, 100]
        prev_line_bbox = [0, 0, 90, 100]
        next_line_bbox = [0, 0, 110, 100]
        avg_char_width = 10
        direction = 2
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_not_right_aligned_with_prev_line(self):
        curr_line_bbox = [0, 0, 100, 100]
        prev_line_bbox = [0, 0, 80, 100]
        next_line_bbox = None
        avg_char_width = 10
        direction = 0
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_not_right_aligned_with_next_line(self):
        curr_line_bbox = [0, 0, 100, 100]
        prev_line_bbox = None
        next_line_bbox = [0, 0, 120, 100]
        avg_char_width = 10
        direction = 1
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_not_right_aligned_with_both_lines(self):
        curr_line_bbox = [0, 0, 100, 100]
        prev_line_bbox = [0, 0, 80, 100]
        next_line_bbox = [0, 0, 120, 100]
        avg_char_width = 10
        direction = 2
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_invalid_direction(self):
        curr_line_bbox = [0, 0, 100, 100]
        prev_line_bbox = None
        next_line_bbox = None
        avg_char_width = 10
        direction = 3
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)


# Test is_line_left_aligned_from_neighbors function
class TestIsLineLeftAlignedFromNeighbors(unittest.TestCase):

    def test_left_aligned_with_prev_line(self):
        curr_line_bbox = [10, 20, 30, 40]
        prev_line_bbox = [5, 20, 30, 40]
        next_line_bbox = None
        avg_char_width = 5.0
        direction = 0
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_left_aligned_with_next_line(self):
        curr_line_bbox = [10, 20, 30, 40]
        prev_line_bbox = None
        next_line_bbox = [15, 20, 30, 40]
        avg_char_width = 5.0
        direction = 1
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_left_aligned_with_both_lines(self):
        curr_line_bbox = [10, 20, 30, 40]
        prev_line_bbox = [5, 20, 30, 40]
        next_line_bbox = [15, 20, 30, 40]
        avg_char_width = 5.0
        direction = 2
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_not_left_aligned_with_prev_line(self):
        curr_line_bbox = [10, 20, 30, 40]
        prev_line_bbox = [5, 20, 30, 40]
        next_line_bbox = None
        avg_char_width = 5.0
        direction = 0
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_not_left_aligned_with_next_line(self):
        curr_line_bbox = [10, 20, 30, 40]
        prev_line_bbox = None
        next_line_bbox = [15, 20, 30, 40]
        avg_char_width = 5.0
        direction = 1
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_not_left_aligned_with_both_lines(self):
        curr_line_bbox = [10, 20, 30, 40]
        prev_line_bbox = [5, 20, 30, 40]
        next_line_bbox = [15, 20, 30, 40]
        avg_char_width = 5.0
        direction = 2
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)

    def test_invalid_direction(self):
        curr_line_bbox = [10, 20, 30, 40]
        prev_line_bbox = None
        next_line_bbox = None
        avg_char_width = 5.0
        direction = 3
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
