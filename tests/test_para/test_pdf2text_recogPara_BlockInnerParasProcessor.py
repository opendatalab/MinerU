import unittest

from magic_pdf.post_proc.detect_para import BlockTerminationProcessor

# from ... pdf2text_recogPara import BlockInnerParasProcessor # another way to import

"""
Execute the following command to run the test under directory code-clean:

    python -m tests.test_para.test_pdf2text_recogPara_ClassName
    
    or
    
    pytest -v -s app/pdf_toolbox/tests/test_para/test_pdf2text_recogPara_BlockInnerParasProcessor.py
    
"""


class TestIsConsistentLines(unittest.TestCase):
    def setUp(self):
        self.obj = BlockTerminationProcessor()

    def test_consistent_with_prev_line(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = {"spans": [{"size": 12, "font": "Arial"}]}
        next_line = None
        consistent_direction = 0
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertTrue(result)

    def test_consistent_with_next_line(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = None
        next_line = {"spans": [{"size": 12, "font": "Arial"}]}
        consistent_direction = 1
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertTrue(result)

    def test_consistent_with_both_lines(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = {"spans": [{"size": 12, "font": "Arial"}]}
        next_line = {"spans": [{"size": 12, "font": "Arial"}]}
        consistent_direction = 2
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertTrue(result)

    def test_inconsistent_with_prev_line(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = {"spans": [{"size": 14, "font": "Arial"}]}
        next_line = None
        consistent_direction = 0
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertFalse(result)

    def test_inconsistent_with_next_line(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = None
        next_line = {"spans": [{"size": 14, "font": "Arial"}]}
        consistent_direction = 1
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertFalse(result)

    def test_inconsistent_with_both_lines(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = {"spans": [{"size": 14, "font": "Arial"}]}
        next_line = {"spans": [{"size": 14, "font": "Arial"}]}
        consistent_direction = 2
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertFalse(result)

    def test_invalid_consistent_direction(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = None
        next_line = None
        consistent_direction = 3
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertFalse(result)

    def test_possible_start_of_para(self):
        curr_line = {"bbox": (0, 0, 100, 10)}
        prev_line = {"bbox": (0, 20, 100, 30)}
        next_line = {"bbox": (0, 40, 100, 50)}
        X0 = 0
        X1 = 100
        avg_char_width = 5
        avg_font_size = 10

        result, _, _ = self.obj._is_possible_start_of_para(
            curr_line, prev_line, next_line, X0, X1, avg_char_width, avg_font_size
        )
        self.assertTrue(result)

    def test_not_possible_start_of_para(self):
        curr_line = {"bbox": (0, 0, 100, 10)}
        prev_line = {"bbox": (0, 20, 100, 30)}
        next_line = {"bbox": (0, 40, 100, 50)}
        X0 = 0
        X1 = 100
        avg_char_width = 5
        avg_font_size = 10

        result, _, _ = self.obj._is_possible_start_of_para(curr_line, prev_line, next_line, X0, X1, avg_char_width, avg_font_size)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
