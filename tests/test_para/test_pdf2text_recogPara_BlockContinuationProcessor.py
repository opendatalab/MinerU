import unittest

from magic_pdf.post_proc.detect_para import BlockContinuationProcessor

# from ... pdf2text_recogPara import BlockContinuationProcessor # another way to import

"""
Execute the following command to run the test under directory code-clean:

    python -m tests.test_para.test_pdf2text_recogPara_ClassName
    
    or
    
    pytest -v -s app/pdf_toolbox/tests/test_para/test_pdf2text_recogPara_BlockContinuationProcessor.py
    
"""


class TestIsAlphabetChar(unittest.TestCase):
    def setUp(self):
        self.obj = BlockContinuationProcessor()

    def test_is_alphabet_char(self):
        char = "A"
        result = self.obj._is_alphabet_char(char)
        self.assertTrue(result)

    def test_is_not_alphabet_char(self):
        char = "1"
        result = self.obj._is_alphabet_char(char)
        self.assertFalse(result)


class TestIsChineseChar(unittest.TestCase):
    def setUp(self):
        self.obj = BlockContinuationProcessor()

    def test_is_chinese_char(self):
        char = "中"
        result = self.obj._is_chinese_char(char)
        self.assertTrue(result)

    def test_is_not_chinese_char(self):
        char = "A"
        result = self.obj._is_chinese_char(char)
        self.assertFalse(result)


class TestIsOtherLetterChar(unittest.TestCase):
    def setUp(self):
        self.obj = BlockContinuationProcessor()

    def test_is_other_letter_char(self):
        char = "Ä"
        result = self.obj._is_other_letter_char(char)
        self.assertTrue(result)

    def test_is_not_other_letter_char(self):
        char = "A"
        result = self.obj._is_other_letter_char(char)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
