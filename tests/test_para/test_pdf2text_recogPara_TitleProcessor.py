import json
import unittest

from utils_for_test_para import UtilsForTestPara
from magic_pdf.post_proc.detect_para import TitleProcessor

# from ... pdf2text_recogPara import * # another way to import

"""
Execute the following command to run the test under directory code-clean:

    python -m tests.test_para.test_pdf2text_recogPara_ClassName
    
    or 
    
    pytest -v -s app/pdf_toolbox/tests/test_para/test_pdf2text_recogPara_TitleProcessor.py
    
"""


class TestTitleProcessor(unittest.TestCase):
    def setUp(self):
        self.title_processor = TitleProcessor()
        self.utils = UtilsForTestPara()
        self.preproc_out_jsons = self.utils.read_preproc_out_jfiles()

    def test_batch_process_blocks_detect_titles(self):
        """
        Test the function detect_titles with preprocessed output JSON
        """
        for preproc_out_json in self.preproc_out_jsons:
            with open(preproc_out_json, "r", encoding="utf-8") as f:
                preproc_dict = json.load(f)
                preproc_dict["statistics"] = {}
                result = self.title_processor.batch_detect_titles(preproc_dict)
                for page_id, blocks in preproc_dict.items():
                    if page_id.startswith("page_"):
                        pass
                    else:
                        continue

    def test_batch_process_blocks_recog_title_level(self):
        """
        Test the function batch_process_blocks_recog_title_level with preprocessed output JSON
        """
        for preproc_out_json in self.preproc_out_jsons:
            with open(preproc_out_json, "r", encoding="utf-8") as f:
                preproc_dict = json.load(f)
                preproc_dict["statistics"] = {}
                result = self.title_processor.batch_recog_title_level(preproc_dict)
                for page_id, blocks in preproc_dict.items():
                    if page_id.startswith("page_"):
                        pass
                    else:
                        continue


if __name__ == "__main__":
    unittest.main()
