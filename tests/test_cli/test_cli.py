import pytest
import os
from conf import conf
import subprocess
from lib import common
import logging
pdf_res_path = conf.conf["pdf_res_path"]
code_path = conf.conf["code_path"]
pdf_dev_path = conf.conf["pdf_dev_path"]
class TestCli:
   
    def test_pdf_specify_dir(self):
        """
        输入pdf和指定目录的模型结果
        """
        cmd = 'cd %s && export PYTHONPATH=. && find %s -type f -name "*.pdf" | xargs -I{} python magic_pdf/cli/magicpdf.py  pdf-command  --pdf {}' % (code_path, pdf_dev_path)
        logging.info(cmd)
        common.check_shell(cmd)
        common.count_folders_and_check_contents(pdf_res_path)      
   

    def test_pdf_specify_jsonl(self):
        """
        输入jsonl, 默认方式解析
        """
        cmd = "cd %s && export PYTHONPATH=. && python magic_pdf/cli/magicpdf.py  json-command --json 's3://llm-process-pperf/ebook_index_textbook_40k/中高考&竞赛知识点/part-663f1ef5e7c1-009416.jsonl?bytes=0,1133972'" % (code_path)
        logging.info(cmd)
        common.check_shell(cmd)
        common.count_folders_and_check_contents(pdf_res_path)

    def test_pdf_specify_jsonl_txt(self):
        """
        输入jsonl, txt方式解析  
        """
        cmd = "cd %s && export PYTHONPATH=. && python magic_pdf/cli/magicpdf.py  json-command --json 's3://llm-process-pperf/ebook_index_textbook_40k/中高考&竞赛知识点/part-663f1ef5e7c1-009416.jsonl?bytes=0,1133972' --method txt" % (code_path)
        logging.info(cmd)
        common.check_shell(cmd)
        common.count_folders_and_check_contents(pdf_res_path)
    
    def test_pdf_specify_jsonl_ocr(self):
        """
        输入jsonl, ocr方式解析
        """
        cmd = "cd %s && export PYTHONPATH=. && python magic_pdf/cli/magicpdf.py  json-command --json 's3://llm-process-pperf/ebook_index_textbook_40k/中高考&竞赛知识点/part-663f1ef5e7c1-009416.jsonl?bytes=0,1133972' --method ocr" % (code_path)
        logging.info(cmd)
        common.check_shell(cmd)
        common.count_folders_and_check_contents(pdf_res_path)
 
 
if __name__ == "__main__":
    pytest.main() 
