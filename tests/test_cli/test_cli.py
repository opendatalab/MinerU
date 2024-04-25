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
        输入jsonl
        """
        cmd = "cd %s && export PYTHONPATH=. && python " 

 

if __name__ == "__main__":
    pytest.main() 
