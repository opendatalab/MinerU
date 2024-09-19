"""
test table case
"""
import os
import shutil
import json
from lib import calculate_score
import pytest
from conf import conf

code_path = os.environ.get('GITHUB_WORKSPACE')
pdf_dev_path = conf.conf["pdf_dev_path"]
pdf_res_path = conf.conf["pdf_res_path"]

class TestTable():
    """
    test table
    """
    def test_paddle_table_master_cuda(self):
        """
        select table: paddle table master,mode is cuda
        """
    def test_paddle_table_master_cpu(self):
        """
        select table: paddle table master, mode is cpu
        """
    def test_st_table_cuda(self):
        """
        select table: ST, mode is cuda 
        """

    def test_st_table_cpu(self):
        """
        select table: ST, mode is cpu
        """

    def test_close_table_cuda(self):
        """
        close table, mode is cuda
        """
    



def get_score():
    """
    get score
    """
    score = calculate_score.Scoring(os.path.join(pdf_dev_path, "result.json"))
    score.calculate_similarity_total("mineru", pdf_dev_path)
    res = score.summary_scores()
    return res


