import pytest
import os
from conf import conf
import os
import json
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from lib import common

pdf_res_path = conf.conf["pdf_res_path"]
code_path = conf.conf["code_path"]
pdf_dev_path = conf.conf["pdf_dev_path"]
class TestCli:
    """
    test cli
    """
    def test_pdf_sdk(self):
        """
        pdf sdk 方式解析
        """
        demo_names = list()
        pdf_path = os.path.join(pdf_dev_path, "pdf")
        for pdf_file in os.listdir(pdf_path):
            if pdf_file.endswith('.pdf'):
                demo_names.append(pdf_file.split('.')[0])
        for demo_name in demo_names:
            model_path = os.path.join(pdf_dev_path, f"{demo_name}_model.json")
            pdf_path = os.path.join(pdf_dev_path, "pdf", f"{demo_name}.pdf")
            pdf_bytes = open(pdf_path, "rb").read()
            model_json = json.loads(open(model_path, "r", encoding="utf-8").read())
            image_writer = DiskReaderWriter(pdf_dev_path)
            image_dir = str(os.path.basename(pdf_dev_path))
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            pipe.pipe_classify()
            pipe.pipe_parse()
            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
            dir_path = os.path.join(pdf_dev_path, "mineru")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            res_path = os.path.join(dir_path, f"{demo_name}.md")
            with open(res_path, "w+", encoding="utf-8") as f:
                f.write(md_content)
            common.count_folders_and_check_contents(res_path)
        
    # def test_pdf_specify_jsonl(self):
    #     """
    #     输入jsonl, 默认方式解析
    #     """
    #     cmd = "cd %s && export PYTHONPATH=. && python magic_pdf/cli/magicpdf.py  json-command --json 's3://llm-process-pperf/ebook_index_textbook_40k/中高考&竞赛知识点/part-663f1ef5e7c1-009416.jsonl?bytes=0,1133972'" % (code_path)
    #     logging.info(cmd)
    #     common.check_shell(cmd)
    #     #common.count_folders_and_check_contents(pdf_res_path)

    # def test_pdf_specify_jsonl_txt(self):
    #     """
    #     输入jsonl, txt方式解析
    #     """
    #     cmd = "cd %s && export PYTHONPATH=. && python magic_pdf/cli/magicpdf.py  json-command --json 's3://llm-process-pperf/ebook_index_textbook_40k/中高考&竞赛知识点/part-663f1ef5e7c1-009416.jsonl?bytes=0,1133972' --method txt" % (code_path)
    #     logging.info(cmd)
    #     common.check_shell(cmd)
    #     #common.count_folders_and_check_contents(pdf_res_path)
    #
    # def test_pdf_specify_jsonl_ocr(self):
    #     """
    #     输入jsonl, ocr方式解析
    #     """
    #     cmd = "cd %s && export PYTHONPATH=. && python magic_pdf/cli/magicpdf.py  json-command --json 's3://llm-process-pperf/ebook_index_textbook_40k/中高考&竞赛知识点/part-663f1ef5e7c1-009416.jsonl?bytes=0,1133972' --method ocr" % (code_path)
    #     logging.info(cmd)
    #     common.check_shell(cmd)
    #     #common.count_folders_and_check_contents(pdf_res_path)
 
 
if __name__ == "__main__":
    pytest.main() 
