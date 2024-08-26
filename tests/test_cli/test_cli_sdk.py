import pytest
import os
from conf import conf
import os
import json
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from lib import common
import logging
pdf_res_path = conf.conf["pdf_res_path"]
code_path = conf.conf["code_path"]
pdf_dev_path = conf.conf["pdf_dev_path"]
class TestCli:
    """
    test cli
    """
    def test_pdf_auto_sdk(self):
        """
        pdf sdk auto test
        """
        demo_names = list()
        pdf_path = os.path.join(pdf_dev_path, "pdf")
        for pdf_file in os.listdir(pdf_path):
            if pdf_file.endswith('.pdf'):
                demo_names.append(pdf_file.split('.')[0])
        for demo_name in demo_names:
            pdf_path = os.path.join(pdf_dev_path, "pdf", f"{demo_name}.pdf")
            pdf_bytes = open(pdf_path, "rb").read()
            image_writer = DiskReaderWriter(pdf_dev_path)
            image_dir = str(os.path.basename(pdf_dev_path))
            jso_useful_key = {"_pdf_type": "", "model_list": []}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            pipe.pipe_classify()
            pipe.pipe_parse()
            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
            dir_path = os.path.join(pdf_dev_path, "mineru")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            res_path = os.path.join(dir_path, f"{demo_name}.md")
            common.delete_file(res_path)
            with open(res_path, "w+", encoding="utf-8") as f:
                f.write(md_content)
            common.count_folders_and_check_contents(res_path)
        
    def test_pdf_ocr_sdk(self):
        """
        pdf sdk ocr test
        """
        demo_names = list()
        pdf_path = os.path.join(pdf_dev_path, "pdf")
        for pdf_file in os.listdir(pdf_path):
            if pdf_file.endswith('.pdf'):
                demo_names.append(pdf_file.split('.')[0])
        for demo_name in demo_names:
            pdf_path = os.path.join(pdf_dev_path, "pdf", f"{demo_name}.pdf")
            pdf_bytes = open(pdf_path, "rb").read()
            image_writer = DiskReaderWriter(pdf_dev_path)
            image_dir = str(os.path.basename(pdf_dev_path))
            jso_useful_key = {"_pdf_type": "ocr", "model_list": []}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            pipe.pipe_classify()
            pipe.pipe_parse()
            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
            dir_path = os.path.join(pdf_dev_path, "mineru")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            res_path = os.path.join(dir_path, f"{demo_name}.md")
            common.delete_file(res_path)
            with open(res_path, "w+", encoding="utf-8") as f:
                f.write(md_content)
            common.count_folders_and_check_contents(res_path)


    def test_pdf_txt_sdk(self):
        """
        pdf sdk txt test
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
            jso_useful_key = {"_pdf_type": "txt", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            pipe.pipe_classify()
            pipe.pipe_parse()
            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
            dir_path = os.path.join(pdf_dev_path, "mineru")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            res_path = os.path.join(dir_path, f"{demo_name}.md")
            common.delete_file(res_path)
            with open(res_path, "w+", encoding="utf-8") as f:
                f.write(md_content)
            common.count_folders_and_check_contents(res_path)


    def test_pdf_cli_auto(self):
        """
        magic_pdf cli test auto
        """
        demo_names = []
        pdf_path = os.path.join(pdf_dev_path, "pdf")
        for pdf_file in os.listdir(pdf_path):
            if pdf_file.endswith('.pdf'):
                demo_names.append(pdf_file.split('.')[0])
        for demo_name in demo_names:
            res_path = os.path.join(pdf_dev_path, "mineru")
            common.delete_file(res_path)
            cmd = "magic-pdf -p %s -o %s -m %s" % (os.path.join(pdf_path, f"{demo_name}.pdf"), res_path, "auto")
            logging.info(cmd)
            os.system(cmd)
            common.count_folders_and_check_contents(res_path)


    def test_pdf_clit_txt(self):
        """
        magic_pdf cli test txt
        """
        demo_names = []
        pdf_path = os.path.join(pdf_dev_path, "pdf")
        for pdf_file in os.listdir(pdf_path):
            if pdf_file.endswith('.pdf'):
                demo_names.append(pdf_file.split('.')[0])
        for demo_name in demo_names:
            res_path = os.path.join(pdf_dev_path, "mineru")
            common.delete_file(res_path)
            cmd = "magic-pdf -p %s -o %s -m %s" % (os.path.join(pdf_path, f"{demo_name}.pdf"), res_path, "txt")
            logging.info(cmd)
            os.system(cmd)
            common.count_folders_and_check_contents(res_path)



    def test_pdf_clit_ocr(self):
        """
        magic_pdf cli test ocr
        """
        demo_names = []
        pdf_path = os.path.join(pdf_dev_path, "pdf")
        for pdf_file in os.listdir(pdf_path):
            if pdf_file.endswith('.pdf'):
                demo_names.append(pdf_file.split('.')[0])
        for demo_name in demo_names:
            res_path = os.path.join(pdf_dev_path, "mineru")
            common.delete_file(res_path)
            cmd = "magic-pdf -p %s -o %s -m %s" % (os.path.join(pdf_path, f"{demo_name}.pdf"), res_path, "ocr")
            logging.info(cmd)
            os.system(cmd)
            common.count_folders_and_check_contents(res_path)



 
if __name__ == "__main__":
    pytest.main() 
