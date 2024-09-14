import os
import json
import fire

from loguru import logger

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
import concurrent.futures
import time

import magic_pdf.model as model_config 
model_config.__use_inside_model__ = True



def process(pdf_list):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    pid = os.getpid()
    for index, pdf_path in enumerate(pdf_list):
        if index == 1:
            # We start to record time after the pipe processes the first pdf,
            # since it creates model instances when processing the first one,
            # and creating model instances is very time consuming
            start = time.perf_counter()
        try:
            pdf_bytes = open(pdf_path, "rb").read()
            model_json = []  # model_json传空list使用内置模型解析
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            local_image_dir = os.path.join(current_script_dir, str(pid), 'images')
            image_dir = str(os.path.basename(local_image_dir))
            image_writer = DiskReaderWriter(local_image_dir)
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            pipe.pipe_classify()
            """如果没有传入有效的模型数据，则使用内置model解析"""
            if len(model_json) == 0:
                if model_config.__use_inside_model__:
                    pipe.pipe_analyze()
                else:
                    logger.error("need model list input")
                    exit(1)
            pipe.pipe_parse()
            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
            # TODO: postprocessing md_content
        except Exception as e:
            logger.exception(e)
    end = time.perf_counter()
    logger.warning(f'>>>>> {pid} spent {(end-start):2f} s on processing {len(pdf_list) - 1} PDFs')


def file_ext(file_path):
    return os.path.splitext(file_path)[-1]


def main(file_path: str, max_proc_num: int=4):
    with open(file_path) as f:
        file_paths = f.readlines()
        pdf_file_list = [_path.strip() for _path in file_paths if file_ext(_path.strip()) == '.pdf']
        proc_file_list = []
        files_per_proc = (len(pdf_file_list) + max_proc_num - 1) // max_proc_num
        for i in range(0, len(pdf_file_list), files_per_proc):
            file_list_per_proc = pdf_file_list[i : i + files_per_proc]
            proc_file_list.append(file_list_per_proc)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(process, proc_file_list)


if __name__ == '__main__':
    fire.Fire(main)