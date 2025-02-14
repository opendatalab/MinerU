'''
Author: FutureMeng be_loving@163.com
Date: 2024-11-20 11:10:37
LastEditors: FutureMeng futuremeng@gmail.com
LastEditTime: 2025-02-14 09:06:54
FilePath: /MinerU/services/fastapi/app/magic_pdf_parse_util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json
import datetime
import shutil

from loguru import logger

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

from . import redis_util
from . import http_util

def pdf_parse(
        md5_value,
        pdf_bytes: bytes,
        parse_method: str = 'auto',
        cbUrl: str = None,
        cbkey: str = None,
        model_json_path: str = None,
        output_dir: str = None
):
    try:
        file_info = redis_util.get_file_info(md5_value)
        if not file_info:
            return
        if file_info["state"] != "waiting":
            return
        redis_util.set_parse_parsing(md5_value)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        foldname = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if output_dir:
            output_path = os.path.join(output_dir, foldname)
        else:
            output_path = os.path.join(current_script_dir, foldname)

        output_image_path = os.path.join(output_path, 'images')
        local_md_dir = os.path.join(output_path, 'output')

        # 获取图片的父路径，为的是以相对路径保存到 .md 和 conent_list.json 文件中
        image_path_parent = os.path.basename(output_image_path)

        if model_json_path:
            # 读取已经被模型解析后的pdf文件的 json 原始数据，list 类型
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # 执行解析步骤
        image_writer = DiskReaderWriter(output_image_path)

        ds = PymuDocDataset(pdf_bytes)

        ## inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)

            ## pipeline
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

            ## pipeline
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        ### draw model result on each page
        infer_result.draw_model(os.path.join(local_md_dir, f"model.pdf"))

        ### get model inference result
        model_inference_result = infer_result.get_infer_res()

        ### get markdown content
        md_content = pipe_result.get_markdown(image_dir)

        ### get content list content
        content_list = pipe_result.get_content_list(image_dir)

        # delete fold
        shutil.rmtree(output_path)
        redis_util.set_parse_parsed(md5_value, content_list, md_content)
        http_util.post_result_callback(cbUrl, cbkey, md5_value, content_list, md_content)
    except Exception as e:
        redis_util.set_parse_failed(md5_value)
        logger.exception(e)