'''
Author: FutureMeng be_loving@163.com
Date: 2024-11-20 11:10:37
LastEditors: FutureMeng be_loving@163.com
LastEditTime: 2024-11-20 20:21:22
FilePath: \MinerU\services\fastapi\app\magic_pdf_parse_util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json
import datetime
import shutil

from loguru import logger

from magic_pdf.libs.draw_bbox import draw_layout_bbox, draw_span_bbox
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

from . import redis_util

def pdf_parse(
        md5_value,
        pdf_bytes: bytes,
        parse_method: str = 'auto',
        model_json_path: str = None,
        output_dir: str = None
):
    """
    执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到 pdf 文件所在的目录
    :param parse_method: 解析方法， 共 auto、ocr、txt 三种，默认 auto，如果效果不好，可以尝试 ocr
    :param model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
    :param is_json_md_dump: 是否将解析后的数据写入到 .json 和 .md 文件中，默认 True，会将不同阶段的数据写入到不同的 .json 文件中（共3个.json文件），md内容会保存到 .md 文件中
    :param output_dir: 输出结果的目录地址，会生成一个以 pdf 文件名命名的文件夹并保存所有结果
    """
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

        # 获取图片的父路径，为的是以相对路径保存到 .md 和 conent_list.json 文件中
        image_path_parent = os.path.basename(output_image_path)

        if model_json_path:
            # 读取已经被模型解析后的pdf文件的 json 原始数据，list 类型
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # 执行解析步骤
        image_writer = DiskReaderWriter(output_image_path)

        # 选择解析方式
        # jso_useful_key = {"_pdf_type": "", "model_list": model_json}
        # pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        if parse_method == "auto":
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            shutil.rmtree(output_path)
            redis_util.set_parse_failed(md5_value)
            logger.error("unknown parse method, only auto, ocr, txt allowed")
            return

        # 执行分类
        pipe.pipe_classify()

        # 如果没有传入模型数据，则使用内置模型解析
        if not model_json:
            pipe.pipe_analyze()  # 解析

        # 执行解析
        pipe.pipe_parse()

        # 保存 text 和 md 格式的结果
        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

        # delete fold
        shutil.rmtree(output_path)
        redis_util.set_parse_parsed(md5_value, content_list, md_content)

    except Exception as e:
        redis_util.set_parse_failed(md5_value)
        logger.exception(e)
