import json
import os
import sys
import time

from loguru import logger
from pathlib import Path

from magic_pdf.libs.config_reader import get_s3_config_dict
from magic_pdf.pdf_parse_by_ocr import parse_pdf_by_ocr
from demo.demo_commons import get_json_from_local_or_s3
from magic_pdf.dict2md.ocr_mkcontent import (
    ocr_mk_mm_markdown_with_para,
    make_standard_format_with_para
)
from magic_pdf.libs.commons import join_path, read_file, formatted_time


def save_markdown(markdown_text, input_filepath):
    # 获取输入文件的目录
    directory = os.path.dirname(input_filepath)
    # 获取输入文件的文件名（不带扩展名）
    base_name = os.path.basename(input_filepath)
    file_name_without_ext = os.path.splitext(base_name)[0]
    # 定义输出文件的路径
    output_filepath = os.path.join(directory, f"{file_name_without_ext}.md")

    # 将Markdown文本写入.md文件
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write(markdown_text)


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def ocr_local_parse(ocr_pdf_path, ocr_json_file_path):
    try:
        ocr_pdf_model_info = read_json_file(ocr_json_file_path)
        pth = Path(ocr_json_file_path)
        book_name = pth.name
        pdf_bytes = read_file(ocr_pdf_path, None)
        ocr_parse_core(book_name, pdf_bytes, ocr_pdf_model_info)
    except Exception as e:
        logger.exception(e)


def ocr_online_parse(book_name, start_page_id=0, debug_mode=True):
    try:
        json_object = get_json_from_local_or_s3(book_name)
        # logger.info(json_object)
        s3_pdf_path = json_object["file_location"]
        s3_config = get_s3_config_dict(s3_pdf_path)
        pdf_bytes = read_file(s3_pdf_path, s3_config)
        ocr_pdf_model_info = json_object.get("doc_layout_result")
        ocr_parse_core(book_name, pdf_bytes, ocr_pdf_model_info)
    except Exception as e:
        logger.exception(e)


def ocr_parse_core(book_name, pdf_bytes, ocr_pdf_model_info, start_page_id=0):
    save_tmp_path = os.path.join(os.path.dirname(__file__), "../..", "tmp", "unittest")
    save_path = join_path(save_tmp_path, "md")
    save_path_with_bookname = os.path.join(save_path, book_name)
    text_content_save_path = f"{save_path_with_bookname}/book.md"
    pdf_info_dict, parse_time = ocr_parse_pdf_core(pdf_bytes, ocr_pdf_model_info, book_name, start_page_id=start_page_id, debug_mode=True)

    parent_dir = os.path.dirname(text_content_save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # markdown_content = mk_nlp_markdown(pdf_info_dict)
    markdown_content = ocr_mk_mm_markdown_with_para(pdf_info_dict)
    # markdown_pagination = ocr_mk_mm_markdown_with_para_and_pagination(pdf_info_dict)

    with open(text_content_save_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    standard_format = make_standard_format_with_para(pdf_info_dict)
    standard_format_save_path = f"{save_path_with_bookname}/standard_format.txt"
    with open(standard_format_save_path, "w", encoding="utf-8") as f:
        # 将standard_format dump成json文本并保存
        f.write(json.dumps(standard_format, ensure_ascii=False))


def ocr_parse_pdf_core(pdf_bytes, model_output_json_list, book_name, start_page_id=0, debug_mode=False):
    start_time = time.time()  # 记录开始时间
    # 先打印一下book_name和解析开始的时间
    logger.info(
        f"book_name is:{book_name},start_time is:{formatted_time(start_time)}",
        file=sys.stderr,
    )
    pdf_info_dict = parse_pdf_by_ocr(
        pdf_bytes,
        model_output_json_list,
        "",
        book_name,
        pdf_model_profile=None,
        start_page_id=start_page_id,
        debug_mode=debug_mode,
    )
    end_time = time.time()  # 记录完成时间
    parse_time = int(end_time - start_time)  # 计算执行时间
    # 解析完成后打印一下book_name和耗时
    logger.info(
        f"book_name is:{book_name},end_time is:{formatted_time(end_time)},cost_time is:{parse_time}",
        file=sys.stderr,
    )

    return pdf_info_dict, parse_time


if __name__ == '__main__':
    pdf_path = r"/home/cxu/workspace/Magic-PDF/ocr_demo/j.1540-627x.2006.00176.x.pdf"
    json_file_path = r"/home/cxu/workspace/Magic-PDF/ocr_demo/j.1540-627x.2006.00176.x.json"
    # ocr_local_parse(pdf_path, json_file_path)
    book_name = "科数网/edu_00011318"
    ocr_online_parse(book_name)
    
    pass
