import json
import os
import sys
from pathlib import Path

import click

from demo.demo_commons import get_json_from_local_or_s3, write_json_to_local, local_jsonl_path, local_json_path
from magic_pdf.dict2md.mkcontent import mk_mm_markdown, mk_universal_format
from magic_pdf.filter.pdf_classify_by_type import classify

from magic_pdf.filter.pdf_meta_scan import pdf_meta_scan
from magic_pdf.libs.commons import join_path, read_file
from loguru import logger

from magic_pdf.libs.config_reader import get_s3_config_dict
from magic_pdf.pdf_parse_by_txt import parse_pdf_by_txt
from magic_pdf.spark.base import get_data_source


def demo_parse_pdf(book_name=None, start_page_id=0, debug_mode=True):
    json_object = get_json_from_local_or_s3(book_name)

    s3_pdf_path = json_object.get("file_location")
    s3_config = get_s3_config_dict(s3_pdf_path)
    pdf_bytes = read_file(s3_pdf_path, s3_config)
    model_output_json_list = json_object.get("doc_layout_result")
    data_source = get_data_source(json_object)
    file_id = json_object.get("file_id")
    junk_img_bojids = json_object["pdf_meta"]["junk_img_bojids"]
    save_path = ""
    pdf_info_dict = parse_pdf_by_txt(
        pdf_bytes,
        model_output_json_list,
        save_path,
        f"{data_source}/{file_id}",
        pdf_model_profile=None,
        start_page_id=start_page_id,
        junk_img_bojids=junk_img_bojids,
        debug_mode=debug_mode,
    )

    write_json_to_local(pdf_info_dict, book_name)
    content_list = mk_universal_format(pdf_info_dict)
    markdown_content = mk_mm_markdown(content_list)
    if book_name is not None:
        save_tmp_path = os.path.join(os.path.dirname(__file__), "../..", "tmp", "unittest", "md", book_name)
        uni_format_save_path = join_path(save_tmp_path, "book" + ".json")
        markdown_save_path = join_path(save_tmp_path, "book" + ".md")
        with open(uni_format_save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content_list, ensure_ascii=False, indent=4))
        with open(markdown_save_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    else:
        logger.info(json.dumps(content_list, ensure_ascii=False))


def demo_classify_by_type(book_name=None, debug_mode=True):
    json_object = get_json_from_local_or_s3(book_name)

    pdf_meta = json_object.get("pdf_meta")
    total_page = pdf_meta["total_page"]
    page_width = pdf_meta["page_width_pts"]
    page_height = pdf_meta["page_height_pts"]
    img_sz_list = pdf_meta["image_info_per_page"]
    img_num_list = pdf_meta["imgs_per_page"]
    text_len_list = pdf_meta["text_len_per_page"]
    text_layout_list = pdf_meta["text_layout_per_page"]
    is_text_pdf, results = classify(
        total_page,
        page_width,
        page_height,
        img_sz_list,
        text_len_list,
        img_num_list,
        text_layout_list,
    )
    logger.info(f"is_text_pdf: {is_text_pdf}")
    logger.info(json.dumps(results, ensure_ascii=False))
    write_json_to_local(results, book_name)


def demo_meta_scan(book_name=None, debug_mode=True):
    json_object = get_json_from_local_or_s3(book_name)

    s3_pdf_path = json_object.get("file_location")
    s3_config = get_s3_config_dict(s3_pdf_path)
    pdf_bytes = read_file(s3_pdf_path, s3_config)
    res = pdf_meta_scan(pdf_bytes)

    logger.info(json.dumps(res, ensure_ascii=False))
    write_json_to_local(res, book_name)


def demo_test5():
    with open(local_json_path, "r", encoding="utf-8") as json_file:
        json_line = json_file.read()
        jso = json.loads(json_line)
    img_list_len = len(jso["content"]["image_info_per_page"])
    logger.info(f"img_list_len: {img_list_len}")


def read_more_para_test_samples(type="scihub"):
    # 读取多段落测试样本
    curr_dir = Path(__file__).parent
    files_path = ""
    if type == "gift":
        relative_path = "../tests/assets/more_para_test_samples/gift_files.txt"
        files_path = os.path.join(curr_dir, relative_path)

    if type == "scihub":
        relative_path = "../tests/assets/more_para_test_samples/scihub_files.txt"
        files_path = os.path.join(curr_dir, relative_path)

    if type == "zlib":
        relative_path = "../tests/assets/more_para_test_samples/zlib_files.txt"
        files_path = os.path.join(curr_dir, relative_path)

    # check if file exists
    if not os.path.exists(files_path):
        print("File not exist!")
        sys.exit(0)

    with open(files_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # print("lines", lines)

    return lines


def batch_test_more_para(type="scihub"):
    # 批量测试多段落
    para_test_files = read_more_para_test_samples(type)
    for file in para_test_files:
        file = file.strip()
        print(file)
        demo_parse_pdf(book_name=file)


@click.command()
@click.option("--book-name", help="s3上pdf文件的路径")
def main(book_name: str):
    demo_parse_pdf(book_name, start_page_id=0)


if __name__ == "__main__":
    main()
