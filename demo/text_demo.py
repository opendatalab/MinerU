import json
import os
import sys
from pathlib import Path

import click

from demo.demo_commons import get_json_from_local_or_s3, write_json_to_local, local_jsonl_path, local_json_path
from magic_pdf.dict2md.mkcontent import mk_mm_markdown
from magic_pdf.pipeline import (
    meta_scan,
    classify_by_type,
    parse_pdf,
    pdf_intermediate_dict_to_markdown,
    save_tables_to_s3,
)
from magic_pdf.libs.commons import join_path
from loguru import logger




def demo_parse_pdf(book_name=None, start_page_id=0, debug_mode=True):
    json_object = get_json_from_local_or_s3(book_name)

    jso = parse_pdf(json_object, start_page_id=start_page_id, debug_mode=debug_mode)
    logger.info(f"pdf_parse_time: {jso['parse_time']}")

    write_json_to_local(jso, book_name)

    jso_md = pdf_intermediate_dict_to_markdown(jso, debug_mode=debug_mode)
    content = jso_md.get("content_list")
    markdown_content = mk_mm_markdown(content)
    if book_name is not None:
        save_tmp_path = os.path.join(os.path.dirname(__file__), "../..", "tmp", "unittest", "md", book_name)
        uni_format_save_path = join_path(save_tmp_path,  "book" + ".json")
        markdown_save_path = join_path(save_tmp_path,  "book" + ".md")
        with open(uni_format_save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False, indent=4))
        with open(markdown_save_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
    else:
        logger.info(json.dumps(content, ensure_ascii=False))


def demo_save_tables(book_name=None, start_page_id=0, debug_mode=True):
    json_object = get_json_from_local_or_s3(book_name)

    jso = parse_pdf(json_object, start_page_id=start_page_id, debug_mode=debug_mode)
    logger.info(f"pdf_parse_time: {jso['parse_time']}")

    write_json_to_local(jso, book_name)

    save_tables_to_s3(jso, debug_mode=debug_mode)


def demo_classify_by_type(book_name=None, debug_mode=True):
    json_object = get_json_from_local_or_s3(book_name)

    jso = classify_by_type(json_object, debug_mode=debug_mode)

    logger.info(json.dumps(jso, ensure_ascii=False))
    logger.info(f"classify_time: {jso['classify_time']}")
    write_json_to_local(jso, book_name)


def demo_meta_scan(book_name=None, debug_mode=True):
    json_object = get_json_from_local_or_s3(book_name)

    # doc_layout_check=False
    jso = meta_scan(json_object, doc_layout_check=True)

    logger.info(json.dumps(jso, ensure_ascii=False))
    logger.info(f"meta_scan_time: {jso['meta_scan_time']}")
    write_json_to_local(jso, book_name)


def demo_meta_scan_from_jsonl():
    with open(local_jsonl_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            jso = json.loads(line)
            jso = meta_scan(jso)
            logger.info(f"pdf_path: {jso['content']['pdf_path']}")
            logger.info(f"read_file_time: {jso['read_file_time']}")
            logger.info(f"meta_scan_time: {jso['meta_scan_time']}")


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
