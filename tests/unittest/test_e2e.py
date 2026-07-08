# Copyright (c) Opendatalab. All rights reserved.
import json
import os

from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from loguru import logger

from mineru.parser.base import ParseResult
from mineru.render import render_content_list, render_markdown
from mineru.render.writer import FileBasedDataWriter


def write_infer_result(
    pdf_file_name,
    local_image_dir,
    local_md_dir,
    middle_json,
    model_list,
):
    md_writer = FileBasedDataWriter(local_md_dir)
    pdf_info = middle_json
    image_dir = str(os.path.basename(local_image_dir))

    md_content_str = render_markdown(pdf_info, image_dir)
    md_writer.write_string(
        f"{pdf_file_name}.md",
        md_content_str,
    )

    content_list = render_content_list(pdf_info, image_dir)
    md_writer.write_string(
        f"{pdf_file_name}_content_list.json",
        json.dumps(content_list, ensure_ascii=False, indent=4),
    )

    md_writer.write_string(
        f"{pdf_file_name}_middle.json",
        json.dumps(ParseResult(pages=pdf_info).to_dict(), ensure_ascii=False, indent=4),
    )

    md_writer.write_string(
        f"{pdf_file_name}_model.json",
        json.dumps(model_list, ensure_ascii=False, indent=4),
    )

    logger.info(f"local output dir is {local_md_dir}")


def validate_html(html_content):
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return True
    except Exception as e:
        return False


def assert_content(content_path, parse_method="txt"):
    content_list = []
    with open(content_path, "r", encoding="utf-8") as file:
        content_list = json.load(file)
        logger.info(content_list)
    type_set = set()
    for content_dict in content_list:
        match content_dict["type"]:
            # 图片校验，只校验 Caption
            case "image":
                type_set.add("image")
                assert (
                    fuzz.ratio(
                        content_dict["image_caption"][0],
                        "Figure 1: Figure Caption",
                    )
                    > 90
                )
            # 表格校验，校验 Caption，表格格式和表格内容
            case "table":
                type_set.add("table")
                assert (
                    fuzz.ratio(
                        content_dict["table_caption"][0],
                        "Table 1: Table Caption",
                    )
                    > 90
                )
                assert validate_html(content_dict["table_body"])
                target_str_list = [
                    "Model",
                    "Testing",
                    "Error",
                    "Linear",
                    "Regression",
                    "0.98740",
                    "1321.2",
                    "Gray",
                    "Prediction",
                    "0.00617",
                    "687",
                ]
                correct_count = 0
                for target_str in target_str_list:
                    if target_str in content_dict["table_body"]:
                        correct_count += 1
                if parse_method == "txt" or parse_method == "ocr":
                    assert correct_count > 0.9 * len(target_str_list)
                elif parse_method == "vlm":
                    assert correct_count > 0.7 * len(target_str_list)
                else:
                    assert False
            # 公式校验，检测是否含有公式元素
            case "equation":
                type_set.add("equation")
                target_str_list = ["$$", "lambda", "frac", "bar"]
                for target_str in target_str_list:
                    assert target_str in content_dict["text"]
            # 文本校验，文本相似度超过90
            case "text":
                type_set.add("text")
                assert (
                    fuzz.ratio(
                        content_dict["text"],
                        "Trump graduated from the Wharton School of the University of Pennsylvania with a bachelor's degree in 1968. He became president of his father's real estate business in 1971 and renamed it The Trump Organization.",
                    )
                    > 90
                )
    assert len(type_set) >= 4
