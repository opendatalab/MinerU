# Copyright (c) Opendatalab. All rights reserved.
import json
import os
from pathlib import Path
from loguru import logger
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from mineru.cli.common import (
    convert_pdf_bytes_to_bytes,
    prepare_env,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import (
    doc_analyze_streaming as pipeline_doc_analyze_streaming,
)
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)


def test_pipeline_with_two_config():
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, "pdfs")
    output_dir = os.path.join(__dir__, "output")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]

    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob("*"):
        if doc_path.suffix in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    # os.environ["MINERU_MODEL_SOURCE"] = "modelscope"

    pdf_file_names = []
    pdf_bytes_list = []
    p_lang_list = []
    for path in doc_path_list:
        file_name = str(Path(path).stem)
        pdf_bytes = read_fn(path)
        pdf_file_names.append(file_name)
        pdf_bytes_list.append(pdf_bytes)
        p_lang_list.append("en")
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes(pdf_bytes)
        pdf_bytes_list[idx] = new_pdf_bytes

    run_pipeline_parse(
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        output_dir,
        parse_method="txt",
    )
    res_json_path = (
        Path(__file__).parent / "output" / "test" / "txt" / "test_content_list.json"
    ).as_posix()
    assert_content(res_json_path, parse_method="txt")
    run_pipeline_parse(
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        output_dir,
        parse_method="ocr",
    )
    res_json_path = (
        Path(__file__).parent / "output" / "test" / "ocr" / "test_content_list.json"
    ).as_posix()
    assert_content(res_json_path, parse_method="ocr")


def run_pipeline_parse(
    pdf_file_names,
    pdf_bytes_list,
    p_lang_list,
    output_dir,
    parse_method,
):
    image_writer_list = []
    output_info = []
    for pdf_file_name in pdf_file_names:
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer_list.append(FileBasedDataWriter(local_image_dir))
        output_info.append((pdf_file_name, local_image_dir, local_md_dir))

    def on_doc_ready(doc_index, model_list, middle_json, ocr_enable):
        del ocr_enable
        pdf_file_name, local_image_dir, local_md_dir = output_info[doc_index]
        write_infer_result(
            pdf_file_name,
            local_image_dir,
            local_md_dir,
            middle_json,
            model_list,
        )

    pipeline_doc_analyze_streaming(
        pdf_bytes_list,
        image_writer_list,
        p_lang_list,
        on_doc_ready,
        parse_method=parse_method,
    )


def write_infer_result(
    pdf_file_name,
    local_image_dir,
    local_md_dir,
    middle_json,
    model_list,
):
    md_writer = FileBasedDataWriter(local_md_dir)
    pdf_info = middle_json["pdf_info"]
    image_dir = str(os.path.basename(local_image_dir))

    md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
    md_writer.write_string(
        f"{pdf_file_name}.md",
        md_content_str,
    )

    content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
    md_writer.write_string(
        f"{pdf_file_name}_content_list.json",
        json.dumps(content_list, ensure_ascii=False, indent=4),
    )

    md_writer.write_string(
        f"{pdf_file_name}_middle.json",
        json.dumps(middle_json, ensure_ascii=False, indent=4),
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
