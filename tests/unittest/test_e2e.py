# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path
from loguru import logger
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,
    prepare_env,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make


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
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
        pdf_bytes_list[idx] = new_pdf_bytes

    # 获取 pipline 分析结果, 分别测试 txt 和 ocr 两种解析方法的结果
    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
        pipeline_doc_analyze(
            pdf_bytes_list,
            p_lang_list,
            parse_method="txt",
        )
    )
    write_infer_result(
        infer_results,
        all_image_lists,
        all_pdf_docs,
        lang_list,
        ocr_enabled_list,
        pdf_file_names,
        output_dir,
        parse_method="txt",
    )
    res_json_path = (
        Path(__file__).parent / "output" / "test" / "txt" / "test_content_list.json"
    ).as_posix()
    assert_content(res_json_path, parse_method="txt")
    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
        pipeline_doc_analyze(
            pdf_bytes_list,
            p_lang_list,
            parse_method="ocr",
        )
    )
    write_infer_result(
        infer_results,
        all_image_lists,
        all_pdf_docs,
        lang_list,
        ocr_enabled_list,
        pdf_file_names,
        output_dir,
        parse_method="ocr",
    )
    res_json_path = (
        Path(__file__).parent / "output" / "test" / "ocr" / "test_content_list.json"
    ).as_posix()
    assert_content(res_json_path, parse_method="ocr")


# def test_vlm_transformers_with_default_config():
#     __dir__ = os.path.dirname(os.path.abspath(__file__))
#     pdf_files_dir = os.path.join(__dir__, "pdfs")
#     output_dir = os.path.join(__dir__, "output")
#     pdf_suffixes = [".pdf"]
#     image_suffixes = [".png", ".jpeg", ".jpg"]
#
#     doc_path_list = []
#     for doc_path in Path(pdf_files_dir).glob("*"):
#         if doc_path.suffix in pdf_suffixes + image_suffixes:
#             doc_path_list.append(doc_path)
#
#     # os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
#
#     pdf_file_names = []
#     pdf_bytes_list = []
#     p_lang_list = []
#     for path in doc_path_list:
#         file_name = str(Path(path).stem)
#         pdf_bytes = read_fn(path)
#         pdf_file_names.append(file_name)
#         pdf_bytes_list.append(pdf_bytes)
#         p_lang_list.append("en")
#
#     for idx, pdf_bytes in enumerate(pdf_bytes_list):
#         pdf_file_name = pdf_file_names[idx]
#         pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
#         local_image_dir, local_md_dir = prepare_env(
#             output_dir, pdf_file_name, parse_method="vlm"
#         )
#         image_writer, md_writer = FileBasedDataWriter(
#             local_image_dir
#         ), FileBasedDataWriter(local_md_dir)
#         middle_json, infer_result = vlm_doc_analyze(
#             pdf_bytes, image_writer=image_writer, backend="transformers"
#         )
#
#         pdf_info = middle_json["pdf_info"]
#
#         image_dir = str(os.path.basename(local_image_dir))
#
#         md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)
#         md_writer.write_string(
#             f"{pdf_file_name}.md",
#             md_content_str,
#         )
#
#         content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
#         md_writer.write_string(
#             f"{pdf_file_name}_content_list.json",
#             json.dumps(content_list, ensure_ascii=False, indent=4),
#         )
#
#         md_writer.write_string(
#             f"{pdf_file_name}_middle.json",
#             json.dumps(middle_json, ensure_ascii=False, indent=4),
#         )
#
#         md_writer.write_string(
#             f"{pdf_file_name}_model.json",
#             json.dumps(infer_result, ensure_ascii=False, indent=4),
#         )
#
#         logger.info(f"local output dir is {local_md_dir}")
#         res_json_path = (
#             Path(__file__).parent / "output" / "test" / "vlm" / "test_content_list.json"
#         ).as_posix()
#         assert_content(res_json_path, parse_method="vlm")


def write_infer_result(
    infer_results,
    all_image_lists,
    all_pdf_docs,
    lang_list,
    ocr_enabled_list,
    pdf_file_names,
    output_dir,
    parse_method,
):
    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(
            output_dir, pdf_file_name, parse_method
        )
        image_writer, md_writer = FileBasedDataWriter(
            local_image_dir
        ), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_doc = all_pdf_docs[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        middle_json = pipeline_result_to_middle_json(
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            _lang,
            _ocr_enable,
            True,
        )

        pdf_info = middle_json["pdf_info"]

        image_dir = str(os.path.basename(local_image_dir))
        # 写入 md 文件
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
            json.dumps(model_json, ensure_ascii=False, indent=4),
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
