# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path

from cryptography.hazmat.backends.openssl import backend
from loguru import logger

from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,
    prepare_env,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
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
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


class TestE2E:

    def test_pipeline_with_two_config(self):
        def do_parse(
            output_dir,  # Output directory for storing parsing results
            pdf_file_names: list[str],  # List of PDF file names to be parsed
            pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
            p_lang_list: list[
                str
            ],  # List of languages for each PDF, default is 'ch' (Chinese)
            parse_method="auto",  # The method for parsing PDF, default is 'auto'
            formula_enable=True,  # Enable formula parsing
            table_enable=True,  # Enable table parsing
            f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
            f_draw_span_bbox=True,  # Whether to draw span bounding boxes
            f_dump_md=True,  # Whether to dump markdown files
            f_dump_middle_json=True,  # Whether to dump middle JSON files
            f_dump_model_output=True,  # Whether to dump model output files
            f_dump_orig_pdf=True,  # Whether to dump original PDF files
            f_dump_content_list=True,  # Whether to dump content list files
            f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
            start_page_id=0,  # Start page ID for parsing, default is 0
            end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
        ):
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, start_page_id, end_page_id
                )
                pdf_bytes_list[idx] = new_pdf_bytes

            (
                infer_results,
                all_image_lists,
                all_pdf_docs,
                lang_list,
                ocr_enabled_list,
            ) = pipeline_doc_analyze(
                pdf_bytes_list,
                p_lang_list,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )

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
                    formula_enable,
                )

                pdf_info = middle_json["pdf_info"]

                pdf_bytes = pdf_bytes_list[idx]
                if f_draw_layout_bbox:
                    draw_layout_bbox(
                        pdf_info,
                        pdf_bytes,
                        local_md_dir,
                        f"{pdf_file_name}_layout.pdf",
                    )

                if f_draw_span_bbox:
                    draw_span_bbox(
                        pdf_info,
                        pdf_bytes,
                        local_md_dir,
                        f"{pdf_file_name}_span.pdf",
                    )

                if f_dump_orig_pdf:
                    md_writer.write(
                        f"{pdf_file_name}_origin.pdf",
                        pdf_bytes,
                    )

                if f_dump_md:
                    image_dir = str(os.path.basename(local_image_dir))
                    md_content_str = pipeline_union_make(
                        pdf_info, f_make_md_mode, image_dir
                    )
                    md_writer.write_string(
                        f"{pdf_file_name}.md",
                        md_content_str,
                    )

                if f_dump_content_list:
                    image_dir = str(os.path.basename(local_image_dir))
                    content_list = pipeline_union_make(
                        pdf_info, MakeMode.CONTENT_LIST, image_dir
                    )
                    md_writer.write_string(
                        f"{pdf_file_name}_content_list.json",
                        json.dumps(content_list, ensure_ascii=False, indent=4),
                    )

                if f_dump_middle_json:
                    md_writer.write_string(
                        f"{pdf_file_name}_middle.json",
                        json.dumps(middle_json, ensure_ascii=False, indent=4),
                    )

                if f_dump_model_output:
                    md_writer.write_string(
                        f"{pdf_file_name}_model.json",
                        json.dumps(model_json, ensure_ascii=False, indent=4),
                    )

                logger.info(f"local output dir is {local_md_dir}")

        def parse_doc(
            path_list: list[Path],
            output_dir,
            lang="ch",
            method="auto",
            start_page_id=0,
            end_page_id=None,
        ):
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            for path in path_list:
                file_name = str(Path(path).stem)
                pdf_bytes = read_fn(path)
                file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
                lang_list.append(lang)
            # 运行两次 do_parse，分别是开启公式和表格解析和不开启
            do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                parse_method=method,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )
            do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                parse_method=method,
                table_enable=False,
                formula_enable=False,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )

        __dir__ = os.path.dirname(os.path.abspath(__file__))
        pdf_files_dir = os.path.join(__dir__, "pdfs")
        output_dir = os.path.join(__dir__, "output")
        pdf_suffixes = [".pdf"]
        image_suffixes = [".png", ".jpeg", ".jpg"]

        doc_path_list = []
        for doc_path in Path(pdf_files_dir).glob("*"):
            if doc_path.suffix in pdf_suffixes + image_suffixes:
                doc_path_list.append(doc_path)

        os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
        parse_doc(doc_path_list, output_dir)

    def test_vlm_transformers_with_default_config(self):
        def do_parse(
            output_dir,  # Output directory for storing parsing results
            pdf_file_names: list[str],  # List of PDF file names to be parsed
            pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
            server_url=None,  # Server URL for vlm-sglang-client backend
            f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
            f_dump_md=True,  # Whether to dump markdown files
            f_dump_middle_json=True,  # Whether to dump middle JSON files
            f_dump_model_output=True,  # Whether to dump model output files
            f_dump_orig_pdf=True,  # Whether to dump original PDF files
            f_dump_content_list=True,  # Whether to dump content list files
            f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
            start_page_id=0,  # Start page ID for parsing, default is 0
            end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
        ):
            backend = "transformers"
            f_draw_span_bbox = False
            parse_method = "vlm"
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                pdf_file_name = pdf_file_names[idx]
                pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, start_page_id, end_page_id
                )
                local_image_dir, local_md_dir = prepare_env(
                    output_dir, pdf_file_name, parse_method
                )
                image_writer, md_writer = FileBasedDataWriter(
                    local_image_dir
                ), FileBasedDataWriter(local_md_dir)
                middle_json, infer_result = vlm_doc_analyze(
                    pdf_bytes,
                    image_writer=image_writer,
                    backend=backend,
                    server_url=server_url,
                )

                pdf_info = middle_json["pdf_info"]

                if f_draw_layout_bbox:
                    draw_layout_bbox(
                        pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
                    )

                if f_draw_span_bbox:
                    draw_span_bbox(
                        pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf"
                    )

                if f_dump_orig_pdf:
                    md_writer.write(
                        f"{pdf_file_name}_origin.pdf",
                        pdf_bytes,
                    )

                if f_dump_md:
                    image_dir = str(os.path.basename(local_image_dir))
                    md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                    md_writer.write_string(
                        f"{pdf_file_name}.md",
                        md_content_str,
                    )

                if f_dump_content_list:
                    image_dir = str(os.path.basename(local_image_dir))
                    content_list = vlm_union_make(
                        pdf_info, MakeMode.CONTENT_LIST, image_dir
                    )
                    md_writer.write_string(
                        f"{pdf_file_name}_content_list.json",
                        json.dumps(content_list, ensure_ascii=False, indent=4),
                    )

                if f_dump_middle_json:
                    md_writer.write_string(
                        f"{pdf_file_name}_middle.json",
                        json.dumps(middle_json, ensure_ascii=False, indent=4),
                    )

                if f_dump_model_output:
                    model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                    md_writer.write_string(
                        f"{pdf_file_name}_model_output.txt",
                        model_output,
                    )

                logger.info(f"local output dir is {local_md_dir}")

        def parse_doc(
            path_list: list[Path],
            output_dir,
            lang="ch",
            server_url=None,
            start_page_id=0,
            end_page_id=None,
        ):
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            for path in path_list:
                file_name = str(Path(path).stem)
                pdf_bytes = read_fn(path)
                file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
                lang_list.append(lang)
            do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )

        __dir__ = os.path.dirname(os.path.abspath(__file__))
        pdf_files_dir = os.path.join(__dir__, "pdfs")
        output_dir = os.path.join(__dir__, "output")
        pdf_suffixes = [".pdf"]
        image_suffixes = [".png", ".jpeg", ".jpg"]

        doc_path_list = []
        for doc_path in Path(pdf_files_dir).glob("*"):
            if doc_path.suffix in pdf_suffixes + image_suffixes:
                doc_path_list.append(doc_path)

        os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
        parse_doc(doc_path_list, output_dir)
