# Copyright (c) Opendatalab. All rights reserved.
import io
import json
import os
from pathlib import Path

import pypdfium2 as pdfium
from loguru import logger
from ..api.vlm_middle_json_mkcontent import union_make
from ..backend.vlm.vlm_analyze import doc_analyze
from ..data.data_reader_writer import FileBasedDataWriter
from ..utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from ..utils.enum_class import MakeMode
from ..utils.pdf_image_tools import images_bytes_to_pdf_bytes

pdf_suffixes = [".pdf"]
image_suffixes = [".png", ".jpeg", ".jpg"]


def read_fn(path: Path):
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        if path.suffix in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif path.suffix in pdf_suffixes:
            return file_bytes
        else:
            raise Exception(f"Unknown file suffix: {path.suffix}")


def prepare_env(output_dir, pdf_file_name):
    local_parent_dir = os.path.join(output_dir, pdf_file_name)

    local_image_dir = os.path.join(str(local_parent_dir), "images")
    local_md_dir = local_parent_dir
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None):

    # 从字节数据加载PDF
    pdf = pdfium.PdfDocument(pdf_bytes)

    # 确定结束页
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(pdf) - 1
    if end_page_id > len(pdf) - 1:
        logger.warning("end_page_id is out of range, use pdf_docs length")
        end_page_id = len(pdf) - 1

    # 创建一个新的PDF文档
    output_pdf = pdfium.PdfDocument.new()

    # 选择要导入的页面索引
    page_indices = list(range(start_page_id, end_page_id + 1))

    # 从原PDF导入页面到新PDF
    output_pdf.import_pages(pdf, page_indices)

    # 将新PDF保存到内存缓冲区
    output_buffer = io.BytesIO()
    output_pdf.save(output_buffer)

    # 获取字节数据
    output_bytes = output_buffer.getvalue()

    return output_bytes


def do_parse(
    output_dir,
    pdf_file_name,
    pdf_bytes,
    backend="pipeline",
    model_path="jinzhenj/OEEzRkQ3RTAtMDMx-0415",  # TODO: change to formal path after release.
    server_url=None,
    f_draw_layout_bbox=True,
    f_draw_span_bbox=False,
    f_dump_md=True,
    f_dump_middle_json=True,
    f_dump_model_output=True,
    f_dump_orig_pdf=True,
    f_dump_content_list=True,
    f_make_md_mode=MakeMode.MM_MD,
    start_page_id=0,
    end_page_id=None,
):
    if backend == 'pipeline':
        f_draw_span_bbox = True

    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name)
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

    middle_json, infer_result = doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url)
    pdf_info = middle_json["pdf_info"]

    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    if f_dump_md:
        image_dir = str(os.path.basename(local_image_dir))
        md_content_str = union_make(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        image_dir = str(os.path.basename(local_image_dir))
        content_list = union_make(pdf_info, MakeMode.STANDARD_FORMAT, image_dir)
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

    return infer_result


if __name__ == "__main__":
    pdf_path = "../../demo/demo2.pdf"
    with open(pdf_path, "rb") as f:
        try:
            result = do_parse("./output", Path(pdf_path).stem, f.read())
        except Exception as e:
            logger.exception(e)
        # dict转成json
        print(json.dumps(result, ensure_ascii=False, indent=4))
