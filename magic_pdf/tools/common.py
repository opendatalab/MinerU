import os

import click
import fitz
from loguru import logger

import magic_pdf.model as model_config
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.libs.draw_bbox import draw_char_bbox
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.models import InferenceResult

# from io import BytesIO
# from pypdf import PdfReader, PdfWriter


def prepare_env(output_dir, pdf_file_name, method):
    local_parent_dir = os.path.join(output_dir, pdf_file_name, method)

    local_image_dir = os.path.join(str(local_parent_dir), 'images')
    local_md_dir = local_parent_dir
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


# def convert_pdf_bytes_to_bytes_by_pypdf(pdf_bytes, start_page_id=0, end_page_id=None):
#     # 将字节数据包装在 BytesIO 对象中
#     pdf_file = BytesIO(pdf_bytes)
#     # 读取 PDF 的字节数据
#     reader = PdfReader(pdf_file)
#     # 创建一个新的 PDF 写入器
#     writer = PdfWriter()
#     # 将所有页面添加到新的 PDF 写入器中
#     end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(reader.pages) - 1
#     if end_page_id > len(reader.pages) - 1:
#         logger.warning("end_page_id is out of range, use pdf_docs length")
#         end_page_id = len(reader.pages) - 1
#     for i, page in enumerate(reader.pages):
#         if start_page_id <= i <= end_page_id:
#             writer.add_page(page)
#     # 创建一个字节缓冲区来存储输出的 PDF 数据
#     output_buffer = BytesIO()
#     # 将 PDF 写入字节缓冲区
#     writer.write(output_buffer)
#     # 获取字节缓冲区的内容
#     converted_pdf_bytes = output_buffer.getvalue()
#     return converted_pdf_bytes


def convert_pdf_bytes_to_bytes_by_pymupdf(pdf_bytes, start_page_id=0, end_page_id=None):
    document = fitz.open('pdf', pdf_bytes)
    output_document = fitz.open()
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else len(document) - 1
    )
    if end_page_id > len(document) - 1:
        logger.warning('end_page_id is out of range, use pdf_docs length')
        end_page_id = len(document) - 1
    output_document.insert_pdf(document, from_page=start_page_id, to_page=end_page_id)
    output_bytes = output_document.tobytes()
    return output_bytes


def do_parse(
    output_dir,
    pdf_file_name,
    pdf_bytes,
    model_list,
    parse_method,
    debug_able,
    f_draw_span_bbox=True,
    f_draw_layout_bbox=True,
    f_dump_md=True,
    f_dump_middle_json=True,
    f_dump_model_json=True,
    f_dump_orig_pdf=True,
    f_dump_content_list=True,
    f_make_md_mode=MakeMode.MM_MD,
    f_draw_model_bbox=False,
    f_draw_line_sort_bbox=False,
    f_draw_char_bbox=False,
    start_page_id=0,
    end_page_id=None,
    lang=None,
    layout_model=None,
    formula_enable=None,
    table_enable=None,
):
    if debug_able:
        logger.warning('debug mode is on')
        f_draw_model_bbox = True
        f_draw_line_sort_bbox = True
        # f_draw_char_bbox = True

    pdf_bytes = convert_pdf_bytes_to_bytes_by_pymupdf(
        pdf_bytes, start_page_id, end_page_id
    )

    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )
    image_dir = str(os.path.basename(local_image_dir))

    ds = PymuDocDataset(pdf_bytes, lang=lang)

    if len(model_list) == 0:
        if model_config.__use_inside_model__:
            if parse_method == 'auto':
                if ds.classify() == SupportedPdfParseMethod.TXT:
                    infer_result = ds.apply(
                        doc_analyze,
                        ocr=False,
                        lang=ds._lang,
                        layout_model=layout_model,
                        formula_enable=formula_enable,
                        table_enable=table_enable,
                    )
                    pipe_result = infer_result.pipe_txt_mode(
                        image_writer, debug_mode=True, lang=ds._lang
                    )
                else:
                    infer_result = ds.apply(
                        doc_analyze,
                        ocr=True,
                        lang=ds._lang,
                        layout_model=layout_model,
                        formula_enable=formula_enable,
                        table_enable=table_enable,
                    )
                    pipe_result = infer_result.pipe_ocr_mode(
                        image_writer, debug_mode=True, lang=ds._lang
                    )

            elif parse_method == 'txt':
                infer_result = ds.apply(
                    doc_analyze,
                    ocr=False,
                    lang=ds._lang,
                    layout_model=layout_model,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                )
                pipe_result = infer_result.pipe_txt_mode(
                    image_writer, debug_mode=True, lang=ds._lang
                )
            elif parse_method == 'ocr':
                infer_result = ds.apply(
                    doc_analyze,
                    ocr=True,
                    lang=ds._lang,
                    layout_model=layout_model,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                )
                pipe_result = infer_result.pipe_ocr_mode(
                    image_writer, debug_mode=True, lang=ds._lang
                )
            else:
                logger.error('unknown parse method')
                exit(1)
        else:
            logger.error('need model list input')
            exit(2)
    else:

        infer_result = InferenceResult(model_list, ds)
        if parse_method == 'ocr':
            pipe_result = infer_result.pipe_ocr_mode(
                image_writer, debug_mode=True, lang=ds._lang
            )
        elif parse_method == 'txt':
            pipe_result = infer_result.pipe_txt_mode(
                image_writer, debug_mode=True, lang=ds._lang
            )
        else:
            if ds.classify() == SupportedPdfParseMethod.TXT:
                pipe_result = infer_result.pipe_txt_mode(
                        image_writer, debug_mode=True, lang=ds._lang
                    )
            else:
                pipe_result = infer_result.pipe_ocr_mode(
                        image_writer, debug_mode=True, lang=ds._lang
                    )


    if f_draw_model_bbox:
        infer_result.draw_model(
            os.path.join(local_md_dir, f'{pdf_file_name}_model.pdf')
        )

    if f_draw_layout_bbox:
        pipe_result.draw_layout(
            os.path.join(local_md_dir, f'{pdf_file_name}_layout.pdf')
        )
    if f_draw_span_bbox:
        pipe_result.draw_span(os.path.join(local_md_dir, f'{pdf_file_name}_spans.pdf'))

    if f_draw_line_sort_bbox:
        pipe_result.draw_line_sort(
            os.path.join(local_md_dir, f'{pdf_file_name}_line_sort.pdf')
        )

    if f_draw_char_bbox:
        draw_char_bbox(pdf_bytes, local_md_dir, f'{pdf_file_name}_char_bbox.pdf')

    if f_dump_md:
        pipe_result.dump_md(
            md_writer,
            f'{pdf_file_name}.md',
            image_dir,
            drop_mode=DropMode.NONE,
            md_make_mode=f_make_md_mode,
        )

    if f_dump_middle_json:
        pipe_result.dump_middle_json(md_writer, f'{pdf_file_name}_middle.json')

    if f_dump_model_json:
        infer_result.dump_model(md_writer, f'{pdf_file_name}_model.json')

    if f_dump_orig_pdf:
        md_writer.write(
            f'{pdf_file_name}_origin.pdf',
            pdf_bytes,
        )

    if f_dump_content_list:
        pipe_result.dump_content_list(
            md_writer,
            f'{pdf_file_name}_content_list.json',
            image_dir
        )

    logger.info(f'local output dir is {local_md_dir}')


parse_pdf_methods = click.Choice(['ocr', 'txt', 'auto'])
