import copy
import json as json_parse
import os

import click
from loguru import logger

import magic_pdf.model as model_config
from magic_pdf.libs.draw_bbox import (draw_layout_bbox, draw_span_bbox,
                                      drow_model_bbox)
from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter


def prepare_env(output_dir, pdf_file_name, method):
    local_parent_dir = os.path.join(output_dir, pdf_file_name, method)

    local_image_dir = os.path.join(str(local_parent_dir), 'images')
    local_md_dir = local_parent_dir
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


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
    f_dump_content_list=False,
    f_make_md_mode=MakeMode.MM_MD,
    f_draw_model_bbox=False,
    start_page_id=0,
    end_page_id=None,
):
    if debug_able:
        logger.warning('debug mode is on')
        f_dump_content_list = True
        f_draw_model_bbox = True

    orig_model_list = copy.deepcopy(model_list)
    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name,
                                                parse_method)

    image_writer, md_writer = DiskReaderWriter(
        local_image_dir), DiskReaderWriter(local_md_dir)
    image_dir = str(os.path.basename(local_image_dir))

    if parse_method == 'auto':
        jso_useful_key = {'_pdf_type': '', 'model_list': model_list}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer, is_debug=True,
                       start_page_id=start_page_id, end_page_id=end_page_id)
    elif parse_method == 'txt':
        pipe = TXTPipe(pdf_bytes, model_list, image_writer, is_debug=True,
                       start_page_id=start_page_id, end_page_id=end_page_id)
    elif parse_method == 'ocr':
        pipe = OCRPipe(pdf_bytes, model_list, image_writer, is_debug=True,
                       start_page_id=start_page_id, end_page_id=end_page_id)
    else:
        logger.error('unknown parse method')
        exit(1)

    pipe.pipe_classify()

    if len(model_list) == 0:
        if model_config.__use_inside_model__:
            pipe.pipe_analyze()
            orig_model_list = copy.deepcopy(pipe.model_list)
        else:
            logger.error('need model list input')
            exit(2)

    pipe.pipe_parse()
    pdf_info = pipe.pdf_mid_data['pdf_info']
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, pdf_file_name)
    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, pdf_file_name)
    if f_draw_model_bbox:
        drow_model_bbox(copy.deepcopy(orig_model_list), pdf_bytes, local_md_dir, pdf_file_name)

    md_content = pipe.pipe_mk_markdown(image_dir,
                                       drop_mode=DropMode.NONE,
                                       md_make_mode=f_make_md_mode)
    if f_dump_md:
        md_writer.write(
            content=md_content,
            path=f'{pdf_file_name}.md',
            mode=AbsReaderWriter.MODE_TXT,
        )

    if f_dump_middle_json:
        md_writer.write(
            content=json_parse.dumps(pipe.pdf_mid_data,
                                     ensure_ascii=False,
                                     indent=4),
            path=f'{pdf_file_name}_middle.json',
            mode=AbsReaderWriter.MODE_TXT,
        )

    if f_dump_model_json:
        md_writer.write(
            content=json_parse.dumps(orig_model_list,
                                     ensure_ascii=False,
                                     indent=4),
            path=f'{pdf_file_name}_model.json',
            mode=AbsReaderWriter.MODE_TXT,
        )

    if f_dump_orig_pdf:
        md_writer.write(
            content=pdf_bytes,
            path=f'{pdf_file_name}_origin.pdf',
            mode=AbsReaderWriter.MODE_BIN,
        )

    content_list = pipe.pipe_mk_uni_format(image_dir, drop_mode=DropMode.NONE)
    if f_dump_content_list:
        md_writer.write(
            content=json_parse.dumps(content_list,
                                     ensure_ascii=False,
                                     indent=4),
            path=f'{pdf_file_name}_content_list.json',
            mode=AbsReaderWriter.MODE_TXT,
        )

    logger.info(f'local output dir is {local_md_dir}')


parse_pdf_methods = click.Choice(['ocr', 'txt', 'auto'])
