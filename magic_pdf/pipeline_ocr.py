import sys
import time

from loguru import logger

from magic_pdf.dict2md.ocr_mkcontent import ocr_mk_mm_markdown, ocr_mk_nlp_markdown_with_para, \
    ocr_mk_mm_markdown_with_para_and_pagination, ocr_mk_mm_markdown_with_para, ocr_mk_mm_standard_format, \
    make_standard_format_with_para
from magic_pdf.libs.commons import join_path, s3_image_save_path, formatted_time
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.pdf_parse_by_ocr import parse_pdf_by_ocr
from magic_pdf.spark.base import get_data_source, exception_handler, get_pdf_bytes, get_bookname
from magic_pdf.spark.s3 import get_s3_config


def ocr_pdf_intermediate_dict_to_markdown(jso: dict, debug_mode=False) -> dict:
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            book_name = join_path(get_data_source(jso), jso["file_id"])
            logger.info(f"book_name is:{book_name} need drop", file=sys.stderr)
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso["pdf_intermediate_dict"]
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        markdown_content = ocr_mk_mm_markdown(pdf_intermediate_dict)
        jso["content"] = markdown_content
        logger.info(
            f"book_name is:{get_data_source(jso)}/{jso['file_id']},markdown content length is {len(markdown_content)}",
            file=sys.stderr,
        )
        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def ocr_pdf_intermediate_dict_to_markdown_with_para(jso: dict, debug_mode=False) -> dict:
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            book_name = join_path(get_data_source(jso), jso["file_id"])
            logger.info(f"book_name is:{book_name} need drop", file=sys.stderr)
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso["pdf_intermediate_dict"]
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        # markdown_content = ocr_mk_mm_markdown_with_para(pdf_intermediate_dict)
        markdown_content = ocr_mk_nlp_markdown_with_para(pdf_intermediate_dict)
        jso["content"] = markdown_content
        logger.info(
            f"book_name is:{get_data_source(jso)}/{jso['file_id']},markdown content length is {len(markdown_content)}",
            file=sys.stderr,
        )
        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def ocr_pdf_intermediate_dict_to_markdown_with_para_and_pagination(jso: dict, debug_mode=False) -> dict:
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            book_name = join_path(get_data_source(jso), jso["file_id"])
            logger.info(f"book_name is:{book_name} need drop", file=sys.stderr)
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso["pdf_intermediate_dict"]
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        markdown_content = ocr_mk_mm_markdown_with_para_and_pagination(pdf_intermediate_dict)
        jso["content"] = markdown_content
        logger.info(
            f"book_name is:{get_data_source(jso)}/{jso['file_id']},markdown content length is {len(markdown_content)}",
            file=sys.stderr,
        )
        # 把无用的信息清空
        # jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        # jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def ocr_pdf_intermediate_dict_to_markdown_with_para_for_qa(
        jso: dict, debug_mode=False
) -> dict:
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            book_name = join_path(get_data_source(jso), jso["file_id"])
            logger.info(f"book_name is:{book_name} need drop", file=sys.stderr)
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso["pdf_intermediate_dict"]
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        markdown_content = ocr_mk_mm_markdown_with_para(pdf_intermediate_dict)
        jso["content_ocr"] = markdown_content
        logger.info(
            f"book_name is:{get_data_source(jso)}/{jso['file_id']},markdown content length is {len(markdown_content)}",
            file=sys.stderr,
        )
        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["mid_json_ocr"] = pdf_intermediate_dict
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def ocr_pdf_intermediate_dict_to_standard_format(jso: dict, debug_mode=False) -> dict:
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            book_name = join_path(get_data_source(jso), jso["file_id"])
            logger.info(f"book_name is:{book_name} need drop", file=sys.stderr)
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso["pdf_intermediate_dict"]
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        standard_format = ocr_mk_mm_standard_format(pdf_intermediate_dict)
        jso["content_list"] = standard_format
        logger.info(
            f"book_name is:{get_data_source(jso)}/{jso['file_id']},content_list length is {len(standard_format)}",
            file=sys.stderr,
        )
        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def ocr_pdf_intermediate_dict_to_standard_format_with_para(jso: dict, debug_mode=False) -> dict:
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            book_name = join_path(get_data_source(jso), jso["file_id"])
            logger.info(f"book_name is:{book_name} need drop", file=sys.stderr)
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso["pdf_intermediate_dict"]
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        standard_format = make_standard_format_with_para(pdf_intermediate_dict)
        jso["content_list"] = standard_format
        logger.info(
            f"book_name is:{get_data_source(jso)}/{jso['file_id']},content_list length is {len(standard_format)}",
            file=sys.stderr,
        )
        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def ocr_parse_pdf_core(pdf_bytes, model_output_json_list, book_name, start_page_id=0, debug_mode=False):
    save_path = s3_image_save_path
    image_s3_config = get_s3_config(save_path)
    start_time = time.time()  # 记录开始时间
    # 先打印一下book_name和解析开始的时间
    logger.info(
        f"book_name is:{book_name},start_time is:{formatted_time(start_time)}",
        file=sys.stderr,
    )
    pdf_info_dict = parse_pdf_by_ocr(
        pdf_bytes,
        model_output_json_list,
        save_path,
        book_name,
        pdf_model_profile=None,
        image_s3_config=image_s3_config,
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


# 专门用来跑被drop的pdf，跑完之后需要把need_drop字段置为false
def ocr_dropped_parse_pdf(jso: dict, start_page_id=0, debug_mode=False) -> dict:
    if not jso.get("need_drop", False):
        return jso
    else:
        try:
            pdf_bytes = get_pdf_bytes(jso)
            model_output_json_list = jso.get("doc_layout_result")
            book_name = get_bookname(jso)
            pdf_info_dict, parse_time = ocr_parse_pdf_core(
                pdf_bytes, model_output_json_list, book_name, start_page_id=start_page_id, debug_mode=debug_mode
            )
            jso["pdf_intermediate_dict"] = JsonCompressor.compress_json(pdf_info_dict)
            jso["parse_time"] = parse_time
            jso["need_drop"] = False
        except Exception as e:
            jso = exception_handler(jso, e)
        return jso


def ocr_parse_pdf(jso: dict, start_page_id=0, debug_mode=False) -> dict:
    # 检测debug开关
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            return jso
    try:
        pdf_bytes = get_pdf_bytes(jso)
        model_output_json_list = jso.get("doc_layout_result")
        book_name = get_bookname(jso)
        pdf_info_dict, parse_time = ocr_parse_pdf_core(pdf_bytes, model_output_json_list, book_name,
                                                       start_page_id=start_page_id, debug_mode=debug_mode)
        jso["pdf_intermediate_dict"] = JsonCompressor.compress_json(pdf_info_dict)
        jso["parse_time"] = parse_time
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso
