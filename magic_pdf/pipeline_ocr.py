import sys

from loguru import logger

from magic_pdf.dict2md.ocr_mkcontent import ocr_mk_mm_markdown, ocr_mk_nlp_markdown_with_para, \
    ocr_mk_mm_markdown_with_para_and_pagination, ocr_mk_mm_markdown_with_para, ocr_mk_mm_standard_format, \
    make_standard_format_with_para
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.spark.base import get_data_source, exception_handler


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