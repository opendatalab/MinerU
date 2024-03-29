"""
文本型pdf转化为统一清洗格式
"""

# TODO 移动到spark/目录下

from loguru import logger
from magic_pdf.dict2md.mkcontent import mk_mm_markdown, mk_universal_format
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.spark import exception_handler, get_data_source


def txt_pdf_to_standard_format(jso: dict, debug_mode=False) -> dict:
    """
    变成统一的标准格式
    """
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            book_name = join_path(get_data_source(jso), jso["file_id"])
            logger.info(f"book_name is:{book_name} need drop")
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso["pdf_intermediate_dict"]
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        standard_format = mk_universal_format(pdf_intermediate_dict)
        jso["content_list"] = standard_format
        logger.info(f"book_name is:{get_data_source(jso)}/{jso['file_id']},content_list length is {len(standard_format)}",)
        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def txt_pdf_to_mm_markdown_format(jso: dict, debug_mode=False) -> dict:
    """
    变成多模态的markdown格式
    """
    if debug_mode:
        pass
    else:  # 如果debug没开，则检测是否有needdrop字段
        if jso.get("need_drop", False):
            book_name = join_path(get_data_source(jso), jso["file_id"])
            logger.info(f"book_name is:{book_name} need drop")
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso["pdf_intermediate_dict"]
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        standard_format = mk_universal_format(pdf_intermediate_dict)
        mm_content = mk_mm_markdown(standard_format)
        jso["content_list"] = mm_content
        logger.info(f"book_name is:{get_data_source(jso)}/{jso['file_id']},content_list length is {len(standard_format)}",)
        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso