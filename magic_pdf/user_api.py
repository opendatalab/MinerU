
"""
用户输入：
    model数组，每个元素代表一个页面
    pdf在s3的路径
    截图保存的s3位置

然后：
    1）根据s3路径，调用spark集群的api,拿到ak,sk,endpoint，构造出s3PDFReader
    2）根据用户输入的s3地址，调用spark集群的api,拿到ak,sk,endpoint，构造出s3ImageWriter

其余部分至于构造s3cli, 获取ak,sk都在code-clean里写代码完成。不要反向依赖！！！

"""
from loguru import logger

from magic_pdf.io import AbsReaderWriter
from magic_pdf.pdf_parse_by_ocr import parse_pdf_by_ocr
from magic_pdf.pdf_parse_by_txt import parse_pdf_by_txt


def parse_txt_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: AbsReaderWriter, is_debug=False, start_page=0, *args,
                  **kwargs):
    """
    解析文本类pdf
    """
    pdf_info_dict = parse_pdf_by_txt(
        pdf_bytes,
        pdf_models,
        imageWriter,
        start_page_id=start_page,
        debug_mode=is_debug,
    )

    pdf_info_dict["parse_type"] = "txt"

    return pdf_info_dict


def parse_ocr_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: AbsReaderWriter, is_debug=False, start_page=0, *args,
                  **kwargs):
    """
    解析ocr类pdf
    """
    pdf_info_dict = parse_pdf_by_ocr(
        pdf_bytes,
        pdf_models,
        imageWriter,
        start_page_id=start_page,
        debug_mode=is_debug,
    )

    pdf_info_dict["_parse_type"] = "ocr"

    return pdf_info_dict


def parse_union_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: AbsReaderWriter, is_debug=False, start_page=0,
                    *args, **kwargs):
    """
    ocr和文本混合的pdf，全部解析出来
    """

    def parse_pdf(method):
        try:
            return method(
                pdf_bytes,
                pdf_models,
                imageWriter,
                start_page_id=start_page,
                debug_mode=is_debug,
            )
        except Exception as e:
            logger.error(f"{method.__name__} error: {e}")
            return None

    pdf_info_dict = parse_pdf(parse_pdf_by_txt)

    if pdf_info_dict is None or pdf_info_dict.get("_need_drop", False):
        logger.warning(f"parse_pdf_by_txt drop or error, switch to parse_pdf_by_ocr")
        pdf_info_dict = parse_pdf(parse_pdf_by_ocr)
        if pdf_info_dict is None:
            raise Exception("Both parse_pdf_by_txt and parse_pdf_by_ocr failed.")
        else:
            pdf_info_dict["_parse_type"] = "ocr"
    else:
        pdf_info_dict["_parse_type"] = "txt"

    return pdf_info_dict
