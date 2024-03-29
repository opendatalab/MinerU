
from loguru import logger

from magic_pdf.libs.commons import read_file
from magic_pdf.libs.drop_reason import DropReason

from magic_pdf.spark.s3 import get_s3_config


def get_data_source(jso: dict):
    data_source = jso.get("data_source")
    if data_source is None:
        data_source = jso.get("file_source")
    return data_source


def get_data_type(jso: dict):
    data_type = jso.get("data_type")
    if data_type is None:
        data_type = jso.get("file_type")
    return data_type


def get_bookid(jso: dict):
    book_id = jso.get("bookid")
    if book_id is None:
        book_id = jso.get("original_file_id")
    return book_id


def exception_handler(jso: dict, e):
    logger.exception(e)
    jso["need_drop"] = True
    jso["drop_reason"] = DropReason.Exception
    jso["exception"] = f"ERROR: {e}"
    return jso


def get_bookname(jso: dict):
    data_source = get_data_source(jso)
    file_id = jso.get("file_id")
    book_name = f"{data_source}/{file_id}"
    return book_name


def get_pdf_bytes(jso: dict):
    pdf_s3_path = jso.get("file_location")
    s3_config = get_s3_config(pdf_s3_path)
    pdf_bytes = read_file(pdf_s3_path, s3_config)
    return pdf_bytes