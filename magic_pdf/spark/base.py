

from loguru import logger

from magic_pdf.libs.drop_reason import DropReason


def get_data_source(jso: dict):
    data_source = jso.get("data_source")
    if data_source is None:
        data_source = jso.get("file_source")
    return data_source


def exception_handler(jso: dict, e):
    logger.exception(e)
    jso["need_drop"] = True
    jso["drop_reason"] = DropReason.Exception
    jso["exception"] = f"ERROR: {e}"
    return jso

