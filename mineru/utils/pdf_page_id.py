# Copyright (c) Opendatalab. All rights reserved.
from loguru import logger


def get_end_page_id(end_page_id, pdf_page_num):
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
    if end_page_id > pdf_page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = pdf_page_num - 1
    return end_page_id
