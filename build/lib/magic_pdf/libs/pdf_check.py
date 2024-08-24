from io import BytesIO
import re
import fitz
import numpy as np
from loguru import logger
from pdfminer.high_level import extract_text


def calculate_sample_count(total_page: int):
    """
    根据总页数和采样率计算采样页面的数量。
    """
    select_page_cnt = min(10, total_page)
    return select_page_cnt


def extract_pages(src_pdf_bytes: bytes):
    pdf_docs = fitz.open("pdf", src_pdf_bytes)
    total_page = len(pdf_docs)
    if total_page == 0:
        # 如果PDF没有页面，直接返回空文档
        logger.warning("PDF is empty, return empty document")
        return fitz.Document()
    select_page_cnt = calculate_sample_count(total_page)

    page_num = np.random.choice(total_page, select_page_cnt, replace=False)
    sample_docs = fitz.Document()
    try:
        for index in page_num:
            sample_docs.insert_pdf(pdf_docs, from_page=int(index), to_page=int(index))
    except Exception as e:
        logger.exception(e)
    return sample_docs


def detect_invalid_chars(src_pdf_bytes: bytes) -> bool:
    """"
    检测PDF中是否包含非法字符
    """
    '''pdfminer比较慢,需要先随机抽取10页左右的sample'''
    sample_docs = extract_pages(src_pdf_bytes)
    sample_pdf_bytes = sample_docs.tobytes()
    sample_pdf_file_like_object = BytesIO(sample_pdf_bytes)
    text = extract_text(sample_pdf_file_like_object)
    text = text.replace("\n", "")
    # logger.info(text)
    '''乱码文本用pdfminer提取出来的文本特征是(cid:xxx)'''
    cid_pattern = re.compile(r'\(cid:\d+\)')
    matches = cid_pattern.findall(text)
    cid_count = len(matches)
    cid_len = sum(len(match) for match in matches)
    text_len = len(text)
    if text_len == 0:
        cid_chars_radio = 0
    else:
        cid_chars_radio = cid_count/(cid_count + text_len - cid_len)
    logger.info(f"cid_count: {cid_count}, text_len: {text_len}, cid_chars_radio: {cid_chars_radio}")
    '''当一篇文章存在5%以上的文本是乱码时,认为该文档为乱码文档'''
    if cid_chars_radio > 0.05:
        return False  # 乱码文档
    else:
        return True   # 正常文档
