# Copyright (c) Opendatalab. All rights reserved.
import re
from io import BytesIO
import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams


def classify(pdf_bytes):
    """
    判断PDF文件是可以直接提取文本还是需要OCR

    Args:
        pdf_bytes: PDF文件的字节数据

    Returns:
        str: 'txt' 表示可以直接提取文本，'ocr' 表示需要OCR
    """
    try:
        # 从字节数据加载PDF
        sample_pdf_bytes = extract_pages(pdf_bytes)
        pdf = pdfium.PdfDocument(sample_pdf_bytes)

        # 获取PDF页数
        page_count = len(pdf)

        # 如果PDF页数为0，直接返回OCR
        if page_count == 0:
            return 'ocr'

        # 总字符数
        total_chars = 0
        # 清理后的总字符数
        cleaned_total_chars = 0
        # 检查的页面数（最多检查10页）
        pages_to_check = min(page_count, 10)

        # 检查前几页的文本
        for i in range(pages_to_check):
            page = pdf[i]
            text_page = page.get_textpage()
            text = text_page.get_text_bounded()
            total_chars += len(text)

            # 清理提取的文本，移除空白字符
            cleaned_text = re.sub(r'\s+', '', text)
            cleaned_total_chars += len(cleaned_text)

        # 计算平均每页字符数
        # avg_chars_per_page = total_chars / pages_to_check
        avg_cleaned_chars_per_page = cleaned_total_chars / pages_to_check

        # 设置阈值：如果每页平均少于50个有效字符，认为需要OCR
        chars_threshold = 50

        # logger.debug(f"PDF分析: 平均每页{avg_chars_per_page:.1f}字符, 清理后{avg_cleaned_chars_per_page:.1f}字符")

        if (avg_cleaned_chars_per_page < chars_threshold) or detect_invalid_chars(sample_pdf_bytes):
            return 'ocr'
        else:
            return 'txt'
    except Exception as e:
        logger.error(f"判断PDF类型时出错: {e}")
        # 出错时默认使用OCR
        return 'ocr'


def extract_pages(src_pdf_bytes: bytes) -> bytes:
    """
    从PDF字节数据中随机提取最多10页，返回新的PDF字节数据

    Args:
        src_pdf_bytes: PDF文件的字节数据

    Returns:
        bytes: 提取页面后的PDF字节数据
    """

    # 从字节数据加载PDF
    pdf = pdfium.PdfDocument(src_pdf_bytes)

    # 获取PDF页数
    total_page = len(pdf)
    if total_page == 0:
        # 如果PDF没有页面，直接返回空文档
        logger.warning("PDF is empty, return empty document")
        return b''

    # 选择最多10页
    select_page_cnt = min(10, total_page)

    # 从总页数中随机选择页面
    page_indices = np.random.choice(total_page, select_page_cnt, replace=False).tolist()

    # 创建一个新的PDF文档
    sample_docs = pdfium.PdfDocument.new()

    try:
        # 将选择的页面导入新文档
        sample_docs.import_pages(pdf, page_indices)

        # 将新PDF保存到内存缓冲区
        output_buffer = BytesIO()
        sample_docs.save(output_buffer)

        # 获取字节数据
        return output_buffer.getvalue()
    except Exception as e:
        logger.exception(e)
        return b''  # 出错时返回空字节


def detect_invalid_chars(sample_pdf_bytes: bytes) -> bool:
    """"
    检测PDF中是否包含非法字符
    """
    '''pdfminer比较慢,需要先随机抽取10页左右的sample'''
    # sample_pdf_bytes = extract_pages(src_pdf_bytes)
    sample_pdf_file_like_object = BytesIO(sample_pdf_bytes)
    laparams = LAParams(
        line_overlap=0.5,
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        boxes_flow=None,
        detect_vertical=False,
        all_texts=False,
    )
    text = extract_text(pdf_file=sample_pdf_file_like_object, laparams=laparams)
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
    # logger.debug(f"cid_count: {cid_count}, text_len: {text_len}, cid_chars_radio: {cid_chars_radio}")
    '''当一篇文章存在5%以上的文本是乱码时,认为该文档为乱码文档'''
    if cid_chars_radio > 0.05:
        return True  # 乱码文档
    else:
        return False   # 正常文档


if __name__ == '__main__':
    with open('/Users/myhloli/pdf/luanma2x10.pdf', 'rb') as f:
        p_bytes = f.read()
        logger.info(f"PDF分类结果: {classify(p_bytes)}")