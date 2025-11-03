# Copyright (c) Opendatalab. All rights reserved.
import io

import pypdfium2 as pdfium
from loguru import logger


def get_end_page_id(end_page_id, pdf_page_num):
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
    if end_page_id > pdf_page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = pdf_page_num - 1
    return end_page_id


def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None):
    pdf = pdfium.PdfDocument(pdf_bytes)
    output_pdf = pdfium.PdfDocument.new()
    try:
        end_page_id = get_end_page_id(end_page_id, len(pdf))

        # 选择要导入的页面索引
        page_indices = list(range(start_page_id, end_page_id + 1))

        # 从原PDF导入页面到新PDF
        output_pdf.import_pages(pdf, page_indices)

        # 将新PDF保存到内存缓冲区
        output_buffer = io.BytesIO()
        output_pdf.save(output_buffer)

        # 获取字节数据
        output_bytes = output_buffer.getvalue()
    except Exception as e:
        logger.warning(f"Error in converting PDF bytes: {e}, Using original PDF bytes.")
        output_bytes = pdf_bytes

    pdf.close()
    output_pdf.close()
    return output_bytes
