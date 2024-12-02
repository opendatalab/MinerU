from loguru import logger

from magic_pdf.config.ocr_content_type import ContentType
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.pdf_image_tools import cut_image


def ocr_cut_image_and_table(spans, page, page_id, pdf_bytes_md5, imageWriter):
    def return_path(type):
        return join_path(pdf_bytes_md5, type)

    for span in spans:
        span_type = span['type']
        if span_type == ContentType.Image:
            if not check_img_bbox(span['bbox']) or not imageWriter:
                continue
            span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('images'),
                                           imageWriter=imageWriter)
        elif span_type == ContentType.Table:
            if not check_img_bbox(span['bbox']) or not imageWriter:
                continue
            span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('tables'),
                                           imageWriter=imageWriter)

    return spans


def check_img_bbox(bbox) -> bool:
    if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
        logger.warning(f'image_bboxes: 错误的box, {bbox}')
        return False
    return True
