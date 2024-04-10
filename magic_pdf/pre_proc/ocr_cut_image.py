from magic_pdf.libs.commons import join_path
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.libs.pdf_image_tools import cut_image


def cut_image_and_table(spans, page, page_id, pdf_bytes_md5, imageWriter):

    def return_path(type):
        return join_path(pdf_bytes_md5, type)

    for span in spans:
        span_type = span['type']
        if span_type == ContentType.Image:
            span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('images'), imageWriter=imageWriter)
        elif span_type == ContentType.Table:
            span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('tables'), imageWriter=imageWriter)

    return spans
