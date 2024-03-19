from magic_pdf.libs.commons import join_path
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.libs.pdf_image_tools import cut_image


def cut_image_and_table(spans, page, page_id, book_name, save_path, img_s3_client):
    def s3_return_path(type):
        return join_path(book_name, type)

    def img_save_path(type):
        return join_path(save_path, s3_return_path(type))

    for span in spans:
        span_type = span['type']
        if span_type == ContentType.Image:
            span['image_path'] = cut_image(span['bbox'], page_id, page, img_save_path('images'), s3_return_path=s3_return_path('images'), img_s3_client=img_s3_client)
        elif span_type == ContentType.Table:
            span['image_path'] = cut_image(span['bbox'], page_id, page, img_save_path('tables'), s3_return_path=s3_return_path('tables'), img_s3_client=img_s3_client)

    return spans
