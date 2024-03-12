from magic_pdf.libs.commons import join_path
from magic_pdf.libs.pdf_image_tools import cut_image


def cut_image_and_table(spans, page, page_id, book_name, save_path):
    def s3_return_path(type):
        return join_path(book_name, type)

    def img_save_path(type):
        return join_path(save_path, s3_return_path(type))

    for span in spans:
        span_type = span['type']
        if span_type == 'image':
            span['image_path'] = cut_image(span['bbox'], page_id, page, img_save_path('images'))
        elif span_type == 'table':
            span['image_path'] = cut_image(span['bbox'], page_id, page, img_save_path('tables'))

    return spans
