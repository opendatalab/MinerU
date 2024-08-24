from loguru import logger

from magic_pdf.libs.commons import join_path
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.libs.pdf_image_tools import cut_image


def ocr_cut_image_and_table(spans, page, page_id, pdf_bytes_md5, imageWriter):
    def return_path(type):
        return join_path(pdf_bytes_md5, type)

    for span in spans:
        span_type = span['type']
        if span_type == ContentType.Image:
            if not check_img_bbox(span['bbox']):
                continue
            span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('images'),
                                           imageWriter=imageWriter)
        elif span_type == ContentType.Table:
            if not check_img_bbox(span['bbox']):
                continue
            span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('tables'),
                                           imageWriter=imageWriter)

    return spans


def txt_save_images_by_bboxes(page_num: int, page, pdf_bytes_md5: str,
                              image_bboxes: list, images_overlap_backup: list, table_bboxes: list,
                              equation_inline_bboxes: list,
                              equation_interline_bboxes: list, imageWriter) -> dict:
    """
    返回一个dict, key为bbox, 值是图片地址
    """
    image_info = []
    image_backup_info = []
    table_info = []
    inline_eq_info = []
    interline_eq_info = []

    # 图片的保存路径组成是这样的： {s3_or_local_path}/{book_name}/{images|tables|equations}/{page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg

    def return_path(type):
        return join_path(pdf_bytes_md5, type)

    for bbox in image_bboxes:
        if not check_img_bbox(bbox):
            continue
        image_path = cut_image(bbox, page_num, page, return_path("images"), imageWriter)
        image_info.append({"bbox": bbox, "image_path": image_path})

    for bbox in images_overlap_backup:
        if not check_img_bbox(bbox):
            continue
        image_path = cut_image(bbox, page_num, page, return_path("images"), imageWriter)
        image_backup_info.append({"bbox": bbox, "image_path": image_path})

    for bbox in table_bboxes:
        if not check_img_bbox(bbox):
            continue
        image_path = cut_image(bbox, page_num, page, return_path("tables"), imageWriter)
        table_info.append({"bbox": bbox, "image_path": image_path})

    return image_info, image_backup_info, table_info, inline_eq_info, interline_eq_info


def check_img_bbox(bbox) -> bool:
    if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
        logger.warning(f"image_bboxes: 错误的box, {bbox}")
        return False
    return True
