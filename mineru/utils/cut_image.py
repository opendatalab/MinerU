from loguru import logger

from .pdf_image_tools import cut_image


def cut_image_and_table(span, page_pil_img, page_img_md5, page_id, image_writer, scale=2):

    def return_path(path_type):
        return f"{path_type}/{page_img_md5}"

    span_type = span["type"]

    if not check_img_bbox(span["bbox"]) or not image_writer:
        span["image_path"] = ""
    else:
        span["image_path"] = cut_image(
            span["bbox"], page_id, page_pil_img, return_path=return_path(span_type), image_writer=image_writer, scale=scale
        )

    return span


def check_img_bbox(bbox) -> bool:
    if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
        logger.warning(f"image_bboxes: 错误的box, {bbox}")
        return False
    return True
