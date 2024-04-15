
from magic_pdf.io import AbsReaderWriter
from magic_pdf.libs.commons import fitz
from loguru import logger
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.hash_utils import compute_sha256


def cut_image(bbox: tuple, page_num: int, page: fitz.Page, return_path, imageWriter:AbsReaderWriter):
    """
    从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径
    save_path：需要同时支持s3和本地, 图片存放在save_path下，文件名是: {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。
    """
    # 拼接文件名
    filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"

    # 老版本返回不带bucket的路径
    img_path = join_path(return_path, filename) if return_path is not None else None

    # 新版本生成平铺路径
    img_hash256_path = f"{compute_sha256(img_path)}.jpg"

    # 将坐标转换为fitz.Rect对象
    rect = fitz.Rect(*bbox)
    # 配置缩放倍数为3倍
    zoom = fitz.Matrix(3, 3)
    # 截取图片
    pix = page.get_pixmap(clip=rect, matrix=zoom)

    byte_data = pix.tobytes(output='jpeg', jpg_quality=95)

    imageWriter.write(byte_data, path=img_hash256_path, mode="binary")
    imageWriter.write(content=byte_data, path=img_hash256_path, mode="binary")

    return img_hash256_path


def save_images_by_bboxes(page_num: int, page: fitz.Page, pdf_bytes_md5: str,
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
        if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
            logger.warning(f"image_bboxes: 错误的box, {bbox}")
            continue

        image_path = cut_image(bbox, page_num, page, return_path("images"), imageWriter)
        image_info.append({"bbox": bbox, "image_path": image_path})

    for bbox in images_overlap_backup:
        if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
            logger.warning(f"images_overlap_backup: 错误的box, {bbox}")
            continue
        image_path = cut_image(bbox, page_num, page, return_path("images"), imageWriter)
        image_backup_info.append({"bbox": bbox, "image_path": image_path})

    for bbox in table_bboxes:
        if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
            logger.warning(f"table_bboxes: 错误的box, {bbox}")
            continue
        image_path = cut_image(bbox, page_num, page, return_path("tables"), imageWriter)
        table_info.append({"bbox": bbox, "image_path": image_path})

    return image_info, image_backup_info, table_info, inline_eq_info, interline_eq_info
