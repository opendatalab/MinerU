import os
from pathlib import Path
from typing import Tuple
import io

# from app.common.s3 import get_s3_client
from magic_pdf.libs.commons import fitz
from loguru import logger
from magic_pdf.libs.commons import parse_bucket_key, join_path
from magic_pdf.libs.hash_utils import compute_sha256


def cut_image(bbox: Tuple, page_num: int, page: fitz.Page, return_path, imageWriter, upload_switch=True):
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

    imageWriter.write(data=byte_data, path=img_hash256_path, mode="binary")

    return img_hash256_path


def save_images_by_bboxes(book_name: str, page_num: int, page: fitz.Page, save_path: str,
                            image_bboxes: list, images_overlap_backup:list, table_bboxes: list, equation_inline_bboxes: list,
                            equation_interline_bboxes: list, img_s3_client) -> dict:
    """
    返回一个dict, key为bbox, 值是图片地址
    """
    image_info = []
    image_backup_info = []
    table_info = []
    inline_eq_info = []
    interline_eq_info = []

    # 图片的保存路径组成是这样的： {s3_or_local_path}/{book_name}/{images|tables|equations}/{page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg
    s3_return_image_path = join_path(book_name, "images")
    image_save_path = join_path(save_path, s3_return_image_path)

    s3_return_table_path = join_path(book_name, "tables")
    table_save_path = join_path(save_path, s3_return_table_path)

    s3_return_equations_inline_path = join_path(book_name, "equations_inline")
    equation_inline_save_path = join_path(save_path, s3_return_equations_inline_path)

    s3_return_equation_interline_path = join_path(book_name, "equation_interline")
    equation_interline_save_path = join_path(save_path, s3_return_equation_interline_path)


    for bbox in image_bboxes:
        if any([bbox[0]>=bbox[2], bbox[1]>=bbox[3]]):
            logger.warning(f"image_bboxes: 错误的box, {bbox}")
            continue
        
        image_path = cut_image(bbox, page_num, page, image_save_path, s3_return_image_path, img_s3_client)
        image_info.append({"bbox": bbox, "image_path": image_path})
        
    for bbox in images_overlap_backup:
        if any([bbox[0]>=bbox[2], bbox[1]>=bbox[3]]):
            logger.warning(f"images_overlap_backup: 错误的box, {bbox}")
            continue
        image_path = cut_image(bbox, page_num, page, image_save_path, s3_return_image_path, img_s3_client)
        image_backup_info.append({"bbox": bbox, "image_path": image_path})

    for bbox in table_bboxes:
        if any([bbox[0]>=bbox[2], bbox[1]>=bbox[3]]):
            logger.warning(f"table_bboxes: 错误的box, {bbox}")
            continue
        image_path = cut_image(bbox, page_num, page, table_save_path, s3_return_table_path, img_s3_client)
        table_info.append({"bbox": bbox, "image_path": image_path})

    for bbox in equation_inline_bboxes:
        if any([bbox[0]>=bbox[2], bbox[1]>=bbox[3]]):
            logger.warning(f"equation_inline_bboxes: 错误的box, {bbox}")
            continue
        image_path = cut_image(bbox[:4], page_num, page, equation_inline_save_path, s3_return_equations_inline_path, img_s3_client, upload_switch=False)
        inline_eq_info.append({'bbox':bbox[:4], "image_path":image_path, "latex_text":bbox[4]})

    for bbox in equation_interline_bboxes:
        if any([bbox[0]>=bbox[2], bbox[1]>=bbox[3]]):
            logger.warning(f"equation_interline_bboxes: 错误的box, {bbox}")
            continue
        image_path = cut_image(bbox[:4], page_num, page, equation_interline_save_path, s3_return_equation_interline_path, img_s3_client, upload_switch=False)
        interline_eq_info.append({"bbox":bbox[:4], "image_path":image_path, "latex_text":bbox[4]})

    return image_info, image_backup_info,  table_info, inline_eq_info, interline_eq_info