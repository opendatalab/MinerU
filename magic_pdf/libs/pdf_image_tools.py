import os
from pathlib import Path
from typing import Tuple
import io

# from app.common.s3 import get_s3_client
from magic_pdf.libs.commons import fitz
from loguru import logger
from magic_pdf.libs.commons import parse_bucket_key, join_path
from magic_pdf.libs.hash_utils import compute_sha256


def cut_image(bbox: Tuple, page_num: int, page: fitz.Page, save_parent_path: str, s3_return_path=None, img_s3_client=None, upload_switch=True):
    """
    从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径
    save_path：需要同时支持s3和本地, 图片存放在save_path下，文件名是: {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。
    """
    # 拼接文件名
    filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}.jpg"

    # 老版本返回不带bucket的路径
    s3_img_path = join_path(s3_return_path, filename) if s3_return_path is not None else None

    # 新版本生成s3的平铺路径
    s3_img_hash256_path = f"{compute_sha256(s3_img_path)}.jpg"

    # 打印图片文件名
    # print(f"Saved {image_save_path}")

    #检查坐标
    # x_check = int(bbox[2]) - int(bbox[0])
    # y_check = int(bbox[3]) - int(bbox[1])
    # if x_check <= 0 or y_check <= 0:
    #
    #     if image_save_path.startswith("s3://"):
    #         logger.exception(f"传入图片坐标有误，x1<x0或y1<y0,{s3_img_path}")
    #         return s3_img_path
    #     else:
    #         logger.exception(f"传入图片坐标有误，x1<x0或y1<y0,{image_save_path}")
    #         return image_save_path


    # 将坐标转换为fitz.Rect对象
    rect = fitz.Rect(*bbox)
    # 配置缩放倍数为3倍
    zoom = fitz.Matrix(3, 3)
    # 截取图片
    pix = page.get_pixmap(clip=rect, matrix=zoom)

    if save_parent_path.startswith("s3://"):
        if not upload_switch:
            pass
        else:
            """图片保存到s3"""
            # 从save_parent_path获取bucket_name
            bucket_name, bucket_key = parse_bucket_key(save_parent_path)
            # 平铺路径赋值给bucket_key
            bucket_key = s3_img_hash256_path

            # 将字节流上传到s3
            byte_data = pix.tobytes(output='jpeg', jpg_quality=95)
            file_obj = io.BytesIO(byte_data)
            if img_s3_client is not None:
                img_s3_client.upload_fileobj(file_obj, bucket_name, bucket_key)
                # 每个图片上传任务都创建一个新的client
                # img_s3_client_once = get_s3_client(image_save_path)
                # img_s3_client_once.upload_fileobj(file_obj, bucket_name, bucket_key)
            else:
                logger.exception("must input img_s3_client")
        # return s3_img_path # 早期版本要求返回不带bucket的路径
        s3_image_save_path = f"s3://{bucket_name}/{s3_img_hash256_path}"  # 新版本返回平铺的s3路径
        return s3_image_save_path
    else:
        # 保存图片到本地
        # 先检查一下image_save_path的父目录是否存在，如果不存在，就创建
        local_image_save_path = join_path(save_parent_path, filename)
        parent_dir = os.path.dirname(local_image_save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        pix.save(local_image_save_path, jpg_quality=95)
        # 为了直接能在markdown里看，这里把地址改为相对于mardown的地址
        pth = Path(local_image_save_path)
        local_image_save_path = f"{pth.parent.name}/{pth.name}"
        return local_image_save_path


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