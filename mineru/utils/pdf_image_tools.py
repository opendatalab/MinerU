# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO

import pypdfium2 as pdfium
from loguru import logger
from PIL import Image

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.pdf_reader import image_to_b64str, image_to_bytes, page_to_image
from .hash_utils import str_sha256


def pdf_page_to_image(page: pdfium.PdfPage, dpi=200) -> dict:
    """Convert pdfium.PdfDocument to image, Then convert the image to base64.

    Args:
        page (_type_): pdfium.PdfPage
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img_base64': str, 'img_pil': pil_img, 'scale': float }
    """
    pil_img, scale = page_to_image(page, dpi=dpi)
    img_base64 = image_to_b64str(pil_img)

    image_dict = {
        "img_base64": img_base64,
        "img_pil": pil_img,
        "scale": scale,
    }
    return image_dict


def load_images_from_pdf(
    pdf_bytes: bytes,
    dpi=200,
    start_page_id=0,
    end_page_id=None,
):
    images_list = []
    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    pdf_page_num = len(pdf_doc)
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
    if end_page_id > pdf_page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = pdf_page_num - 1

    for index in range(0, pdf_page_num):
        if start_page_id <= index <= end_page_id:
            page = pdf_doc[index]
            image_dict = pdf_page_to_image(page, dpi=dpi)
            images_list.append(image_dict)

    return images_list, pdf_doc


def cut_image(bbox: tuple, page_num: int, page_pil_img, return_path, image_writer: FileBasedDataWriter, scale=2):
    """从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径 save_path：需要同时支持s3和本地,
    图片存放在save_path下，文件名是:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。"""

    # 拼接文件名
    filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"

    # 老版本返回不带bucket的路径
    img_path = f"{return_path}_{filename}" if return_path is not None else None

    # 新版本生成平铺路径
    img_hash256_path = f"{str_sha256(img_path)}.jpg"
    # img_hash256_path = f'{img_path}.jpg'

    crop_img = get_crop_img(bbox, page_pil_img, scale=scale)

    img_bytes = image_to_bytes(crop_img, image_format="JPEG")

    image_writer.write(img_hash256_path, img_bytes)
    return img_hash256_path


def get_crop_img(bbox: tuple, pil_img, scale=2):
    scale_bbox = (
        int(bbox[0] * scale),
        int(bbox[1] * scale),
        int(bbox[2] * scale),
        int(bbox[3] * scale),
    )
    return pil_img.crop(scale_bbox)


def images_bytes_to_pdf_bytes(image_bytes):
    # 内存缓冲区
    pdf_buffer = BytesIO()

    # 载入并转换所有图像为 RGB 模式
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # 第一张图保存为 PDF，其余追加
    image.save(pdf_buffer, format="PDF", save_all=True)

    # 获取 PDF bytes 并重置指针（可选）
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes
