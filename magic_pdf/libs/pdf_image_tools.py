from io import BytesIO
import cv2
import fitz
import numpy as np
from PIL import Image
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.hash_utils import compute_sha256


def cut_image(bbox: tuple, page_num: int, page: fitz.Page, return_path, imageWriter: DataWriter):
    """从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径 save_path：需要同时支持s3和本地,
    图片存放在save_path下，文件名是:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。"""
    # 拼接文件名
    filename = f'{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}'

    # 老版本返回不带bucket的路径
    img_path = join_path(return_path, filename) if return_path is not None else None

    # 新版本生成平铺路径
    img_hash256_path = f'{compute_sha256(img_path)}.jpg'

    # 将坐标转换为fitz.Rect对象
    rect = fitz.Rect(*bbox)
    # 配置缩放倍数为3倍
    zoom = fitz.Matrix(3, 3)
    # 截取图片
    pix = page.get_pixmap(clip=rect, matrix=zoom)

    byte_data = pix.tobytes(output='jpeg', jpg_quality=95)

    imageWriter.write(img_hash256_path, byte_data)

    return img_hash256_path


def cut_image_to_pil_image(bbox: tuple, page: fitz.Page, mode="pillow"):

    # 将坐标转换为fitz.Rect对象
    rect = fitz.Rect(*bbox)
    # 配置缩放倍数为3倍
    zoom = fitz.Matrix(3, 3)
    # 截取图片
    pix = page.get_pixmap(clip=rect, matrix=zoom)

    if mode == "cv2":
        # 直接转换为numpy数组供cv2使用
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # PyMuPDF使用RGB顺序，而cv2使用BGR顺序
        if pix.n == 3 or pix.n == 4:
            image_result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            image_result = img_array
    elif mode == "pillow":
        # 将字节数据转换为文件对象
        image_file = BytesIO(pix.tobytes(output='png'))
        # 使用 Pillow 打开图像
        image_result = Image.open(image_file)
    else:
        raise ValueError(f"mode: {mode} is not supported.")

    return image_result