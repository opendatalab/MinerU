# Copyright (c) Opendatalab. All rights reserved.
import base64
from io import BytesIO

from loguru import logger
from PIL import Image
from pypdfium2 import PdfBitmap, PdfPage
from mineru.utils.pdfium_guard import pdfium_guard


def page_to_image(
    page: PdfPage,
    dpi: int = 200,
    max_width_or_height: int = 3500,  # changed from 4500 to 3500
) -> (Image.Image, float):
    with pdfium_guard():
        scale = dpi / 72

        long_side_length = max(*page.get_size())
        if (long_side_length*scale) > max_width_or_height:
            scale = max_width_or_height / long_side_length

        bitmap: PdfBitmap = page.render(scale=scale)  # type: ignore

        image = bitmap.to_pil()
        try:
            bitmap.close()
        except Exception as e:
            logger.error(f"Failed to close bitmap: {e}")
    return image, scale


def image_to_bytes(
    image: Image.Image,
    # image_format: str = "PNG",  # 也可以用 "JPEG"
    image_format: str = "JPEG",
) -> bytes:
    with BytesIO() as image_buffer:
        image.save(image_buffer, format=image_format)
        return image_buffer.getvalue()


def image_to_b64str(
    image: Image.Image,
    # image_format: str = "PNG",  # 也可以用 "JPEG"
    image_format: str = "JPEG",
) -> str:
    image_bytes = image_to_bytes(image, image_format)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
