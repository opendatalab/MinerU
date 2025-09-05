# Copyright (c) Opendatalab. All rights reserved.
import base64
from io import BytesIO

from loguru import logger
from PIL import Image
from pypdfium2 import PdfBitmap, PdfDocument, PdfPage


def page_to_image(
    page: PdfPage,
    dpi: int = 200,
    max_width_or_height: int = 3500,  # changed from 4500 to 3500
) -> (Image.Image, float):
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
    return base64.b64encode(image_bytes).decode("utf-8")


def base64_to_pil_image(
    base64_str: str,
) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    with BytesIO(image_bytes) as image_buffer:
        return Image.open(image_buffer).convert("RGB")


def pdf_to_images(
    pdf: str | bytes | PdfDocument,
    dpi: int = 200,
    max_width_or_height: int = 3500,
    start_page_id: int = 0,
    end_page_id: int | None = None,
) -> list[Image.Image]:
    doc = pdf if isinstance(pdf, PdfDocument) else PdfDocument(pdf)
    page_num = len(doc)

    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else page_num - 1
    if end_page_id > page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = page_num - 1

    images = []
    try:
        for i in range(start_page_id, end_page_id + 1):
            image, _ = page_to_image(doc[i], dpi, max_width_or_height)
            images.append(image)
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return images


def pdf_to_images_bytes(
    pdf: str | bytes | PdfDocument,
    dpi: int = 200,
    max_width_or_height: int = 3500,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    # image_format: str = "PNG",  # 也可以用 "JPEG"
    image_format: str = "JPEG",
) -> list[bytes]:
    images = pdf_to_images(pdf, dpi, max_width_or_height, start_page_id, end_page_id)
    return [image_to_bytes(image, image_format) for image in images]


def pdf_to_images_b64strs(
    pdf: str | bytes | PdfDocument,
    dpi: int = 200,
    max_width_or_height: int = 3500,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    # image_format: str = "PNG",  # 也可以用 "JPEG"
    image_format: str = "JPEG",
) -> list[str]:
    images = pdf_to_images(pdf, dpi, max_width_or_height, start_page_id, end_page_id)
    return [image_to_b64str(image, image_format) for image in images]
