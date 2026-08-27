# Copyright (c) Opendatalab. All rights reserved.
from loguru import logger
from PIL import Image
from pypdfium2 import PdfBitmap, PdfPage

from .pdfium import pdfium_guard


def page_to_image(
    page: PdfPage,
    dpi: int = 200,
    max_width_or_height: int = 3500,  # changed from 4500 to 3500
) -> tuple[Image.Image, float]:
    with pdfium_guard():
        scale = dpi / 72

        long_side_length = max(*page.get_size())
        if (long_side_length * scale) > max_width_or_height:
            scale = max_width_or_height / long_side_length

        bitmap: PdfBitmap | None = None
        try:
            bitmap = page.render(scale=scale)  # type: ignore
            image = bitmap.to_pil().copy()
        finally:
            if bitmap is not None:
                try:
                    bitmap.close()
                except Exception as e:
                    logger.error(f"Failed to close bitmap: {e}")
    return image, scale


__all__ = ["page_to_image"]
