# Copyright (c) Opendatalab. All rights reserved.
"""Flash 各格式复用的轻量图片编码能力。"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def image_to_bytes(
    image: Image.Image,
    image_format: str = "JPEG",
) -> bytes:
    """按指定格式把 Pillow 图片编码为字节。"""
    with BytesIO() as image_buffer:
        image.save(image_buffer, format=image_format)
        return image_buffer.getvalue()


def image_to_b64str(
    image: Image.Image,
    image_format: str = "JPEG",
) -> str:
    """按指定格式把 Pillow 图片编码为 data URI。"""
    image_bytes = image_to_bytes(image, image_format)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


__all__ = ["image_to_b64str", "image_to_bytes"]
