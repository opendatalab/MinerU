# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Any

from loguru import logger

from ..types import BBox, Span
from .image_payload import ImagePayloadCache, image_path_from_key
from .pdf_image_tools import get_crop_img
from .pdf_reader import image_to_bytes


def cut_image_and_table(
    span: Span,
    page_pil_img: Any,
    page_img_md5: str,
    page_id: int,
    scale: int = 2,
    image_cache: ImagePayloadCache | None = None,
) -> Span:

    def return_path(path_type: str) -> str:
        return f"{path_type}/{page_img_md5}"

    span_type = span.type

    if not check_img_bbox(span.bbox):
        span.image_path = ""
    else:
        filename = f"{page_id}_{int(span.bbox[0])}_{int(span.bbox[1])}_{int(span.bbox[2])}_{int(span.bbox[3])}"
        path_key = f"{return_path(span_type)}_{filename}"
        crop_img = get_crop_img(span.bbox, page_pil_img, scale=scale)
        img_bytes = image_to_bytes(crop_img, image_format="JPEG")
        span.image_path = image_path_from_key(path_key, "jpg")
        if image_cache is not None:
            image_cache.register_bytes(img_bytes, "jpeg", image_path=span.image_path)

    return span


def check_img_bbox(bbox: BBox) -> bool:
    if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
        logger.warning(f"image_bboxes: 错误的box, {bbox}")
        return False
    return True
