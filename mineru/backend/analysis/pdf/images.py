# Copyright (c) Opendatalab. All rights reserved.
"""共享 PDF 图像服务，并保留 MinerU 的渲染资源配置。"""

from __future__ import annotations
import os
from typing import Any, Literal
from docgale.document.pdf import images as _images
from docgale.document.pdf.images import DEFAULT_PDF_IMAGE_DPI
from docgale.document.pdf.images import (
    ImageType,
    pdf_page_to_image,
    shutdown_pdf_render_executor,
    load_images_from_pdf_core,
    crop_img,
    get_crop_img,
    get_crop_np_img,
)


def _positive_int_env(name: str, default: int) -> int:
    """保留现有正整数配置校验与缺省值。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def get_load_images_timeout() -> int:
    """读取宿主 PDF 页面渲染超时。"""
    return _positive_int_env("MINERU_PDF_RENDER_TIMEOUT", 300)


def get_load_images_threads() -> int:
    """读取宿主 PDF 页面渲染并发限制。"""
    return _positive_int_env("MINERU_PDF_RENDER_THREADS", 3)


def load_images_from_pdf_bytes_range(
    pdf_bytes: bytes,
    dpi: int = DEFAULT_PDF_IMAGE_DPI,
    start_page_id: int = 0,
    end_page_id: int = 0,
    image_type: Literal["pil_img", "base64_img"] = "pil_img",
    timeout: int | None = None,
    threads: int | None = None,
) -> list[dict[str, Any]]:
    """向独立引擎显式传入宿主配置，复用其唯一渲染进程池。"""
    return _images.load_images_from_pdf_bytes_range(
        pdf_bytes,
        dpi=dpi,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        image_type=image_type,
        timeout=get_load_images_timeout() if timeout is None else timeout,
        threads=get_load_images_threads() if threads is None else threads,
    )


__all__ = [
    "ImageType",
    "get_load_images_timeout",
    "get_load_images_threads",
    "pdf_page_to_image",
    "shutdown_pdf_render_executor",
    "load_images_from_pdf_bytes_range",
    "load_images_from_pdf_core",
    "crop_img",
    "get_crop_img",
    "get_crop_np_img",
]
