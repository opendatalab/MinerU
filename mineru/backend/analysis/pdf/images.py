# Copyright (c) Opendatalab. All rights reserved.
"""共享 PDF 图像服务，并保留 MinerU 的渲染资源配置。"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Literal
from docvortex.document.pdf import images as _images
from docvortex.document.pdf.images import DEFAULT_PDF_IMAGE_DPI
from docvortex.document.pdf.images import (
    ImageType,
    pdf_page_to_image,
    shutdown_pdf_render_executor,
    load_images_from_pdf_core,
    crop_img,
    get_crop_img,
    get_crop_np_img,
)


if TYPE_CHECKING:
    from docvortex.document.pdf import PDFDocument, PDFRenderSession


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


def get_document_render_session(document: PDFDocument) -> PDFRenderSession | None:
    """按共享后端选择复用文档会话，选中新接口但引擎缺失时明确报错。"""
    selector = getattr(_images, "get_pdf_render_backend", None)
    if selector is None:
        if os.environ.get("DOCVORTEX_PDF_RENDER_BACKEND", "legacy").strip().lower() != "legacy":
            raise RuntimeError("Installed DocVortex does not support PDF render sessions")
        return None
    if selector() != "session":
        return None
    factory = getattr(document, "get_render_session", None)
    if not callable(factory):
        raise RuntimeError("PDF document does not support get_render_session; upgrade DocVortex")
    return factory(threads=get_load_images_threads(), timeout=get_load_images_timeout())


def load_images_from_pdf_bytes_range(
    pdf_bytes: bytes,
    dpi: int = DEFAULT_PDF_IMAGE_DPI,
    start_page_id: int = 0,
    end_page_id: int = 0,
    image_type: Literal["pil_img", "base64_img"] = "pil_img",
    timeout: int | None = None,
    threads: int | None = None,
    *,
    document: PDFDocument | None = None,
    session: PDFRenderSession | None = None,
) -> list[dict[str, Any]]:
    """向独立引擎传入宿主配置，按后端复用同一文档的渲染会话。"""
    if getattr(_images, "get_pdf_render_backend", None) is None:
        if session is not None or os.environ.get("DOCVORTEX_PDF_RENDER_BACKEND", "legacy").strip().lower() != "legacy":
            raise RuntimeError("Installed DocVortex does not support PDF render sessions")
    if session is None and document is not None:
        session = get_document_render_session(document)
    return _images.load_images_from_pdf_bytes_range(
        pdf_bytes,
        dpi=dpi,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        image_type=image_type,
        timeout=get_load_images_timeout() if timeout is None else timeout,
        threads=get_load_images_threads() if threads is None else threads,
        **({"session": session} if session is not None else {}),
    )


__all__ = [
    "ImageType",
    "get_document_render_session",
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
