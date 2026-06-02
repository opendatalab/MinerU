# Copyright (c) Opendatalab. All rights reserved.
import threading
from io import BytesIO
from contextlib import contextmanager
from typing import Any, Callable, Sequence, TypeVar

from loguru import logger

from mineru.utils.pdf_page_id import get_end_page_id


_pdfium_lock = threading.RLock()

T = TypeVar("T")


@contextmanager
def pdfium_guard():
    with _pdfium_lock:
        yield


def open_pdfium_document(
    opener: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    with pdfium_guard():
        return opener(*args, **kwargs)


def get_pdfium_document_page_count(pdf_doc) -> int:
    with pdfium_guard():
        return len(pdf_doc)


def close_pdfium_document(pdf_doc) -> None:
    if pdf_doc is None:
        return
    with pdfium_guard():
        pdf_doc.close()


def close_pdfium_child(pdfium_obj) -> None:
    """显式关闭 PDFium 子对象，避免依赖 weakref/finalizer 延迟释放 native 资源。"""
    if pdfium_obj is None:
        return
    close = getattr(pdfium_obj, "close", None)
    if callable(close):
        with pdfium_guard():
            close()


def close_pdfium_objects_safely(*pdfium_objs, owner: str = "pdfium cleanup") -> None:
    """清理多个 PDFium 对象时逐个尝试关闭，避免前一个关闭失败阻断后续对象释放。"""
    for pdfium_obj in pdfium_objs:
        if pdfium_obj is None:
            continue
        try:
            close_pdfium_child(pdfium_obj)
        except Exception as exc:
            logger.warning(f"Failed to close PDFium object during {owner}: {exc}")


def get_loadable_pdfium_page_indices(
    src_pdf_bytes: bytes,
    start_page_id: int = 0,
    end_page_id: int | None = None,
) -> tuple[list[int], list[int]]:
    """逐页探测 PDFium 可加载页面，返回可保留页和损坏页的 0-based 索引。"""
    import pypdfium2 as pdfium

    loadable_page_indices = []
    broken_page_indices = []
    pdf_doc = None

    try:
        with pdfium_guard():
            pdf_doc = pdfium.PdfDocument(src_pdf_bytes)
            total_page_count = len(pdf_doc)
            if total_page_count == 0:
                return [], []

            normalized_start_page_id = max(0, start_page_id)
            normalized_end_page_id = get_end_page_id(end_page_id, total_page_count)
            if normalized_start_page_id > normalized_end_page_id:
                return [], []

            for page_index in range(
                normalized_start_page_id,
                normalized_end_page_id + 1,
            ):
                page = None
                try:
                    page = pdf_doc[page_index]
                    page.get_size()
                    loadable_page_indices.append(page_index)
                except Exception:
                    broken_page_indices.append(page_index)
                finally:
                    close_pdfium_child(page)
    finally:
        close_pdfium_document(pdf_doc)

    return loadable_page_indices, broken_page_indices


def rewrite_pdf_bytes_with_pdfium(
    src_pdf_bytes: bytes,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    page_indices: Sequence[int] | None = None,
) -> bytes:
    import pypdfium2 as pdfium

    pdf_doc = None
    output_doc = None
    try:
        with pdfium_guard():
            pdf_doc = pdfium.PdfDocument(src_pdf_bytes)
            total_page_count = len(pdf_doc)
            if total_page_count == 0:
                return b""

            if page_indices is not None:
                normalized_page_indices = sorted(
                    {
                        page_index
                        for page_index in page_indices
                        if 0 <= page_index < total_page_count
                    }
                )
                if not normalized_page_indices:
                    return b""
            else:
                normalized_end_page_id = get_end_page_id(end_page_id, total_page_count)
                normalized_page_indices = list(
                    range(start_page_id, normalized_end_page_id + 1)
                )

            output_doc = pdfium.PdfDocument.new()
            output_doc.import_pages(pdf_doc, normalized_page_indices)

            output_buffer = BytesIO()
            output_doc.save(output_buffer)
            return output_buffer.getvalue()
    finally:
        close_pdfium_objects_safely(
            output_doc,
            pdf_doc,
            owner="rewrite_pdf_bytes_with_pdfium",
        )
