# Copyright (c) Opendatalab. All rights reserved.
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Iterator, Sequence, TypeVar

from loguru import logger


_pdfium_lock = threading.RLock()

T = TypeVar("T")


@dataclass
class PdfiumRewriteResult:
    """记录 PDFium 安全重写结果，供调用方按实际保留页修正原始页号。"""

    pdf_bytes: bytes
    retained_page_indices: list[int] | None = None
    broken_page_indices: list[int] = field(default_factory=list)
    used_original: bool = False


@contextmanager
def pdfium_guard() -> Iterator[None]:
    with _pdfium_lock:
        yield


def open_pdfium_document(
    opener: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    with pdfium_guard():
        return opener(*args, **kwargs)


def get_pdfium_document_page_count(pdf_doc: Any) -> int:
    with pdfium_guard():
        return len(pdf_doc)


def close_pdfium_document(pdf_doc: Any) -> None:
    if pdf_doc is None:
        return
    with pdfium_guard():
        pdf_doc.close()


def close_pdfium_child(pdfium_obj: Any) -> None:
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
            normalized_end_page_id = (
                end_page_id
                if end_page_id is not None and end_page_id >= 0
                else total_page_count - 1
            )
            if normalized_end_page_id > total_page_count - 1:
                normalized_end_page_id = total_page_count - 1
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


def _normalize_rewrite_page_indices(
    total_page_count: int,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    page_indices: Sequence[int] | None = None,
) -> list[int]:
    """按 rewrite_pdf_bytes_with_pdfium 的规则归一化实际导出的 0-based 页号。"""
    if total_page_count == 0:
        return []

    if page_indices is not None:
        return sorted({int(page_index) for page_index in page_indices if 0 <= int(page_index) < total_page_count})

    normalized_start_page_id = max(0, start_page_id)
    normalized_end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else total_page_count - 1
    )
    if normalized_end_page_id > total_page_count - 1:
        normalized_end_page_id = total_page_count - 1
    if normalized_start_page_id > normalized_end_page_id:
        return []
    return list(range(normalized_start_page_id, normalized_end_page_id + 1))


def _get_rewrite_page_indices_from_pdf(
    src_pdf_bytes: bytes,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    page_indices: Sequence[int] | None = None,
) -> list[int]:
    """读取源 PDF 页数并计算本次重写会保留的原始页号。"""
    import pypdfium2 as pdfium

    pdf_doc = None
    try:
        with pdfium_guard():
            pdf_doc = pdfium.PdfDocument(src_pdf_bytes)
            return _normalize_rewrite_page_indices(
                len(pdf_doc),
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                page_indices=page_indices,
            )
    finally:
        close_pdfium_document(pdf_doc)


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

            normalized_page_indices = _normalize_rewrite_page_indices(
                total_page_count,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                page_indices=page_indices,
            )
            if not normalized_page_indices:
                return b""

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


def safe_rewrite_pdf_bytes_with_pdfium(
    src_pdf_bytes: bytes,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    page_indices: Sequence[int] | None = None,
) -> bytes:
    """安全重写 PDF 字节；常规重写失败时跳过损坏页并保留可加载页面。"""
    return safe_rewrite_pdf_bytes_with_pdfium_result(
        src_pdf_bytes,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        page_indices=page_indices,
    ).pdf_bytes


def safe_rewrite_pdf_bytes_with_pdfium_result(
    src_pdf_bytes: bytes,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    page_indices: Sequence[int] | None = None,
) -> PdfiumRewriteResult:
    """安全重写 PDF 字节，并返回重写后 PDF 对应的原始页号映射。"""
    try:
        rebuilt_pdf_bytes = rewrite_pdf_bytes_with_pdfium(
            src_pdf_bytes,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            page_indices=page_indices,
        )
        if rebuilt_pdf_bytes:
            retained_page_indices = _get_rewrite_page_indices_from_pdf(
                src_pdf_bytes,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                page_indices=page_indices,
            )
            return PdfiumRewriteResult(
                pdf_bytes=rebuilt_pdf_bytes,
                retained_page_indices=retained_page_indices,
            )
        logger.warning("PDFium rewrite returned empty bytes, trying to skip broken pages.")
    except Exception as fallback_error:
        logger.warning(
            f"Error in converting PDF bytes with pdfium: {fallback_error}, trying to skip broken pages."
        )

    try:
        if page_indices is not None:
            requested_page_indices = sorted(
                {int(page_index) for page_index in page_indices if int(page_index) >= 0}
            )
            if not requested_page_indices:
                logger.warning("PDFium safe rewrite received no valid requested pages, using original PDF bytes.")
                return PdfiumRewriteResult(pdf_bytes=src_pdf_bytes, retained_page_indices=None, used_original=True)
            probe_start_page_id = requested_page_indices[0]
            probe_end_page_id = requested_page_indices[-1]
        else:
            requested_page_indices = None
            probe_start_page_id = start_page_id
            probe_end_page_id = end_page_id

        loadable_page_indices, broken_page_indices = get_loadable_pdfium_page_indices(
            src_pdf_bytes,
            start_page_id=probe_start_page_id,
            end_page_id=probe_end_page_id,
        )
        if requested_page_indices is not None:
            requested_page_index_set = set(requested_page_indices)
            loadable_page_indices = [
                page_index
                for page_index in loadable_page_indices
                if page_index in requested_page_index_set
            ]
            broken_page_indices = [
                page_index
                for page_index in broken_page_indices
                if page_index in requested_page_index_set
            ]

        if broken_page_indices:
            skipped_pages = [page_index + 1 for page_index in broken_page_indices]
            logger.warning(f"Skipped broken PDF pages during PDFium rewrite: {skipped_pages}")
        if not loadable_page_indices:
            logger.warning("PDFium skip-broken-page rewrite found no loadable pages, using original PDF bytes.")
            return PdfiumRewriteResult(
                pdf_bytes=src_pdf_bytes,
                retained_page_indices=None,
                broken_page_indices=broken_page_indices,
                used_original=True,
            )

        rebuilt_pdf_bytes = rewrite_pdf_bytes_with_pdfium(
            src_pdf_bytes,
            start_page_id=probe_start_page_id,
            end_page_id=probe_end_page_id,
            page_indices=loadable_page_indices,
        )
        if rebuilt_pdf_bytes:
            return PdfiumRewriteResult(
                pdf_bytes=rebuilt_pdf_bytes,
                retained_page_indices=loadable_page_indices,
                broken_page_indices=broken_page_indices,
            )
        logger.warning("PDFium skip-broken-page rewrite returned empty bytes, using original PDF bytes.")
    except Exception as fallback_error:
        logger.warning(
            f"Error in converting PDF bytes with skip-broken-page fallback: {fallback_error}, using original PDF bytes."
        )
    return PdfiumRewriteResult(pdf_bytes=src_pdf_bytes, retained_page_indices=None, used_original=True)
