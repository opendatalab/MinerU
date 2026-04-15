# Copyright (c) Opendatalab. All rights reserved.
import threading
from io import BytesIO
from contextlib import contextmanager
from typing import Any, Callable, Sequence, TypeVar

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
        if output_doc is not None:
            close_pdfium_document(output_doc)
        if pdf_doc is not None:
            close_pdfium_document(pdf_doc)
