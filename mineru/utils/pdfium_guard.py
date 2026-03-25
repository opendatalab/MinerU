import threading
from io import BytesIO
from contextlib import contextmanager
from typing import Any, Callable, Sequence, TypeVar


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

            if page_indices is None:
                normalized_page_indices = list(range(total_page_count))
            else:
                normalized_page_indices = sorted(
                    {
                        page_index
                        for page_index in page_indices
                        if 0 <= page_index < total_page_count
                    }
                )
                if not normalized_page_indices:
                    return b""

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
