import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, TypeVar


_pdfium_lock = threading.RLock()

T = TypeVar("T")


@contextmanager
def pdfium_guard():
    with _pdfium_lock:
        yield


@asynccontextmanager
async def aio_pdfium_guard():
    await asyncio.to_thread(_pdfium_lock.acquire)
    try:
        yield
    finally:
        _pdfium_lock.release()


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
