"""Flash tier PDF extractor — CPU-only text extraction via pypdfium2."""

from __future__ import annotations

import pypdfium2

from mineru.utils.pdfium_guard import close_pdfium_objects_safely, pdfium_guard


def extract_pages_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> list[str]:
    """Extract plain text from each PDF page, preserving empty pages."""

    with pdfium_guard():
        pdf = pypdfium2.PdfDocument(filepath)
    try:
        with pdfium_guard():
            total = len(pdf)
        end = total if end_page is None else min(end_page, total)

        pages: list[str] = []
        for i in range(start_page, end):
            page = None
            tp = None
            try:
                with pdfium_guard():
                    page = pdf[i]
                    tp = page.get_textpage()
                    pages.append(tp.get_text_range() or "")
            finally:
                close_pdfium_objects_safely(tp, page, owner="flash page text extraction")
        return pages
    finally:
        close_pdfium_objects_safely(pdf, owner="flash pdf extraction")


def extract_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> str:
    """Extract plain text from a PDF file using pypdfium2.

    Args:
        filepath: Path to the PDF file.
        start_page: 0-based start page (inclusive).
        end_page: 0-based end page (exclusive).  None means all pages.

    Returns:
        Extracted text as a string, one page per `\\n\\n`.
    """

    return "\n\n".join(text for text in extract_pages_text(filepath, start_page, end_page) if text)
