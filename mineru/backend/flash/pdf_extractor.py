"""Flash tier PDF extractor — CPU-only text extraction."""

from __future__ import annotations

import pypdfium2

from ...utils.pdfium_guard import (
    close_pdfium_child,
    close_pdfium_document,
    pdfium_guard,
)


def extract_pages_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> list[str]:
    """Extract plain text from each PDF page, preserving empty pages."""

    pages: list[str] = []
    pdf_doc = None
    try:
        with pdfium_guard():
            pdf_doc = pypdfium2.PdfDocument(filepath)
            try:
                page_count = pdf_doc.page_count
            except AttributeError:
                page_count = len(pdf_doc)
            end = page_count if end_page is None else min(end_page, page_count)
            for page_idx in range(start_page, end):
                page = None
                textpage = None
                try:
                    page = pdf_doc[page_idx]
                    textpage = page.get_textpage()
                    pages.append(textpage.get_text_range() or "")
                finally:
                    close_pdfium_child(textpage)
                    close_pdfium_child(page)
    finally:
        close_pdfium_document(pdf_doc)
    return pages
