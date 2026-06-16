"""Flash tier PDF extractor — CPU-only text extraction via pypdfium2."""

from __future__ import annotations

import pypdfium2


def extract_pages_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> list[str]:
    """Extract plain text from each PDF page, preserving empty pages."""

    pdf = pypdfium2.PdfDocument(filepath)
    try:
        total = len(pdf)
        end = total if end_page is None else min(end_page, total)

        pages: list[str] = []
        for i in range(start_page, end):
            page = pdf[i]
            tp = page.get_textpage()
            pages.append(tp.get_text_range() or "")
        return pages
    finally:
        pdf.close()


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
