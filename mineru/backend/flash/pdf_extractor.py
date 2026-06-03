"""Flash tier PDF extractor — CPU-only text extraction via pypdfium2."""

from __future__ import annotations


def extract_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> str:
    """Extract plain text from a PDF file using pypdfium2.

    Args:
        filepath: Path to the PDF file.
        start_page: 0-based start page (inclusive).
        end_page: 0-based end page (exclusive).  None means all pages.

    Returns:
        Extracted text as a string, one page per `\\n\\n`.
    """
    import pypdfium2

    pdf = pypdfium2.PdfDocument(filepath)
    try:
        total = len(pdf)
        end = total if end_page is None else min(end_page, total)

        parts: list[str] = []
        for i in range(start_page, end):
            page = pdf[i]
            tp = page.get_textpage()
            text = tp.get_text_range()
            if text:
                parts.append(text)

        return "\n\n".join(parts)
    finally:
        pdf.close()
