"""Flash tier PDF extractor — CPU-only text extraction."""

from __future__ import annotations

from ...utils.pdf_document import PDFDocument


def extract_pages_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> list[str]:
    """Extract plain text from each PDF page, preserving empty pages."""

    pages: list[str] = []
    with PDFDocument(filepath) as pdf_doc:
        page_count = pdf_doc.page_count
        end = page_count if end_page is None else min(end_page, page_count)

        # TODO: jzj here has issue
        for page_idx in range(start_page, end):
            pages.append(pdf_doc.get_page_text(page_idx))
    return pages
