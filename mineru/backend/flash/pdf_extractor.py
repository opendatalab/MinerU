"""Flash tier PDF extractor — CPU-only text extraction."""

from __future__ import annotations

import logging
import time

from ...utils.pdf_document import PDFDocument

logger = logging.getLogger(__name__)


def extract_pages_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> list[str]:
    """Extract plain text from each PDF page, preserving empty pages."""

    pages: list[str] = []
    open_started_at = time.perf_counter()
    logger.debug("Flash PDFium document open started")
    with PDFDocument(filepath) as pdf_doc:
        page_count = pdf_doc.page_count
        logger.debug(
            "Flash PDFium document open completed pages=%d elapsed_ms=%d",
            page_count,
            round((time.perf_counter() - open_started_at) * 1000),
        )
        end = page_count if end_page is None else min(end_page, page_count)

        # TODO: jzj here has issue
        for page_idx in range(start_page, end):
            page_started_at = time.perf_counter()
            logger.debug("Flash PDFium text extraction started page_index=%d", page_idx)
            text = pdf_doc.get_page_text(page_idx)
            pages.append(text)
            logger.debug(
                "Flash PDFium text extraction completed page_index=%d text_chars=%d elapsed_ms=%d",
                page_idx,
                len(text),
                round((time.perf_counter() - page_started_at) * 1000),
            )
    return pages
