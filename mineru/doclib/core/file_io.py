"""File I/O utilities: SHA-256, stat, metadata extraction."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import pypdfium2

# Optional office doc support
try:
    from docx import Document
except ImportError:
    Document = None  # type: ignore[assignment]
try:
    from pptx import Presentation
except ImportError:
    Presentation = None  # type: ignore[assignment]
try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None  # type: ignore[assignment]

# Metadata truncation limits
TRUNC_TITLE = 500
TRUNC_AUTHOR = 200
TRUNC_SUBJECT = 1000
TRUNC_KEYWORDS = 1000


# ── SHA-256 ────────────────────────────────────────────────────────


def _sha256_sync(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


async def compute_sha256(filepath: str) -> str:
    return await asyncio.to_thread(_sha256_sync, filepath)


# ── file stat ──────────────────────────────────────────────────────


async def get_file_stat(filepath: str) -> dict:
    def _stat() -> dict:
        st = os.stat(filepath)
        return {
            "size_bytes": st.st_size,
            "mtime_ms": int(st.st_mtime * 1000),
            "birthtime_ms": int(st.st_birthtime * 1000)
            if hasattr(st, "st_birthtime")
            else int(st.st_ctime * 1000),
        }

    return await asyncio.to_thread(_stat)


# ── metadata extraction ────────────────────────────────────────────


async def extract_metadata(filepath: str) -> dict:
    """Extract metadata from file.  Returns dict with keys matching docs table columns.
    All string fields are protectively truncated."""
    ext = Path(filepath).suffix.lower().lstrip(".")

    result = {
        "mime_type": None,
        "page_count": None,
        "title": None,
        "author": None,
        "subject": None,
        "keywords": None,
        "is_encrypted": 0,
        "is_scanned": 0,
    }

    if ext == "pdf":
        await _extract_pdf_meta(filepath, result)

    elif ext in ("docx", "pptx", "xlsx"):
        await _extract_office_meta(filepath, ext, result)

    # truncate all string fields
    for field, limit in [
        ("title", TRUNC_TITLE),
        ("author", TRUNC_AUTHOR),
        ("subject", TRUNC_SUBJECT),
        ("keywords", TRUNC_KEYWORDS),
    ]:
        val = result.get(field)
        if val and len(val) > limit:
            result[field] = val[:limit]

    return result


# ── PDF metadata ───────────────────────────────────────────────────


async def _extract_pdf_meta(filepath: str, result: dict) -> None:
    def _extract() -> None:
        pdf = pypdfium2.PdfDocument(filepath)
        try:
            result["page_count"] = len(pdf)

            meta = pdf.get_metadata_dict() or {}
            result["title"] = meta.get("Title") or None
            result["author"] = meta.get("Author") or None
            result["subject"] = meta.get("Subject") or None
            result["keywords"] = meta.get("Keywords") or None

            # pypdfium2 doesn't expose is_encrypted directly;
            # trying to access pages on an encrypted doc raises.
        finally:
            pdf.close()

    await asyncio.to_thread(_extract)


# ── Office metadata ────────────────────────────────────────────────


async def _extract_office_meta(filepath: str, ext: str, result: dict) -> None:
    def _extract() -> None:
        if ext == "docx":
            if Document is None:
                return
            doc = Document(filepath)
            cp = doc.core_properties
            result["title"] = cp.title or None
            result["author"] = cp.author or None
            result["subject"] = cp.subject or None
            result["keywords"] = cp.keywords or None
            # DOCX has no fixed page count; leave as None (will be set to 1 later by caller)

        elif ext == "pptx":
            if Presentation is None:
                return
            prs = Presentation(filepath)
            cp = prs.core_properties
            result["title"] = cp.title or None
            result["author"] = cp.author or None
            result["subject"] = cp.subject or None
            result["keywords"] = cp.keywords or None
            result["page_count"] = len(prs.slides)

        elif ext == "xlsx":
            if load_workbook is None:
                return
            wb = load_workbook(filepath, read_only=True)
            cp = wb.properties
            result["title"] = cp.title or None
            result["author"] = cp.creator or None
            result["subject"] = cp.subject or None
            result["keywords"] = cp.keywords or None
            result["page_count"] = len(wb.sheetnames)
            wb.close()

    await asyncio.to_thread(_extract)
