from __future__ import annotations

import asyncio
from contextlib import contextmanager

from mineru.doclib.core import file_io


def test_extract_pdf_meta_serializes_pdfium_calls(monkeypatch) -> None:
    state = {"guard_depth": 0, "closed_doc": False}

    @contextmanager
    def _fake_guard():
        state["guard_depth"] += 1
        try:
            yield
        finally:
            state["guard_depth"] -= 1

    class _FakePdf:
        def __len__(self) -> int:
            assert state["guard_depth"] > 0
            return 7

        def get_metadata_dict(self) -> dict[str, str]:
            assert state["guard_depth"] > 0
            return {
                "Title": "Test Title",
                "Author": "Test Author",
                "Subject": "Test Subject",
                "Keywords": "alpha,beta",
            }

    monkeypatch.setattr(file_io, "pdfium_guard", _fake_guard)
    monkeypatch.setattr(file_io, "open_pdfium_document", lambda opener, filepath: _FakePdf())
    monkeypatch.setattr(file_io, "close_pdfium_document", lambda pdf: state.__setitem__("closed_doc", True))

    result = {
        "page_count": None,
        "title": None,
        "author": None,
        "subject": None,
        "keywords": None,
        "is_image_based": 0,
    }
    asyncio.run(file_io._extract_pdf_meta("dummy.pdf", result))

    assert result["page_count"] == 7
    assert result["title"] == "Test Title"
    assert result["author"] == "Test Author"
    assert result["subject"] == "Test Subject"
    assert result["keywords"] == "alpha,beta"
    assert state["closed_doc"] is True
