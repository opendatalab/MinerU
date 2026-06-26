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


def test_extract_pdf_metadata_open_failure_uses_open_failed(monkeypatch) -> None:
    def _fail_open(opener, filepath):
        raise RuntimeError("cannot open pdf")

    monkeypatch.setattr(file_io, "open_pdfium_document", _fail_open)

    try:
        asyncio.run(file_io.extract_metadata("dummy.pdf"))
    except file_io.MetadataExtractionError as exc:
        assert exc.code == "open_failed"
        assert "cannot open pdf" in str(exc)
    else:
        raise AssertionError("PDF open failure should be classified")


def test_extract_pdf_metadata_read_failure_uses_read_metadata_failed(monkeypatch) -> None:
    state = {"closed_doc": False}

    @contextmanager
    def _fake_guard():
        yield

    class _FakePdf:
        def __len__(self) -> int:
            return 7

        def get_metadata_dict(self) -> dict[str, str]:
            raise RuntimeError("cannot read title")

    monkeypatch.setattr(file_io, "pdfium_guard", _fake_guard)
    monkeypatch.setattr(file_io, "open_pdfium_document", lambda opener, filepath: _FakePdf())
    monkeypatch.setattr(file_io, "close_pdfium_document", lambda pdf: state.__setitem__("closed_doc", True))

    try:
        asyncio.run(file_io.extract_metadata("dummy.pdf"))
    except file_io.MetadataExtractionError as exc:
        assert exc.code == "read_metadata_failed"
        assert "cannot read title" in str(exc)
    else:
        raise AssertionError("PDF metadata read failure should be classified")

    assert state["closed_doc"] is True
