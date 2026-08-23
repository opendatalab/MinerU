from __future__ import annotations

import asyncio

from mineru.doclib.core import file_io


def test_extract_pdf_meta_serializes_pdfium_calls(monkeypatch) -> None:
    assert not hasattr(file_io, "open_pdfium_document")

    state = {"closed_doc": False}

    class _FakePdf:
        @property
        def page_count(self) -> int:
            return 7

        @property
        def metadata(self) -> dict[str, str]:
            return {
                "Title": "Test Title",
                "Author": "Test Author",
                "Subject": "Test Subject",
                "Keywords": "alpha,beta",
            }

        def close(self) -> None:
            state["closed_doc"] = True

    monkeypatch.setattr(file_io, "PDFDocument", lambda filepath: _FakePdf())

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
    def _fail_open(filepath):
        raise RuntimeError("cannot open pdf")

    monkeypatch.setattr(file_io, "PDFDocument", _fail_open)

    try:
        asyncio.run(file_io.extract_metadata("dummy.pdf"))
    except file_io.MetadataExtractionError as exc:
        assert exc.code == "open_failed"
        assert "cannot open pdf" in str(exc)
    else:
        raise AssertionError("PDF open failure should be classified")


def test_extract_pdf_metadata_read_failure_uses_read_metadata_failed(monkeypatch) -> None:
    state = {"closed_doc": False}

    class _FakePdf:
        @property
        def page_count(self) -> int:
            return 7

        @property
        def metadata(self) -> dict[str, str]:
            raise RuntimeError("cannot read title")

        def close(self) -> None:
            state["closed_doc"] = True

    monkeypatch.setattr(file_io, "PDFDocument", lambda filepath: _FakePdf())

    try:
        asyncio.run(file_io.extract_metadata("dummy.pdf"))
    except file_io.MetadataExtractionError as exc:
        assert exc.code == "read_metadata_failed"
        assert "cannot read title" in str(exc)
    else:
        raise AssertionError("PDF metadata read failure should be classified")

    assert state["closed_doc"] is True
