from __future__ import annotations

import pytest

from mineru.errors import MineruError
from mineru.parser.pdf import PdfFlashParser


def test_flash_parser_preserves_page_indices_for_empty_pages(monkeypatch) -> None:
    from mineru.backend.flash import pdf_extractor

    parser = PdfFlashParser()
    monkeypatch.setattr(PdfFlashParser, "_pdf_bytes_to_tempfile", staticmethod(lambda pdf_bytes: "dummy.pdf"))
    monkeypatch.setattr(pdf_extractor, "extract_pages_text", lambda filepath: ["first", "", "third"])

    pages, model_output = parser._run_analysis(b"%PDF", page_index_map=[10, 12, 13])

    assert model_output is None
    assert [page.page_idx for page in pages] == [10, 12, 13]
    assert [len(page.para_blocks) for page in pages] == [1, 0, 1]


def test_flash_parser_unlinks_temp_pdf_after_success(monkeypatch, tmp_path) -> None:
    from mineru.backend.flash import pdf_extractor

    temp_pdf = tmp_path / "flash-success.pdf"
    temp_pdf.write_bytes(b"%PDF")
    parser = PdfFlashParser()
    monkeypatch.setattr(PdfFlashParser, "_pdf_bytes_to_tempfile", staticmethod(lambda pdf_bytes: str(temp_pdf)))
    monkeypatch.setattr(pdf_extractor, "extract_pages_text", lambda filepath: ["page text"])

    parser._run_analysis(b"%PDF")

    assert not temp_pdf.exists()


def test_flash_parser_unlinks_temp_pdf_after_extractor_failure(monkeypatch, tmp_path) -> None:
    from mineru.backend.flash import pdf_extractor

    temp_pdf = tmp_path / "flash-failure.pdf"
    temp_pdf.write_bytes(b"%PDF")
    parser = PdfFlashParser()
    monkeypatch.setattr(PdfFlashParser, "_pdf_bytes_to_tempfile", staticmethod(lambda pdf_bytes: str(temp_pdf)))

    def _raise(_filepath: str) -> list[str]:
        raise RuntimeError("extract failed")

    monkeypatch.setattr(pdf_extractor, "extract_pages_text", _raise)

    with pytest.raises(RuntimeError, match="extract failed"):
        parser._run_analysis(b"%PDF")

    assert not temp_pdf.exists()


def test_pdf_parser_rejects_explicit_page_range_with_no_selected_pages(monkeypatch) -> None:
    from mineru.utils import pdf_document, pdfium_guard

    class _FakePdfDocument:
        def __init__(self, pdf_bytes: bytes) -> None:
            self.page_count = 5

    parser = PdfFlashParser()
    monkeypatch.setattr(pdf_document, "PDFDocument", _FakePdfDocument)
    monkeypatch.setattr(pdfium_guard, "safe_rewrite_pdf_bytes_with_pdfium_result", lambda *args, **kwargs: pytest.fail("rewrite called"))

    with pytest.raises(MineruError) as exc_info:
        parser._maybe_adjust_pdf_bytes(b"%PDF", "pdf", "6")

    assert exc_info.value.code == "page_range_invalid"


def test_flash_pdf_extractor_serializes_pdfium_calls(monkeypatch) -> None:
    from mineru.backend.flash import pdf_extractor

    assert not hasattr(pdf_extractor, "open_pdfium_document")
    assert not hasattr(pdf_extractor, "pdfium_guard")

    state = {"entered": False, "closed": False, "requested_pages": []}

    class _FakePdfDocument:
        def __init__(self, filepath: str) -> None:
            assert filepath == "dummy.pdf"
            self.page_count = 2

        def __enter__(self) -> "_FakePdfDocument":
            state["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            state["closed"] = True

        def get_page_text(self, page_idx: int) -> str:
            assert state["entered"] is True
            state["requested_pages"].append(page_idx)
            return "page text"

    monkeypatch.setattr(pdf_extractor, "PDFDocument", _FakePdfDocument)

    assert pdf_extractor.extract_pages_text("dummy.pdf") == ["page text", "page text"]
    assert state["requested_pages"] == [0, 1]
    assert state["closed"] is True
