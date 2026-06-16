from __future__ import annotations

from mineru.parser.pdf import PdfFlashParser


def test_flash_parser_preserves_page_indices_for_empty_pages(monkeypatch) -> None:
    from mineru.backend.flash import pdf_extractor

    parser = PdfFlashParser()
    parser._page_indices = [10, 12, 13]

    monkeypatch.setattr(PdfFlashParser, "_pdf_bytes_to_tempfile", staticmethod(lambda pdf_bytes: "dummy.pdf"))
    monkeypatch.setattr(pdf_extractor, "extract_pages_text", lambda filepath: ["first", "", "third"])

    pages = parser._run_analysis(b"%PDF", image_writer=None)

    assert [page.page_idx for page in pages] == [10, 12, 13]
    assert [len(page.para_blocks) for page in pages] == [1, 0, 1]
