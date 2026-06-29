from __future__ import annotations

from contextlib import contextmanager

from mineru.parser.pdf import PdfFlashParser


def test_flash_parser_preserves_page_indices_for_empty_pages(monkeypatch) -> None:
    from mineru.backend.flash import pdf_extractor

    parser = PdfFlashParser()
    monkeypatch.setattr(PdfFlashParser, "_pdf_bytes_to_tempfile", staticmethod(lambda pdf_bytes: "dummy.pdf"))
    monkeypatch.setattr(pdf_extractor, "extract_pages_text", lambda filepath: ["first", "", "third"])

    pages = parser._run_analysis(b"%PDF", page_index_map=[10, 12, 13])

    assert [page.page_idx for page in pages] == [10, 12, 13]
    assert [len(page.para_blocks) for page in pages] == [1, 0, 1]


def test_flash_pdf_extractor_serializes_pdfium_calls(monkeypatch) -> None:
    from mineru.backend.flash import pdf_extractor

    assert not hasattr(pdf_extractor, "open_pdfium_document")

    state = {"guard_depth": 0, "closed_children": 0, "closed_doc": False}

    @contextmanager
    def _fake_guard():
        state["guard_depth"] += 1
        try:
            yield
        finally:
            state["guard_depth"] -= 1

    class _FakeTextPage:
        def get_text_range(self) -> str:
            assert state["guard_depth"] > 0
            return "page text"

        def close(self) -> None:
            state["closed_children"] += 1

    class _FakePage:
        def get_textpage(self) -> _FakeTextPage:
            assert state["guard_depth"] > 0
            return _FakeTextPage()

        def close(self) -> None:
            state["closed_children"] += 1

    class _FakePdf:
        def __len__(self) -> int:
            assert state["guard_depth"] > 0
            return 2

        def __getitem__(self, index: int) -> _FakePage:
            assert state["guard_depth"] > 0
            assert index in (0, 1)
            return _FakePage()

    monkeypatch.setattr(pdf_extractor, "pdfium_guard", _fake_guard)
    monkeypatch.setattr(pdf_extractor.pypdfium2, "PdfDocument", lambda filepath: _FakePdf())
    monkeypatch.setattr(pdf_extractor, "close_pdfium_child", lambda obj: obj.close())
    monkeypatch.setattr(pdf_extractor, "close_pdfium_document", lambda pdf: state.__setitem__("closed_doc", True))

    assert pdf_extractor.extract_pages_text("dummy.pdf") == ["page text", "page text"]
    assert state["closed_children"] == 4
    assert state["closed_doc"] is True
