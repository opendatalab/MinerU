from __future__ import annotations

from typing import Any

from mineru.utils import pdf_document


class _TrackingLock:
    def __init__(self) -> None:
        self.depth = 0

    def __enter__(self) -> None:
        self.depth += 1

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.depth -= 1


def test_pdf_document_methods_keep_page_access_inside_pdfium_lock(monkeypatch) -> None:
    lock = _TrackingLock()
    monkeypatch.setattr(pdf_document, "_pdfium_lock", lock)

    events: list[str] = []

    class _FakeBitmap:
        def to_pil(self) -> str:
            events.append(f"bitmap.to_pil:{lock.depth}")
            return "image"

        def close(self) -> None:
            events.append(f"bitmap.close:{lock.depth}")

    class _FakePage:
        def get_bbox(self) -> tuple[float, float, float, float]:
            events.append(f"page.get_bbox:{lock.depth}")
            return (0.0, 10.0, 20.0, 0.0)

        def render(self, *, scale: int) -> _FakeBitmap:
            events.append(f"page.render:{lock.depth}:{scale}")
            return _FakeBitmap()

    class _FakeDoc:
        def __init__(self) -> None:
            self.page = _FakePage()

        def __len__(self) -> int:
            events.append(f"doc.__len__:{lock.depth}")
            return 1

        def __getitem__(self, page_idx: int) -> _FakePage:
            events.append(f"doc.__getitem__:{lock.depth}:{page_idx}")
            return self.page

        def close(self) -> None:
            events.append(f"doc.close:{lock.depth}")

    fake_doc = _FakeDoc()
    monkeypatch.setattr(pdf_document, "open_pdfium_document", lambda opener, pdf_bytes: fake_doc)
    monkeypatch.setattr(pdf_document, "get_page_chars", lambda page: {"depth": lock.depth})
    monkeypatch.setattr(
        pdf_document,
        "get_text_quality_signal_pdfium",
        lambda doc, page_indices: {"depth": lock.depth, "page_indices": page_indices},
    )

    doc = pdf_document.PDFDocument(b"%PDF")

    assert doc.page_size(0) == (20.0, 10.0)
    assert doc.render_page(0, scale=3) == "image"
    assert doc.get_page_chars(0) == {"depth": 1}
    assert doc.get_text_quality() == {"depth": 1, "page_indices": [0]}

    assert "doc.__getitem__:2:0" in events
    assert "page.get_bbox:1" in events
    assert "page.render:1:3" in events
    assert "bitmap.to_pil:1" in events
    assert "bitmap.close:1" in events
    assert "doc.__len__:1" in events
