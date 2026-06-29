from __future__ import annotations

from typing import Any

from pdftext.schema import Bbox
from PIL import Image

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
        def to_pil(self) -> Image.Image:
            events.append(f"bitmap.to_pil:{lock.depth}")
            return Image.new("RGB", (2, 2), "white")

        def close(self) -> None:
            events.append(f"bitmap.close:{lock.depth}")

    class _FakePage:
        def get_bbox(self) -> tuple[float, float, float, float]:
            events.append(f"page.get_bbox:{lock.depth}")
            return (0.0, 10.0, 20.0, 0.0)

        def get_size(self) -> tuple[int, int]:
            events.append(f"page.get_size:{lock.depth}")
            return 20, 10

        def get_textpage(self) -> "_FakeTextPage":
            events.append(f"page.get_textpage:{lock.depth}")
            return _FakeTextPage()

        def get_rotation(self) -> int:
            events.append(f"page.get_rotation:{lock.depth}")
            return 0

        def render(self, *, scale: float) -> _FakeBitmap:
            events.append(f"page.render:{lock.depth}:{scale}")
            return _FakeBitmap()

    class _FakeTextPage:
        def close(self) -> None:
            events.append(f"textpage.close:{lock.depth}")

    class _FakeDoc:
        def __init__(self, pdf_bytes: bytes) -> None:
            events.append(f"doc.open:{lock.depth}:{pdf_bytes!r}")
            self.page = _FakePage()

        def __len__(self) -> int:
            events.append(f"doc.__len__:{lock.depth}")
            return 1

        def __getitem__(self, page_idx: int) -> _FakePage:
            events.append(f"doc.__getitem__:{lock.depth}:{page_idx}")
            return self.page

        def close(self) -> None:
            events.append(f"doc.close:{lock.depth}")

    def fake_get_chars(textpage: _FakeTextPage, page_bbox: list[float], page_rotation: int) -> list[dict[str, Any]]:
        """记录文本抽取时的锁深度，避免依赖旧模块级 get_page_chars 钩子。"""
        events.append(f"get_chars:{lock.depth}:{page_bbox}:{page_rotation}")
        return [
            {
                "char": "A",
                "bbox": Bbox([0.0, 0.0, 1.0, 1.0]),
                "rotation": 0,
                "font": {"name": "Helvetica", "flags": 0, "size": 10, "weight": 400},
                "char_idx": 0,
            }
        ]

    monkeypatch.setattr(pdf_document.pdfium, "PdfDocument", _FakeDoc)
    monkeypatch.setattr(pdf_document, "get_chars", fake_get_chars, raising=False)
    monkeypatch.setattr(pdf_document, "pdftext_get_chars", fake_get_chars, raising=False)

    doc = pdf_document.PDFDocument(b"%PDF")

    assert doc.page_size(0) == (20.0, 10.0)
    image = doc.render_page(0, scale=3)
    assert image.pil_image.size == (2, 2)
    assert image.scale == 3
    assert doc.get_page_chars(0)[0]["char"] == "A"

    assert any(event.startswith("doc.open:") and not event.startswith("doc.open:0:") for event in events)
    assert any(event.startswith("doc.__getitem__:") and not event.startswith("doc.__getitem__:0:") for event in events)
    assert "page.get_bbox:1" in events
    assert "page.get_size:1" in events
    assert "page.render:1:3" in events
    assert "bitmap.to_pil:1" in events
    assert "bitmap.close:1" in events
    assert "page.get_textpage:1" in events
    assert "get_chars:1:[0.0, 10.0, 20.0, 0.0]:0" in events
    assert "textpage.close:1" in events


def test_pdf_document_does_not_expose_legacy_compat_hooks() -> None:
    assert not hasattr(pdf_document, "pdf_page_to_image")
    assert not hasattr(pdf_document, "open_pdfium_document")
    assert not hasattr(pdf_document, "get_text_quality_signal_pdfium")
    assert not hasattr(pdf_document.PDFDocument, "get_text_quality")
    assert pdf_document.PDFDocument._pdf_doc.fset is None
