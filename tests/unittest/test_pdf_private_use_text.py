"""验证三个私用区的检测、真实 Rust 回填与最终 ASCII 空格占位。"""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest
import reportlab
from docvortex import _native
from docvortex.analyzers.pdf import PDFTextEvidence
from docvortex.document.pdf import PDFDocument, PDFPageTextGeometry
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from mineru.backend.analysis.pdf.text import content, native
from mineru.backend.analysis.pdf.text.models import _AnalyzeLine, _AnalyzeSpan
from mineru.types import BlockType, ContentType


def _mapped_pdf(text: str) -> bytes:
    """用嵌入字体和真实 UTF-16 ToUnicode 映射构造 PUA，避免缺字被制 PDF 工具改为零。"""
    private_chars = list(dict.fromkeys(char for char in text if native._is_private_use_char(char)))
    replacements = dict(zip(private_chars, "xyz", strict=False))
    assert len(replacements) == len(private_chars) <= 3
    pdfmetrics.registerFont(TTFont("PrivateUseVera", str(Path(reportlab.__file__).parent / "fonts/Vera.ttf")))
    source = BytesIO()
    canvas = Canvas(source, pagesize=(1024, 100))
    canvas.setFont("PrivateUseVera", 12)
    canvas.drawString(10, 50, "".join(replacements.get(char, char) for char in text))
    canvas.save()
    writer = PdfWriter()
    writer.add_page(PdfReader(source).pages[0])
    for font_ref in writer.pages[0]["/Resources"]["/Font"].values():
        font = font_ref.get_object()
        if "/ToUnicode" not in font:
            continue
        mapping = font["/ToUnicode"].get_data()
        for private, marker in replacements.items():
            original = f"<{ord(marker):02X}> <{ord(marker):04X}>".encode()
            assert original in mapping
            mapping = mapping.replace(original, f"<{ord(marker):02X}> <{private.encode('utf-16-be').hex()}>".encode())
        stream = DecodedStreamObject()
        stream.set_data(mapping)
        font[NameObject("/ToUnicode")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0xDFFF, False),
        (0xE000, True),
        (0xF8FF, True),
        (0xF900, False),
        (0xEFFFF, False),
        (0xF0000, True),
        (0xFFFFD, True),
        (0xFFFFE, False),
        (0xFFFFF, False),
        (0x100000, True),
        (0x10FFFD, True),
        (0x10FFFE, False),
        (0x10FFFF, False),
    ],
)
def test_private_use_range_boundaries(code: int, expected: bool) -> None:
    """只包含三个正式私用区，边界外字符和非字符码点不被扩大替换。"""
    assert native._is_private_use_char(chr(code)) is expected


@pytest.mark.parametrize("private", ["\ue000", "\uf8ff", "\U000f0000", "\U000ffffd", "\U00100000", "\U0010fffd", "\U001001b0"])
@pytest.mark.parametrize("kind", ["single", "consecutive", "sparse", "ratio"])
def test_owned_pua_counts_and_ocr_decision_match_python(private: str, kind: str) -> None:
    """三个私用区真实提取后，Rust 与 Python 的计数、连续长度和原阈值决策一致。"""
    text = {
        "single": f"A0{private}93B",
        "consecutive": f"A{private}{private}B",
        "sparse": f"{'A' * 80}{private}B{private}C",
        "ratio": f"A{private}B{private}C",
    }[kind]
    with PDFDocument(_mapped_pdf(text)) as document:
        owner = document[0].get_text_snapshot()
    geometry = owner.materialize_geometry()
    assert "".join(char["char"] for char in geometry.chars) == text
    original = pickle.dumps(geometry.chars)
    boxes = [char["bbox"] for char in geometry.chars]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    results = []
    before = _native.text_snapshot_stats()[2]
    for current in (None, owner):
        span = _AnalyzeSpan(ContentType.TEXT, box, metadata={"chars": [], "height": box[3] - box[1], "width": box[2] - box[0]})
        pending = native.fill_char_in_spans(
            [span],
            geometry.chars,
            12.0,
            tight_bboxes=geometry.tight_bboxes,
            origins=geometry.origins,
            detect_scripts=False,
            _native_text=current,
        )
        assert bool(pending) is (kind in {"consecutive", "ratio"})
        assert private in span.content  # 必须先检测原文，不能在 OCR 决策前替换。
        results.append((asdict(span), bool(pending)))
    assert _native.text_snapshot_stats()[2] == before + 1
    assert results[0] == results[1]
    assert pickle.dumps(geometry.chars) == original


@pytest.mark.parametrize("mode", ["no_ocr", "success", "low_confidence", "low_contrast", "ocr_pua"])
def test_pipeline_replaces_residual_pua_after_ocr(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """完整窗口出口覆盖不识别、识别成功和两种恢复原文路径，公式不参与替换。"""
    original = "0\U001001b093  A\ue000B\U000f0000C"
    span = _AnalyzeSpan(ContentType.TEXT, (0.0, 0.0, 100.0, 20.0), content=original, score=0.9)
    formula = _AnalyzeSpan(ContentType.INLINE_EQUATION, (100.0, 0.0, 120.0, 20.0), content="x\ue000")
    lines = {0: [_AnalyzeLine((0.0, 0.0, 120.0, 20.0), [span, formula])]}
    context = MagicMock()
    if mode != "no_ocr":
        span.metadata[native.POST_OCR_FALLBACK_CONTENT_KEY] = original
        span.metadata[native.POST_OCR_FALLBACK_SCORE_KEY] = span.score
        span.metadata[native.POST_OCR_REASON_KEY] = native.POST_OCR_REASON_PRIVATE_USE_TEXT
    if mode == "low_contrast":
        monkeypatch.setattr(native, "get_crop_img", lambda *_args: np.zeros((20, 100, 3), dtype=np.uint8))
        native._prepare_post_ocr_spans([span], [span], object(), 1.0)
    elif mode != "no_ocr":
        span.image = np.zeros((20, 100, 3), dtype=np.uint8)
        span.content = ""
        recognized = "0.93  ABC" if mode == "success" else original
        context.ocr_model.ocr.return_value = [[(recognized, 0.1 if mode == "low_confidence" else 0.99)]]
    monkeypatch.setattr(content, "_build_page_text_formula_spans", lambda *_args: [])
    monkeypatch.setattr(content, "_fill_native_pdf_text_spans", lambda _page, spans, *_args, **_kwargs: spans)
    monkeypatch.setattr(content, "_group_page_spans_by_block", lambda *_args: lines)
    monkeypatch.setattr(
        content,
        "prepare_text_evidence",
        lambda *_args, **_kwargs: PDFTextEvidence((120.0, 100.0), PDFPageTextGeometry([], {}, {})),
    )
    model = [[{"type": BlockType.TEXT, "bbox": [0.0, 0.0, 1.0, 1.0]}]]
    content._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [MagicMock(size=(120.0, 100.0))],
        model,
        [[]],
        [[]],
        "txt",
        "medium",
        {BlockType.TEXT},
        context,
    )
    expected = "0.93  ABC" if mode == "success" else "0 93  A B C"
    assert span.content == expected
    assert formula.content == "x\ue000"
    assert expected in str(model[0][0]["content"])
    assert "093" not in span.content
    assert native.POST_OCR_FALLBACK_CONTENT_KEY not in span.metadata
    assert context.ocr_model.ocr.call_count == (0 if mode in {"no_ocr", "low_contrast"} else 1)


def test_replacement_preserves_non_pua_and_existing_whitespace() -> None:
    """逐码点替换不折叠原空白，不影响非 BMP 汉字、表情或非字符码点。"""
    text = "  A\t中\U00020000😀\ue000\U000f0000\U001001b0\U0010fffeB\n "
    assert native._replace_private_use_chars(text) == "  A\t中\U00020000😀   \U0010fffeB\n "
