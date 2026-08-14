# Copyright (c) Opendatalab. All rights reserved.
import pytest
from pdftext.schema import Bbox

from mineru.utils import pdf_text_tool


def test_restore_pdfium_surrogate_pairs_recovers_supplementary_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证合法 surrogate pair 可恢复，真实替换符与孤立 surrogate 不被误判。"""
    raw_codes = [ord("A"), 0xD835, 0xDF03, 0xFFFD, 0xD835, ord("B")]

    class _FakeTextPage:
        raw = object()

        def count_chars(self) -> int:
            """返回测试用 PDFium 字符数量。"""
            return len(raw_codes)

    monkeypatch.setattr(
        pdf_text_tool.pdfium_c,
        "FPDFText_GetUnicode",
        lambda _textpage, char_idx: raw_codes[char_idx],
    )
    chars = [
        {
            "char": text,
            "bbox": Bbox([float(char_idx), 0.0, float(char_idx + 1), 1.0]),
            "rotation": 0,
            "font": {},
            "char_idx": char_idx,
        }
        for char_idx, text in enumerate(["A", "\uFFFD", "\uFFFD", "\uFFFD", "\ud835", "B"])
    ]

    restored = pdf_text_tool._restore_pdfium_surrogate_pairs(chars, _FakeTextPage())

    assert [char["char"] for char in restored] == ["A", "𝜃", "\uFFFD", "\uFFFD", "B"]
    assert [char["char_idx"] for char in restored] == [0, 1, 3, 4, 5]
