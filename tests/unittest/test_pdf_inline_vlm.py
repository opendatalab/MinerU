"""验证 VLM 行内切片、混合文本保护和单片 MFR 回退。"""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from docvortex.document.pdf import PDFDocument
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from mineru.backend.analysis.pdf.inline_recognition import (
    finish_inline_recognition,
    normalize_inline_text,
    prepare_inline_recognition,
)
from mineru.backend.analysis.pdf.normalization import _natural_language_spans
from mineru.backend.analysis.pdf.text.content import (
    _apply_window_post_ocr,
    _build_page_text_formula_spans,
    _fill_native_pdf_text_spans,
    _lines_to_block_content,
)
from mineru.backend.analysis.pdf.text.lines import group_spans_to_lines
from mineru.model.vlm.errors import EngineDeadError
from mineru.integrations.docvortex import build_metadata
from mineru.parser.base import ParseResult
from mineru.render import render_html
from mineru.types import MiddleJson, PageInfo, Producer, TextBlock


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("α ≤ 5", "α ≤ 5"),
        (r"前 \(x_1\)，后 \[y^2\]。", r"前 \(x_1\)，后 \(y^2\)。"),
        (r"\( (m_1, m_n) \) . (D)", r"\((m_1, m_n)\) . (D)"),
        (r"US$5 and $10", r"US$5 and $10"),
        (r"\$4.48 ± \$1.02", "$4.48 ± $1.02"),
        (r"price \$4.48; \(x+\$1\); \[y\] then \$1.02", r"price $4.48; \(x+\$1\); \(y\) then $1.02"),
        (r"literal \\$5 and \\\$6", r"literal \\$5 and \\$6"),
        (r"C:\data\file.txt \alpha \% \_ \$5", r"C:\data\file.txt \alpha \% \_ $5"),
        (r"\(\text{\$4.48}\) and \$1.02", r"\(\text{\$4.48}\) and $1.02"),
        (r"literal \\(x\\) and \(y\)", r"literal \\(x\\) and \(y\)"),
        ("(p =", "(p ="),
        (r"\((p =\)", r"\((p =\)"),
    ],
)
def test_inline_text_normalization(source: str, expected: str) -> None:
    """仅处理模型公式定界符，正文括号、Unicode、美元及转义文字不猜测补全。"""
    assert normalize_inline_text(source) == expected


@pytest.mark.parametrize("source", [None, "", "  ", "[Non-Text]", r"\(\)", r"\(x", r"x\)", r"\[x\)", r"\(a\(b\)\)"])
def test_invalid_inline_response_requests_fallback(source: str | None) -> None:
    """空结果及残缺定界符必须触发回退，不能在遮罩后静默丢字。"""
    with pytest.raises(ValueError):
        normalize_inline_text(source)


def test_semantic_formula_split_respects_backslash_parity() -> None:
    """转义定界符保留文字，公式边界前的成对反斜杠保留在原内容中。"""
    assert _natural_language_spans(r"literal \\(x\\) and \(y\)") == [
        {"type": "text", "content": r"literal \\(x\\) and "},
        {"type": "equation_inline", "content": "y"},
    ]
    assert _natural_language_spans(r"\\\(x\\\)") == [
        {"type": "text", "content": r"\\"},
        {"type": "equation_inline", "content": "x\\\\"},
    ]


def test_vlm_currency_roundtrip_keeps_semantic_text_and_renderer_escaping_separate() -> None:
    """真实回填与序列化后金额为纯文本，Markdown 只转义一次，HTML 不泄漏反斜杠。"""
    raw = r"\$4.48 ± \$1.02"
    with Image.new("RGB", (100, 30)) as image:
        batch = prepare_inline_recognition([[_formula([1, 1, 99, 29])]], [[]], [image])
        try:
            formulas = finish_inline_recognition(batch, [raw], SimpleNamespace(), [], page_offset=0)
            spans = _build_page_text_formula_spans(formulas[0], [], (100, 30), 1.0)
            content = _lines_to_block_content(group_spans_to_lines(spans), "text")
        finally:
            batch.close()
    semantic = _natural_language_spans(content)
    assert semantic == [{"type": "text", "content": "$4.48 ± $1.02"}]
    middle = MiddleJson(
        pages=[PageInfo(page_idx=0, blocks=[TextBlock(type="text", index=0, bbox=(0, 0, 1, 1), content=semantic)])],
        is_full_document=True,
        metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version="test")},
        extensions=build_metadata(effort="high", parse_mode="txt"),
    )
    result = ParseResult.from_json(ParseResult(middle).to_json())
    assert result.pages[0].blocks[0].content[0].content == "$4.48 ± $1.02"
    assert result.markdown() == raw
    rendered = render_html(result.middle_json)
    assert "$4.48 ± $1.02" in rendered
    assert r"\$" not in rendered


def test_currency_unescaping_does_not_change_native_text_or_mfr_latex() -> None:
    """修复限定在 VLM 文本入口，原生文本与 MFR 公式的美元转义保持原义。"""
    assert _natural_language_spans(r"literal \$5") == [{"type": "text", "content": r"literal \$5"}]
    spans = _build_page_text_formula_spans([{**_formula([1, 1, 20, 10]), "latex": r"\$5"}], [], (100, 30), 1.0)
    content = _lines_to_block_content(group_spans_to_lines(spans), "text")
    assert _natural_language_spans(content) == [{"type": "equation_inline", "content": r"\$5"}]


def _formula(bbox: list[float]) -> dict[str, Any]:
    """构造可追踪的原始检测框。"""
    return {"label": "inline_formula", "bbox": bbox, "score": 0.9}


def test_crop_selection_and_mixed_results_preserve_original_masks() -> None:
    """以最终表格过滤切片，保持 MFR 裁边规则且不改原始检测数据。"""
    layouts = [[_formula([1.2, 2.8, 9.1, 8.2]), _formula([30, 30, 40, 40]), _formula([200, 0, 210, 10])], []]
    before = deepcopy(layouts)
    with Image.new("RGB", (100, 100)) as image:
        batch = prepare_inline_recognition(layouts, [[{"type": "table", "bbox": [0.2, 0.2, 0.5, 0.5]}], []], [image, image])
        try:
            assert len(batch.crops) == 1
            assert batch.images[0].size == (9, 7)
            result = finish_inline_recognition(batch, [r"\(x\) . (D)"], SimpleNamespace(), [], page_offset=0)
            assert result[0][0]["content"] == r"\(x\) . (D)"
            assert result[0][0]["content_format"] == "text_with_inline_math"
            assert result[1] == []
            assert layouts == before
            crop = batch.images[0]
        finally:
            batch.close()
        with pytest.raises(ValueError, match="closed image"):
            crop.getpixel((0, 0))


def test_only_failed_crops_reach_mfr_in_page_order() -> None:
    """只把不同页的失败切片交给 MFR，识别相同文本也不能错配几何落点。"""
    layout = [[_formula([1, 1, 10, 10]), _formula([12, 1, 20, 10])], [_formula([1, 1, 10, 10])]]
    with Image.new("RGB", (40, 40)) as image:
        batch = prepare_inline_recognition(layout, [[], []], [image, image])
        mfr = MagicMock(return_value=[[{**layout[0][1], "latex": "x_2"}], [{**layout[1][0], "latex": "x_3"}]])
        try:
            result = finish_inline_recognition(
                batch,
                ["α", None, ValueError("request failed")],
                SimpleNamespace(mfr_model=SimpleNamespace(batch_predict=mfr)),
                [np.zeros((40, 40, 3))] * 2,
                page_offset=64,
            )
            inputs = mfr.call_args.args[0]
            assert [[f["bbox"] for f in page] for page in inputs] == [[layout[0][1]["bbox"]], [layout[1][0]["bbox"]]]
            assert mfr.call_args.kwargs["interline_enable"] is False
            assert result[0][0]["content"] == "α"
            assert result[0][1]["latex"] == "x_2"
            assert result[1][0]["latex"] == "x_3"
            assert "content_format" not in result[0][1]
        finally:
            batch.close()


@pytest.mark.parametrize("failure", ["empty", "count", "identity", "engine", "vlm_count"])
def test_fallback_failure_and_contract_errors_are_not_swallowed(failure: str) -> None:
    """MFR 二次失败、返回错配及引擎死亡必须显式报错。"""
    formula = _formula([1, 1, 10, 10])
    with Image.new("RGB", (20, 20)) as image:
        batch = prepare_inline_recognition([[formula]], [[]], [image])
        answer = {**formula, "latex": "" if failure == "empty" else "x"}
        if failure == "identity":
            answer["bbox"] = [2, 2, 3, 3]
        mfr = MagicMock(return_value=[] if failure == "count" else [[answer]])
        outputs = [] if failure == "vlm_count" else [EngineDeadError("dead") if failure == "engine" else None]
        try:
            with pytest.raises((ValueError, EngineDeadError)):
                finish_inline_recognition(
                    batch, outputs, SimpleNamespace(mfr_model=SimpleNamespace(batch_predict=mfr)), [], page_offset=0
                )
            if failure in {"engine", "vlm_count"}:
                mfr.assert_not_called()
        finally:
            batch.close()


@pytest.mark.parametrize("recognized", ["α ≤ 5", r"\(m_n\) . (D)", r"literal \\(x\\) and \(y\)"])
def test_native_and_post_ocr_cannot_overwrite_vlm_text(recognized: str) -> None:
    """用真实 PDF 原生字符验证几何占位保护，最终才拆出没有几何字段的语义 span。"""
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(200, 80))
    canvas.setFont("Helvetica", 12)
    canvas.drawString(10, 40, "before WRONG after")
    canvas.save()
    with PDFDocument(stream.getvalue()) as document, Image.new("RGB", (200, 80)) as image:
        page = document[0]
        chars = [char for char in page.get_chars() if char["char"] not in "\r\n"]

        def bbox(selected: list[dict[str, Any]]) -> list[float]:
            """由真实字符框计算三段互不重叠的检测位置。"""
            return [
                min(c["bbox"][0] for c in selected),
                min(c["bbox"][1] for c in selected),
                max(c["bbox"][2] for c in selected),
                max(c["bbox"][3] for c in selected),
            ]

        formula = {**_formula(bbox(chars[7:12])), "content": recognized, "content_format": "text_with_inline_math"}
        ocr = [{"bbox": bbox(chars[:6]), "text": ""}, {"bbox": bbox(chars[13:]), "text": ""}]
        spans = _build_page_text_formula_spans([formula], ocr, page.size, 1.0)
        spans = _fill_native_pdf_text_spans(page, spans, image, 1.0, page.size)
        lines = group_spans_to_lines(spans)
        model = SimpleNamespace(ocr_model=SimpleNamespace(ocr=MagicMock()))
        _apply_window_post_ocr(model, [{0: lines}])
        model.ocr_model.ocr.assert_not_called()
        content = _lines_to_block_content(lines, "text")
        assert "WRONG" not in content
        assert content == f"before {recognized} after"
        semantic = _natural_language_spans(content)
        assert all(set(span) <= {"type", "content"} for span in semantic)
        if recognized == "α ≤ 5":
            assert semantic == [{"type": "text", "content": "before α ≤ 5 after"}]
        else:
            assert sum(span["type"] == "equation_inline" for span in semantic) == 1
            assert semantic[-1]["content"].endswith(" after")
