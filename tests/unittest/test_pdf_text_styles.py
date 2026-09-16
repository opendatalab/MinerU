from __future__ import annotations

import html
from typing import Any
from unittest.mock import MagicMock

import pytest
from _span_test_utils import inline_text
from docvortex.analyzers.pdf import (
    PDFTextEvidence,
    PDFTextLinkLine,
    PDFTextLinkRange,
    PDFTextScriptLine,
    PDFTextScriptRange,
    PDFTextStyleLine,
    PDFTextStyleRange,
)
from docvortex.document.pdf import PDFPageTextGeometry

from mineru.backend.analysis.pdf.text import content as text_content
from mineru.backend.analysis.pdf.text.models import _AnalyzeLine, _AnalyzeSpan
from mineru.backend.analysis.pdf.text.native import txt_spans_extract
from mineru.types import (
    BlockType,
    ContentType,
)


def _span_snapshot(content: Any) -> str:
    """把结构化 Span 序列化为便于沿用既有精确区间断言的只读快照。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for span in content:
        if not isinstance(span, dict):
            continue
        span_type = span.get("type")
        value = span.get("content")
        if span_type == "text" and isinstance(value, str):
            styles = span.get("styles")
            if isinstance(styles, list) and styles:
                parts.append(f'<text style="{",".join(str(style) for style in styles)}">{value}</text>')
            else:
                parts.append(value)
        elif span_type == "equation_inline" and isinstance(value, str):
            parts.append(f"<eq>{value}</eq>")
        elif span_type == "code_inline" and isinstance(value, str):
            parts.append(f"<code>{value}</code>")
        elif span_type == "hyperlink":
            url = span.get("url")
            if isinstance(url, str):
                parts.append(f"<hyperlink>{_span_snapshot(value)}<url>{html.escape(url, quote=False)}</url></hyperlink>")
    return "".join(parts)


def test_hybrid_txt_reuses_loaded_chars_and_applies_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Medium/High/XHigh 复用 chars，并应用允许的组合样式。"""

    page_chars = [{"char": "d", "bbox": (10.0, 10.0, 16.0, 20.0), "char_idx": 0}]
    page_text_geometry = PDFPageTextGeometry(
        chars=page_chars,  # type: ignore[arg-type]
        tight_bboxes={0: (10.0, 10.0, 16.0, 20.0)},
        origins={0: (10.0, 20.0)},
    )
    style_lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 52.0, 20.0),
            "deleted",
            (
                PDFTextStyleRange(
                    0,
                    7,
                    ("bold", "italic", "underline", "strikethrough"),
                ),
            ),
            0,
        )
    ]
    link_lines = [
        PDFTextLinkLine(
            (10.0, 10.0, 52.0, 20.0),
            "deleted",
            (PDFTextLinkRange(0, 7, "https://hybrid.example.test"),),
            0,
        )
    ]
    build_styles = MagicMock(
        return_value=PDFTextEvidence((100.0, 100.0), page_text_geometry, tuple(style_lines), tuple(link_lines))
    )
    observed_geometry: list[object] = []

    def fake_fill_native(
        _pdf_page: object,
        _page_spans: object,
        _page_image: object,
        _scale: object,
        _page_size: object,
        *,
        page_text_geometry: object,
        detect_scripts: bool,
    ) -> list[_AnalyzeSpan]:
        """记录传入原生回填的字符几何，并返回稳定文本 span。"""

        observed_geometry.append(page_text_geometry)
        assert detect_scripts is False
        return [
            _AnalyzeSpan(
                type=ContentType.TEXT,
                bbox=(10.0, 10.0, 52.0, 20.0),
                content="deleted",
                score=1.0,
            )
        ]

    monkeypatch.setattr(
        text_content,
        "prepare_text_evidence",
        build_styles,
    )
    monkeypatch.setattr(text_content, "_fill_native_pdf_text_spans", fake_fill_native)
    monkeypatch.setattr(
        text_content,
        "_build_page_text_formula_spans",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        text_content,
        "_group_page_spans_by_block",
        lambda *_args: {
            0: [
                _AnalyzeLine(
                    bbox=(10.0, 10.0, 52.0, 20.0),
                    spans=[
                        _AnalyzeSpan(
                            type=ContentType.TEXT,
                            bbox=(10.0, 10.0, 52.0, 20.0),
                            content="deleted",
                            score=1.0,
                        )
                    ],
                )
            ]
        },
    )
    pdf_page = MagicMock(size=(100.0, 100.0))
    pdf_page.get_char_count.return_value = 1
    model_list = [[{"type": BlockType.TEXT, "bbox": [0.1, 0.1, 0.52, 0.2], "content": ""}]]

    text_content._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [pdf_page],
        model_list,
        [[]],
        [[]],
        "txt",
        "medium",
        {BlockType.TEXT},
        MagicMock(),
    )

    assert observed_geometry == [page_text_geometry]
    assert _span_snapshot(model_list[0][0]["content"]) == (
        '<hyperlink><text style="bold,strikethrough">deleted</text><url>https://hybrid.example.test</url></hyperlink>'
    )


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
def test_hybrid_txt_efforts_apply_dehyphenated_links(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
) -> None:
    """验证 Medium、High 与 XHigh 共享的 TXT 回填路径应用断词链接投影。"""

    target = f"https://{effort}.example.test"
    link_lines = [
        PDFTextLinkLine(
            (10.0, 10.0, 52.0, 20.0),
            "inter-",
            (PDFTextLinkRange(0, 6, target),),
            0,
        ),
        PDFTextLinkLine(
            (10.0, 30.0, 52.0, 40.0),
            "national",
            (PDFTextLinkRange(0, 8, target),),
            1,
        ),
    ]
    monkeypatch.setattr(
        text_content,
        "prepare_text_evidence",
        lambda *_args, **_kwargs: PDFTextEvidence(
            (100.0, 100.0), PDFPageTextGeometry([], {}, {}), tuple([]), tuple(link_lines), tuple([])
        ),
    )
    monkeypatch.setattr(
        text_content,
        "_build_page_text_formula_spans",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        text_content,
        "_fill_native_pdf_text_spans",
        lambda _page, spans, *_args, **_kwargs: spans,
    )
    monkeypatch.setattr(
        text_content,
        "_group_page_spans_by_block",
        lambda *_args: {
            0: [
                _AnalyzeLine(
                    bbox=(10.0, 10.0, 52.0, 20.0),
                    spans=[
                        _AnalyzeSpan(
                            type=ContentType.TEXT,
                            bbox=(10.0, 10.0, 52.0, 20.0),
                            content="inter-",
                            score=1.0,
                        )
                    ],
                ),
                _AnalyzeLine(
                    bbox=(10.0, 30.0, 52.0, 40.0),
                    spans=[
                        _AnalyzeSpan(
                            type=ContentType.TEXT,
                            bbox=(10.0, 30.0, 52.0, 40.0),
                            content="national",
                            score=1.0,
                        )
                    ],
                ),
            ]
        },
    )
    monkeypatch.setattr(text_content, "_apply_window_post_ocr", lambda *_args: None)
    pdf_page = MagicMock(size=(100.0, 100.0))
    pdf_page.get_char_count.return_value = 1
    model_list = [
        [
            {
                "type": BlockType.TEXT,
                "bbox": [0.1, 0.1, 0.52, 0.4],
                "content": "",
            }
        ]
    ]

    text_content._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [pdf_page],
        model_list,
        [[]],
        [[]],
        "txt",
        effort,
        {BlockType.TEXT},
        MagicMock(),
    )

    assert _span_snapshot(model_list[0][0]["content"]) == (f"<hyperlink>international<url>{target}</url></hyperlink>")


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
def test_hybrid_txt_efforts_apply_shared_script_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
) -> None:
    """验证 Medium、High 与 XHigh 的 TXT 文本统一应用整行脚本 sidecar。"""

    page_geometry = PDFPageTextGeometry([], {}, {})
    script_line = PDFTextScriptLine(
        bbox=(10.0, 10.0, 30.0, 20.0),
        text="x2",
        script_ranges=(PDFTextScriptRange(1, 2, "superscript", (20.0, 10.0, 25.0, 15.0), 1, False),),
        source_index=0,
        angle=0,
    )
    build_evidence = MagicMock(return_value=PDFTextEvidence((100.0, 100.0), page_geometry, scripts=(script_line,)))
    monkeypatch.setattr(text_content, "prepare_text_evidence", build_evidence)
    monkeypatch.setattr(text_content, "_build_page_text_formula_spans", lambda *_args: [])

    def fake_fill_native(*_args: object, detect_scripts: bool, **_kwargs: object) -> list[_AnalyzeSpan]:
        """确认 Hybrid TXT 不再使用 layout span 内的旧脚本分类。"""

        assert detect_scripts is False
        return [_AnalyzeSpan(type=ContentType.TEXT, bbox=(10.0, 10.0, 30.0, 20.0), content="x2", score=1.0)]

    monkeypatch.setattr(text_content, "_fill_native_pdf_text_spans", fake_fill_native)
    monkeypatch.setattr(
        text_content,
        "_group_page_spans_by_block",
        lambda *_args: {
            0: [
                _AnalyzeLine(
                    bbox=(10.0, 10.0, 30.0, 20.0),
                    spans=[_AnalyzeSpan(type=ContentType.TEXT, bbox=(10.0, 10.0, 30.0, 20.0), content="x2", score=1.0)],
                )
            ]
        },
    )
    monkeypatch.setattr(text_content, "_apply_window_post_ocr", lambda *_args: None)
    pdf_page = MagicMock(size=(100.0, 100.0))
    pdf_page.get_char_count.return_value = 2
    model_list = [[{"type": BlockType.TEXT, "bbox": [0.1, 0.1, 0.3, 0.2], "content": ""}]]

    text_content._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [pdf_page],
        model_list,
        [[]],
        [[]],
        "txt",
        effort,  # type: ignore[arg-type]
        {BlockType.TEXT},
        MagicMock(),
    )

    assert model_list[0][0]["content"] == [
        {"type": "text", "content": "x"},
        {"type": "text", "content": "2", "styles": ["superscript"]},
    ]


def test_ocr_path_does_not_collect_or_apply_pdf_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 OCR 路径不会读取或应用 PDF 原生文本样式与超链接。"""

    build_styles = MagicMock()
    monkeypatch.setattr(
        text_content,
        "prepare_text_evidence",
        build_styles,
    )
    monkeypatch.setattr(
        text_content,
        "_build_page_text_formula_spans",
        lambda *_args: [
            _AnalyzeSpan(
                type=ContentType.TEXT,
                bbox=(10.0, 10.0, 52.0, 20.0),
                content="deleted",
                score=1.0,
            )
        ],
    )
    monkeypatch.setattr(
        text_content,
        "_group_page_spans_by_block",
        lambda *_args: {
            0: [
                _AnalyzeLine(
                    bbox=(10.0, 10.0, 52.0, 20.0),
                    spans=[
                        _AnalyzeSpan(
                            type=ContentType.TEXT,
                            bbox=(10.0, 10.0, 52.0, 20.0),
                            content="deleted",
                            score=1.0,
                        )
                    ],
                )
            ]
        },
    )
    model_list = [[{"type": BlockType.TEXT, "bbox": [0.1, 0.1, 0.52, 0.2], "content": ""}]]

    text_content._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [MagicMock(size=(100.0, 100.0))],
        model_list,
        [[]],
        [[]],
        "ocr",
        "medium",
        {BlockType.TEXT},
        MagicMock(),
    )

    build_styles.assert_not_called()
    assert inline_text(model_list[0][0]["content"]) == "deleted"


def test_high_char_count_txt_path_skips_pdf_text_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证超高字符页走 post-OCR 兜底时不读取样式或 Link 注解。"""

    build_enrichment = MagicMock()
    monkeypatch.setattr(
        text_content,
        "prepare_text_evidence",
        build_enrichment,
    )
    plain_span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(10.0, 10.0, 52.0, 20.0),
        content="plain",
        score=1.0,
    )
    monkeypatch.setattr(
        text_content,
        "_build_page_text_formula_spans",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        text_content,
        "_fill_native_pdf_text_spans",
        lambda *_args, **_kwargs: [plain_span],
    )
    monkeypatch.setattr(
        text_content,
        "_group_page_spans_by_block",
        lambda *_args: {
            0: [
                _AnalyzeLine(
                    bbox=plain_span.bbox,
                    spans=[plain_span],
                )
            ]
        },
    )
    pdf_page = MagicMock(size=(100.0, 100.0))
    pdf_page.get_char_count.return_value = text_content.MAX_NATIVE_TEXT_CHARS_PER_PAGE + 1
    model_list = [
        [
            {
                "type": BlockType.TEXT,
                "bbox": [0.1, 0.1, 0.52, 0.2],
                "content": "",
            }
        ]
    ]

    text_content._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [pdf_page],
        model_list,
        [[]],
        [[]],
        "txt",
        "medium",
        {BlockType.TEXT},
        MagicMock(),
    )

    build_enrichment.assert_not_called()
    assert inline_text(model_list[0][0]["content"]) == "plain"


def test_native_span_fill_does_not_read_preloaded_page_chars_twice() -> None:
    """验证样式检测已加载 chars 后，原生 span 回填不会再次读取当前页字符。"""

    page_chars = [
        {
            "char": char,
            "bbox": (10.0 + index * 6.0, 10.0, 16.0 + index * 6.0, 20.0),
            "char_idx": index,
            "rotation": 0.0,
            "font": {"name": "Helvetica", "flags": 0},
        }
        for index, char in enumerate("deleted")
    ]
    span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(10.0, 10.0, 52.0, 20.0),
        score=1.0,
    )
    pdf_page = MagicMock()
    pdf_page.get_char_count.return_value = len(page_chars)

    result = txt_spans_extract(
        pdf_page,
        [span],
        object(),
        1.0,
        [(0.0, 0.0, 100.0, 100.0, None, None, None, BlockType.TEXT)],
        [],
        page_chars=page_chars,
    )

    assert result == [span]
    assert span.content == "deleted"
    pdf_page.get_chars.assert_not_called()
    pdf_page.get_chars_with_geometry.assert_not_called()
