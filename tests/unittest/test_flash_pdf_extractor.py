from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from mineru.backend.flash import pdf_extractor
from mineru.backend.flash.native_pdf import (
    pipeline,
)


def _install_pdf_document(
    monkeypatch: pytest.MonkeyPatch,
    *,
    page_count: int = 1,
    classified_mode: str = "txt",
) -> tuple[MagicMock, MagicMock]:
    """安装支持上下文管理的 PDFDocument 替身并返回文档与管理器。"""

    pdf_doc = MagicMock()
    pdf_doc.page_count = page_count
    pdf_doc.classify.return_value = classified_mode
    context_manager = MagicMock()
    context_manager.__enter__.return_value = pdf_doc
    context_manager.__exit__.return_value = False
    monkeypatch.setattr(pdf_extractor, "PDFDocument", MagicMock(return_value=context_manager))
    return pdf_doc, context_manager


def _install_hybrid_analyze(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_list: list[list[dict[str, Any]]],
) -> MagicMock:
    """安装 Hybrid analyze 模块替身，避免路由测试加载真实模型。"""

    hybrid_doc_analyze = MagicMock(return_value=([object()], model_list))
    analyze_module = ModuleType("mineru.backend.hybrid.analyze")
    analyze_module.doc_analyze = hybrid_doc_analyze  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mineru.backend.hybrid.analyze", analyze_module)
    return hybrid_doc_analyze


@pytest.mark.parametrize(
    ("parse_mode", "expected_classify_calls"),
    [
        ("ocr", 0),
        ("auto", 1),
    ],
)
def test_ocr_mode_delegates_to_hybrid_low(
    monkeypatch: pytest.MonkeyPatch,
    parse_mode: str,
    expected_classify_calls: int,
) -> None:
    """验证显式或自动判定的 OCR 模式都精确委托 Hybrid low。"""

    pdf_bytes = b"%PDF-1.7\n"
    page_index_map = [7]
    expected_model_list = [[{"type": "text", "content": "hybrid"}]]
    pdf_doc, context_manager = _install_pdf_document(
        monkeypatch,
        classified_mode="ocr",
    )
    hybrid_doc_analyze = _install_hybrid_analyze(
        monkeypatch,
        model_list=expected_model_list,
    )

    result = pdf_extractor.doc_analyze(
        pdf_bytes,
        parse_mode=parse_mode,  # type: ignore[arg-type]
        page_index_map=page_index_map,
    )

    assert result is expected_model_list
    assert pdf_doc.classify.call_count == expected_classify_calls
    assert context_manager.__exit__.call_count == 1
    hybrid_doc_analyze.assert_called_once_with(
        pdf_bytes,
        effort="low",
        parse_mode="ocr",
        page_index_map=page_index_map,
    )


@pytest.mark.parametrize(
    ("parse_mode", "expected_classify_calls"),
    [
        ("txt", 0),
        ("auto", 1),
    ],
)
def test_txt_mode_keeps_native_flash_path(
    monkeypatch: pytest.MonkeyPatch,
    parse_mode: str,
    expected_classify_calls: int,
) -> None:
    """验证显式或自动判定的文本模式继续使用 Flash 原生路径。"""

    expected_model_list = [[{"type": "text", "content": "native"}]]
    pdf_doc, _context_manager = _install_pdf_document(
        monkeypatch,
        classified_mode="txt",
    )
    native_analyze = MagicMock(return_value=expected_model_list)
    monkeypatch.setattr(pipeline, "_analyze_native_document", native_analyze)
    hybrid_doc_analyze = _install_hybrid_analyze(
        monkeypatch,
        model_list=[[{"type": "text", "content": "unexpected"}]],
    )

    result = pdf_extractor.doc_analyze(
        b"%PDF-1.7\n",
        parse_mode=parse_mode,  # type: ignore[arg-type]
    )

    assert result is expected_model_list
    assert pdf_doc.classify.call_count == expected_classify_calls
    native_analyze.assert_called_once_with(pdf_doc)
    hybrid_doc_analyze.assert_not_called()
def test_invalid_parse_mode_is_rejected_before_opening_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证非法解析模式在打开 PDF 前保持原有报错行为。"""

    pdf_document = MagicMock()
    monkeypatch.setattr(pdf_extractor, "PDFDocument", pdf_document)

    with pytest.raises(ValueError, match="parse_mode invalid is not supported"):
        pdf_extractor.doc_analyze(b"%PDF-1.7\n", parse_mode="invalid")  # type: ignore[arg-type]

    pdf_document.assert_not_called()


def test_page_index_map_mismatch_is_rejected_before_hybrid_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证页映射长度不匹配时不会进入 Hybrid low。"""

    _install_pdf_document(monkeypatch, page_count=2, classified_mode="ocr")
    hybrid_doc_analyze = _install_hybrid_analyze(monkeypatch, model_list=[])

    with pytest.raises(ValueError, match="Flash page_index_map length mismatch"):
        pdf_extractor.doc_analyze(
            b"%PDF-1.7\n",
            parse_mode="ocr",
            page_index_map=[0],
        )

    hybrid_doc_analyze.assert_not_called()


def test_marginal_header_row_does_not_break_two_column_reading_order() -> None:
    """验证页眉页码独立按视觉行排序，正文仍保持左栏后右栏的阅读顺序。"""

    blocks = [
        {"type": "header", "bbox": (80.0, 2.0, 120.0, 12.0), "angle": 0, "content": "center"},
        {"type": "page_number", "bbox": (2.0, 0.0, 12.0, 14.0), "angle": 0, "content": "page"},
        {"type": "header", "bbox": (175.0, 1.0, 198.0, 13.0), "angle": 0, "content": "volume"},
        {"type": "text", "bbox": (0.0, 20.0, 80.0, 40.0), "angle": 0, "content": "left one"},
        {"type": "text", "bbox": (0.0, 45.0, 80.0, 65.0), "angle": 0, "content": "left two"},
        {"type": "text", "bbox": (120.0, 20.0, 200.0, 40.0), "angle": 0, "content": "right one"},
        {"type": "text", "bbox": (120.0, 45.0, 200.0, 65.0), "angle": 0, "content": "right two"},
    ]

    sorted_blocks = pipeline._sort_blocks_with_visual_row_groups(
        blocks,
        (200.0, 100.0),
    )

    assert [block["content"] for block in sorted_blocks] == [
        "page",
        "center",
        "volume",
        "left one",
        "left two",
        "right one",
        "right two",
    ]


def test_overlapping_span_captions_follow_visual_center_order() -> None:
    """验证同一跨栏带中轻微重叠的图注按视觉中心由上到下稳定排序。"""

    common = {
        "type": "text",
        "angle": 0,
        "_lane_interval": (20.0, 180.0),
        "_lane_is_span": True,
        "_line_heights": [10.0],
    }
    later = {**common, "bbox": (30.0, 58.0, 170.0, 70.0), "content": "later"}
    earlier = {**common, "bbox": (70.0, 50.0, 130.0, 62.0), "content": "earlier"}

    stabilized = pipeline._stabilize_overlapping_lane_order(
        [later, earlier],
        (200.0, 100.0),
    )

    assert [block["content"] for block in stabilized] == ["earlier", "later"]
