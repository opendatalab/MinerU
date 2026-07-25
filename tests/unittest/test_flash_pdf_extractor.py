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
