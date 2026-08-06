from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

from mineru.model.flash import FlashModel
from mineru.model.flash.native_pdf import pipeline
from mineru.utils.pdf_document import PDFDocument


def test_flash_model_predict_returns_native_model_list_without_owning_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证模型传递同一 PDFDocument，且不负责分类或关闭调用方文档。"""

    pdf_doc = MagicMock(spec=PDFDocument)
    expected_model_list = [[{"type": "text", "content": "native"}]]
    native_analyze = MagicMock(return_value=expected_model_list)
    monkeypatch.setattr(pipeline, "_analyze_native_document", native_analyze)

    result = FlashModel().predict(pdf_doc)

    assert result is expected_model_list
    native_analyze.assert_called_once_with(pdf_doc)
    pdf_doc.classify.assert_not_called()
    pdf_doc.close.assert_not_called()


def test_flash_model_is_public_and_old_native_pdf_import_is_removed() -> None:
    """验证新模型入口公开可用，旧 backend native_pdf 路径不再兼容。"""

    assert importlib.import_module("mineru.model.flash").FlashModel is FlashModel
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mineru.backend.flash.native_pdf")
