"""验证分类下沉后仍严格保留 MinerU Flash 的模式路由。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mineru.backend.analysis.pdf import pipeline


@pytest.mark.parametrize(
    ("parse_mode", "classification", "native", "classification_calls"),
    [("txt", "ocr", True, 0), ("ocr", "txt", False, 0), ("auto", "txt", True, 1), ("auto", "ocr", False, 1)],
)
def test_flash_routes_classification_and_native_analysis_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    parse_mode: str,
    classification: str,
    native: bool,
    classification_calls: int,
) -> None:
    """显式模式跳过分类，auto 才采用分类结果，OCR 仍初始化宿主推理路径。"""
    calls: list[str] = []
    observed: list[dict[str, object]] = []

    class Document:
        """记录公开文档访问与分类，不依赖真实推理模型。"""

        def __init__(self, data: bytes) -> None:
            """接收由宿主传入的输入字节。"""
            assert data == b"pdf"

        def classify(self) -> str:
            """只有 auto 路由才允许调用分类。"""
            calls.append("classify")
            return classification

        def close(self) -> None:
            """记录文档在所有正常路径都及时关闭。"""
            calls.append("close")

    def get_model() -> SimpleNamespace:
        """记录宿主 OCR 模型入口，原生分支不应经过这里。"""
        calls.append("ocr-model")
        return SimpleNamespace(device="cpu")

    def process(_data: bytes, _document: Document, **options: object) -> list[list[dict[str, object]]]:
        """记录实际传给处理窗口的路由决定。"""
        observed.append(options)
        return [[]]

    monkeypatch.setattr(pipeline, "PDFDocument", Document)
    monkeypatch.setattr(pipeline, "HybridLocalModelContextSingleton", lambda: SimpleNamespace(get_model=get_model))
    monkeypatch.setattr(pipeline, "process_pdf_windows", process)
    monkeypatch.setattr(pipeline, "clean_memory", lambda *_args: None)
    result = pipeline.analyze_pdf(b"pdf", effort="flash", parse_mode=parse_mode)
    assert calls.count("classify") == classification_calls
    assert ("ocr-model" in calls) is not native
    assert observed[0]["flash_txt_mode"] is native
    assert observed[0]["parse_mode"] == ("txt" if native else "ocr")
    assert result.parse_mode == ("txt" if native else "ocr")
    assert calls[-1] == "close"
