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
        return [
            [
                {"type": "text", "content": "Ａ１", "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}]},
                {"type": "table", "content": "<table><tr><td>Ｂ２</td></tr></table>"},
            ]
        ]

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
    assert result.model_list[0][0]["content"] == [{"type": "text", "content": "A1"}]
    assert result.model_list[0][1]["content"] == "<table><tr><td>B2</td></tr></table>"


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
@pytest.mark.parametrize("parse_mode", ["txt", "ocr"])
def test_hybrid_and_vlm_pdf_outputs_share_docvortex_text_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
    parse_mode: str,
) -> None:
    """推理和原生回填的编排留在宿主，所有非 Flash PDF 出口都使用相同文字清洗。"""
    # 只隔离模型推理，真实执行 PDF 管线的公共出口与协议规范化。
    monkeypatch.setattr(pipeline, "PDFDocument", lambda _data: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        pipeline, "HybridLocalModelContextSingleton", lambda: SimpleNamespace(get_model=lambda: SimpleNamespace(device="cpu"))
    )
    monkeypatch.setattr(pipeline, "get_vlm_predictor", lambda _config: (object(), "test"))
    monkeypatch.setattr(pipeline, "clean_memory", lambda _device: None)

    def process(_data: bytes, _document: object, **options: object) -> list[list[dict[str, object]]]:
        """模拟文字回填后的页，确认非原生路由不会被本次清洗切换。"""
        assert options["flash_txt_mode"] is False
        return [
            [
                {"type": "text", "content": "Ａ１", "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}]},
                {"type": "table", "content": "<table><tr><td>Ｂ２<eq>Ｃ３</eq></td></tr></table>"},
            ]
        ]

    monkeypatch.setattr(pipeline, "process_pdf_windows", process)
    result = pipeline.analyze_pdf(b"pdf", effort=effort, parse_mode=parse_mode)
    assert result.effort == effort and result.parse_mode == parse_mode
    assert result.model_list[0][0]["content"] == [{"type": "text", "content": "A1"}]
    assert result.model_list[0][1]["content"] == "<table><tr><td>B2<eq>Ｃ３</eq></td></tr></table>"
