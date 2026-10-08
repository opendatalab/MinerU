"""窗口级 VLM 解码异常隔离：截断输出损坏多字节序列时按窗口降级，不再中断整份文档。"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from mineru.backend.analysis.pdf import window
from mineru.model.vlm.contracts import VlmPredictor


def _decode_error() -> UnicodeDecodeError:
    """构造截断解码异常：汉字 UTF-8 多字节序列在输出上限处被截断。"""
    return UnicodeDecodeError("utf-8", b"\xe5\x8f\xb7\xe6\x8a\xa4", 2, 3, "unexpected end of data")


def _state(vl_blocks: list[list[dict[str, Any]]], index: int = 0) -> window._WindowInputs:
    """构造单页窗口资源，聚焦失败隔离编排而非真实准备流程。"""
    image = Image.new("RGB", (200, 200))
    return window._WindowInputs(
        window._ProcessingWindow(index, 1, index, index),
        [{"img_pil": image, "scale": 1.0}],
        [MagicMock()],
        [image],
        [],
        [[]],
        vl_blocks,
        vl_blocks,
        [],
        [None],
        [None],
        None,
    )


def _fake_finish(recorder: dict[str, Any]) -> Any:
    """记录回填入参并返回单页空结果的替身。"""

    def finish(_state: Any, result: Any, *, effort: str, parse_mode: str, hybrid_model: Any) -> list[list[dict[str, Any]]]:
        """effort 与 result 是降级断言的观测点。"""
        recorder["result"] = result
        recorder["effort"] = effort
        return [[]]

    return finish


@pytest.mark.parametrize("effort", ["high", "xhigh"])
def test_sync_window_falls_back_to_local_layout_on_vlm_failure(
    effort: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同步路径：VLM 推理异常按窗口降级为本地布局并按 medium 回填，不再上抛。"""
    state = _state([[{"type": "text_block"}]])
    monkeypatch.setattr(window, "_prepare_pdf_window", lambda *_args, **_kwargs: state)
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(window, "_finish_pdf_window", _fake_finish(recorder))
    predictor = MagicMock(spec=VlmPredictor)
    if effort == "high":
        predictor.batch_extract_with_layout.side_effect = _decode_error()
    else:
        predictor.batch_two_step_extract.side_effect = _decode_error()

    out = window._process_pdf_window(
        b"",
        MagicMock(),
        window._ProcessingWindow(0, 1, 0, 0),
        page_count=1,
        effort=effort,
        parse_mode="txt",
        image_analysis=False,
        hybrid_model=MagicMock(),
        vlm_predictor=predictor,
    )

    assert out == [[]]
    assert recorder["effort"] == "medium"
    assert recorder["result"] is state.vl_style_layout_blocks
    if effort == "high":
        predictor.batch_extract_with_layout.assert_called_once()
    else:
        predictor.batch_two_step_extract.assert_called_once()


def test_document_survives_vlm_window_failure_across_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """入口路径：第一个窗口 VLM 失败后降级，第二个窗口继续推理，整份文档保持完整。"""
    monkeypatch.setenv("MINERU_PROCESSING_WINDOW_SIZE", "1")
    first = _state([[{"page": "local-0"}]], index=0)
    second = _state([[{"page": "local-1"}]], index=1)
    states = iter([first, second])
    monkeypatch.setattr(window, "_prepare_pdf_window", lambda *_args, **_kwargs: next(states))
    finishes: list[tuple[Any, str]] = []

    def finish(_state: Any, result: Any, *, effort: str, parse_mode: str, hybrid_model: Any) -> list[list[dict[str, Any]]]:
        """逐窗口记录回填 effort 与结果来源。"""
        finishes.append((result, effort))
        return [["done"]]

    monkeypatch.setattr(window, "_finish_pdf_window", finish)
    predictor = MagicMock(spec=VlmPredictor)
    predictor.batch_extract_with_layout.side_effect = [_decode_error(), [["vlm-page-1"]]]

    out = window.process_pdf_windows(
        b"",
        MagicMock(page_count=2),
        effort="high",
        parse_mode="txt",
        image_analysis=False,
        flash_txt_mode=False,
        hybrid_model=MagicMock(),
        vlm_predictor=predictor,
    )

    assert out == [["done"], ["done"]]
    assert [effort for _, effort in finishes] == ["medium", "high"]
    assert finishes[0][0] is first.vl_style_layout_blocks
    assert finishes[1][0] == [["vlm-page-1"]]
    assert predictor.batch_extract_with_layout.call_count == 2


def test_async_window_falls_back_to_local_layout_on_vlm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """异步路径：推理异常同样按窗口降级为本地布局，取消语义保持不变。"""
    state = _state([[{"type": "text_block"}]])
    monkeypatch.setattr(window, "_prepare_locked_window", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(window, "_inference_options", lambda *_args: {})
    monkeypatch.setattr(window, "get_document_render_session", lambda *_args: None)
    monkeypatch.setattr(window, "trim_process_heap", lambda: None)
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(window, "_finish_locked_window", _fake_finish(recorder))

    async def run() -> list[list[dict[str, Any]]]:
        """在事件循环内执行异步入口。"""
        predictor = MagicMock(spec=VlmPredictor)
        predictor.aio_batch_extract_with_layout = AsyncMock(side_effect=_decode_error())
        return await window.aio_process_pdf_windows(
            b"",
            MagicMock(page_count=1),
            effort="high",
            parse_mode="txt",
            image_analysis=False,
            hybrid_model=MagicMock(),
            vlm_predictor=predictor,
        )

    out = asyncio.run(run())

    assert out == [[]]
    assert recorder["effort"] == "medium"
    assert recorder["result"] is state.vl_style_layout_blocks
