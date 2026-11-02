"""验证真实窗口编排的最终表格归属、模型锁和切片生命周期。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from mineru.backend.analysis.pdf import ocr, window


@pytest.mark.parametrize("effort", ["high", "xhigh"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_window_inline_inference_runs_outside_local_lock(
    monkeypatch: pytest.MonkeyPatch, effort: str, is_async: bool, fallback: bool
) -> None:
    """同步与异步入口在最终版面过滤后识别，仅失败片段在本地模型锁内回退。"""
    locked = []
    observed_crops = []
    observed_results = []
    layouts = [[{"label": "inline_formula", "bbox": box} for box in [[50, 50, 60, 60], [10, 10, 15, 15], [20, 10, 25, 15]]]]
    # 本地 table 故意覆盖外侧公式，实际归属必须由最终 VLM 表格决定。
    layouts[0].append({"label": "table", "bbox": [0, 0, 30, 30]})
    final_blocks = [[{"type": "text", "bbox": [0, 0, 0.3, 0.3]}, {"type": "table", "bbox": [0.4, 0.4, 0.8, 0.8]}]]

    @contextmanager
    def stage(_device: str) -> Iterator[None]:
        """显式观察本地模型锁，不依赖线程重入锁的实现细节。"""
        assert not locked
        locked.append(True)
        try:
            yield
        finally:
            locked.pop()

    def prepare(*args: Any, **kwargs: Any) -> window._WindowInputs:
        """只替代渲染与 layout，继续使用真实窗口转换、推理和资源关闭。"""
        assert locked
        picture = Image.new("RGB", (100, 100))
        return window._WindowInputs(
            window._ProcessingWindow(0, 1, 0, 0),
            [{"img_pil": picture, "scale": 1}],
            [object()],
            [picture],
            [np.asarray(picture).copy()],
            deepcopy(layouts),
            [],
            [],
            [[]],
            None,
        )

    def main_infer(**kwargs: Any) -> list[list[dict[str, Any]]]:
        """主 VLM 也必须在本地模型锁之外执行。"""
        assert not locked
        return deepcopy(final_blocks)

    def inline_infer(images: list[Image.Image]) -> list[str | None]:
        """最终表格内公式不得进入 VLM，并保存切片引用核对关闭时机。"""
        assert not locked
        assert len(images) == 2
        observed_crops.extend(images)
        assert all(image.getpixel((0, 0)) == (0, 0, 0) for image in images)
        return ["α", None if fallback else r"\(x\) . (D)"]

    async def async_main(**kwargs: Any) -> list[list[dict[str, Any]]]:
        """异步主推理复用同一版面及锁断言。"""
        return main_infer(**kwargs)

    async def async_inline(images: list[Image.Image]) -> list[str | None]:
        """让真实异步窗口经过一次等待，再验证图片仍可读取。"""
        await asyncio.sleep(0)
        return inline_infer(images)

    def mfr_predict(inputs: list[list[dict[str, Any]]], *args: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        """回退必须持有本地模型锁，且只接收第二个表外切片。"""
        assert locked
        assert [item["bbox"] for item in inputs[0]] == [[20, 10, 25, 15]]
        return [[{**item, "latex": "x"} for item in inputs[0]]]

    def fill(*args: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        """正文阶段观察按原顺序合并后的 VLM 与 MFR 结果。"""
        assert locked
        observed_results.extend(kwargs["recognized_inline_formulas"][0])
        return args[2]

    model = SimpleNamespace(device="cpu")
    if fallback:
        model.mfr_model = SimpleNamespace(batch_predict=mfr_predict)
    predictor = SimpleNamespace(
        batch_extract_with_layout=main_infer,
        batch_two_step_extract=main_infer,
        aio_batch_extract_with_layout=async_main,
        aio_batch_two_step_extract=async_main,
        batch_recognize_text=inline_infer,
        aio_batch_recognize_text=async_inline,
    )
    monkeypatch.setattr(window, "local_model_stage", stage)
    monkeypatch.setattr(window, "_prepare_pdf_window", prepare)
    monkeypatch.setattr(window, "_process_text_and_formulas", fill)
    monkeypatch.setattr(window, "_attach_visual_block_images", MagicMock())
    monkeypatch.setattr(window, "_supplement_missing_image_block_containers", MagicMock())
    monkeypatch.setattr(window, "get_document_render_session", lambda document: None)
    monkeypatch.setattr(ocr, "_apply_seal_ocr", MagicMock())
    options = {
        "effort": effort,
        "parse_mode": "txt",
        "image_analysis": False,
        "hybrid_model": model,
        "vlm_predictor": predictor,
    }
    if is_async:
        result = asyncio.run(window.aio_process_pdf_windows(b"", SimpleNamespace(page_count=1), **options))
    else:
        result = window._process_pdf_window(
            b"", SimpleNamespace(page_count=1), window._ProcessingWindow(0, 1, 0, 0), page_count=1, **options
        )
    assert len(result) == 1
    assert observed_results[0]["content"] == "α"
    assert observed_results[1].get("latex") == ("x" if fallback else "")
    assert [item["bbox"] for item in observed_results] == [[10, 10, 15, 15], [20, 10, 25, 15]]
    for image in observed_crops:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))
