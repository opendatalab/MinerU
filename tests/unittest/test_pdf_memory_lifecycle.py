"""验证窗口临时引用释放、结果素材保留和文档级回收顺序。"""

from __future__ import annotations

import asyncio
import base64
import weakref
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from mineru.backend import analyze
from mineru.backend.analysis.pdf import pipeline, window
from mineru.model.runtime import memory


def _window_probe(monkeypatch: pytest.MonkeyPatch, failure: str = "") -> SimpleNamespace:
    """隔离推理模型，保留真实数组、表格裁图和素材编码以检查窗口边界。"""
    probe = SimpleNamespace(arrays=[], crops=[], images=[], events=[])
    original_collect = window._collect_table_items
    original_attach = window._attach_visual_block_images

    def fail_at(stage: str) -> None:
        """在指定阶段抛出可辨识的业务异常。"""
        if failure == stage:
            raise RuntimeError(stage)

    def render(**_options: Any) -> list[dict[str, Any]]:
        """下一窗口分配图片之前，检查上一窗口图像数组与表格视图已释放。"""
        assert all(ref() is None for ref in probe.arrays + probe.crops)
        fail_at("render")
        image = Image.new("RGB", (40, 40), "red")
        probe.images.append(image)
        probe.events.append("render")
        return [{"img_pil": image, "scale": 1}]

    def layout(_images: list[Image.Image], *, batch_size: int) -> list[list[dict[str, Any]]]:
        """返回可由真实表格裁图函数处理的像素坐标。"""
        assert batch_size > 0
        fail_at("layout")
        return [[{"label": "table", "bbox": [0, 0, 20, 20]}]]

    def collect(layouts: list[list[dict[str, Any]]], arrays: list[np.ndarray]) -> list[dict[str, Any]]:
        """只用弱引用观察独立页图数组及由真实函数产生的表格裁图。"""
        probe.arrays.extend(weakref.ref(array) for array in arrays)
        items = original_collect(layouts, arrays)
        probe.crops.extend(weakref.ref(item["table_img"]) for item in items)
        return items

    def blocks(*_args: Any, **_kwargs: Any) -> list[list[dict[str, Any]]]:
        """生成最小视觉块，供真实 JPEG 编码和公共协议转换消费。"""
        return [[{"type": "image", "bbox": [0.0, 0.0, 1.0, 1.0], "content": ""}]]

    def orient(*_args: Any) -> None:
        """模拟表格方向推理的异常边界。"""
        fail_at("orientation")

    def text(*args: Any) -> list[list[dict[str, Any]]]:
        """保持已有块与素材归属，模拟 OCR 异常。"""
        fail_at("ocr")
        return args[2]

    def vlm(**_kwargs: Any) -> list[list[dict[str, Any]]]:
        """模拟 VLM 推理成功或失败，但不持有传入页图。"""
        fail_at("vlm")
        return blocks()

    def attach(*args: Any, **kwargs: Any) -> None:
        """真实编码独立 JPEG 素材，然后覆盖裁图阶段异常。"""
        original_attach(*args, **kwargs)
        fail_at("crop")

    def trimmed() -> bool:
        """正常窗口回收前必须已释放临时数组，异常时只检查容器清理。"""
        if not failure:
            assert all(ref() is None for ref in probe.arrays + probe.crops)
        probe.events.append("trim")
        return False

    monkeypatch.setattr(window, "_configured_window_size", lambda default: 1)
    monkeypatch.setattr(window, "_get_window_pdf_pages", lambda *_args: [object()])
    monkeypatch.setattr(window, "load_images_from_pdf_bytes_range", render)
    monkeypatch.setattr(window, "_collect_table_items", collect)
    monkeypatch.setattr(window, "_apply_table_orientations", orient)
    monkeypatch.setattr(window, "_build_vl_style_layout_blocks", blocks)
    monkeypatch.setattr(window, "_process_text_and_formulas", text)
    monkeypatch.setattr(window, "_process_flash_ocr", text)
    monkeypatch.setattr(window, "_apply_seal_ocr", lambda *_args: None)
    monkeypatch.setattr(window, "_convert_vlm_results_to_model_list", lambda value: value)
    monkeypatch.setattr(window, "_attach_visual_block_images", attach)
    monkeypatch.setattr(window, "trim_process_heap", trimmed)
    probe.model = SimpleNamespace(device="cpu", layout_model=SimpleNamespace(batch_predict=layout))
    probe.predictor = SimpleNamespace(batch_extract_with_layout=vlm)
    return probe


def _run_windows(probe: SimpleNamespace, *, effort: str = "medium", pages: int = 2) -> list[list[dict[str, Any]]]:
    """通过真实窗口入口分析模拟页面，不初始化任何神经模型。"""
    document = SimpleNamespace(page_count=pages)
    return window.process_pdf_windows(
        b"pdf",
        document,
        effort=effort,
        parse_mode="ocr",
        image_analysis=True,
        flash_txt_mode=False,
        hybrid_model=probe.model,
        vlm_predictor=probe.predictor,
    )


@pytest.mark.parametrize("effort", ["flash", "medium", "high"])
def test_previous_window_arrays_die_before_next_render(monkeypatch: pytest.MonkeyPatch, effort: str) -> None:
    """跨窗口释放完整数组与裁图，输出中的真实 JPEG 数据仍完整可读。"""
    probe = _window_probe(monkeypatch)
    result = _run_windows(probe, effort=effort)
    assert probe.events == ["render", "trim", "render", "trim"]
    assert len(probe.arrays) == len(probe.crops) == 2
    assert len(result) == 2
    for page in result:
        with Image.open(BytesIO(base64.b64decode(page[0]["image_base64"].split(",", 1)[1]))) as image:
            assert image.size == (40, 40)
    for image in probe.images:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize("failure", ["render", "layout", "orientation", "ocr", "vlm", "crop"])
def test_window_failure_closes_images_and_clears_owned_containers(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """异常栈保留时也清空本层图像容器，并在每种失败路径尝试外层回收。"""
    probe = _window_probe(monkeypatch, failure)
    with pytest.raises(RuntimeError, match=failure) as caught:
        _run_windows(probe, effort="high" if failure == "vlm" else "medium")
    assert probe.events[-1] == "trim"
    for image in probe.images:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))
    for entry in caught.traceback:
        if entry.name == "_process_pdf_window":
            for name in ("images_list", "images_pil_list", "np_images", "table_items"):
                assert entry.frame.f_locals[name] == []


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("failure", ["", "analysis", "close", "device"])
@pytest.mark.parametrize("enabled", [False, True])
def test_document_cleanup_order_and_global_switch(
    monkeypatch: pytest.MonkeyPatch, native: bool, failure: str, enabled: bool
) -> None:
    """正常和异常路径均按文档、设备、CPU 堆顺序清理，Flash 原生不走设备清理。"""
    events: list[str] = []

    def stage(name: str) -> None:
        """记录阶段并注入所选错误，检查后续清理仍然执行。"""
        events.append(name)
        if failure == name:
            raise RuntimeError(name)

    def process(*_args: Any, **_kwargs: Any) -> list[Any]:
        """返回空文档以独立检查生命周期，不涉及协议内容。"""
        stage("analysis")
        return []

    def trim(_pad: int) -> int:
        """记录真实开关控制后的底层回收调用。"""
        events.append("trim")
        return 1

    monkeypatch.setenv("MINERU_MALLOC_TRIM", "1" if enabled else "0")
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(memory, "_get_malloc_trim", lambda: trim)
    monkeypatch.setattr(pipeline, "PDFDocument", lambda _data: SimpleNamespace(close=lambda: stage("close")))
    monkeypatch.setattr(pipeline, "process_pdf_windows", process)
    monkeypatch.setattr(pipeline, "clean_memory", lambda _device: stage("device"))
    monkeypatch.setattr(
        pipeline,
        "HybridLocalModelContextSingleton",
        lambda: SimpleNamespace(get_model=lambda: SimpleNamespace(device="cpu")),
    )
    if failure and not (native and failure == "device"):
        with pytest.raises(RuntimeError, match=failure):
            pipeline.analyze_pdf(b"pdf", effort="flash", parse_mode="txt" if native else "ocr")
    else:
        pipeline.analyze_pdf(b"pdf", effort="flash", parse_mode="txt" if native else "ocr")
    assert events == ["analysis", "close"] + ([] if native else ["device"]) + (["trim"] if enabled else [])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_window_and_document_hooks_share_switch(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, asynchronous: bool
) -> None:
    """同步和异步公共入口的两个窗口与文档回收均受唯一开关控制。"""
    probe = _window_probe(monkeypatch)
    monkeypatch.setattr(window, "trim_process_heap", memory.trim_process_heap)
    monkeypatch.setenv("MINERU_MALLOC_TRIM", "1" if enabled else "0")
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform="linux"))
    malloc_trim = Mock(return_value=1)
    monkeypatch.setattr(memory, "_get_malloc_trim", lambda: malloc_trim)
    document = Mock(page_count=2, close=Mock())
    monkeypatch.setattr(window, "_get_window_pdf_pages", lambda *_args: [object()])
    monkeypatch.setattr(pipeline, "PDFDocument", lambda _data: document)
    monkeypatch.setattr(pipeline, "HybridLocalModelContextSingleton", lambda: SimpleNamespace(get_model=lambda: probe.model))
    monkeypatch.setattr(pipeline, "clean_memory", lambda _device: None)
    if asynchronous:
        middle, model = asyncio.run(analyze.aio_doc_analyze(b"pdf", effort="medium", parse_mode="ocr"))
    else:
        middle, model = analyze.doc_analyze(b"pdf", effort="medium", parse_mode="ocr")
    assert len(middle.pages) == len(model.pages) == 2
    assert malloc_trim.call_count == (3 if enabled else 0)


def test_allocator_failure_preserves_window_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实回收工具吞掉 allocator 故障，外层仍收到原始推理异常。"""
    probe = _window_probe(monkeypatch, "ocr")
    monkeypatch.setattr(window, "trim_process_heap", memory.trim_process_heap)
    monkeypatch.setenv("MINERU_MALLOC_TRIM", "1")
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(memory, "_get_malloc_trim", lambda: Mock(side_effect=RuntimeError("allocator")))
    with pytest.raises(RuntimeError, match="^ocr$"):
        _run_windows(probe)
