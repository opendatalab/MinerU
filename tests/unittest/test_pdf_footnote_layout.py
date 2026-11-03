"""真实脚注 PDF 从 layout 到同步/异步识别、原生回填及导出的回归。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from docvortex.document.pdf import PDFDocument
from PIL import Image

from mineru.backend.analysis.pdf import ocr, window
from mineru.backend.analysis.pdf.layout import _build_vl_style_layout_blocks
from mineru.backend.analysis.pdf.normalization import _normalize_pdf_model_list
from mineru.backend.postprocess.document import build_middle_json
from mineru.model.layout.pp_doclayout_v2_base import PPDocLayoutV2PostProcessor
from mineru.parser.base import ParseResult
from mineru.types import ModelJson

FIXTURE = Path(__file__).parents[1] / "fixtures/emnlp614_footnote"


@pytest.mark.parametrize(
    ("effort", "is_async", "fallback"),
    [("medium", False, False), ("high", False, False), ("high", True, False), ("high", False, True), ("high", True, True)],
)
def test_real_footnote_survives_recognition_and_native_fill(
    monkeypatch: pytest.MonkeyPatch, effort: str, is_async: bool, fallback: bool
) -> None:
    """冻结推理边界但执行真实回填，确保完整脚注、公式和两行内容均只出现一次。"""
    fixture = json.loads((FIXTURE / "detection.json").read_text())
    source = (FIXTURE / "page7.pdf").read_bytes()
    formula_bbox = fixture["raw_layout"]["torch"][0]["bbox"]
    latex = "h_0(d_0) = 5, h_1(d_0) = 15, h_0(d_1) = 10"
    crop_images = []
    mfr_calls = []

    def prepare(*args: Any, **kwargs: Any) -> window._WindowInputs:
        """在真实渲染页面上复放原始检测框，其余窗口阶段继续使用生产实现。"""
        image = document.render_page(0, scale=fixture["scale"]).pil_image
        assert list(image.size) == fixture["render_size"]
        layout = PPDocLayoutV2PostProcessor._apply_layout_post_process(
            PPDocLayoutV2PostProcessor._apply_paddlex_filter_boxes(fixture["raw_layout"]["torch"], drop_inline_formula=False),
            image_size=image.size[::-1],
        )
        blocks = _build_vl_style_layout_blocks([layout], [image])
        return window._WindowInputs(
            window._ProcessingWindow(0, 1, 0, 0),
            [{"img_pil": image, "scale": fixture["scale"]}],
            [document[0]],
            [image],
            [np.asarray(image).copy()],
            [layout],
            blocks,
            deepcopy(blocks),
            [[]],
            [None],
        )

    def main_infer(**kwargs: Any) -> list[list[dict[str, Any]]]:
        """Standard 主识别只收到完整脚注，行内公式由独立切片识别。"""
        blocks = kwargs["blocks_list"]
        assert [[block["type"] for block in page] for page in blocks] == [["page_footnote"]]
        return deepcopy(blocks)

    def inline_infer(images: list[Image.Image]) -> list[str | None]:
        """检查真实公式裁剪被送入 VLM，并覆盖空响应的局部回退。"""
        assert len(images) == 1 and images[0].size == (411, 33)
        crop_images.extend(images)
        return [None if fallback else fixture["vlm_text"]]

    async def async_main(**kwargs: Any) -> list[list[dict[str, Any]]]:
        """异步主识别复用相同的输入契约。"""
        return main_infer(**kwargs)

    async def async_inline(images: list[Image.Image]) -> list[str | None]:
        """异步切片在一次调度后仍能交付给识别器。"""
        await asyncio.sleep(0)
        return inline_infer(images)

    def mfr_infer(inputs: list[list[dict[str, Any]]], *args: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Basic 使用完整 MFR，Standard 仅在失败时回退，均保留公式位置。"""
        assert [[item["bbox"] for item in page] for page in inputs] == [[formula_bbox]]
        assert kwargs["interline_enable"] is (effort == "medium")
        mfr_calls.append(deepcopy(inputs))
        return [[{**item, "latex": latex} for item in inputs[0]]]

    def detect(
        context: Any,
        images: list[np.ndarray],
        blocks: list[list[dict[str, Any]]],
        formulas: list[list[dict[str, Any]]],
        need_rec_img: bool,
        types: set[str],
    ) -> list[list[dict[str, Any]]]:
        """复放公式遮罩后的 OCR 框，确保改标修复也保留 OCR 所用公式位置。"""
        assert [item["bbox"] for item in formulas[0]] == [formula_bbox]
        assert len(blocks[0]) == 1 and blocks[0][0]["type"] == "page_footnote"
        return deepcopy(fixture["ocr_detection"])

    context = SimpleNamespace(
        device="cpu",
        mfr_model=SimpleNamespace(batch_predict=mfr_infer),
        ocr_model=SimpleNamespace(ocr=Mock(side_effect=AssertionError("Unexpected OCR-rec fallback"))),
    )
    predictor = SimpleNamespace(
        batch_extract_with_layout=main_infer,
        batch_recognize_text=inline_infer,
        aio_batch_extract_with_layout=async_main,
        aio_batch_recognize_text=async_inline,
    )
    monkeypatch.setattr(window, "_prepare_pdf_window", prepare)
    monkeypatch.setattr(window, "get_document_render_session", lambda document: None)
    monkeypatch.setattr(ocr, "_ocr_det", detect)
    with PDFDocument(source) as document:
        options = {
            "effort": effort,
            "parse_mode": "txt",
            "image_analysis": False,
            "hybrid_model": context,
            "vlm_predictor": predictor if effort == "high" else None,
        }
        if is_async:
            result = asyncio.run(window.aio_process_pdf_windows(source, document, **options))
        else:
            result = window._process_pdf_window(source, document, window._ProcessingWindow(0, 1, 0, 0), page_count=1, **options)
    _normalize_pdf_model_list(result)
    assert len(result[0]) == 1
    block = result[0][0]
    assert block["type"] == "page_footnote"
    assert [span["content"] for span in block["content"] if span["type"] == "equation_inline"] == [latex]
    text = "".join(span["content"] for span in block["content"])
    assert text == f"9For example, if {latex} and h1(d1) = 10 then s0 > s1."
    assert len(mfr_calls) == int(effort == "medium" or fallback)
    assert len(crop_images) == int(effort == "high")
    for image in crop_images:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))
    model = ModelJson(
        pages=result,
        page_index_map=[6],
        metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
    )
    parsed = ParseResult(build_middle_json(model))
    markdown = parsed.markdown()
    assert len(parsed.pages[0].blocks) == 1
    assert markdown.count('data-block-type="page_footnote"') == 1
    assert markdown.count("For example, if") == 1
    assert markdown.count(f"${latex}$") == 1
    assert "and h<sub>1</sub>(d<sub>1</sub>) = 10 then s0 > s1." in markdown
    assert "10 ( ) =" not in markdown
