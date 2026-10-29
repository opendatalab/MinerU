"""验证乱码表格通过公开接口拒绝后，basic/standard 保留模型回退输入。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from docvortex.document.pdf import PDFPageTextGeometry, PDFPageVectorGeometry
from PIL import Image

from mineru.backend.analysis.pdf import tables


@pytest.mark.parametrize("effort", ["medium", "high"])
@pytest.mark.parametrize("clean", [False, True])
def test_real_table_text_quality_controls_model_fallback(effort: Literal["medium", "high"], clean: bool) -> None:
    """真实乱码被拒绝时完整保留模型输入；同结构正常文本继续采用原生结果。"""
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/native_pdf_table_text_quality.json").read_text(encoding="utf-8")
    )
    assert fixture["source_sha256"] == "c2b7ef637fefefbe9a85b5a33681137a41121983a93419d27bb9707bdf3bf4c0"
    chars = fixture["chars"]
    if clean:
        for char in chars:
            char["char"] = "".join("5" if ord(c) < 32 and c not in "\t\r\n" else c for c in char["char"])
    width, height = fixture["page_size"]
    page = MagicMock()
    page.size = (width, height)
    page.get_chars_with_geometry.return_value = PDFPageTextGeometry(chars, {}, {})
    page.get_vector_geometry.return_value = PDFPageVectorGeometry(
        drawing_lines=fixture["drawing_lines"], path_infos=fixture["rectangles"]
    )
    bbox = [v / (width if i % 2 == 0 else height) for i, v in enumerate(fixture["bbox"])]
    block = {"type": "table", "bbox": bbox, "angle": 0}
    text = {"type": "text", "bbox": [0.4, 0.8, 0.5, 0.81], "content": "keep source text"}
    formula = {"type": "equation", "bbox": [0.6, 0.8, 0.7, 0.81]}
    formula_layout = {"label": "inline_formula", "bbox": [0.6 * width, 0.8 * height, 0.7 * width, 0.81 * height]}
    blocks, layout = [[block, text, formula]], [[formula_layout]]
    with Image.new("RGB", (int(width), int(height)), "white") as image:
        summary = tables._apply_native_txt_table_priority(
            blocks, layout, [page], [{"img_pil": image, "scale": 1.0}], effort=effort
        )
    assert summary.errors == 0
    if clean:
        assert summary.accepted == 1 and summary.rejected == 0
        assert block["content"].startswith("<table")
        assert blocks == [[block]] and layout == [[]]
    else:
        assert summary.accepted == 0 and summary.rejected == 1
        assert "content" not in block
        assert blocks == [[block, text, formula]] and layout == [[formula_layout]]
    if effort == "high":
        vlm, native = tables._split_native_high_table_blocks(blocks)
        assert (vlm, native) == (([[]], [[block]]) if clean else (blocks, [[]]))
    else:
        tasks = tables._collect_medium_table_tasks(blocks, layout, [np.zeros((int(height), int(width), 3), dtype=np.uint8)])
        assert len(tasks) == (0 if clean else 1)
        if tasks:
            assert tasks[0]["table_block"] is block
