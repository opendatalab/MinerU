# Copyright (c) Opendatalab. All rights reserved.
"""验证 Native PDF 表格结构恢复和 Flash OCR 表格投影。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from mineru.backend.analysis.pdf import tables as pdf_tables
from mineru.model.flash.native_pdf import models as flash_models
from mineru.model.flash.native_pdf import tables as flash_tables
from mineru.types import BlockType


def test_flash_materialization_prefers_native_html_and_keeps_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Flash 只替换表体 content，不改变候选认领语义。"""

    source = flash_models._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            flash_models._LineItem(
                text="A B",
                bbox=(10.0, 20.0, 80.0, 30.0),
                angle=0,
                source_index=7,
                effective_height=10.0,
            )
        ],
        chars=[],
        drawing_lines=[],
    )
    candidate = flash_models._TableCandidate(
        bbox=(0.0, 10.0, 90.0, 40.0),
        local_bbox=(0.0, 10.0, 90.0, 40.0),
        angle=0,
        score=1.0,
        core_bbox=(0.0, 10.0, 90.0, 40.0),
        line_indices={7},
    )
    html = "<table><tbody><tr><td>A</td><td>B</td></tr></tbody></table>"
    projection = MagicMock(return_value="fallback")
    monkeypatch.setattr(flash_tables, "_recover_native_table_html", MagicMock(return_value=html))
    monkeypatch.setattr(flash_tables, "project_pdf_table_text", projection)

    blocks, annotations, claimed = flash_tables._materialize_table_blocks(
        source,
        [candidate],
    )

    assert annotations == []
    assert claimed == {7}
    assert blocks[0]["content"] == html
    assert "cell_merge" not in blocks[0]
    projection.assert_not_called()


def test_flash_ocr_projects_table_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Flash OCR 保持现有 OCR 空间投影路径。"""

    monkeypatch.setattr(pdf_tables, "run_ocr_inference", MagicMock(return_value=[[]]))
    monkeypatch.setattr(pdf_tables, "project_ocr_table_text", MagicMock(return_value="OCR TABLE"))
    context = MagicMock()
    context.get_ocr_model.return_value = SimpleNamespace(ocr=object())
    model_list = [[{"type": BlockType.TABLE, "bbox": [0.1, 0.1, 0.9, 0.9]}]]
    image = Image.new("RGB", (100, 100), "white")
    try:
        pdf_tables._fill_flash_ocr_table_contents(
            [{"img_pil": image}],
            model_list,
            context,
        )
    finally:
        image.close()

    assert model_list[0][0]["content"] == "OCR TABLE"
