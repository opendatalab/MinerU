# Copyright (c) Opendatalab. All rights reserved.
"""验证 Native PDF 表格结构恢复在 Flash/Low pipeline 中的采用与回退。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from mineru.backend.analysis.pdf import tables as low_tables
from mineru.model.flash.native_pdf import models as flash_models
from mineru.model.flash.native_pdf import tables as flash_tables
from mineru.types import BlockType
from mineru.utils.native_pdf_table import NativeTableResult


def _native_result(html: str) -> NativeTableResult:
    """构造 pipeline 采用分支需要的最小原生结构结果。"""

    return NativeTableResult(
        html=html,
        rows=2,
        cols=2,
        cells=(),
        source="vector_grid",
        confidence=1.0,
    )


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


def test_low_txt_prefers_native_html_and_caches_page_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Low TXT 采用原生 HTML，并按页只读取一次字符、drawing 和 Path。"""

    html = "<table><tbody><tr><td>A</td><td>B</td></tr></tbody></table>"
    recover = MagicMock(return_value=_native_result(html))
    projection = MagicMock(return_value="fallback")
    monkeypatch.setattr(low_tables, "recover_native_pdf_table", recover)
    monkeypatch.setattr(low_tables, "project_pdf_table_text", projection)
    pdf_page = MagicMock()
    pdf_page.size = (100.0, 100.0)
    pdf_page.rotation = 0
    pdf_page.get_chars.return_value = []
    pdf_page.get_drawing_lines.return_value = []
    pdf_page.get_path_infos.return_value = []
    model_list = [
        [
            {"type": BlockType.TABLE, "bbox": [0.1, 0.1, 0.9, 0.9], "angle": 0},
            {"type": BlockType.TABLE, "bbox": [0.2, 0.2, 0.8, 0.8], "angle": 0},
        ]
    ]
    image = Image.new("RGB", (100, 100), "white")
    try:
        low_tables._fill_low_table_contents(
            [{"img_pil": image}],
            [pdf_page],
            model_list,
            "txt",
            MagicMock(),
        )
    finally:
        image.close()

    assert [block["content"] for block in model_list[0]] == [html, html]
    assert all("cell_merge" not in block for block in model_list[0])
    assert recover.call_count == 2
    projection.assert_not_called()
    pdf_page.get_chars.assert_called_once_with()
    pdf_page.get_drawing_lines.assert_called_once_with()
    pdf_page.get_path_infos.assert_called_once_with()


def test_low_txt_rejection_keeps_existing_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Low TXT 原生候选主动放弃后仍逐字采用旧空间投影。"""

    monkeypatch.setattr(low_tables, "recover_native_pdf_table", MagicMock(return_value=None))
    projection = MagicMock(return_value="A       B")
    monkeypatch.setattr(low_tables, "project_pdf_table_text", projection)
    pdf_page = MagicMock()
    pdf_page.size = (100.0, 100.0)
    pdf_page.rotation = 0
    pdf_page.get_chars.return_value = []
    pdf_page.get_drawing_lines.return_value = []
    pdf_page.get_path_infos.return_value = []
    model_list = [[{"type": BlockType.TABLE, "bbox": [0.1, 0.1, 0.9, 0.9]}]]
    image = Image.new("RGB", (100, 100), "white")
    try:
        low_tables._fill_low_table_contents(
            [{"img_pil": image}],
            [pdf_page],
            model_list,
            "txt",
            MagicMock(),
        )
    finally:
        image.close()

    assert model_list[0][0]["content"] == "A       B"
    projection.assert_called_once()


def test_low_ocr_never_invokes_native_structure_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Low OCR 保持现有 OCR 空间投影路径且不调用原生结构器。"""

    recover = MagicMock(side_effect=AssertionError("native recovery must not run"))
    monkeypatch.setattr(low_tables, "recover_native_pdf_table", recover)
    monkeypatch.setattr(low_tables, "run_ocr_inference", MagicMock(return_value=[[]]))
    monkeypatch.setattr(low_tables, "project_ocr_table_text", MagicMock(return_value="OCR TABLE"))
    context = MagicMock()
    context.get_ocr_model.return_value = SimpleNamespace(ocr=object())
    model_list = [[{"type": BlockType.TABLE, "bbox": [0.1, 0.1, 0.9, 0.9]}]]
    image = Image.new("RGB", (100, 100), "white")
    try:
        low_tables._fill_low_table_contents(
            [{"img_pil": image}],
            [MagicMock()],
            model_list,
            "ocr",
            context,
        )
    finally:
        image.close()

    recover.assert_not_called()
    assert model_list[0][0]["content"] == "OCR TABLE"


def test_low_txt_compensates_pdf_page_dictionary_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Low 原生结构器使用视觉角度而不改变表块原始角度字段。"""

    recover = MagicMock(return_value=_native_result("<table></table>"))
    monkeypatch.setattr(low_tables, "recover_native_pdf_table", recover)
    pdf_page = MagicMock()
    pdf_page.size = (100.0, 80.0)
    pdf_page.rotation = 90
    pdf_page.get_chars.return_value = []
    pdf_page.get_drawing_lines.return_value = []
    pdf_page.get_path_infos.return_value = []
    table_block = {
        "type": BlockType.TABLE,
        "bbox": [0.1, 0.1, 0.9, 0.9],
        "angle": 270,
    }
    image = Image.new("RGB", (100, 80), "white")
    try:
        low_tables._fill_low_table_contents(
            [{"img_pil": image}],
            [pdf_page],
            [[table_block]],
            "txt",
            MagicMock(),
        )
    finally:
        image.close()

    native_input = recover.call_args.args[0]
    assert native_input.angle == 0
    assert table_block["angle"] == 270
