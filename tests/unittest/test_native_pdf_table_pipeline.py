# Copyright (c) Opendatalab. All rights reserved.
"""验证 Native PDF 表格结构恢复和 Flash OCR 表格投影。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from mineru.backend.analysis.pdf import formulas as pdf_formulas
from mineru.backend.analysis.pdf import layout as pdf_layout
from mineru.backend.analysis.pdf import tables as pdf_tables
from mineru.backend.analysis.pdf import window as pdf_window
from mineru.model.flash.pdf import models as flash_models
from mineru.model.flash.pdf import tables as flash_tables
from mineru.model.flash.pdf.document import PDFDocument, PDFPageTextGeometry
from mineru.types import RAW_FORMULA_NUMBER, BlockType


_PROJECT_ROOT = Path(__file__).parents[2]


def _build_native_pdf_page(*, width: float = 100.0, height: float = 100.0) -> MagicMock:
    """构造带完整原生表格页面接口的测试替身。"""

    page = MagicMock()
    page.size = (width, height)
    page.get_chars.return_value = []
    page.get_chars_with_geometry.return_value = PDFPageTextGeometry(
        chars=[],
        tight_bboxes={},
        origins={},
    )
    page.get_drawing_lines.return_value = []
    page.get_path_infos.return_value = []
    return page


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


def test_medium_native_table_priority_accepts_html_and_removes_internal_text_and_formulas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Medium 原生命中后原子清理表内文本和公式并保留表外对象。"""

    html = "<table><tbody><tr><td>A</td></tr></tbody></table>"
    recover = MagicMock(return_value=SimpleNamespace(html=html, source="vector_grid", confidence=1.0))
    monkeypatch.setattr(pdf_tables, "recover_native_pdf_table", recover)
    table_block = {
        "type": BlockType.TABLE,
        "bbox": [0.1, 0.2, 0.9, 0.8],
        "angle": 0,
    }
    internal_text = {
        "type": BlockType.TEXT,
        "bbox": [0.2, 0.3, 0.8, 0.4],
    }
    internal_equation = {
        "type": BlockType.EQUATION,
        "bbox": [0.2, 0.45, 0.6, 0.55],
    }
    internal_formula_number = {
        "type": RAW_FORMULA_NUMBER,
        "bbox": [0.7, 0.45, 0.8, 0.55],
    }
    outside_equation = {
        "type": BlockType.EQUATION,
        "bbox": [0.01, 0.01, 0.08, 0.08],
    }
    caption = {
        "type": BlockType.TABLE_CAPTION,
        "bbox": [0.1, 0.1, 0.9, 0.18],
        "content": "caption",
    }
    footnote = {
        "type": BlockType.TABLE_FOOTNOTE,
        "bbox": [0.1, 0.82, 0.9, 0.9],
        "content": "footnote",
    }
    layout_res = [
        {"label": "table", "bbox": [10, 40, 90, 160]},
        {"label": "inline_formula", "bbox": [20, 70, 30, 80]},
        {"label": "display_formula", "bbox": [40, 90, 60, 110]},
        {"label": "formula_number", "bbox": [70, 90, 80, 110]},
        {"label": "inline_formula", "bbox": [1, 1, 5, 5]},
    ]
    model_list = [[outside_equation, caption, table_block, internal_text, internal_equation, internal_formula_number, footnote]]
    page = _build_native_pdf_page(width=100.0, height=200.0)
    image = Image.new("RGB", (100, 200), "white")
    try:
        summary = pdf_tables._apply_native_txt_table_priority(
            model_list,
            [layout_res],
            [page],
            [{"img_pil": image, "scale": 2.0}],
            effort="medium",
        )
    finally:
        image.close()

    assert table_block["content"] == html
    assert model_list == [[outside_equation, caption, table_block, footnote]]
    assert [item["label"] for item in layout_res] == ["table", "inline_formula"]
    assert pdf_formulas._build_formula_inputs([layout_res]) == [
        [
            {
                "label": "inline_formula",
                "bbox": [1, 1, 5, 5],
                "score": 0.0,
                "latex": "",
            }
        ]
    ]
    assert summary == pdf_tables._NativeTablePrioritySummary(
        total=1,
        accepted=1,
        removed_internal_text=1,
        removed_formula_blocks=2,
        removed_formula_layout_items=3,
    )
    table_input = recover.call_args.args[0]
    assert table_input.table_bbox == pytest.approx((10.0, 40.0, 90.0, 160.0))
    assert table_input.page_size == (100.0, 200.0)
    assert table_input.angle == 0
    page.get_chars_with_geometry.assert_called_once_with()
    page.get_drawing_lines.assert_called_once_with()
    page.get_path_infos.assert_called_once_with()


@pytest.mark.parametrize(
    ("extra_blocks", "layout_res"),
    [
        ([{"type": BlockType.IMAGE, "bbox": [0.2, 0.2, 0.4, 0.4]}], []),
        ([{"type": BlockType.CODE, "bbox": [0.2, 0.2, 0.4, 0.4]}], []),
    ],
)
def test_hybrid_native_table_priority_falls_back_for_complex_content(
    monkeypatch: pytest.MonkeyPatch,
    extra_blocks: list[dict[str, object]],
    layout_res: list[dict[str, object]],
) -> None:
    """验证 Medium 图片和代码复杂内容继续保留现有模型回落。"""

    recover = MagicMock()
    monkeypatch.setattr(pdf_tables, "recover_native_pdf_table", recover)
    table_block = {
        "type": BlockType.TABLE,
        "bbox": [0.1, 0.1, 0.9, 0.9],
        "angle": 0,
    }
    image = Image.new("RGB", (100, 100), "white")
    try:
        summary = pdf_tables._apply_native_txt_table_priority(
            [[table_block, *extra_blocks]],
            [[*layout_res]],
            [_build_native_pdf_page()],
            [{"img_pil": image, "scale": 1.0}],
            effort="medium",
        )
    finally:
        image.close()

    assert "content" not in table_block
    assert summary == pdf_tables._NativeTablePrioritySummary(
        total=1,
        complex_fallbacks=1,
    )
    recover.assert_not_called()


def test_high_native_table_priority_ignores_layout_formula_and_removes_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 High 不采信 layout 公式框，并在原生命中后移除重复对象。"""

    html = "<table><tbody><tr><td>T</td></tr></tbody></table>"
    recover = MagicMock(return_value=SimpleNamespace(html=html, source="vector_grid", confidence=1.0))
    monkeypatch.setattr(pdf_tables, "recover_native_pdf_table", recover)
    table_block = {"type": BlockType.TABLE, "bbox": [0.1, 0.1, 0.9, 0.9], "angle": 0}
    equation_block = {"type": BlockType.EQUATION, "bbox": [0.2, 0.2, 0.4, 0.4]}
    layout_res = [{"label": "inline_formula", "bbox": [20, 20, 40, 40]}]
    page_blocks = [table_block, equation_block]
    image = Image.new("RGB", (100, 100), "white")
    try:
        summary = pdf_tables._apply_native_txt_table_priority(
            [page_blocks],
            [layout_res],
            [_build_native_pdf_page()],
            [{"img_pil": image, "scale": 1.0}],
            effort="high",
        )
    finally:
        image.close()

    assert table_block["content"] == html
    assert page_blocks == [table_block]
    assert layout_res == []
    assert summary == pdf_tables._NativeTablePrioritySummary(
        total=1,
        accepted=1,
        removed_formula_blocks=1,
        removed_formula_layout_items=1,
    )
    vlm_blocks, accepted_tables = pdf_tables._split_native_high_table_blocks([page_blocks])
    assert vlm_blocks == [[]]
    assert accepted_tables == [[table_block]]
    recover.assert_called_once()


@pytest.mark.parametrize("effort", ["medium", "high"])
@pytest.mark.parametrize("angle", [90, 180, 270])
def test_hybrid_native_table_priority_attempts_rotated_table(
    monkeypatch: pytest.MonkeyPatch,
    effort: Literal["medium", "high"],
    angle: int,
) -> None:
    """验证 Medium/High 会把标准旋转角传入原生表格恢复器。"""

    html = "<table><tbody><tr><td>rotated</td></tr></tbody></table>"
    recover = MagicMock(return_value=SimpleNamespace(html=html, source="vector_grid", confidence=1.0))
    monkeypatch.setattr(pdf_tables, "recover_native_pdf_table", recover)
    table_block = {
        "type": BlockType.TABLE,
        "bbox": [0.1, 0.1, 0.9, 0.9],
        "angle": angle,
    }
    image = Image.new("RGB", (100, 100), "white")
    try:
        summary = pdf_tables._apply_native_txt_table_priority(
            [[table_block]],
            [[]],
            [_build_native_pdf_page()],
            [{"img_pil": image, "scale": 1.0}],
            effort=effort,
        )
    finally:
        image.close()

    assert table_block["content"] == html
    assert summary == pdf_tables._NativeTablePrioritySummary(
        total=1,
        accepted=1,
    )
    recover.assert_called_once()
    assert recover.call_args.args[0].angle == angle


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_hybrid_native_table_priority_accepts_real_rotated_table(
    effort: Literal["medium", "high"],
) -> None:
    """验证真实 270 度表格在 Medium/High 中都能直接生成原生 HTML。"""

    manifest = json.loads((_PROJECT_ROOT / "tests" / "fixtures" / "native_pdf_table_demo_manifest.json").read_text())
    target = next(
        item
        for item in manifest["tables"]
        if item["file"] == "demo1.pdf" and item["page_index"] == 4 and item["table_index"] == 0
    )
    pdf_path = _PROJECT_ROOT / manifest["source_root"] / target["file"]
    with PDFDocument(pdf_path.read_bytes()) as document:
        page = document[target["page_index"]]
        image = Image.new("RGB", (round(page.size[0]), round(page.size[1])), "white")
        table_block = {
            "type": BlockType.TABLE,
            "bbox": target["bbox"],
            "angle": target["angle"],
        }
        try:
            summary = pdf_tables._apply_native_txt_table_priority(
                [[table_block]],
                [[]],
                [page],
                [{"img_pil": image, "scale": 1.0}],
                effort=effort,
            )
        finally:
            image.close()

    assert summary == pdf_tables._NativeTablePrioritySummary(total=1, accepted=1)
    assert table_block["content"].count("<tr>") == target["rows"]
    assert table_block["content"].startswith("<table><tbody>")


def test_hybrid_native_table_priority_keeps_none_and_exception_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证规则拒绝或异常时逐表保留现有模型路径。"""

    recover = MagicMock(side_effect=[None, RuntimeError("broken")])
    monkeypatch.setattr(pdf_tables, "recover_native_pdf_table", recover)
    tables = [
        {"type": BlockType.TABLE, "bbox": [0.05, 0.1, 0.45, 0.9], "angle": 270},
        {"type": BlockType.TABLE, "bbox": [0.55, 0.1, 0.95, 0.9], "angle": 90},
    ]
    equation_block = {"type": BlockType.EQUATION, "bbox": [0.1, 0.2, 0.3, 0.4]}
    page_blocks = [tables[0], equation_block, tables[1]]
    layout_res = [{"label": "inline_formula", "bbox": [10, 20, 30, 40]}]
    page = _build_native_pdf_page()
    image = Image.new("RGB", (100, 100), "white")
    try:
        summary = pdf_tables._apply_native_txt_table_priority(
            [page_blocks],
            [layout_res],
            [page],
            [{"img_pil": image, "scale": 1.0}],
            effort="medium",
        )
    finally:
        image.close()

    assert all("content" not in block for block in tables)
    assert page_blocks == [tables[0], equation_block, tables[1]]
    assert layout_res == [{"label": "inline_formula", "bbox": [10, 20, 30, 40]}]
    assert summary == pdf_tables._NativeTablePrioritySummary(
        total=2,
        rejected=1,
        errors=1,
    )
    assert recover.call_count == 2
    assert [call.args[0].angle for call in recover.call_args_list] == [270, 90]
    page.get_chars_with_geometry.assert_called_once_with()


def test_medium_table_tasks_skip_native_html_and_keep_model_fallback() -> None:
    """验证 Medium 只为未命中的表格构造 OCR 和结构模型任务。"""

    native_table = {
        "type": BlockType.TABLE,
        "bbox": [0.0, 0.0, 0.4, 1.0],
        "angle": 0,
        "content": "<table><tbody><tr><td>native</td></tr></tbody></table>",
    }
    fallback_table = {
        "type": BlockType.TABLE,
        "bbox": [0.6, 0.0, 1.0, 1.0],
        "angle": 0,
    }

    tasks = pdf_tables._collect_medium_table_tasks(
        [[native_table, fallback_table]],
        [[]],
        [np.zeros((40, 100, 3), dtype=np.uint8)],
    )

    assert len(tasks) == 1
    assert tasks[0]["table_block"] is fallback_table
    assert native_table["content"].startswith("<table>")


def test_medium_formula_processing_skips_mfr_after_native_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证原生命中清空公式输入后不再加载或调用 Medium MFR。"""

    native_table = {
        "type": BlockType.TABLE,
        "bbox": [0.1, 0.1, 0.9, 0.9],
        "content": "<table><tbody><tr><td>native formula text</td></tr></tbody></table>",
    }
    model_list = [[native_table]]
    image = Image.new("RGB", (100, 100), "white")
    local_model_context = MagicMock()
    local_model_context.mfr_model = MagicMock()
    medium_table_recognition = MagicMock()
    monkeypatch.setattr(pdf_window, "_apply_medium_table_recognition", medium_table_recognition)
    monkeypatch.setattr(pdf_window, "_apply_medium_display_formula_results", MagicMock())
    monkeypatch.setattr(pdf_window, "_apply_medium_formula_number_ocr", MagicMock())
    monkeypatch.setattr(pdf_window, "_ocr_det", MagicMock(return_value=[[]]))
    monkeypatch.setattr(
        pdf_window,
        "_fill_window_block_content_and_lines",
        MagicMock(return_value=model_list),
    )

    try:
        result = pdf_window._process_text_and_formulas(
            [{"img_pil": image, "scale": 1.0}],
            [_build_native_pdf_page()],
            model_list,
            "txt",
            "medium",
            local_model_context,
            [[]],
        )
    finally:
        image.close()

    assert result is model_list
    local_model_context.mfr_model.batch_predict.assert_not_called()
    assert medium_table_recognition.call_args.args[2] == [[]]


def test_high_native_tables_restore_source_order_and_drop_private_marker() -> None:
    """验证 High 原生表格按源顺序回插且临时字段不进入 ModelJson。"""

    blocks = [
        {"type": BlockType.TEXT, "bbox": [0.0, 0.0, 1.0, 0.1]},
        {
            "type": BlockType.TABLE,
            "bbox": [0.0, 0.1, 1.0, 0.4],
            "content": "<table><tbody><tr><td>native</td></tr></tbody></table>",
        },
        {"type": BlockType.TABLE_CAPTION, "bbox": [0.0, 0.4, 1.0, 0.5]},
        {"type": BlockType.TABLE, "bbox": [0.0, 0.5, 1.0, 0.9]},
    ]

    vlm_blocks, accepted_tables = pdf_tables._split_native_high_table_blocks([blocks])
    assert [block["type"] for block in vlm_blocks[0]] == [
        BlockType.TEXT,
        BlockType.TABLE_CAPTION,
        BlockType.TABLE,
    ]

    vlm_blocks[0][-1]["content"] = "<table><tbody><tr><td>fallback</td></tr></tbody></table>"
    restored = pdf_tables._restore_native_high_table_blocks(vlm_blocks, accepted_tables)
    converted = pdf_layout._convert_vlm_results_to_model_list(restored)

    assert [block["type"] for block in converted[0]] == [
        BlockType.TEXT,
        BlockType.TABLE,
        BlockType.TABLE_CAPTION,
        BlockType.TABLE,
    ]
    assert converted[0][1]["content"].endswith("native</td></tr></tbody></table>")
    assert converted[0][3]["content"].endswith("fallback</td></tr></tbody></table>")
    assert all(pdf_tables._NATIVE_HIGH_SOURCE_ORDER_KEY not in block for block in converted[0])


def test_high_txt_window_excludes_native_table_from_vlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 High TXT 只把规则回落表送入 VLM，并原位保留命中表。"""

    native_html = "<table><tbody><tr><td>native</td></tr></tbody></table>"
    fallback_html = "<table><tbody><tr><td>fallback</td></tr></tbody></table>"
    layout_results = [
        [
            {"label": "table", "bbox": [0, 0, 40, 100], "angle": 0},
            {"label": "table", "bbox": [60, 0, 100, 100], "angle": 0},
        ]
    ]
    page_image = Image.new("RGB", (100, 100), "white")
    fake_document = MagicMock(page_count=1)
    fake_document.__getitem__.return_value = _build_native_pdf_page()
    hybrid_model = MagicMock()
    hybrid_model.layout_model.batch_predict.return_value = layout_results
    vlm_predictor = MagicMock()
    cached_geometry = PDFPageTextGeometry(chars=[], tight_bboxes={}, origins={})

    def fake_native_priority(
        model_list: list[list[dict[str, object]]],
        _images_layout_res: object,
        _pdf_pages: object,
        _images_list: object,
        *,
        effort: object,
        page_text_geometries: object,
    ) -> object:
        """只命中首个表格，构造同页混合短路场景。"""

        assert effort == "high"
        assert isinstance(page_text_geometries, list)
        page_text_geometries[0] = cached_geometry
        model_list[0][0]["content"] = native_html
        return pdf_tables._NativeTablePrioritySummary(total=2, accepted=1, rejected=1)

    def fake_high_extract(*, blocks_list: list[list[dict[str, object]]], **_kwargs: object) -> object:
        """校验 VLM 只收到回落表，并模拟现有模型 HTML。"""

        assert len(blocks_list[0]) == 1
        assert blocks_list[0][0]["bbox"] == [0.6, 0.0, 1.0, 1.0]
        blocks_list[0][0]["content"] = fallback_html
        return blocks_list

    def keep_window_model_list(
        _images_list: object,
        _pdf_pages: object,
        model_list: list[list[dict[str, object]]],
        _parse_mode: object,
        _effort: object,
        _local_model_context: object,
        _images_layout_res: object,
        _page_text_geometries: object,
    ) -> list[list[dict[str, object]]]:
        """跳过与本测试无关的正文和公式回填。"""

        assert isinstance(_page_text_geometries, list)
        assert _page_text_geometries[0] is cached_geometry
        return model_list

    vlm_predictor.batch_extract_with_layout.side_effect = fake_high_extract
    monkeypatch.setattr(pdf_window, "_configured_window_size", lambda default: 1)
    monkeypatch.setattr(
        pdf_window,
        "load_images_from_pdf_bytes_range",
        MagicMock(return_value=[{"img_pil": page_image, "scale": 1.0}]),
    )
    monkeypatch.setattr(pdf_window, "_apply_table_orientations", MagicMock())
    monkeypatch.setattr(pdf_window, "_apply_native_txt_table_priority", fake_native_priority)
    monkeypatch.setattr(pdf_window, "_process_text_and_formulas", keep_window_model_list)
    monkeypatch.setattr(pdf_window, "_apply_seal_ocr", MagicMock())
    monkeypatch.setattr(pdf_window, "_attach_visual_block_images", MagicMock())

    result = pdf_window.process_pdf_windows(
        b"fake-pdf",
        fake_document,
        effort="high",
        parse_mode="txt",
        image_analysis=False,
        flash_txt_mode=False,
        hybrid_model=hybrid_model,
        vlm_predictor=vlm_predictor,
    )

    assert [block["content"] for block in result[0]] == [native_html, fallback_html]
    assert all(pdf_tables._NATIVE_HIGH_SOURCE_ORDER_KEY not in block for block in result[0])
    vlm_predictor.batch_extract_with_layout.assert_called_once()
    with pytest.raises(ValueError, match="closed image"):
        page_image.getpixel((0, 0))
