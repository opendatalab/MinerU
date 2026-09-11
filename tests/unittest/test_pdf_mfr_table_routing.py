"""验证最终表格版面对 MFR 输入的归属约束。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from mineru.backend.analysis.pdf import window
from mineru.backend.analysis.pdf.formulas import _build_formula_inputs
from mineru.backend.analysis.pdf.tables import _build_table_external_mfr_inputs


def _formula(bbox: list[float], label: str = "inline_formula") -> dict[str, Any]:
    """构造与 MFD 输出一致的测试公式。"""
    return {"label": label, "bbox": bbox, "score": 0.9, "latex": ""}


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("table_bbox", [[0.1, 0.2, 0.9, 0.8], [10, 40, 90, 160]])
def test_final_table_geometry_preserves_external_formulas_and_inputs(angle: int, table_bbox: list[float]) -> None:
    """旋转不改变页面坐标；保留正文、表题、表注及中心位于表外的交叠框。"""
    formulas = [
        [
            _formula([30, 60, 40, 80]),
            _formula([30, 20, 40, 30]),
            _formula([30, 170, 40, 180]),
            _formula([1, 60, 5, 80]),
            _formula([5, 60, 12, 80]),
            _formula([30, 20, 40, 30], "display_formula"),
        ],
        [_formula([30, 60, 40, 80])],
    ]
    blocks = [[{"type": "table", "bbox": table_bbox, "angle": angle}], []]
    original = deepcopy((formulas, blocks))
    result = _build_table_external_mfr_inputs(formulas, blocks, [(100, 200), (100, 200)])
    assert result == [formulas[0][1:5], formulas[1]]
    assert (formulas, blocks) == original
    result[0][0]["bbox"][0] = -1
    assert (formulas, blocks) == original


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        1,
        [],
        [0, 0, 1],
        [0, 0, 0, 1],
        [0.8, 0.8, 0.1, 0.1],
        [0, 0, float("inf"), 1],
        [0, 0, float("nan"), 1],
        [0, 0, "bad", 1],
    ],
)
def test_invalid_table_bbox_does_not_remove_formulas(bbox: Any) -> None:
    """无效表格框不能导致合法公式被排除。"""
    formulas = [[_formula([20, 20, 30, 30])]]
    assert _build_table_external_mfr_inputs(formulas, [[{"type": "table", "bbox": bbox}]], [(100, 100)]) == formulas


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
@pytest.mark.parametrize("parse_mode", ["txt", "ocr"])
@pytest.mark.parametrize("content", ["", "<table><tr><td>recognized</td></tr></table>"])
def test_window_mfr_routing_and_original_ocr_masks(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
    parse_mode: str,
    content: str,
) -> None:
    """验证档位路由、空 VLM 结果与原始 OCR 遮罩；Basic 规则清理由已有测试覆盖。"""
    layout = [[_formula([20, 20, 30, 30]), _formula([1, 1, 5, 5]), _formula([40, 40, 50, 50], "display_formula")]]
    blocks = [[{"type": "table", "bbox": [0.1, 0.1, 0.9, 0.9], "content": content}]]
    original = deepcopy(layout)
    mfr = MagicMock(side_effect=lambda inputs, *_args, **_kwargs: deepcopy(inputs))
    enabled = effort == "medium" or parse_mode == "txt"
    context = SimpleNamespace(mfr_model=SimpleNamespace(batch_predict=mfr)) if enabled else SimpleNamespace()
    det = MagicMock(return_value=[[]])
    fill = MagicMock(return_value=blocks)
    monkeypatch.setattr(window, "_ocr_det", det)
    monkeypatch.setattr(window, "_fill_window_block_content_and_lines", fill)
    for name in (
        "_apply_medium_table_recognition",
        "_apply_medium_display_formula_results",
        "_apply_medium_formula_number_ocr",
        "_apply_ocr_rec_results",
    ):
        monkeypatch.setattr(window, name, MagicMock())
    with Image.new("RGB", (100, 100)) as image:
        window._process_text_and_formulas(
            [{"img_pil": image, "scale": 1.0}],
            [MagicMock()],
            blocks,
            parse_mode,
            effort,
            context,
            layout,
        )
    if enabled:
        expected = _build_formula_inputs(layout) if effort == "medium" else [[layout[0][1]]]
        assert mfr.call_args.args[0] == expected
        assert mfr.call_args.kwargs["interline_enable"] is (effort == "medium")
    else:
        mfr.assert_not_called()
    assert det.call_args.args[3] == _build_formula_inputs(layout)
    assert layout == original
    expected_inline = [layout[0][:2]] if effort == "medium" or not enabled else [[layout[0][1]]]
    assert fill.call_args.args[3] == expected_inline


@pytest.mark.parametrize("effort", ["high", "xhigh"])
@pytest.mark.parametrize("final_table", [True, False])
def test_final_layout_controls_filter_and_empty_results_skip_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
    final_table: bool,
) -> None:
    """最终表格框优先于本地框；全空输入不得加载 MFR 或恢复表内 sidecar。"""
    formula = _formula([20, 20, 30, 30])
    # 本地表格与最终表格刻意不一致，确保只根据最终 model_list 决策。
    local_table_bbox = [70, 70, 90, 90] if final_table else [10, 10, 50, 50]
    layout = [[formula, {"label": "table", "bbox": local_table_bbox}], [_formula([10, 10, 20, 20], "display_formula")]]
    blocks = [[{"type": "table", "bbox": [0.1, 0.1, 0.5, 0.5]}] if final_table else [], []]
    mfr = MagicMock(side_effect=lambda inputs, *_args, **_kwargs: deepcopy(inputs))
    context = SimpleNamespace() if final_table else SimpleNamespace(mfr_model=SimpleNamespace(batch_predict=mfr))
    fill = MagicMock(return_value=blocks)
    det = MagicMock(return_value=[[], []])
    monkeypatch.setattr(window, "_ocr_det", det)
    monkeypatch.setattr(window, "_fill_window_block_content_and_lines", fill)
    with Image.new("RGB", (100, 100)) as image:
        window._process_text_and_formulas(
            [{"img_pil": image, "scale": 1.0}] * 2,
            [MagicMock()] * 2,
            blocks,
            "txt",
            effort,
            context,
            layout,
        )
    assert fill.call_args.args[3] == ([[], []] if final_table else [[formula], []])
    assert det.call_args.args[3] == _build_formula_inputs(layout)
    assert mfr.call_count == (0 if final_table else 1)


def test_demo1_frozen_mfr_inputs_exclude_only_29_table_formulas() -> None:
    """固定真实检测输入，验证第 6/9 页排除 28/1 个且两档剩余 45 个一致。"""
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/demo1_mfr_table_inputs.json").read_text())
    results = {}
    for tier, data in fixture["tiers"].items():
        result = _build_table_external_mfr_inputs(data["formulas"], data["tables"], data["page_sizes"])
        counts = [len(page) for page in result]
        assert counts == [2, 0, 14, 14, 1, 0, 6, 4, 4, 0, 0, 0, 0]
        before = [sum(f["label"] == "inline_formula" for f in page) for page in data["formulas"]]
        expected_removed = [0, 0, 0, 0, 0, 28, 0, 0, 1, 0, 0, 0, 0] if tier == "advanced" else [0] * 13
        assert [a - b for a, b in zip(before, counts, strict=True)] == expected_removed
        results[tier] = result
    assert results["standard"] == results["advanced"]
