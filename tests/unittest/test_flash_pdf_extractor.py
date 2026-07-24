from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from mineru.backend.flash import pdf_extractor


def _install_pdf_document(
    monkeypatch: pytest.MonkeyPatch,
    *,
    page_count: int = 1,
    classified_mode: str = "txt",
) -> tuple[MagicMock, MagicMock]:
    """安装支持上下文管理的 PDFDocument 替身并返回文档与管理器。"""

    pdf_doc = MagicMock()
    pdf_doc.page_count = page_count
    pdf_doc.classify.return_value = classified_mode
    context_manager = MagicMock()
    context_manager.__enter__.return_value = pdf_doc
    context_manager.__exit__.return_value = False
    monkeypatch.setattr(pdf_extractor, "PDFDocument", MagicMock(return_value=context_manager))
    return pdf_doc, context_manager


def _install_hybrid_analyze(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_list: list[list[dict[str, Any]]],
) -> MagicMock:
    """安装 Hybrid analyze 模块替身，避免路由测试加载真实模型。"""

    hybrid_doc_analyze = MagicMock(return_value=([object()], model_list))
    analyze_module = ModuleType("mineru.backend.hybrid.analyze")
    analyze_module.doc_analyze = hybrid_doc_analyze  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mineru.backend.hybrid.analyze", analyze_module)
    return hybrid_doc_analyze


@pytest.mark.parametrize(
    ("parse_mode", "expected_classify_calls"),
    [
        ("ocr", 0),
        ("auto", 1),
    ],
)
def test_ocr_mode_delegates_to_hybrid_low(
    monkeypatch: pytest.MonkeyPatch,
    parse_mode: str,
    expected_classify_calls: int,
) -> None:
    """验证显式或自动判定的 OCR 模式都精确委托 Hybrid low。"""

    pdf_bytes = b"%PDF-1.7\n"
    page_index_map = [7]
    expected_model_list = [[{"type": "text", "content": "hybrid"}]]
    pdf_doc, context_manager = _install_pdf_document(
        monkeypatch,
        classified_mode="ocr",
    )
    hybrid_doc_analyze = _install_hybrid_analyze(
        monkeypatch,
        model_list=expected_model_list,
    )

    result = pdf_extractor.doc_analyze(
        pdf_bytes,
        parse_mode=parse_mode,  # type: ignore[arg-type]
        page_index_map=page_index_map,
    )

    assert result is expected_model_list
    assert pdf_doc.classify.call_count == expected_classify_calls
    assert context_manager.__exit__.call_count == 1
    hybrid_doc_analyze.assert_called_once_with(
        pdf_bytes,
        effort="low",
        parse_mode="ocr",
        page_index_map=page_index_map,
    )


@pytest.mark.parametrize(
    ("parse_mode", "expected_classify_calls"),
    [
        ("txt", 0),
        ("auto", 1),
    ],
)
def test_txt_mode_keeps_native_flash_path(
    monkeypatch: pytest.MonkeyPatch,
    parse_mode: str,
    expected_classify_calls: int,
) -> None:
    """验证显式或自动判定的文本模式继续使用 Flash 原生路径。"""

    expected_model_list = [[{"type": "text", "content": "native"}]]
    pdf_doc, _context_manager = _install_pdf_document(
        monkeypatch,
        classified_mode="txt",
    )
    native_analyze = MagicMock(return_value=expected_model_list)
    monkeypatch.setattr(pdf_extractor, "_analyze_native_document", native_analyze)
    hybrid_doc_analyze = _install_hybrid_analyze(
        monkeypatch,
        model_list=[[{"type": "text", "content": "unexpected"}]],
    )

    result = pdf_extractor.doc_analyze(
        b"%PDF-1.7\n",
        parse_mode=parse_mode,  # type: ignore[arg-type]
    )

    assert result is expected_model_list
    assert pdf_doc.classify.call_count == expected_classify_calls
    native_analyze.assert_called_once_with(pdf_doc)
    hybrid_doc_analyze.assert_not_called()


def test_invalid_parse_mode_is_rejected_before_opening_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证非法解析模式在打开 PDF 前保持原有报错行为。"""

    pdf_document = MagicMock()
    monkeypatch.setattr(pdf_extractor, "PDFDocument", pdf_document)

    with pytest.raises(ValueError, match="parse_mode invalid is not supported"):
        pdf_extractor.doc_analyze(b"%PDF-1.7\n", parse_mode="invalid")  # type: ignore[arg-type]

    pdf_document.assert_not_called()


def test_page_index_map_mismatch_is_rejected_before_hybrid_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证页映射长度不匹配时不会进入 Hybrid low。"""

    _install_pdf_document(monkeypatch, page_count=2, classified_mode="ocr")
    hybrid_doc_analyze = _install_hybrid_analyze(monkeypatch, model_list=[])

    with pytest.raises(ValueError, match="Flash page_index_map length mismatch"):
        pdf_extractor.doc_analyze(
            b"%PDF-1.7\n",
            parse_mode="ocr",
            page_index_map=[0],
        )

    hybrid_doc_analyze.assert_not_called()


def test_flash_extractor_has_no_local_ocr_runtime_logic() -> None:
    """守卫 Flash extractor 不再引入或实现本地 OCR 运行时逻辑。"""

    source = inspect.getsource(pdf_extractor)
    forbidden_tokens = (
        "import numpy",
        "import cv2",
        "project_ocr_table_text",
        "_load_ocr_runtime",
        "_run_full_page_ocr",
        "_project_ocr_candidate",
        "_recognize_rotated_ocr_table",
        "AtomModelSingleton",
        "run_ocr_inference",
        "get_processing_window_size",
        "bgr_image",
        "pixel_quad",
        "ocr_model",
    )

    assert not [token for token in forbidden_tokens if token in source]


def _formula_member(
    text: str,
    bbox: tuple[float, float, float, float],
    source_index: int,
) -> tuple[pdf_extractor._LineItem, tuple[float, float, float, float]]:
    """构造公式块序列化测试使用的文本行及其局部几何。"""

    return (
        pdf_extractor._LineItem(
            text=text,
            bbox=bbox,
            angle=0,
            source_index=source_index,
            effective_height=bbox[3] - bbox[1],
        ),
        bbox,
    )


def test_detached_formula_sidecar_sharing_middle_row_moves_to_trailing_line() -> None:
    """验证纯 bbox 规则会后置与正文共享中间视觉行的远距窄幅 sidecar。"""

    members = [
        _formula_member("numerator", (20.0, 0.0, 60.0, 10.0), 0),
        _formula_member("body", (10.0, 10.0, 40.0, 20.0), 1),
        _formula_member("marker", (100.0, 10.0, 110.0, 20.0), 2),
        _formula_member("denominator", (30.0, 20.0, 70.0, 30.0), 3),
    ]

    block = pdf_extractor._formula_members_to_block(
        members,
        (130.0, 60.0),
        0,
        anchor_source_index=2,
    )

    assert block == {
        "type": "equation",
        "bbox": (10.0, 0.0, 110.0, 30.0),
        "angle": 0,
        "content": "numerator\nbody\ndenominator\nmarker",
    }


@pytest.mark.parametrize("marker", ["(4)", "（4）", "﹙4﹚", "(4）"])
def test_adjacent_parenthesized_formula_number_moves_to_trailing_line(marker: str) -> None:
    """验证贴近公式主体的多种圆括号序号后置，且前导逗号留在原视觉行。"""

    members = [
        _formula_member("numerator", (20.0, 0.0, 90.0, 10.0), 0),
        _formula_member(f", {marker}", (91.0, 0.0, 110.0, 10.0), 1),
        _formula_member("body", (10.0, 10.0, 40.0, 20.0), 2),
        _formula_member("denominator", (30.0, 20.0, 70.0, 30.0), 3),
    ]

    block = pdf_extractor._formula_members_to_block(
        members,
        (130.0, 60.0),
        0,
        anchor_source_index=1,
    )

    assert block == {
        "type": "equation",
        "bbox": (10.0, 0.0, 110.0, 30.0),
        "angle": 0,
        "content": f"numerator,\nbody\ndenominator\n{marker}",
    }


def test_adjacent_square_bracket_formula_sidecar_keeps_visual_order() -> None:
    """验证方括号内容不触发圆括号公式序号规则。"""

    members = [
        _formula_member("numerator", (20.0, 0.0, 90.0, 10.0), 0),
        _formula_member(", [4]", (91.0, 0.0, 110.0, 10.0), 1),
        _formula_member("body", (10.0, 10.0, 40.0, 20.0), 2),
        _formula_member("denominator", (30.0, 20.0, 70.0, 30.0), 3),
    ]

    block = pdf_extractor._formula_members_to_block(
        members,
        (130.0, 60.0),
        0,
        anchor_source_index=1,
    )

    assert block == {
        "type": "equation",
        "bbox": (10.0, 0.0, 110.0, 30.0),
        "angle": 0,
        "content": "numerator, [4]\nbody\ndenominator",
    }


def test_detached_formula_sidecar_on_middle_row_moves_after_denominator() -> None:
    """验证独占中间视觉行的远距窄幅 sidecar 排到分母后且不留下空行。"""

    members = [
        _formula_member("numerator", (20.0, 0.0, 70.0, 10.0), 0),
        _formula_member("sidecar", (100.0, 10.0, 110.0, 20.0), 1),
        _formula_member("denominator", (30.0, 20.0, 65.0, 30.0), 2),
    ]

    block = pdf_extractor._formula_members_to_block(
        members,
        (130.0, 60.0),
        0,
        anchor_source_index=1,
    )

    assert block == {
        "type": "equation",
        "bbox": (20.0, 0.0, 110.0, 30.0),
        "angle": 0,
        "content": "numerator\ndenominator\nsidecar",
    }


def test_detached_formula_sidecar_already_at_end_keeps_visual_row() -> None:
    """验证已经处于内容末尾的离散 sidecar 保持原视觉行格式。"""

    members = [
        _formula_member("formula", (10.0, 0.0, 40.0, 10.0), 0),
        _formula_member("terminal", (100.0, 0.0, 110.0, 10.0), 1),
    ]

    block = pdf_extractor._formula_members_to_block(
        members,
        (130.0, 60.0),
        0,
        anchor_source_index=1,
    )

    assert block == {
        "type": "equation",
        "bbox": (10.0, 0.0, 110.0, 10.0),
        "angle": 0,
        "content": "formula        terminal",
    }


def test_attached_formula_sidecar_keeps_visual_order() -> None:
    """验证与公式主体净空不足的右侧锚点保持原视觉顺序。"""

    members = [
        _formula_member("numerator", (20.0, 0.0, 60.0, 10.0), 0),
        _formula_member("body", (10.0, 10.0, 70.0, 20.0), 1),
        _formula_member("sidecar", (85.0, 10.0, 95.0, 20.0), 2),
        _formula_member("denominator", (30.0, 20.0, 70.0, 30.0), 3),
    ]

    block = pdf_extractor._formula_members_to_block(
        members,
        (120.0, 60.0),
        0,
        anchor_source_index=2,
    )

    assert block == {
        "type": "equation",
        "bbox": (10.0, 0.0, 95.0, 30.0),
        "angle": 0,
        "content": "numerator\nbody   sidecar\ndenominator",
    }


def test_wide_formula_sidecar_keeps_visual_order() -> None:
    """验证宽度超过中位行高限制的远距锚点不会被后置。"""

    members = [
        _formula_member("numerator", (20.0, 0.0, 60.0, 10.0), 0),
        _formula_member("body", (10.0, 10.0, 70.0, 20.0), 1),
        _formula_member("wide", (100.0, 10.0, 126.0, 20.0), 2),
        _formula_member("denominator", (30.0, 20.0, 70.0, 30.0), 3),
    ]

    block = pdf_extractor._formula_members_to_block(
        members,
        (140.0, 60.0),
        0,
        anchor_source_index=2,
    )

    assert block == {
        "type": "equation",
        "bbox": (10.0, 0.0, 126.0, 30.0),
        "angle": 0,
        "content": "numerator\nbody      wide\ndenominator",
    }


def test_non_rightmost_formula_sidecar_keeps_visual_order() -> None:
    """验证未处于公式分量最右侧的锚点不会被后置。"""

    members = [
        _formula_member("numerator", (20.0, 0.0, 120.0, 10.0), 0),
        _formula_member("body", (10.0, 10.0, 40.0, 20.0), 1),
        _formula_member("sidecar", (90.0, 10.0, 100.0, 20.0), 2),
        _formula_member("denominator", (30.0, 20.0, 70.0, 30.0), 3),
    ]

    block = pdf_extractor._formula_members_to_block(
        members,
        (140.0, 60.0),
        0,
        anchor_source_index=2,
    )

    assert block == {
        "type": "equation",
        "bbox": (10.0, 0.0, 120.0, 30.0),
        "angle": 0,
        "content": "numerator\nbody        sidecar\ndenominator",
    }


def _axis_line(
    orientation: str,
    bbox: tuple[float, float, float, float],
) -> pdf_extractor._LocalAxisLine:
    """构造表格候选测试使用的局部横竖线。"""

    return pdf_extractor._LocalAxisLine(
        bbox=bbox,
        original_bbox=bbox,
        orientation=orientation,  # type: ignore[arg-type]
        width=0.0,
    )


@pytest.mark.parametrize(
    ("median_height", "rejected_length", "accepted_length"),
    [
        (3.0, 39.99, 40.0),
        (5.0, 49.99, 50.0),
    ],
)
def test_long_rule_group_uses_40pt_or_ten_times_height_threshold(
    median_height: float,
    rejected_length: float,
    accepted_length: float,
) -> None:
    """验证长横线沿用 40pt 或十倍行高门槛，竖线不参与分组。"""

    rejected_lines = [
        _axis_line("horizontal", (0.0, 10.0, rejected_length, 10.1)),
        _axis_line("horizontal", (0.0, 20.0, accepted_length, 20.1)),
        _axis_line("horizontal", (0.0, 30.0, accepted_length, 30.1)),
        _axis_line("vertical", (10.0, 0.0, 10.1, 40.0)),
    ]
    accepted_lines = [_axis_line("horizontal", (0.0, top, accepted_length, top + 0.1)) for top in (10.0, 20.0, 30.0)]

    assert pdf_extractor._group_long_horizontal_rules(rejected_lines, median_height) == []
    assert len(pdf_extractor._group_long_horizontal_rules(accepted_lines, median_height)) == 1


def _rule_table_fixture(
    rule_count: int = 3,
    *,
    centered_columns: bool = False,
) -> tuple[
    list[pdf_extractor._VisualRow],
    list[pdf_extractor._LineItem],
    list[pdf_extractor._LocalAxisLine],
]:
    """构造无 caption 的规则表格，并可切换为宽度变化的居中列。"""

    rows: list[pdf_extractor._VisualRow] = []
    lines: list[pdf_extractor._LineItem] = []
    source_index = 0
    for row_index, top in enumerate((10.0, 30.0, 50.0)):
        fragments: list[pdf_extractor._Fragment] = []
        if centered_columns:
            half_width = (2.0, 8.0, 14.0)[row_index]
            column_bounds = (
                (30.0 - half_width, 30.0 + half_width),
                (90.0 - half_width, 90.0 + half_width),
            )
        else:
            column_bounds = ((10.0, 20.0), (50.0, 60.0), (90.0, 100.0))
        for column_index, (left, right) in enumerate(column_bounds):
            bbox = (left, top, right, top + 5.0)
            text = f"r{row_index}c{column_index}"
            fragments.append(
                pdf_extractor._Fragment(
                    text=text,
                    bbox=bbox,
                    local_bbox=bbox,
                    line_index=source_index,
                    visual_row_id=row_index,
                )
            )
            lines.append(
                pdf_extractor._LineItem(
                    text=text,
                    bbox=bbox,
                    angle=0,
                    source_index=source_index,
                    effective_height=5.0,
                    visual_row_id=row_index,
                )
            )
            source_index += 1
        rows.append(
            pdf_extractor._VisualRow(
                fragments=fragments,
                center_y=top + 2.5,
                bbox=(10.0, top, 100.0, top + 5.0),
                visual_row_id=row_index,
            )
        )
    axis_lines = [_axis_line("horizontal", (0.0, top, 110.0, top + 0.1)) for top in (8.0, 33.0, 58.0)[:rule_count]]
    return rows, lines, axis_lines


def test_rule_table_candidate_accepts_captionless_regular_text_distribution() -> None:
    """验证三横线和连续稳定列足以识别没有显式标题的表格。"""

    rows, lines, axis_lines = _rule_table_fixture()

    candidates = pdf_extractor._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
    )

    assert len(candidates) == 1
    assert candidates[0].line_indices == set(range(9))


def test_rule_table_candidate_accepts_center_aligned_columns_with_varying_widths() -> None:
    """验证左右边界变化但中心稳定的两列表格仍可形成候选。"""

    rows, lines, axis_lines = _rule_table_fixture(centered_columns=True)

    assert pdf_extractor._count_stable_columns(rows, 5.0) == (2, 1.0)
    candidates = pdf_extractor._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
    )

    assert len(candidates) == 1
    assert candidates[0].line_indices == set(range(6))


def test_two_horizontal_rule_grid_is_deliberately_rejected() -> None:
    """验证即使存在竖线和规则文本，两条长横线也不再形成表格候选。"""

    rows, lines, axis_lines = _rule_table_fixture(rule_count=2)
    axis_lines.extend(
        [
            _axis_line("vertical", (0.0, 8.0, 0.1, 58.0)),
            _axis_line("vertical", (55.0, 8.0, 55.1, 58.0)),
            _axis_line("vertical", (109.9, 8.0, 110.0, 58.0)),
        ]
    )

    candidates = pdf_extractor._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
    )

    assert candidates == []


def test_duplicate_horizontal_paths_do_not_satisfy_three_rule_requirement() -> None:
    """验证同一 y 位置的重复 PDF path 只计为一条横线。"""

    axis_lines = [
        _axis_line("horizontal", (0.0, 10.0, 100.0, 10.1)),
        _axis_line("horizontal", (0.0, 10.5, 100.0, 10.6)),
        _axis_line("horizontal", (0.0, 30.0, 100.0, 30.1)),
    ]

    assert pdf_extractor._group_long_horizontal_rules(axis_lines, 5.0) == []


def test_chart_tick_rows_fail_dense_multi_cell_distribution() -> None:
    """验证多个图表刻度行因纵向不连续而不能冒充规则表格。"""

    rows: list[pdf_extractor._VisualRow] = []
    lines: list[pdf_extractor._LineItem] = []
    source_index = 0
    for row_index, (top, anchors) in enumerate(
        (
            (10.0, (10.0, 30.0, 50.0, 70.0, 90.0)),
            (20.0, (50.0,)),
            (30.0, (45.0,)),
            (75.0, (10.0, 30.0, 50.0, 70.0, 90.0)),
            (85.0, (50.0,)),
            (95.0, (45.0,)),
        )
    ):
        fragments: list[pdf_extractor._Fragment] = []
        for anchor in anchors:
            bbox = (anchor, top, anchor + 5.0, top + 5.0)
            text = f"tick-{source_index}"
            fragments.append(
                pdf_extractor._Fragment(
                    text=text,
                    bbox=bbox,
                    local_bbox=bbox,
                    line_index=source_index,
                    visual_row_id=row_index,
                )
            )
            lines.append(
                pdf_extractor._LineItem(
                    text=text,
                    bbox=bbox,
                    angle=0,
                    source_index=source_index,
                    effective_height=5.0,
                    visual_row_id=row_index,
                )
            )
            source_index += 1
        rows.append(
            pdf_extractor._VisualRow(
                fragments=fragments,
                center_y=top + 2.5,
                bbox=pdf_extractor._bbox_union_many([fragment.bbox for fragment in fragments]),
                visual_row_id=row_index,
            )
        )
    axis_lines = [_axis_line("horizontal", (0.0, top, 110.0, top + 0.1)) for top in (0.0, 55.0, 110.0)]

    candidates = pdf_extractor._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 120.0),
        0,
        5.0,
        axis_lines,
    )

    assert candidates == []


def test_split_table_footnote_marker_is_joined_before_matching() -> None:
    """验证旋转表拆开的 For 与星号脚注在视觉行拼接后可被识别。"""

    row = pdf_extractor._VisualRow(
        fragments=[
            pdf_extractor._Fragment("For", (0.0, 0.0, 10.0, 5.0), (0.0, 0.0, 10.0, 5.0), 0),
            pdf_extractor._Fragment("*rainfall", (12.0, 0.0, 35.0, 5.0), (12.0, 0.0, 35.0, 5.0), 1),
        ],
        center_y=2.5,
        bbox=(0.0, 0.0, 35.0, 5.0),
    )

    assert pdf_extractor._is_table_note_text(pdf_extractor._visual_row_text(row))
    assert pdf_extractor._is_table_note_text("1 Numeric table footnote")


@pytest.mark.parametrize("projection_mode", ["empty", "error"])
def test_failed_table_projection_does_not_claim_text(
    monkeypatch: pytest.MonkeyPatch,
    projection_mode: str,
) -> None:
    """验证投影为空或抛错时完整回滚候选，文本行仍可进入正文路径。"""

    line = pdf_extractor._LineItem(
        text="cell",
        bbox=(10.0, 10.0, 30.0, 20.0),
        angle=0,
        source_index=0,
        effective_height=10.0,
    )
    source = pdf_extractor._PageSource(
        page_size=(100.0, 100.0),
        lines=[line],
        chars=[],
        drawing_lines=[],
    )
    candidate = pdf_extractor._TableCandidate(
        bbox=(0.0, 0.0, 50.0, 50.0),
        local_bbox=(0.0, 0.0, 50.0, 50.0),
        angle=0,
        score=1.0,
        core_bbox=(0.0, 0.0, 50.0, 50.0),
        line_indices={0},
    )
    projection = MagicMock(return_value="")
    if projection_mode == "error":
        projection.side_effect = RuntimeError("projection failed")
    monkeypatch.setattr(pdf_extractor, "project_pdf_table_text", projection)

    blocks, claimed = pdf_extractor._materialize_table_blocks(source, [candidate])

    assert blocks == []
    assert claimed == set()


def _text_line(
    text: str,
    bbox: tuple[float, float, float, float],
    source_index: int,
    *,
    visual_row_id: int | None = None,
    run_index: int = 0,
    split_from_row: bool = False,
    effective_height: float | None = None,
    font_signature: tuple[str, int] | None = None,
    font_coverage: float = 0.0,
    dominant_font_weight: float | None = None,
    semantic_type: str | None = None,
) -> pdf_extractor._LineItem:
    """构造栏带、排版恢复与图形标签测试使用的原生文本行。"""

    return pdf_extractor._LineItem(
        text=text,
        bbox=bbox,
        angle=0,
        source_index=source_index,
        visual_row_id=visual_row_id,
        run_index=run_index,
        effective_height=effective_height or (bbox[3] - bbox[1]),
        font_signature=font_signature,
        font_coverage=font_coverage,
        dominant_font_weight=dominant_font_weight,
        split_from_row=split_from_row,
        semantic_type=semantic_type,
    )


def _prepared_text_page(
    *lines: pdf_extractor._LineItem,
    page_size: tuple[float, float] = (100.0, 100.0),
) -> pdf_extractor._PreparedPage:
    """构造跨页边缘类型测试使用的无容器轻量页面。"""

    return pdf_extractor._PreparedPage(
        page_size=page_size,
        remaining_lines=list(lines),
        table_bboxes=[],
        drawing_lines=[],
        fixed_blocks=[],
    )


def test_repeated_marginals_require_cross_page_evidence_and_separate_page_numbers() -> None:
    """验证重复页眉页脚与递增镜像页码被标注，孤立边缘行和正文不变。"""

    margin_font = ("Margin", 0)
    pages = [
        _prepared_text_page(
            _text_line("Quarterly report 2024 - 1", (20.0, 5.0, 80.0, 10.0), 0, font_signature=margin_font, font_coverage=1.0),
            _text_line("Only on first page", (20.0, 11.0, 80.0, 16.0), 1, font_signature=margin_font, font_coverage=1.0),
            _text_line("Repeated body", (10.0, 30.0, 90.0, 40.0), 2, font_signature=margin_font, font_coverage=1.0),
            _text_line("1", (4.0, 48.0, 6.0, 53.0), 3, font_signature=margin_font, font_coverage=1.0),
            _text_line("10", (5.0, 89.0, 10.0, 94.0), 4, font_signature=margin_font, font_coverage=1.0),
            _text_line("Confidential", (30.0, 95.0, 70.0, 99.0), 5, font_signature=margin_font, font_coverage=1.0),
        ),
        _prepared_text_page(
            _text_line("Quarterly report 2024 - 2", (20.0, 5.0, 80.0, 10.0), 0, font_signature=margin_font, font_coverage=1.0),
            _text_line("Repeated body", (10.0, 30.0, 90.0, 40.0), 1, font_signature=margin_font, font_coverage=1.0),
            _text_line("2", (4.0, 48.0, 6.0, 53.0), 2, font_signature=margin_font, font_coverage=1.0),
            _text_line("11", (90.0, 89.0, 95.0, 94.0), 3, font_signature=margin_font, font_coverage=1.0),
            _text_line("Confidential", (30.0, 95.0, 70.0, 99.0), 4, font_signature=margin_font, font_coverage=1.0),
        ),
        _prepared_text_page(
            _text_line("Quarterly report 2024 - 3", (20.0, 5.0, 80.0, 10.0), 0, font_signature=margin_font, font_coverage=1.0),
            _text_line("Repeated body", (10.0, 30.0, 90.0, 40.0), 1, font_signature=margin_font, font_coverage=1.0),
            _text_line("3", (4.0, 48.0, 6.0, 53.0), 2, font_signature=margin_font, font_coverage=1.0),
            _text_line("12", (5.0, 89.0, 10.0, 94.0), 3, font_signature=margin_font, font_coverage=1.0),
            _text_line("Confidential", (30.0, 95.0, 70.0, 99.0), 4, font_signature=margin_font, font_coverage=1.0),
        ),
    ]

    pdf_extractor._classify_repeated_page_marginals(pages)

    assert [[line.semantic_type for line in page.remaining_lines] for page in pages] == [
        ["header", None, None, None, "page_number", "footer"],
        ["header", None, None, "page_number", "footer"],
        ["header", None, None, "page_number", "footer"],
    ]


def test_page_number_sequence_survives_portrait_to_landscape_edge_change() -> None:
    """验证横竖版切换时连续页码可从底边迁移到侧边，且后续侧边序列继续命中。"""

    pages = [
        _prepared_text_page(
            _text_line("2", (45.0, 126.0, 55.0, 131.0), 0),
            page_size=(100.0, 140.0),
        ),
        _prepared_text_page(
            _text_line("3", (126.0, 75.0, 131.0, 80.0), 0),
            page_size=(140.0, 100.0),
        ),
        _prepared_text_page(
            _text_line("4", (126.0, 75.0, 131.0, 80.0), 0),
            page_size=(140.0, 100.0),
        ),
    ]

    pdf_extractor._classify_repeated_page_marginals(pages)

    assert [page.remaining_lines[0].semantic_type for page in pages] == ["page_number"] * 3


def test_single_page_marginal_content_remains_text() -> None:
    """验证单页顶部和底部文字不会仅凭位置被猜成页眉、页脚或页码。"""

    page = _prepared_text_page(
        _text_line("Page 7", (45.0, 3.0, 55.0, 8.0), 0),
        _text_line("Copyright notice", (20.0, 92.0, 80.0, 98.0), 1),
    )

    pdf_extractor._classify_repeated_page_marginals([page])

    assert [line.semantic_type for line in page.remaining_lines] == [None, None]


@pytest.mark.parametrize("heading_text", ["1. INTRODUCTION", "completely unrelated words"])
def test_paragraph_title_classification_is_independent_of_heading_content(heading_text: str) -> None:
    """验证相同版式和字体的不同内容得到完全相同的段落标题类型。"""

    body_font = ("Body", 0)
    heading_font = ("Heading", 1)
    lines = [
        _text_line("body one", (0.0, 10.0, 100.0, 20.0), 0, font_signature=body_font, font_coverage=1.0),
        _text_line("body two", (0.0, 22.0, 100.0, 32.0), 1, font_signature=body_font, font_coverage=1.0),
        _text_line(heading_text, (0.0, 45.0, 40.0, 55.0), 2, font_signature=heading_font, font_coverage=1.0),
        _text_line("body three", (0.0, 68.0, 100.0, 78.0), 3, font_signature=body_font, font_coverage=1.0),
        _text_line("body four", (0.0, 80.0, 100.0, 90.0), 4, font_signature=body_font, font_coverage=1.0),
        _text_line("body five", (0.0, 92.0, 100.0, 102.0), 5, font_signature=body_font, font_coverage=1.0),
    ]

    pdf_extractor._classify_page_titles(lines, (100.0, 150.0), page_index=1, container_bboxes=[])

    assert lines[2].semantic_type == "paragraph_title"


def test_heading_like_content_with_body_layout_remains_text() -> None:
    """验证标题式字符串在正文几何、正文字体和常规行距下仍保持普通文本。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            f"body {index}",
            (0.0, 10.0 + 12.0 * index, 100.0, 20.0 + 12.0 * index),
            index,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index in range(6)
    ]
    lines[2].text = "1. INTRODUCTION"

    pdf_extractor._classify_page_titles(lines, (100.0, 120.0), page_index=1, container_bboxes=[])

    assert lines[2].semantic_type is None


def test_centered_smaller_paragraph_title_uses_layout_contrast_only() -> None:
    """验证同字体的小字号居中标题可由栏宽和上下留白识别。"""

    body_font = ("Body", 0)
    lines = [
        _text_line("body one", (0.0, 10.0, 100.0, 20.0), 0, font_signature=body_font, font_coverage=1.0),
        _text_line("body two", (0.0, 22.0, 100.0, 32.0), 1, font_signature=body_font, font_coverage=1.0),
        _text_line(
            "neutral label",
            (30.0, 45.0, 70.0, 53.0),
            2,
            effective_height=8.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line("body three", (0.0, 66.0, 100.0, 76.0), 3, font_signature=body_font, font_coverage=1.0),
        _text_line("body four", (0.0, 78.0, 100.0, 88.0), 4, font_signature=body_font, font_coverage=1.0),
        _text_line("body five", (0.0, 90.0, 100.0, 100.0), 5, font_signature=body_font, font_coverage=1.0),
    ]

    pdf_extractor._classify_page_titles(lines, (100.0, 150.0), page_index=1, container_bboxes=[])

    assert lines[2].semantic_type == "paragraph_title"


def test_multiline_document_title_does_not_absorb_author_line() -> None:
    """验证首页两行大字号标题合并为文档标题，较小作者行保持普通文本。"""

    body_font = ("Body", 0)
    title_font = ("Title", 0)
    lines = [
        _text_line(
            "title line one",
            (15.0, 20.0, 85.0, 34.4),
            0,
            effective_height=14.4,
            font_signature=title_font,
            font_coverage=1.0,
        ),
        _text_line(
            "title line two",
            (10.0, 33.8, 90.0, 48.2),
            1,
            effective_height=14.4,
            font_signature=title_font,
            font_coverage=1.0,
        ),
        _text_line(
            "author names",
            (20.0, 60.0, 80.0, 69.1),
            2,
            effective_height=9.1,
            font_signature=title_font,
            font_coverage=1.0,
        ),
        _text_line("body one", (0.0, 110.0, 100.0, 120.0), 3, font_signature=body_font, font_coverage=1.0),
        _text_line("body two", (0.0, 122.0, 100.0, 132.0), 4, font_signature=body_font, font_coverage=1.0),
        _text_line("body three", (0.0, 134.0, 100.0, 144.0), 5, font_signature=body_font, font_coverage=1.0),
    ]

    pdf_extractor._classify_page_titles(lines, (100.0, 200.0), page_index=0, container_bboxes=[])

    assert [line.semantic_type for line in lines[:3]] == ["doc_title", "doc_title", None]


def test_cross_column_document_title_uses_thirteen_tenths_body_height_fallback() -> None:
    """验证首页跨栏居中标题达到正文 1.30 倍时可命中，作者行保持正文类型。"""

    body_font = ("Body", 0)
    title_font = ("Title", 0)
    lines = [
        _text_line(
            "wide title",
            (20.0, 20.0, 180.0, 33.0),
            0,
            effective_height=13.0,
            font_signature=title_font,
            font_coverage=1.0,
        ),
        _text_line(
            "author row",
            (60.0, 42.0, 140.0, 52.0),
            1,
            effective_height=10.0,
            font_signature=title_font,
            font_coverage=1.0,
        ),
    ]
    for row_index, top in enumerate((80.0, 92.0, 104.0, 116.0)):
        lines.extend(
            [
                _text_line(
                    f"left body {row_index}",
                    (10.0, top, 90.0, top + 10.0),
                    2 + row_index * 2,
                    font_signature=body_font,
                    font_coverage=1.0,
                ),
                _text_line(
                    f"right body {row_index}",
                    (110.0, top, 190.0, top + 10.0),
                    3 + row_index * 2,
                    font_signature=body_font,
                    font_coverage=1.0,
                ),
            ]
        )

    pdf_extractor._classify_page_titles(
        lines,
        (200.0, 200.0),
        page_index=0,
        container_bboxes=[],
    )

    assert lines[0].semantic_type == "doc_title"
    assert lines[1].semantic_type != "paragraph_title"


def test_complete_visual_row_promotes_number_and_demotes_inline_body() -> None:
    """验证同字体编号整行晋升，标题字体与正文同排时整行降级并重新合并。"""

    body_font = ("Body", 0)
    title_font = ("Title", 0)
    lines = [
        _text_line("body one", (0.0, 5.0, 100.0, 15.0), 0, font_signature=body_font, font_coverage=1.0),
        _text_line("body two", (0.0, 17.0, 100.0, 27.0), 1, font_signature=body_font, font_coverage=1.0),
        _text_line(
            "2",
            (0.0, 42.0, 8.0, 52.0),
            2,
            visual_row_id=20,
            split_from_row=True,
            font_signature=title_font,
            font_coverage=1.0,
            dominant_font_weight=700.0,
        ),
        _text_line(
            "neutral heading",
            (15.0, 42.0, 60.0, 52.0),
            3,
            visual_row_id=20,
            split_from_row=True,
            font_signature=title_font,
            font_coverage=1.0,
            dominant_font_weight=700.0,
        ),
        _text_line("body three", (0.0, 65.0, 100.0, 75.0), 4, font_signature=body_font, font_coverage=1.0),
        _text_line(
            "inline label",
            (0.0, 82.0, 32.0, 92.0),
            5,
            visual_row_id=30,
            split_from_row=True,
            font_signature=title_font,
            font_coverage=1.0,
            dominant_font_weight=700.0,
        ),
        _text_line(
            "inline body",
            (34.0, 82.0, 100.0, 92.0),
            6,
            visual_row_id=30,
            split_from_row=True,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line("body four", (0.0, 94.0, 100.0, 104.0), 7, font_signature=body_font, font_coverage=1.0),
    ]

    pdf_extractor._classify_page_titles(
        lines,
        (100.0, 140.0),
        page_index=1,
        container_bboxes=[],
    )
    merged = pdf_extractor._merge_title_resolved_visual_rows(lines, (100.0, 140.0))

    numbered_title = next(line for line in merged if line.visual_row_id == 20)
    inline_row = next(line for line in merged if line.visual_row_id == 30)
    assert numbered_title.semantic_type == "paragraph_title"
    assert numbered_title.text == "2 neutral heading"
    assert inline_row.semantic_type is None
    assert inline_row.text == "inline label inline body"


def test_low_coverage_mixed_weight_visual_row_demotes_inline_title() -> None:
    """验证低字体覆盖率的同一视觉行仍按字重冲突识别为行内粗体正文。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            "body one",
            (0.0, 0.0, 100.0, 10.0),
            0,
            font_signature=body_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "body two",
            (0.0, 12.0, 100.0, 22.0),
            1,
            font_signature=body_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "body three",
            (0.0, 24.0, 100.0, 34.0),
            2,
            font_signature=body_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "inline label",
            (0.0, 50.0, 35.0, 62.0),
            3,
            visual_row_id=30,
            split_from_row=True,
            font_signature=("Heading", 0),
            font_coverage=0.6,
            dominant_font_weight=700.0,
        ),
        _text_line(
            "inline body",
            (37.0, 50.0, 100.0, 62.0),
            4,
            visual_row_id=30,
            run_index=1,
            split_from_row=True,
            font_signature=body_font,
            font_coverage=0.6,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "body continuation",
            (0.0, 64.0, 100.0, 74.0),
            5,
            font_signature=body_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
    ]

    pdf_extractor._classify_page_titles(
        lines,
        (100.0, 100.0),
        page_index=1,
        container_bboxes=[],
    )
    merged = pdf_extractor._merge_title_resolved_visual_rows(lines, (100.0, 100.0))

    inline_row = next(line for line in merged if line.visual_row_id == 30)
    assert inline_row.semantic_type is None
    assert inline_row.text == "inline label inline body"


def test_full_width_normal_height_inline_heading_merges_with_body_continuation() -> None:
    """验证满栏正常字号粗体行降为正文，并只与其下方常规正文续接。"""

    body_font = ("Body", 0)
    heading_font = ("Heading", 1)
    lines = [
        _text_line("body one", (0.0, 0.0, 100.0, 10.0), 0, font_signature=body_font, font_coverage=1.0),
        _text_line("body two", (0.0, 12.0, 100.0, 22.0), 1, font_signature=body_font, font_coverage=1.0),
        _text_line(
            "inline heading",
            (0.0, 40.0, 100.0, 50.0),
            2,
            font_signature=heading_font,
            font_coverage=1.0,
            dominant_font_weight=700.0,
        ),
        _text_line(
            "continuation one",
            (0.0, 52.0, 100.0, 62.0),
            3,
            effective_height=8.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line("continuation two", (0.0, 64.0, 100.0, 74.0), 4, font_signature=body_font, font_coverage=1.0),
        _text_line("continuation three", (0.0, 76.0, 100.0, 86.0), 5, font_signature=body_font, font_coverage=1.0),
    ]

    pdf_extractor._classify_page_titles(
        lines,
        (100.0, 120.0),
        page_index=1,
        container_bboxes=[],
    )
    blocks = pdf_extractor._build_text_blocks(lines, [], (100.0, 120.0))

    inline_block = next(block for block in blocks if block["content"].startswith("inline heading"))
    assert lines[2].semantic_type is None
    assert inline_block["type"] == "text"
    assert "continuation one" in inline_block["content"]
    assert "body two" not in inline_block["content"]


def test_dense_same_font_two_run_row_requires_complete_high_occupancy_geometry() -> None:
    """验证双 run 正文仅在同字体、同基线、连续编号且占用充分时恢复。"""

    body_font = ("Body", 0)
    dense_members = [
        _text_line(
            "left",
            (0.0, 0.0, 48.0, 10.0),
            0,
            visual_row_id=10,
            run_index=0,
            split_from_row=True,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "right",
            (52.0, 0.0, 100.0, 10.0),
            1,
            visual_row_id=10,
            run_index=1,
            split_from_row=True,
            font_signature=body_font,
            font_coverage=1.0,
        ),
    ]
    sparse_members = [
        _text_line(
            "sparse left",
            (0.0, 20.0, 30.0, 30.0),
            2,
            visual_row_id=20,
            run_index=0,
            split_from_row=True,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "sparse right",
            (70.0, 20.0, 100.0, 30.0),
            3,
            visual_row_id=20,
            run_index=1,
            split_from_row=True,
            font_signature=body_font,
            font_coverage=1.0,
        ),
    ]
    partial_formula_row = [
        _text_line(
            "formula body",
            (0.0, 40.0, 48.0, 50.0),
            4,
            visual_row_id=30,
            run_index=0,
            split_from_row=True,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "sidecar",
            (52.0, 40.0, 100.0, 50.0),
            5,
            visual_row_id=30,
            run_index=2,
            split_from_row=True,
            font_signature=body_font,
            font_coverage=1.0,
        ),
    ]
    different_font_members = [
        dense_members[0],
        _text_line(
            "different",
            (52.0, 0.0, 100.0, 10.0),
            6,
            visual_row_id=10,
            run_index=1,
            split_from_row=True,
            font_signature=("Math", 0),
            font_coverage=1.0,
        ),
    ]

    assert pdf_extractor._is_dense_same_font_two_run_row(dense_members, (100.0, 100.0))
    assert not pdf_extractor._is_dense_same_font_two_run_row(sparse_members, (100.0, 100.0))
    assert not pdf_extractor._is_dense_same_font_two_run_row(partial_formula_row, (100.0, 100.0))
    assert not pdf_extractor._is_dense_same_font_two_run_row(different_font_members, (100.0, 100.0))

    merged = pdf_extractor._merge_title_resolved_visual_rows(
        dense_members + sparse_members,
        (100.0, 100.0),
    )
    assert [line.text for line in merged if line.visual_row_id == 10] == ["left right"]
    assert len([line for line in merged if line.visual_row_id == 20]) == 2


def test_paragraph_title_detector_does_not_read_line_text() -> None:
    """守卫段落标题候选、打分和邻行扩展不读取文本内容。"""

    source = "\n".join(
        inspect.getsource(function)
        for function in (
            pdf_extractor._classify_page_titles,
            pdf_extractor._infer_lane_body_profile,
            pdf_extractor._classify_document_title,
            pdf_extractor._document_title_uses_page_fallback,
            pdf_extractor._classify_paragraph_titles_in_lane,
            pdf_extractor._visual_row_has_body_style_sibling,
            pdf_extractor._is_near_full_mixed_inline_row,
            pdf_extractor._is_full_width_inline_heading,
            pdf_extractor._has_following_body_row,
            pdf_extractor._unify_visual_row_title_types,
            pdf_extractor._protect_front_matter_title_types,
            pdf_extractor._infer_front_matter_boundary,
            pdf_extractor._normalized_title_gap,
            pdf_extractor._line_near_visual_container,
            pdf_extractor._title_fonts_compatible,
            pdf_extractor._expand_paragraph_title_neighbors,
        )
    )

    assert ".text" not in source


@pytest.mark.parametrize("block_type", ["doc_title", "paragraph_title", "header", "footer", "page_number", "equation"])
def test_output_normalization_preserves_new_flash_types(block_type: str) -> None:
    """验证 Flash 新增文本类型和公式类型不会在归一化阶段退回 text。"""

    block = pdf_extractor._normalize_output_block(
        {"type": block_type, "bbox": (10.0, 20.0, 40.0, 50.0), "angle": 0, "content": "value"},
        (100.0, 100.0),
    )

    assert block is not None
    assert block["type"] == block_type


def test_hanging_indent_groups_neutral_entries_and_ignores_centered_heading() -> None:
    """验证不含序号的重复悬挂缩进逐条分组，且居中标题不参与条目。"""

    body_font = ("Body", 0)
    italic_font = ("BodyItalic", 1)
    lines = [
        _text_line(
            "Centered heading",
            (35.0, 0.0, 85.0, 10.0),
            0,
            font_signature=("Heading", 0),
            font_coverage=1.0,
        ),
        _text_line("Alpha begins", (0.0, 20.0, 100.0, 30.0), 1, font_signature=body_font, font_coverage=1.0),
        _text_line(
            "alpha italic continuation",
            (15.0, 30.0, 100.0, 40.0),
            2,
            font_signature=italic_font,
            font_coverage=1.0,
        ),
        _text_line("alpha closes", (15.0, 40.0, 70.0, 50.0), 3, font_signature=body_font, font_coverage=1.0),
        _text_line("Beta begins", (0.0, 50.0, 100.0, 60.0), 4, font_signature=body_font, font_coverage=1.0),
        _text_line("beta continues", (15.0, 60.0, 100.0, 70.0), 5, font_signature=body_font, font_coverage=1.0),
        _text_line("Gamma begins", (0.0, 70.0, 100.0, 80.0), 6, font_signature=body_font, font_coverage=1.0),
        _text_line("gamma continues", (15.0, 80.0, 100.0, 90.0), 7, font_signature=body_font, font_coverage=1.0),
    ]
    lane = pdf_extractor._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lines],
    )

    group_map = pdf_extractor._build_hanging_indent_group_map(lane, [], [])
    blocks = pdf_extractor._build_text_blocks(lines, [], (120.0, 120.0))

    assert 0 not in group_map
    assert [group_map[index] for index in range(1, 8)] == [0, 0, 0, 1, 1, 2, 2]
    assert [block["content"] for block in blocks] == [
        "Centered heading",
        "Alpha begins alpha italic continuation alpha closes",
        "Beta begins beta continues",
        "Gamma begins gamma continues",
    ]


def test_hanging_indent_accepts_reference_spacing_and_italic_tail() -> None:
    """验证 1.25 倍行高的参考文献间距不会拆掉前一条斜体尾行。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            "Alpha begins",
            (0.0, 0.0, 100.0, 10.0),
            0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "alpha continues",
            (15.0, 10.0, 100.0, 20.0),
            1,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "alpha italic tail",
            (15.0, 20.0, 75.0, 30.0),
            2,
            font_signature=("BodyItalic", 1),
            font_coverage=1.0,
        ),
        _text_line(
            "Beta begins",
            (0.0, 42.5, 100.0, 52.5),
            3,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "beta continues",
            (15.0, 52.5, 100.0, 62.5),
            4,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "Gamma begins",
            (0.0, 75.0, 100.0, 85.0),
            5,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "gamma continues",
            (15.0, 85.0, 100.0, 95.0),
            6,
            font_signature=body_font,
            font_coverage=1.0,
        ),
    ]
    lane = pdf_extractor._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lines],
    )

    group_map = pdf_extractor._build_hanging_indent_group_map(lane, [], [])
    blocks = pdf_extractor._build_text_blocks(lines, [], (120.0, 120.0))

    assert [group_map[index] for index in range(7)] == [0, 0, 0, 1, 1, 2, 2]
    assert [block["content"] for block in blocks] == [
        "Alpha begins alpha continues alpha italic tail",
        "Beta begins beta continues",
        "Gamma begins gamma continues",
    ]


def test_first_line_indent_and_large_gap_do_not_form_hanging_indent_groups() -> None:
    """验证普通首行缩进及跨越大间距的行不会误触发悬挂缩进模式。"""

    lines = [
        _text_line("First paragraph", (15.0, 0.0, 100.0, 10.0), 0),
        _text_line("first continuation", (0.0, 10.0, 100.0, 20.0), 1),
        _text_line("Second paragraph", (15.0, 20.0, 100.0, 30.0), 2),
        _text_line("second continuation", (0.0, 30.0, 100.0, 40.0), 3),
        _text_line("Detached start", (0.0, 70.0, 100.0, 80.0), 4),
        _text_line("detached continuation", (15.0, 80.0, 100.0, 90.0), 5),
    ]
    lane = pdf_extractor._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lines],
    )

    assert pdf_extractor._build_hanging_indent_group_map(lane, [], []) == {}


def test_twelve_point_gutter_keeps_two_text_lanes_and_paragraphs_separate() -> None:
    """验证约 12pt 行高和栏沟仍识别为双栏，左右正文不会交叉拼接。"""

    lines: list[pdf_extractor._LineItem] = []
    for row_index, top in enumerate((100.0, 112.0, 124.0)):
        lines.extend(
            [
                _text_line(f"left-{row_index}", (49.0, top, 300.0, top + 12.0), row_index * 2),
                _text_line(f"right-{row_index}", (312.0, top, 563.0, top + 12.0), row_index * 2 + 1),
            ]
        )

    lanes = pdf_extractor._infer_text_lanes(
        [(line, line.bbox) for line in lines],
        612.0,
        12.0,
    )
    blocks = pdf_extractor._build_text_blocks(lines, [], (612.0, 792.0))

    regular_lanes = [lane for lane in lanes if not lane.is_span]
    assert [(lane.left, lane.right) for lane in regular_lanes] == [
        (49.0, 300.0),
        (312.0, 563.0),
    ]
    assert len(blocks) == 2
    assert all("right-" not in block["content"] for block in blocks if "left-" in block["content"])
    assert all("left-" not in block["content"] for block in blocks if "right-" in block["content"])


def test_cross_column_caption_tail_stays_in_span_lane_and_one_text_block() -> None:
    """验证仅占单栏宽的短 caption 尾行仍回收到连续跨栏 caption。"""

    lines: list[pdf_extractor._LineItem] = [
        _text_line("caption line one", (0.0, 10.0, 200.0, 20.0), 0),
        _text_line("caption line two", (0.0, 22.0, 200.0, 32.0), 1),
        _text_line("caption tail", (0.0, 34.0, 70.0, 44.0), 2),
        _text_line("ordinary left short line", (0.0, 70.0, 70.0, 80.0), 3),
    ]
    for row_index, top in enumerate((100.0, 112.0, 124.0, 136.0, 148.0)):
        lines.extend(
            [
                _text_line(
                    f"left body {row_index}",
                    (0.0, top, 90.0, top + 10.0),
                    4 + 2 * row_index,
                ),
                _text_line(
                    f"right body {row_index}",
                    (110.0, top, 200.0, top + 10.0),
                    5 + 2 * row_index,
                ),
            ]
        )

    lanes = pdf_extractor._infer_text_lanes(
        [(line, line.bbox) for line in lines],
        200.0,
        10.0,
    )
    blocks = pdf_extractor._build_text_blocks(lines, [], (200.0, 180.0))

    span_lane = next(lane for lane in lanes if lane.is_span)
    assert {line.source_index for line, _bbox in span_lane.lines} == {0, 1, 2}
    assert all(line.source_index != 3 for line, _bbox in span_lane.lines)
    caption_blocks = [block for block in blocks if "caption line one" in block["content"]]
    assert len(caption_blocks) == 1
    assert "caption tail" in caption_blocks[0]["content"]


def test_slight_bbox_overlap_contributes_to_gap_estimate_and_separates_caption() -> None:
    """验证轻微纵向重叠按零净空统计，短图例不会与后续长 caption 合并。"""

    body_lines = [
        _text_line("body-0", (312.0, 0.0, 563.0, 12.0), 0),
        _text_line("body-1", (312.0, 11.95, 563.0, 23.95), 1),
        _text_line("body-2", (312.0, 23.9, 563.0, 35.9), 2),
    ]
    legend = _text_line("Right camera", (458.0, 60.0, 509.0, 72.0), 3)
    caption = _text_line(
        "Figure 1: a long caption spanning the full column",
        (312.0, 79.0, 563.0, 91.0),
        4,
    )
    lane = pdf_extractor._TextLane(
        left=312.0,
        right=563.0,
        lines=[*((line, line.bbox) for line in body_lines), (legend, legend.bbox), (caption, caption.bbox)],
    )

    regular_gap, gap_mad = pdf_extractor._estimate_lane_gap(lane)

    assert (regular_gap, gap_mad) == (0.0, 0.0)
    assert not pdf_extractor._should_connect_text_rows(
        (legend, legend.bbox),
        (caption, caption.bbox),
        lane,
        regular_gap,
        gap_mad,
        [],
        [],
    )


def test_local_previous_left_edge_exposes_first_line_indent() -> None:
    """验证局部版心左移后仍能识别下一行相对前一物理行的首行缩进。"""

    previous = _text_line(
        "previous paragraph without punctuation",
        (0.0, 0.0, 80.0, 10.0),
        0,
    )
    current = _text_line(
        "Indented new paragraph",
        (10.0, 12.0, 100.0, 22.0),
        1,
    )
    lane = pdf_extractor._TextLane(
        left=20.0,
        right=100.0,
        lines=[(previous, previous.bbox), (current, current.bbox)],
    )

    assert not pdf_extractor._should_connect_text_rows(
        (previous, previous.bbox),
        (current, current.bbox),
        lane,
        2.0,
        0.0,
        [],
        [],
    )


def test_terminal_full_lane_row_breaks_after_abnormal_clearance() -> None:
    """验证满栏句末行后的净空超过常规间距半行高时强制另起段落。"""

    previous = _text_line("Figure caption ends.", (0.0, 0.0, 100.0, 10.0), 0)
    current = _text_line("new paragraph fills the lane", (0.0, 17.0, 100.0, 27.0), 1)
    lane = pdf_extractor._TextLane(
        left=0.0,
        right=100.0,
        lines=[(previous, previous.bbox), (current, current.bbox)],
    )

    assert not pdf_extractor._should_connect_text_rows(
        (previous, previous.bbox),
        (current, current.bbox),
        lane,
        1.0,
        0.0,
        [],
        [],
    )


def test_effective_height_connects_body_line_after_tall_math_glyph() -> None:
    """验证高数学字形拉长原始 bbox 时仍按有效行高连接下一正文行。"""

    previous = _text_line(
        "support window Ωp centered at the pixel",
        (312.0, 100.0, 563.0, 118.82),
        0,
        effective_height=12.0,
    )
    current = _text_line(
        "by",
        (312.0, 112.0, 325.0, 124.0),
        1,
        effective_height=12.0,
    )
    lane = pdf_extractor._TextLane(
        left=312.0,
        right=563.0,
        lines=[(previous, previous.bbox), (current, current.bbox)],
    )

    assert current.bbox[1] - previous.bbox[3] == pytest.approx(-6.82)
    assert pdf_extractor._effective_text_row_gap(
        (previous, previous.bbox),
        (current, current.bbox),
    ) == pytest.approx(0.0)
    assert pdf_extractor._should_connect_text_rows(
        (previous, previous.bbox),
        (current, current.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )


def test_inline_scripts_and_touching_low_coverage_runs_are_recovered() -> None:
    """验证紧贴上下标与低覆盖率同行后缀恢复，同时保留外置公式编号。"""

    script_lines = [
        _text_line("O(ω", (0.0, 0.0, 100.0, 18.8), 0, visual_row_id=0, effective_height=12.0),
        _text_line("2", (100.3, 0.8, 104.0, 7.0), 1, visual_row_id=1, effective_height=6.0),
        _text_line("Di", (0.0, 40.0, 100.0, 52.0), 2, visual_row_id=2, effective_height=12.0),
        _text_line("p", (96.9, 46.0, 101.0, 53.0), 3, visual_row_id=3, effective_height=7.0),
    ]

    merged_scripts = pdf_extractor._merge_native_inline_scripts(script_lines, (200.0, 100.0))

    assert [line.text for line in merged_scripts] == ["O(ω2", "Dip"]

    caption_prefix = _text_line(
        "(4",
        (0.0, 70.0, 12.0, 82.0),
        4,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )
    caption_suffix = _text_line(
        "th row).",
        (12.1, 70.0, 50.0, 82.0),
        5,
        font_signature=("Body", 0),
        font_coverage=0.7,
    )
    formula_body = _text_line(
        "formula",
        (0.0, 90.0, 50.0, 102.0),
        6,
        font_signature=("Math", 0),
        font_coverage=1.0,
    )
    formula_number = _text_line(
        "(4)",
        (51.2, 90.0, 60.0, 102.0),
        7,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )

    assert pdf_extractor._can_merge_same_baseline_pair(
        caption_prefix,
        caption_prefix.bbox,
        caption_suffix,
        caption_suffix.bbox,
        [],
    )
    assert not pdf_extractor._can_merge_same_baseline_pair(
        formula_body,
        formula_body.bbox,
        formula_number,
        formula_number.bbox,
        [],
    )


def test_full_lane_large_height_mismatch_only_recovers_aligned_continuation() -> None:
    """验证满栏混合字体 URL 可续接，而短公式与显式字体样式边界仍分离。"""

    previous = _text_line(
        "video sequences have been made avail-",
        (0.0, 0.0, 100.0, 12.0),
        0,
        effective_height=12.0,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )
    full_width_url = _text_line(
        "able at http://example.test/data",
        (0.0, 12.0, 100.0, 24.0),
        1,
        effective_height=7.69,
        font_signature=("Mono", 0),
        font_coverage=1.0,
    )
    short_formula = _text_line(
        "x = 1",
        (0.0, 12.0, 60.0, 24.0),
        2,
        effective_height=7.69,
        font_signature=("Math", 0),
        font_coverage=1.0,
    )
    styled_reference = _text_line(
        "italic bibliography continuation",
        (0.0, 12.0, 100.0, 24.0),
        3,
        effective_height=7.69,
        font_signature=("BodyItalic", 1),
        font_coverage=1.0,
    )
    lane = pdf_extractor._TextLane(left=0.0, right=100.0)

    assert pdf_extractor._should_connect_text_rows(
        (previous, previous.bbox),
        (full_width_url, full_width_url.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )
    assert not pdf_extractor._should_connect_text_rows(
        (previous, previous.bbox),
        (short_formula, short_formula.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )
    assert not pdf_extractor._should_connect_text_rows(
        (previous, previous.bbox),
        (styled_reference, styled_reference.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )


def test_smaller_footnote_after_abnormal_gap_forces_text_block_break() -> None:
    """验证字号不足前行 88% 且净空偏大时，正文与脚注强制分块。"""

    body = _text_line(
        "body continuation",
        (0.0, 0.0, 100.0, 10.0),
        0,
        effective_height=10.0,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )
    footnote = _text_line(
        "small footnote",
        (0.0, 16.0, 100.0, 24.7),
        1,
        effective_height=8.7,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )
    lane = pdf_extractor._TextLane(
        left=0.0,
        right=100.0,
        lines=[(body, body.bbox), (footnote, footnote.bbox)],
    )

    assert not pdf_extractor._should_connect_text_rows(
        (body, body.bbox),
        (footnote, footnote.bbox),
        lane,
        3.0,
        0.0,
        [],
        [],
    )


def test_detached_formula_anchor_collects_multiline_formula_but_not_body_prefix() -> None:
    """验证低位右缘锚点上溯多行公式，并排除左对齐正文与靠右句点。"""

    body_font = ("Body", 0)
    body_lines = [
        _text_line(
            f"body-{index}",
            (0.0, top, 100.0, top + 10.0),
            index,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index, top in enumerate((0.0, 12.0, 24.0))
    ]
    body_prefix = _text_line(
        "regular prose before formula",
        (0.0, 60.0, 60.0, 70.0),
        3,
        effective_height=10.0,
        font_signature=body_font,
        font_coverage=1.0,
    )
    formula_lines = [
        _text_line("numerator", (20.0, 42.0, 50.0, 52.0), 4, effective_height=10.0),
        _text_line("Fp =", (15.0, 52.0, 45.0, 62.0), 5, effective_height=10.0),
        _text_line("otherwise", (20.0, 65.0, 65.0, 75.0), 6, effective_height=10.0),
        _text_line("0,", (68.0, 72.0, 78.0, 82.0), 7, effective_height=10.0),
    ]
    punctuation = _text_line(".", (93.0, 45.0, 96.0, 55.0), 8, effective_height=10.0)
    number = _text_line("(7)", (91.0, 75.0, 100.0, 85.0), 9, effective_height=10.0)
    lane_lines = [*body_lines, body_prefix, *formula_lines, punctuation, number]
    lane = pdf_extractor._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lane_lines],
    )

    anchors = pdf_extractor._find_formula_spatial_anchors(lane, 10.0)

    assert len(anchors) == 1
    assert anchors[0].line is number
    assert anchors[0].detached_below_body

    anchor_center = pdf_extractor._bbox_center_y(anchors[0].bbox)
    dominant_font = pdf_extractor._infer_formula_body_font(lane, 10.0)
    members = pdf_extractor._grow_formula_spatial_component(
        lane,
        anchors[0],
        anchor_center - 4.75 * 10.0,
        anchor_center + 2.25 * 10.0,
        set(),
        [],
        dominant_font,
        10.0,
    )
    member_texts = {line.text for line, _bbox in members}

    assert member_texts == {"numerator", "Fp =", "otherwise", "0,", "(7)"}
    assert body_prefix.text not in member_texts
    assert punctuation.text not in member_texts


def _drawing_axis_line(
    orientation: str,
    bbox: tuple[float, float, float, float],
) -> pdf_extractor._AxisLine:
    """构造图形容器测试使用的 PDF 绘图线。"""

    return pdf_extractor._AxisLine(
        bbox=bbox,
        width=0.5,
        orientation=orientation,  # type: ignore[arg-type]
    )


def _graphic_source_fixture() -> pdf_extractor._PageSource:
    """构造双框图形、六个标签、拆分 caption 与邻近正文。"""

    lines = [
        _text_line("p = (x, y)", (378.0, 302.0, 421.0, 312.0), 0, visual_row_id=10),
        _text_line("pbar = (xbar, y)", (445.0, 302.0, 488.0, 312.0), 1, visual_row_id=10),
        _text_line("dp", (382.0, 282.0, 403.0, 292.0), 2, visual_row_id=11),
        _text_line("x", (468.0, 282.0, 475.0, 292.0), 3, visual_row_id=11),
        _text_line("Left camera", (360.0, 339.0, 411.0, 350.0), 4, visual_row_id=12),
        _text_line("Right camera", (458.0, 339.0, 509.0, 350.0), 5, visual_row_id=12),
        _text_line(
            "Figure 1: a deliberately long caption that must remain",
            (312.0, 359.0, 500.0, 371.0),
            6,
            visual_row_id=20,
            split_from_row=True,
        ),
        _text_line(
            "independent.",
            (510.0, 359.0, 563.0, 371.0),
            7,
            visual_row_id=20,
            split_from_row=True,
        ),
        _text_line("Nearby prose", (370.0, 225.0, 430.0, 237.0), 8, visual_row_id=9),
    ]
    drawing_lines = [
        _drawing_axis_line("horizontal", (348.0, 272.0, 430.0, 272.5)),
        _drawing_axis_line("horizontal", (348.0, 331.5, 430.0, 332.0)),
        _drawing_axis_line("vertical", (348.0, 272.0, 348.5, 332.0)),
        _drawing_axis_line("vertical", (429.5, 272.0, 430.0, 332.0)),
        _drawing_axis_line("horizontal", (436.0, 272.0, 526.0, 272.5)),
        _drawing_axis_line("horizontal", (436.0, 331.5, 526.0, 332.0)),
        _drawing_axis_line("vertical", (436.0, 272.0, 436.5, 332.0)),
        _drawing_axis_line("vertical", (525.5, 272.0, 526.0, 332.0)),
    ]
    return pdf_extractor._PageSource(
        page_size=(612.0, 792.0),
        lines=lines,
        chars=[],
        drawing_lines=drawing_lines,
    )


def test_double_box_graphic_claims_six_labels_but_not_caption_or_body() -> None:
    """验证双框图形整行聚合六个标签，拆分 caption 和邻近正文均不被部分认领。"""

    blocks, claimed = pdf_extractor._build_graphic_like_blocks(
        _graphic_source_fixture(),
        [],
        set(),
    )

    assert len(blocks) == 1
    assert claimed == set(range(6))
    assert blocks[0]["type"] == "image"
    for expected_text in ("dp", "x", "p = (x, y)", "pbar = (xbar, y)", "Left camera", "Right camera"):
        assert expected_text in blocks[0]["content"]
    assert blocks[0]["content"].count("Left camera") == 1
    assert blocks[0]["content"].count("Right camera") == 1
    assert "Figure 1" not in blocks[0]["content"]
    assert "Nearby prose" not in blocks[0]["content"]


def test_materialized_table_bbox_has_priority_over_graphic_candidate() -> None:
    """验证绘图组件与成功表格框重叠时跳过图形容器并保留全部文本身份。"""

    blocks, claimed = pdf_extractor._build_graphic_like_blocks(
        _graphic_source_fixture(),
        [(340.0, 265.0, 535.0, 355.0)],
        set(),
    )

    assert blocks == []
    assert claimed == set()


def test_form_image_claims_internal_text_and_small_table_but_not_caption() -> None:
    """验证有效 Form 吞并图内文字与小表候选，外部 caption 仍保留。"""

    form_bbox = (10.0, 10.0, 90.0, 65.0)
    source = pdf_extractor._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line("inside row one", (15.0, 20.0, 70.0, 30.0), 0, visual_row_id=1),
            _text_line("inside row two", (15.0, 35.0, 80.0, 45.0), 1, visual_row_id=2),
            _text_line("Figure 1: outside", (10.0, 70.0, 80.0, 80.0), 2, visual_row_id=3),
        ],
        chars=[],
        drawing_lines=[
            _drawing_axis_line("horizontal", (20.0, 15.0, 80.0, 15.5)),
            _drawing_axis_line("horizontal", (20.0, 55.0, 80.0, 55.5)),
            _drawing_axis_line("vertical", (20.0, 15.0, 20.5, 55.0)),
            _drawing_axis_line("vertical", (79.5, 15.0, 80.0, 55.0)),
        ],
        form_bboxes=[form_bbox],
    )

    selected = pdf_extractor._select_form_image_bboxes(source)
    blocks, claimed = pdf_extractor._build_form_image_blocks(source, selected, set())

    assert selected == [form_bbox]
    assert pdf_extractor._form_supersedes_nested_bbox(
        form_bbox,
        (20.0, 20.0, 40.0, 35.0),
    )
    assert not pdf_extractor._form_supersedes_nested_bbox(
        form_bbox,
        (15.0, 15.0, 85.0, 60.0),
    )
    assert claimed == {0, 1}
    assert blocks == [
        {
            "type": "image",
            "bbox": form_bbox,
            "angle": 0,
            "content": "inside row one\ninside row two",
        }
    ]


def test_raster_images_filter_small_objects_avoid_containers_and_claim_text_once() -> None:
    """验证点阵图过滤、容器优先、空 content 和重叠对象的唯一文本归属。"""

    source = pdf_extractor._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line("inside", (25.0, 25.0, 35.0, 35.0), 0, visual_row_id=1),
            _text_line("outside caption", (10.0, 52.0, 50.0, 60.0), 1, visual_row_id=2),
            _text_line("covered container text", (20.0, 70.0, 40.0, 80.0), 2, visual_row_id=3),
        ],
        chars=[],
        drawing_lines=[],
        image_bboxes=[
            (10.0, 10.0, 50.0, 50.0),
            (20.0, 20.0, 40.0, 40.0),
            (60.0, 10.0, 90.0, 40.0),
            (0.0, 90.0, 10.0, 94.0),
            (10.0, 60.0, 50.0, 90.0),
        ],
    )

    blocks, claimed = pdf_extractor._build_raster_image_blocks(
        source,
        [{"type": "table", "bbox": (10.0, 60.0, 50.0, 90.0), "angle": 0, "content": "table"}],
        set(),
    )

    assert len(blocks) == 3
    assert all(block["type"] == "image" and block["angle"] == 0 for block in blocks)
    assert [block["bbox"] for block in blocks] == [
        (10.0, 10.0, 50.0, 50.0),
        (60.0, 10.0, 90.0, 40.0),
        (20.0, 20.0, 40.0, 40.0),
    ]
    assert [block["content"] for block in blocks] == ["", "", "inside"]
    assert claimed == {0}


def test_raster_image_content_is_removed_from_text_and_empty_image_page_is_kept() -> None:
    """验证图内文本只进入 image，caption 保持 text，纯图片页仍输出空 content。"""

    source = pdf_extractor._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line("inside row one", (20.0, 20.0, 50.0, 30.0), 0, visual_row_id=1),
            _text_line("inside row two", (20.0, 35.0, 50.0, 45.0), 1, visual_row_id=2),
            _text_line("Figure 1: outside caption", (10.0, 70.0, 80.0, 80.0), 2, visual_row_id=3),
        ],
        chars=[],
        drawing_lines=[],
        image_bboxes=[(10.0, 10.0, 60.0, 60.0)],
    )

    blocks = pdf_extractor._analyze_page_source(source)
    image_block = next(block for block in blocks if block["type"] == "image")
    text_block = next(block for block in blocks if block["type"] == "text")

    assert image_block["content"] == "inside row one\ninside row two"
    assert text_block["content"] == "Figure 1: outside caption"
    assert sum("inside row" in block["content"] for block in blocks) == 1

    empty_page_blocks = pdf_extractor._analyze_page_source(
        pdf_extractor._PageSource(
            page_size=(100.0, 100.0),
            lines=[],
            chars=[],
            drawing_lines=[],
            image_bboxes=[(10.0, 10.0, 60.0, 60.0)],
        )
    )
    assert empty_page_blocks == [
        {
            "type": "image",
            "bbox": [0.1, 0.1, 0.6, 0.6],
            "angle": 0,
            "content": "",
        }
    ]


def _native_model_list(pdf_name: str) -> list[list[dict[str, Any]]]:
    """运行仓库内数字 PDF 样例并返回 Flash 原生模型输出。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / pdf_name
    with pdf_extractor.PDFDocument(str(pdf_path)) as pdf_doc:
        return pdf_extractor._analyze_native_document(pdf_doc)


def _native_table_counts(pdf_name: str) -> list[int]:
    """返回仓库内数字 PDF 样例的逐页表格块数量。"""

    return [sum(block["type"] == "table" for block in page) for page in _native_model_list(pdf_name)]


def _native_page_source(pdf_name: str, page_idx: int) -> pdf_extractor._PageSource:
    """读取指定样例页并构造候选检测与认领测试使用的页面源。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / pdf_name
    with pdf_extractor.PDFDocument(str(pdf_path)) as pdf_doc:
        page_size = pdf_doc.page_size(page_idx)
        chars = pdf_doc.get_page_chars(page_idx)
        lines = pdf_extractor._build_native_line_items(
            pdf_extractor.get_lines_from_chars(chars),
            page_size,
            page_rotation=pdf_doc.page_rotation(page_idx),
        )
        return pdf_extractor._PageSource(
            page_size=page_size,
            lines=lines,
            chars=chars,
            drawing_lines=pdf_extractor._get_pdf_drawing_lines(pdf_doc, page_idx),
            image_bboxes=pdf_doc.get_page_image_bboxes(page_idx),
            form_bboxes=pdf_doc.get_page_form_bboxes(page_idx),
        )


def _normalized_content_probe(text: str) -> str:
    """去除空白、标点与 dash 差异，生成原生行内容覆盖检查使用的探针。"""

    return "".join(char.casefold() for char in text if char.isalnum())


def test_demo1_keeps_five_real_tables_without_formula_false_positive() -> None:
    """验证 demo1 参考文献逐条分块，第四页公式不误报且五个真实表格保留。"""

    model_list = _native_model_list("demo1.pdf")

    assert [len(page) for page in model_list] == [17, 9, 12, 18, 10, 9, 11, 8, 10, 7, 10, 26, 9]
    assert [sum(block["type"] == "table" for block in page) for page in model_list] == [
        0,
        0,
        0,
        0,
        1,
        2,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
    ]
    assert sum(block["type"] == "doc_title" for page in model_list for block in page) == 1
    assert sum(block["type"] == "header" for page in model_list for block in page) == 12
    assert sum(block["type"] == "page_number" for page in model_list for block in page) == 12
    assert sum(block["type"] == "equation" for page in model_list for block in page) == 7
    assert next(block for block in model_list[0] if block["content"] == "Abstract")["type"] == "paragraph_title"
    assert next(block for block in model_list[6] if block["content"].startswith("4.2."))["type"] == "paragraph_title"


def test_demo1_rotated_table_claims_all_206_lines_without_residual_text() -> None:
    """验证 demo1 第五页旋转表完整认领 206 行且表框内没有残留文本。"""

    source = _native_page_source("demo1.pdf", 4)
    candidates = pdf_extractor._detect_table_candidates(source)
    blocks, claimed = pdf_extractor._materialize_table_blocks(source, candidates)
    rotated_indices = {line.source_index for line in source.lines if line.angle == 270}

    assert len(blocks) == 1
    assert len(rotated_indices) == 206
    assert claimed == rotated_indices
    assert not [
        line
        for line in source.lines
        if line.angle == 270
        and line.source_index not in claimed
        and any(
            candidate.angle == 270
            and pdf_extractor._point_in_bbox(
                (pdf_extractor._bbox_center_x(line.bbox), pdf_extractor._bbox_center_y(line.bbox)),
                candidate.bbox,
            )
            for candidate in candidates
        )
    ]


def test_demo2_rejects_figure_grid_and_keeps_two_real_tables() -> None:
    """验证 demo2 曲线图被拒绝且第四、五页真实表格保留。"""

    assert _native_table_counts("demo2.pdf") == [0, 0, 0, 1, 1, 0]


def test_demo2_page1_forms_sixteen_blocks_and_keeps_figure_caption_separate() -> None:
    """验证 demo2 首页正文自然聚合，六个 Figure 1 标签单块且 caption 独立。"""

    page = _native_model_list("demo2.pdf")[0]
    graphic_block = next(block for block in page if "Left camera" in block["content"])
    caption_block = next(block for block in page if "Figure 1:" in block["content"])

    assert len(page) == 16
    assert not [block for block in page if block["type"] == "table"]
    assert next(block for block in page if block["content"].startswith("Real-time Temporal"))["type"] == "doc_title"
    assert next(block for block in page if block["content"] == "I. INTRODUCTION")["type"] == "paragraph_title"
    for expected_text in ("dp", "¯x", "p =", "¯p =", "Left camera", "Right camera"):
        assert expected_text in graphic_block["content"]
    assert graphic_block is not caption_block
    assert "Figure 1:" not in graphic_block["content"]
    assert "Left camera" not in caption_block["content"]
    assert sum(block["content"].count("Left camera") for block in page) == 1


def test_demo2_pages2_to6_restore_paragraphs_formulas_and_reading_order() -> None:
    """验证 demo2 后续页达到目标块数，正文、公式、caption 与双栏顺序均稳定。"""

    model_list = _native_model_list("demo2.pdf")

    assert [len(page) for page in model_list] == [16, 16, 21, 13, 16, 16]
    assert [sum(block["type"] == "image" for block in page) for page in model_list] == [1, 0, 0, 5, 2, 0]
    assert [sum(block["type"] == "table" for block in page) for page in model_list] == [0, 0, 0, 1, 1, 0]
    assert sum(block["type"] == "equation" for page in model_list for block in page) == 9

    page2 = model_list[1]
    page2_contents = [block["content"] for block in page2]
    humans = next(content for content in page2_contents if content.startswith("Humans group shapes"))
    matching = next(content for content in page2_contents if content.startswith("To identify a match"))
    dissimilarity = next(content for content in page2_contents if content.startswith("where the pixel dissimilarity"))
    formula3 = next(content for content in page2_contents if content.endswith("(3)"))
    assert not [content for content in page2_contents if content.strip() == "by"]
    assert humans.endswith("is given by")
    assert "Sp denotes a set of matching candidates" in matching
    assert "green, and blue components given by" in dissimilarity
    assert "green, and blue" not in formula3

    page3 = model_list[2]
    page3_contents = [block["content"] for block in page3]
    assert page3_contents[12] == "D. Iterative Disparity Refinement"
    assert all(block["bbox"][2] <= 0.5 for block in page3[:12])
    assert "O(ω2) to O(ω)" in page3_contents[0]
    assert "disparity estimates Dip" in page3_contents[13]
    for formula_number in range(4, 10):
        marker = f"({formula_number})"
        assert sum(marker in content for content in page3_contents) == 1
    formula7 = next(content for content in page3_contents if "(7)" in content)
    formula4 = next(content for content in page3_contents if "(4)" in content)
    formula8 = next(content for content in page3_contents if "(8)" in content)
    assert formula4.splitlines()[-1] == "(4)"
    assert formula8.splitlines()[-1] == "(8)"
    assert "(4)" not in "\n".join(formula4.splitlines()[:-1])
    assert "(8)" not in "\n".join(formula8.splitlines()[:-1])
    assert "Fp =" in formula7
    assert "otherwise" in formula7
    assert not [content for content in page3_contents if content.strip() in {"2", "p", "otherwise", "(7)"}]
    assert "available at http://mc2.unl.edu/current-research" in page3_contents[-1]
    assert "/image-processing/. Figure 2" in page3_contents[-1]

    page4_contents = [block["content"] for block in model_list[3]]
    figure2 = next(content for content in page4_contents if content.startswith("Figure 2:"))
    figure3 = next(content for content in page4_contents if content.startswith("Figure 3:"))
    results = next(content for content in page4_contents if content.startswith("The results of temporal stereo"))
    improvements = next(content for content in page4_contents if content.startswith("Significant improvements"))
    assert figure2.endswith("(4th row).")
    assert figure3.endswith("without temporal aggregation.")
    assert results.endswith("methods that operate on pairs of images.")
    assert improvements.endswith("has the effect")

    page5_contents = [block["content"] for block in model_list[4]]
    optimal_feedback = next(content for content in page5_contents if content.startswith("The optimal value"))
    page5_references = [content for content in page5_contents if content.startswith("[")]
    assert "noise ranging between ±0 to ±40" in optimal_feedback
    assert optimal_feedback.endswith("temporal stereo matching is used.")
    assert [content.partition("]")[0] + "]" for content in page5_references] == [
        f"[{number}]" for number in range(1, 6)
    ]

    page6_contents = [block["content"] for block in model_list[5]]
    assert len(page6_contents) == 16
    assert [content.partition("]")[0] + "]" for content in page6_contents] == [
        f"[{number}]" for number in range(6, 22)
    ]


def test_demo2_container_claims_are_pairwise_disjoint() -> None:
    """验证表格、图形和公式阶段按 source_index 唯一认领，不重复消费文本身份。"""

    for page_idx in (1, 2, 3):
        source = _native_page_source("demo2.pdf", page_idx)
        table_candidates = pdf_extractor._detect_table_candidates(source)
        table_blocks, table_claimed = pdf_extractor._materialize_table_blocks(source, table_candidates)
        table_bboxes = [block["bbox"] for block in table_blocks]
        _graphic_blocks, graphic_claimed = pdf_extractor._build_graphic_like_blocks(
            source,
            table_bboxes,
            table_claimed,
        )
        remaining = pdf_extractor._merge_same_baseline_text_lines(
            [
                line
                for line in source.lines
                if line.source_index not in table_claimed | graphic_claimed
            ],
            source.page_size,
            table_bboxes,
        )
        formula_input_indices = {line.source_index for line in remaining}
        _formula_blocks, formula_remaining = pdf_extractor._build_formula_like_blocks(
            remaining,
            table_bboxes,
            source.page_size,
        )
        formula_claimed = formula_input_indices - {line.source_index for line in formula_remaining}

        assert table_claimed.isdisjoint(graphic_claimed)
        assert table_claimed.isdisjoint(formula_claimed)
        assert graphic_claimed.isdisjoint(formula_claimed)
        combined = table_claimed | graphic_claimed | formula_claimed
        assert len(combined) == len(table_claimed) + len(graphic_claimed) + len(formula_claimed)


def test_demo2_page4_groups_five_graphics_and_keeps_table1() -> None:
    """验证 demo2 第四页五个图形区域分别聚合，Table 1 继续优先输出为 table。"""

    page = _native_model_list("demo2.pdf")[3]
    table_blocks = [block for block in page if block["type"] == "table"]
    graphic_markers = ("Frame 30", "Noise: ±0", "Noise: ±20", "Noise: ±40", "Noise ±")
    graphic_blocks = [
        next(block for block in page if block["type"] == "image" and marker in block["content"])
        for marker in graphic_markers
    ]

    assert len(table_blocks) == 1
    assert "Table I:" in table_blocks[0]["content"]
    assert len({id(block) for block in graphic_blocks}) == 5
    assert "Frame 90" in graphic_blocks[0]["content"]
    assert all("Figure" not in block["content"] for block in graphic_blocks)


def test_demo2_table_captions_and_numeric_footnotes_are_not_text_blocks() -> None:
    """验证 demo2 两张表的换行标题和数字脚注全部并入表格投影。"""

    model_list = _native_model_list("demo2.pdf")
    page4_table = next(block for block in model_list[3] if block["type"] == "table")
    page5_table = next(block for block in model_list[4] if block["type"] == "table")
    residual_text = "\n".join(
        block["content"] for page_idx in (3, 4) for block in model_list[page_idx] if block["type"] == "text"
    )

    assert "ral stereo matching." in page4_table["content"]
    assert "1 To enable propagation of disparity information" in page4_table["content"]
    assert "0.01, respectively." in page4_table["content"]
    assert "Noise: ±20" not in page4_table["content"]
    assert "1 Millions of Disparity Estimates per Second." in page5_table["content"]
    assert "2 Assumes 320 × 240 images with 32 disparity levels." in page5_table["content"]
    assert "the avgerage % of bad pixels." in page5_table["content"]
    assert "ral stereo matching." not in residual_text
    assert "Millions of Disparity Estimates per Second." not in residual_text


def test_demo3_keeps_tables_and_covers_every_native_source_line() -> None:
    """验证 demo3 容器、后续页段落边界及每条原生 source line 均保持稳定。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / "demo3.pdf"
    with pdf_extractor.PDFDocument(str(pdf_path)) as pdf_doc:
        model_list = pdf_extractor._analyze_native_document(pdf_doc)
        source_lines_by_page: list[list[pdf_extractor._LineItem]] = []
        for page_idx in range(pdf_doc.page_count):
            page_size = pdf_doc.page_size(page_idx)
            source_lines_by_page.append(
                pdf_extractor._build_native_line_items(
                    pdf_extractor.get_lines_from_chars(pdf_doc.get_page_chars(page_idx)),
                    page_size,
                    page_rotation=pdf_doc.page_rotation(page_idx),
                )
            )

    assert [sum(block["type"] == "table" for block in page) for page in model_list] == [
        2,
        0,
        0,
        0,
        1,
        2,
        2,
        2,
        0,
        0,
    ]
    assert [len(page) for page in model_list] == [
        23,
        15,
        13,
        21,
        19,
        16,
        16,
        15,
        17,
        18,
    ]
    page7_tables = [block for block in model_list[6] if block["type"] == "table"]
    page7_table4 = next(
        block for block in page7_tables if "Number of parameters" in block["content"]
    )
    page7_table4_caption = next(
        block
        for block in model_list[6]
        if block["content"] == "Table 4: Model size comparison."
    )
    page7_inline_body = next(
        block
        for block in model_list[6]
        if block["content"].startswith("Row, Column, & Global Positional IDs.")
    )
    page9_conclusion = next(
        block
        for block in model_list[8]
        if block["content"].startswith("In this paper, we identified")
    )
    page10_first_reference = next(
        block
        for block in model_list[9]
        if block["content"].startswith("Xiang Deng, Huan Sun")
    )
    assert all(
        marker in page7_table4["content"]
        for marker in (
            "Model",
            "TAPASBASE",
            "TABLEFORMERBASE",
            "TAPASLARGE",
            "TABLEFORMERLARGE",
        )
    )
    assert page7_table4_caption["content"] not in page7_table4["content"]
    assert page7_table4["bbox"][3] < page7_table4_caption["bbox"][1]
    assert page7_inline_body["type"] == "text"
    assert "With TAPASBASE" in page7_inline_body["content"]
    assert "To tackle this" in page9_conclusion["content"]
    assert "Acknowledgments" not in page9_conclusion["content"]
    assert "Cong Yu. 2021. TURL:" in page10_first_reference["content"]
    assert "Jacob Devlin" not in page10_first_reference["content"]
    for page, source_lines in zip(model_list, source_lines_by_page, strict=True):
        output_probe = _normalized_content_probe("".join(str(block.get("content") or "") for block in page))
        missing_lines = [
            line.text
            for line in source_lines
            if (line_probe := _normalized_content_probe(line.text)) and line_probe not in output_probe
        ]
        assert not missing_lines


def test_demo3_pages1_and2_fix_title_front_matter_and_embedding_formula() -> None:
    """验证首页标题稳定，第二页标题、公式和栏尾正文各自保持完整。"""

    page1, page2 = _native_model_list("demo3.pdf")[:2]
    title = next(block for block in page1 if block["content"].startswith("TABLEFORMER:"))
    front_matter_contents = {
        "∗",
        "Aditya Gupta† Rahul Goel†",
        "Jingfeng Yang Luheng He†",
        "Shyam Upadhyay† Shachi Paul †",
        "?Georgia Institute of Technology",
        "†Google Assistant",
        "jingfengyangpku@gmail.com tableformer@google.com",
    }
    front_matter = [
        block for block in page1 if block["content"] in front_matter_contents
    ]
    released_code = next(
        block for block in page1 if "TABLEFORMER.md" in block["content"]
    )
    introduction = next(
        block for block in page1 if block["content"].startswith("Recently, semi-structured")
    )
    nutshell = next(
        block for block in page1 if block["content"].startswith("In a nutshell")
    )
    figure_caption = next(
        block for block in page1 if block["content"].startswith("Figure 1:")
    )
    tables_body = next(
        block for block in page1 if block["content"].startswith("tables or rows")
    )

    assert title["type"] == "doc_title"
    assert len(front_matter) == len(front_matter_contents)
    assert all(block["type"] == "text" for block in front_matter)
    assert next(block for block in page1 if block["content"] == "Abstract")["type"] == "paragraph_title"
    assert released_code["type"] == "text"
    assert introduction["content"].endswith("(Eisenschlos et al., 2021; Liu et al., 2021).")
    assert nutshell["type"] == "text"
    assert nutshell["content"].endswith("by serializing")
    assert figure_caption["type"] == "text"
    assert figure_caption["content"].endswith("both questions.")
    assert "tables or rows" not in figure_caption["content"]
    assert tables_body["type"] == "text"

    section_title = next(
        block for block in page2 if block["content"].startswith("2 Preliminaries:")
    )
    equations = [block for block in page2 if block["type"] == "equation"]
    assert section_title["type"] == "paragraph_title"
    assert section_title["content"] == "2 Preliminaries: TAPAS for Table Encoding"
    assert len(equations) == 1
    assert equations[0]["content"].splitlines() == [
        "token ids (W) = {wv1, wv2, · · · , wvn }",
        "positional ids (B) = {b1, b2, · · · , bn}",
        "segment ids (G) = {gseg1, gseg2, · · · , gsegn }",
        "column ids (C) = {ccol1, ccol2, · · · , ccoln}",
        "row ids (R) = {rrow1, rrow2, · · · , rrown }",
        "rank ids (Z) = {zrank1, zrank2, · · · , zrankn}",
    ]
    as_model_blocks = [
        block
        for block in page2
        if "As for the model" in block["content"]
        or "attends to all the tokens." in block["content"]
        or "Let the layer input" in block["content"]
    ]
    assert len(as_model_blocks) == 1
    assert as_model_blocks[0]["type"] == "text"
    assert as_model_blocks[0]["content"].startswith("As for the model")
    assert "attends to all the tokens." in as_model_blocks[0]["content"]
    assert "Let the layer input" in as_model_blocks[0]["content"]


def test_demo3_pages6_7_and10_fix_caption_inline_titles_and_reference_tail() -> None:
    """验证跨栏 caption、行内粗体正文与参考文献尾行均保持正确归属。"""

    model_list = _native_model_list("demo3.pdf")
    page6 = model_list[5]
    page7 = model_list[6]
    page10 = model_list[9]

    table2_caption = next(
        block for block in page6 if block["content"].startswith("Table 2:")
    )
    assert table2_caption["type"] == "text"
    assert "Median of 5 independent runs are reported." in table2_caption["content"]
    assert table2_caption["content"].endswith("not reported in the original paper.")
    assert sum(
        "not reported in the original paper." in block["content"]
        for block in page6
    ) == 1

    attention_bias = next(
        block for block in page7 if block["content"].startswith("Attention Bias Scaling.")
    )
    positional_ids = next(
        block
        for block in page7
        if block["content"].startswith("Row, Column, & Global Positional IDs.")
    )
    formula6 = next(
        block
        for block in page7
        if block["type"] == "equation" and "(6)" in block["content"]
    )
    assert attention_bias["type"] == "text"
    assert attention_bias["content"].endswith("attention score by:")
    assert positional_ids["type"] == "text"
    assert "With TAPASBASE" in positional_ids["content"]
    assert formula6["content"].splitlines()[-1] == "(6)"
    assert not [
        block
        for block in page7
        if block["type"] == "paragraph_title"
        and block["content"].startswith(
            ("Attention Bias Scaling.", "Row, Column, & Global Positional IDs.")
        )
    ]

    ying_reference = next(
        block for block in page10 if block["content"].startswith("Chengxuan Ying")
    )
    yu_reference = next(
        block for block in page10 if block["content"].startswith("Tao Yu")
    )
    assert ying_reference["type"] == yu_reference["type"] == "text"
    assert ying_reference["content"].endswith("arXiv:2106.05234.")
    assert "Tao Yu" not in ying_reference["content"]
    assert not [
        block for block in page10 if block["content"] == "arXiv:2106.05234."
    ]


def test_demo3_page3_form_image_formulas_titles_and_inline_body_are_whole() -> None:
    """验证第三页大 Form、caption、公式、标题及行内粗体都按整体输出。"""

    page = _native_model_list("demo3.pdf")[2]
    image_blocks = [block for block in page if block["type"] == "image"]
    assert len(image_blocks) == 1
    assert not [block for block in page if block["type"] == "table"]
    image_block = image_blocks[0]
    assert "Transformer (Self Attention)" in image_block["content"]
    assert "Screwed Up" in image_block["content"]
    assert "Figure 2:" not in image_block["content"]
    assert all(
        block is image_block
        or not pdf_extractor._point_in_bbox(
            (
                (block["bbox"][0] + block["bbox"][2]) / 2.0,
                (block["bbox"][1] + block["bbox"][3]) / 2.0,
            ),
            tuple(image_block["bbox"]),
        )
        for block in page
    )
    caption_blocks = [
        block
        for block in page
        if "Figure 2:" in block["content"]
        or "types of task independent biases" in block["content"]
    ]
    assert len(caption_blocks) == 1
    assert caption_blocks[0]["type"] == "text"
    assert "This example corresponds to table (a)" in caption_blocks[0]["content"]
    assert "associated text." in caption_blocks[0]["content"]

    formula1 = next(block for block in page if "(1)" in block["content"])
    section3 = next(block for block in page if block["content"].startswith("3 TABLEFORMER:"))
    inline_item = next(block for block in page if block["content"].startswith("2) Per cell positional ids."))
    inline_heading = next(
        block for block in page if block["content"].startswith("Positional Encoding in TABLEFORMER.")
    )
    assert formula1["type"] == "equation"
    assert "Q = HWQ" in formula1["content"] and "K = HWK" in formula1["content"]
    assert section3["type"] == "paragraph_title"
    assert section3["content"] == "3 TABLEFORMER: Robust Structural Table Encoding"
    assert inline_item["type"] == "text" and "To further remove any" in inline_item["content"]
    assert inline_heading["type"] == "text" and "Transformer model" in inline_heading["content"]


def test_demo3_pages4_and5_fix_lists_formula_titles_italics_and_footnotes() -> None:
    """验证第四、五页列表、公式、独立标题、行内标题、斜体续行及脚注边界。"""

    page4, page5 = _native_model_list("demo3.pdf")[3:5]
    left_bullets = [
        block
        for block in page4
        if block["type"] == "text"
        and block["content"].startswith("•")
        and block["bbox"][2] <= 0.5
    ]
    assert len(left_bullets) == 6
    assert all(len(block["content"].split()) > 8 for block in left_bullets)
    attention_biases = next(
        block for block in page4 if block["content"].startswith("Attention Biases in TABLEFORMER.")
    )
    assert attention_biases["type"] == "text"
    assert "13 bias types" in attention_biases["content"]

    formula3 = next(block for block in page4 if "(3)" in block["content"])
    formula4 = next(block for block in page4 if "(4)" in block["content"])
    assert formula3["type"] == formula4["type"] == "equation"
    assert formula3 is not formula4
    assert "A =" in formula3["content"] and "(4)" not in formula3["content"]
    assert "(3)" not in formula4["content"]

    relation_blocks = [
        block
        for block in page4
        if "Relation between TABLEFORMER and ETC." in block["content"]
        or "ETC (Ainslie et al., 2020)" in block["content"]
    ]
    assert len(relation_blocks) == 1
    assert relation_blocks[0]["type"] == "text"
    assert relation_blocks[0]["content"].startswith("Relation between TABLEFORMER and ETC.")
    assert "uses vectors to represent relative position labels" in relation_blocks[0]["content"]

    title4 = next(block for block in page4 if block["content"] == "4 Experimental Setup")
    title41 = next(block for block in page4 if block["content"] == "4.1 Datasets and Evaluation")
    assert title4["type"] == title41["type"] == "paragraph_title"
    for prefix, continuation in (
        ("Table Question Answering.", "conducted experiments"),
        ("Table-Text Entailment.", "TABFACT dataset"),
    ):
        inline_block = next(block for block in page4 if block["content"].startswith(prefix))
        assert inline_block["type"] == "text"
        assert continuation in inline_block["content"]

    assert next(block for block in page5 if block["content"] == "4.2 Baselines")["type"] == "paragraph_title"
    assert next(
        block for block in page5 if block["content"] == "4.3 Perturbing Tables as Augmented Data"
    )["type"] == "paragraph_title"
    italic_body = next(
        block for block in page5 if block["content"].startswith("Could we alleviate")
    )
    assert italic_body["type"] == "text"
    assert italic_body["content"].endswith("without making any")

    final_bullet = next(
        block for block in page5 if block["content"].startswith("• How does TABLEFORMER compare")
    )
    final_footnote = next(
        block for block in page5 if block["content"].startswith("3By perturbation")
    )
    assert final_bullet["type"] == final_footnote["type"] == "text"
    assert "3By perturbation" not in final_bullet["content"]
