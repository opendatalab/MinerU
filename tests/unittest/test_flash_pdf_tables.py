from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mineru.backend.flash.native_pdf import (
    geometry,
    models,
    tables,
)


def _axis_line(
    orientation: str,
    bbox: tuple[float, float, float, float],
) -> models._LocalAxisLine:
    """构造表格候选测试使用的局部横竖线。"""

    return models._LocalAxisLine(
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

    assert tables._group_long_horizontal_rules(rejected_lines, median_height) == []
    assert len(tables._group_long_horizontal_rules(accepted_lines, median_height)) == 1


def _rule_table_fixture(
    rule_count: int = 3,
    *,
    centered_columns: bool = False,
) -> tuple[
    list[models._VisualRow],
    list[models._LineItem],
    list[models._LocalAxisLine],
]:
    """构造无 caption 的规则表格，并可切换为宽度变化的居中列。"""

    rows: list[models._VisualRow] = []
    lines: list[models._LineItem] = []
    source_index = 0
    for row_index, top in enumerate((10.0, 30.0, 50.0)):
        fragments: list[models._Fragment] = []
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
                models._Fragment(
                    text=text,
                    bbox=bbox,
                    local_bbox=bbox,
                    line_index=source_index,
                    visual_row_id=row_index,
                )
            )
            lines.append(
                models._LineItem(
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
            models._VisualRow(
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

    candidates = tables._build_rule_table_candidates(
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

    assert tables._count_stable_columns(rows, 5.0) == (2, 1.0)
    candidates = tables._build_rule_table_candidates(
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

    candidates = tables._build_rule_table_candidates(
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

    assert tables._group_long_horizontal_rules(axis_lines, 5.0) == []


def test_chart_tick_rows_fail_dense_multi_cell_distribution() -> None:
    """验证多个图表刻度行因纵向不连续而不能冒充规则表格。"""

    rows: list[models._VisualRow] = []
    lines: list[models._LineItem] = []
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
        fragments: list[models._Fragment] = []
        for anchor in anchors:
            bbox = (anchor, top, anchor + 5.0, top + 5.0)
            text = f"tick-{source_index}"
            fragments.append(
                models._Fragment(
                    text=text,
                    bbox=bbox,
                    local_bbox=bbox,
                    line_index=source_index,
                    visual_row_id=row_index,
                )
            )
            lines.append(
                models._LineItem(
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
            models._VisualRow(
                fragments=fragments,
                center_y=top + 2.5,
                bbox=geometry._bbox_union_many([fragment.bbox for fragment in fragments]),
                visual_row_id=row_index,
            )
        )
    axis_lines = [_axis_line("horizontal", (0.0, top, 110.0, top + 0.1)) for top in (0.0, 55.0, 110.0)]

    candidates = tables._build_rule_table_candidates(
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

    row = models._VisualRow(
        fragments=[
            models._Fragment("For", (0.0, 0.0, 10.0, 5.0), (0.0, 0.0, 10.0, 5.0), 0),
            models._Fragment("*rainfall", (12.0, 0.0, 35.0, 5.0), (12.0, 0.0, 35.0, 5.0), 1),
        ],
        center_y=2.5,
        bbox=(0.0, 0.0, 35.0, 5.0),
    )

    assert tables._is_table_note_text(tables._visual_row_text(row))
    assert tables._is_table_note_text("1 Numeric table footnote")


@pytest.mark.parametrize("projection_mode", ["empty", "error"])
def test_failed_table_projection_does_not_claim_text(
    monkeypatch: pytest.MonkeyPatch,
    projection_mode: str,
) -> None:
    """验证投影为空或抛错时完整回滚候选，文本行仍可进入正文路径。"""

    line = models._LineItem(
        text="cell",
        bbox=(10.0, 10.0, 30.0, 20.0),
        angle=0,
        source_index=0,
        effective_height=10.0,
    )
    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[line],
        chars=[],
        drawing_lines=[],
    )
    candidate = models._TableCandidate(
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
    monkeypatch.setattr(tables, "project_pdf_table_text", projection)

    blocks, claimed = tables._materialize_table_blocks(source, [candidate])

    assert blocks == []
    assert claimed == set()


