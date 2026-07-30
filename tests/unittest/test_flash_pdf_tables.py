from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mineru.backend.flash.native_pdf import (
    geometry,
    models,
    tables,
)
from mineru.utils.pdf_document import PDFPathInfo


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


def _path_info(
    bbox: tuple[float, float, float, float],
    source_index: int,
    *,
    form_depth: int = 0,
) -> PDFPathInfo:
    """构造填充网格测试使用的矩形 Path。"""

    return PDFPathInfo(
        bbox=bbox,
        segment_count=5,
        fill_visible=True,
        stroke_visible=False,
        form_depth=form_depth,
        source_index=source_index,
    )


def _filled_grid_path_fixture(
    *,
    band_count: int = 5,
    cell_gap: float = 0.0,
    right_edge: float = 495.0,
    form_depth: int = 0,
) -> list[PDFPathInfo]:
    """构造含重复 Path、半行底纹和边缘细条的五行双列填充网格。"""

    output = [
        _path_info((100.0, 100.0, 500.0, 400.0), 0, form_depth=form_depth),
        _path_info((105.0, 101.0, 495.0, 104.0), 1, form_depth=form_depth),
        _path_info((105.0, 396.0, 495.0, 399.0), 2, form_depth=form_depth),
    ]
    row_bounds = [
        (105.0, 160.0),
        (160.0, 230.0),
        (230.0, 280.0),
        (280.0, 335.0),
        (335.0, 395.0),
    ]
    source_index = len(output)
    for row_index, (top, bottom) in enumerate(row_bounds[:band_count]):
        cells = [
            (105.0, top, 300.0, bottom),
            (300.0 + cell_gap, top, right_edge, bottom),
        ]
        for cell in cells:
            output.append(
                _path_info(cell, source_index, form_depth=form_depth)
            )
            source_index += 1
        if row_index == 0:
            output.append(
                _path_info(cells[0], source_index, form_depth=form_depth)
            )
            source_index += 1
        if row_index % 2 == 1:
            middle = 0.5 * (top + bottom)
            for left, _top, right, _bottom in cells:
                for half_bbox in (
                    (left, top, right, middle),
                    (left, middle, right, bottom),
                ):
                    output.append(
                        _path_info(
                            half_bbox,
                            source_index,
                            form_depth=form_depth,
                        )
                    )
                    source_index += 1
    return output


def test_filled_grid_geometry_detects_exact_outer_bbox_without_text() -> None:
    """验证纯 Path 网格在没有文本时仍保留精确外框并清除嵌套副本。"""

    path_infos = _filled_grid_path_fixture()
    rectangles = [path_info.bbox for path_info in path_infos]
    candidates = tables._detect_filled_grid_table_candidates(
        path_infos,
        (1000.0, 1000.0),
    )
    cells = tables._select_maximal_filled_grid_cells(
        rectangles,
        (100.0, 100.0, 500.0, 400.0),
    )

    assert len(cells) == 10
    assert len(candidates) == 1
    assert candidates[0].bbox == (100.0, 100.0, 500.0, 400.0)
    assert candidates[0].core_bbox == candidates[0].bbox
    assert candidates[0].line_indices == set()

    source = models._PageSource(
        page_size=(1000.0, 1000.0),
        lines=[],
        chars=[],
        drawing_lines=[],
        path_infos=path_infos,
    )
    assert [
        candidate.bbox for candidate in tables._detect_table_candidates(source)
    ] == [(100.0, 100.0, 500.0, 400.0)]

    source.lines = [
        models._LineItem(
            text="任意替换内容",
            bbox=(120.0, 120.0, 260.0, 140.0),
            angle=0,
            source_index=0,
            effective_height=20.0,
        )
    ]
    assert [
        candidate.bbox for candidate in tables._detect_table_candidates(source)
    ] == [(100.0, 100.0, 500.0, 400.0)]


def test_filled_grid_geometry_rejects_incomplete_or_interfering_paths() -> None:
    """验证行带不足、横向破损、非根层和强图形重叠均不能生成外框。"""

    page_size = (1000.0, 1000.0)
    assert tables._detect_filled_grid_table_candidates(
        _filled_grid_path_fixture(band_count=3),
        page_size,
    ) == []
    assert tables._detect_filled_grid_table_candidates(
        _filled_grid_path_fixture(right_edge=450.0),
        page_size,
    ) == []
    assert tables._detect_filled_grid_table_candidates(
        _filled_grid_path_fixture(cell_gap=10.0),
        page_size,
    ) == []
    assert tables._detect_filled_grid_table_candidates(
        _filled_grid_path_fixture(form_depth=1),
        page_size,
    ) == []
    assert tables._detect_filled_grid_table_candidates(
        _filled_grid_path_fixture(),
        page_size,
        [(120.0, 120.0, 480.0, 380.0)],
    ) == []


def test_filled_grid_candidate_prevents_rule_bbox_expansion() -> None:
    """验证填充网格优先后，重叠横线候选不能扩大其 Path 外框。"""

    lines: list[models._LineItem] = []
    source_index = 0
    for row_index, top in enumerate((120.0, 190.0, 260.0, 330.0)):
        for left, right in ((120.0, 280.0), (320.0, 480.0)):
            lines.append(
                models._LineItem(
                    text=f"cell-{source_index}",
                    bbox=(left, top, right, top + 20.0),
                    angle=0,
                    source_index=source_index,
                    effective_height=20.0,
                    visual_row_id=row_index,
                )
            )
            source_index += 1
    source = models._PageSource(
        page_size=(1000.0, 1000.0),
        lines=lines,
        chars=[],
        drawing_lines=[
            models._AxisLine(
                bbox=(80.0, 90.0, 520.0, 91.0),
                width=1.0,
                orientation="horizontal",
            ),
            models._AxisLine(
                bbox=(80.0, 410.0, 520.0, 411.0),
                width=1.0,
                orientation="horizontal",
            ),
        ],
        path_infos=_filled_grid_path_fixture(),
    )

    candidates = tables._detect_table_candidates(source)

    assert [candidate.bbox for candidate in candidates] == [
        (100.0, 100.0, 500.0, 400.0)
    ]


def test_filled_grid_materialization_uses_existing_spatial_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证空成员候选由 core bbox 收集文本，不需要建立单元格归属。"""

    lines = [
        models._LineItem(
            text="inside-left",
            bbox=(120.0, 120.0, 240.0, 140.0),
            angle=0,
            source_index=0,
            effective_height=20.0,
        ),
        models._LineItem(
            text="inside-right",
            bbox=(330.0, 120.0, 480.0, 140.0),
            angle=0,
            source_index=1,
            effective_height=20.0,
        ),
        models._LineItem(
            text="outside",
            bbox=(120.0, 430.0, 240.0, 450.0),
            angle=0,
            source_index=2,
            effective_height=20.0,
        ),
    ]
    source = models._PageSource(
        page_size=(1000.0, 1000.0),
        lines=lines,
        chars=[],
        drawing_lines=[],
        path_infos=_filled_grid_path_fixture(),
    )
    candidate = tables._detect_filled_grid_table_candidates(
        source.path_infos,
        source.page_size,
    )[0]
    projection = MagicMock(return_value="inside-left   inside-right")
    monkeypatch.setattr(tables, "project_pdf_table_text", projection)

    blocks, claimed = tables._materialize_table_blocks(source, [candidate])

    assert claimed == {0, 1}
    assert blocks == [
        {
            "type": "table",
            "bbox": (100.0, 100.0, 500.0, 400.0),
            "angle": 0,
            "content": "inside-left   inside-right",
        }
    ]


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

    grouped = tables._group_long_horizontal_rules(rejected_lines, median_height)
    assert len(grouped) == 1
    assert [line.bbox for line in grouped[0]] == [
        (0.0, 20.0, accepted_length, 20.1),
        (0.0, 30.0, accepted_length, 30.1),
    ]
    assert len(tables._group_long_horizontal_rules(accepted_lines, median_height)) == 1


def test_connected_rule_grid_expands_core_and_reclaims_sparse_bottom_row() -> None:
    """验证连续物理网格会补全候选底边，并重新认领稀疏末行文本。"""

    bottom_bbox = (10.0, 66.0, 35.0, 72.0)
    bottom_line = models._LineItem(
        text="sparse bottom",
        bbox=bottom_bbox,
        angle=0,
        source_index=9,
        effective_height=6.0,
        visual_row_id=3,
    )
    bottom_row = models._VisualRow(
        fragments=[
            models._Fragment(
                text=bottom_line.text,
                bbox=bottom_bbox,
                local_bbox=bottom_bbox,
                line_index=9,
                visual_row_id=3,
            )
        ],
        center_y=69.0,
        bbox=bottom_bbox,
        visual_row_id=3,
    )
    candidate = models._TableCandidate(
        bbox=(0.0, 8.0, 110.0, 58.1),
        local_bbox=(0.0, 8.0, 110.0, 58.1),
        angle=0,
        score=8.0,
        core_bbox=(0.0, 8.0, 110.0, 58.1),
        line_indices=set(range(9)),
    )
    axis_lines = [
        *[
            _axis_line("horizontal", (0.0, top, 110.0, top + 0.1))
            for top in (8.0, 58.0, 80.0)
        ],
        *[
            _axis_line("vertical", (left, 8.0, left + 0.1, 80.1))
            for left in (0.0, 55.0, 109.9)
        ],
    ]

    expanded = tables._expand_candidates_to_connected_rule_grids(
        [candidate],
        [bottom_row],
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
        [],
    )

    assert expanded[0].core_bbox == pytest.approx((0.0, 8.0, 110.0, 80.1))
    assert expanded[0].bbox == pytest.approx((0.0, 8.0, 110.0, 80.1))
    assert 9 in expanded[0].line_indices


def test_disconnected_stacked_rule_grids_remain_separate() -> None:
    """验证竖轨未跨越空白间距时，同跨度上下网格仍保持为两张表。"""

    axis_lines = [
        *[
            _axis_line("horizontal", (0.0, top, 110.0, top + 0.1))
            for top in (8.0, 58.0, 90.0, 140.0)
        ],
        *[
            _axis_line("vertical", (left, 8.0, left + 0.1, 58.1))
            for left in (0.0, 55.0, 109.9)
        ],
        *[
            _axis_line("vertical", (left, 90.0, left + 0.1, 140.1))
            for left in (0.0, 55.0, 109.9)
        ],
    ]

    grid_bboxes = tables._connected_rule_grid_bboxes(axis_lines, 5.0)

    assert grid_bboxes == pytest.approx(
        [
            (0.0, 8.0, 110.0, 58.1),
            (0.0, 90.0, 110.0, 140.1),
        ]
    )


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
    rule_tops = (8.0, 58.0) if rule_count == 2 else (8.0, 33.0, 58.0)
    axis_lines = [
        _axis_line("horizontal", (0.0, top, 110.0, top + 0.1))
        for top in rule_tops
    ]
    return rows, lines, axis_lines


def _compact_fully_ruled_table_fixture() -> tuple[
    list[models._VisualRow],
    list[models._LineItem],
    list[models._LocalAxisLine],
]:
    """构造两行四列、三横五竖的紧凑全封闭网格。"""

    rows: list[models._VisualRow] = []
    lines: list[models._LineItem] = []
    column_bounds = (
        (5.0, 20.0),
        (32.0, 48.0),
        (60.0, 75.0),
        (87.0, 105.0),
    )
    source_index = 0
    for row_index, top in enumerate((10.0, 22.0)):
        fragments: list[models._Fragment] = []
        for column_index, (left, right) in enumerate(column_bounds):
            bbox = (left, top, right, top + 5.0)
            text = f"cell-{row_index}-{column_index}"
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
                bbox=geometry._bbox_union_many(
                    [fragment.bbox for fragment in fragments]
                ),
                visual_row_id=row_index,
            )
        )

    axis_lines = [
        *[
            _axis_line("horizontal", (0.0, top, 110.0, top + 0.1))
            for top in (8.0, 20.0, 32.0)
        ],
        *[
            _axis_line("vertical", (left, 8.1, left + 0.1, 32.0))
            for left in (0.0, 27.5, 55.0, 82.5, 109.9)
        ],
    ]
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


def test_compact_fully_ruled_two_row_table_is_accepted() -> None:
    """验证两行表格在三横五竖形成完整网格时可通过严格候选准入。"""

    rows, lines, axis_lines = _compact_fully_ruled_table_fixture()

    candidates = tables._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
    )

    assert len(candidates) == 1
    assert candidates[0].line_indices == set(range(8))
    assert candidates[0].score == 8.0


def test_compact_grid_deduplicates_repeated_vertical_paths() -> None:
    """验证同位置重复竖线路径不会扩大紧凑表格的物理列数和评分。"""

    rows, lines, axis_lines = _compact_fully_ruled_table_fixture()
    axis_lines.extend(
        _axis_line("vertical", (left + 0.3, 8.1, left + 0.4, 32.0))
        for left in (0.0, 27.5, 55.0, 82.5, 109.9)
    )

    candidates = tables._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
    )

    assert len(candidates) == 1
    assert candidates[0].score == 8.0


@pytest.mark.parametrize(
    "failure_mode",
    [
        "horizontal_only",
        "short_verticals",
        "missing_outer",
        "same_cell",
        "center_on_boundary",
    ],
)
def test_compact_two_row_layout_requires_complete_grid(
    failure_mode: str,
) -> None:
    """验证两行文本缺少完整竖向网格或唯一单元格映射时仍保持非表格。"""

    rows, lines, axis_lines = _compact_fully_ruled_table_fixture()
    if failure_mode == "horizontal_only":
        axis_lines = [line for line in axis_lines if line.orientation == "horizontal"]
    elif failure_mode == "short_verticals":
        axis_lines = [
            _axis_line("vertical", (line.bbox[0], 13.0, line.bbox[2], 27.0))
            if line.orientation == "vertical"
            else line
            for line in axis_lines
        ]
    elif failure_mode == "missing_outer":
        axis_lines = [
            line
            for line in axis_lines
            if line.orientation != "vertical" or line.bbox[0] > 1.0
        ]
    elif failure_mode == "same_cell":
        rows[0].fragments[1].bbox = (8.0, 10.0, 18.0, 15.0)
        rows[0].fragments[1].local_bbox = rows[0].fragments[1].bbox
    else:
        rows[0].fragments[1].bbox = (26.5, 10.0, 28.5, 15.0)
        rows[0].fragments[1].local_bbox = rows[0].fragments[1].bbox

    candidates = tables._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
    )

    assert candidates == []


def test_compact_admission_does_not_accept_two_row_column_prose() -> None:
    """验证两行普通双栏文本不会因紧凑表格分支而降低准入门槛。"""

    rows, lines, axis_lines = _compact_fully_ruled_table_fixture()
    for row in rows:
        row.fragments = [row.fragments[0], row.fragments[-1]]
        row.bbox = geometry._bbox_union_many(
            [fragment.bbox for fragment in row.fragments]
        )
    retained_indices = {
        fragment.line_index for row in rows for fragment in row.fragments
    }
    lines = [line for line in lines if line.source_index in retained_indices]
    axis_lines = [line for line in axis_lines if line.orientation == "horizontal"]

    candidates = tables._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
    )

    assert candidates == []


def test_two_horizontal_rule_grid_is_accepted_by_spatial_distribution() -> None:
    """验证两条长横线结合密集稳定列即可形成表格候选。"""

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

    assert len(candidates) == 1
    assert candidates[0].line_indices == set(range(9))


def test_duplicate_horizontal_paths_count_as_one_boundary() -> None:
    """验证同一 y 位置的重复 PDF path 只计为一条边界。"""

    axis_lines = [
        _axis_line("horizontal", (0.0, 10.0, 100.0, 10.1)),
        _axis_line("horizontal", (0.0, 10.5, 100.0, 10.6)),
        _axis_line("horizontal", (0.0, 30.0, 100.0, 30.1)),
    ]

    groups = tables._group_long_horizontal_rules(axis_lines, 5.0)

    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert [line.bbox[1] for line in groups[0]] == [10.0, 30.0]
    assert tables._group_long_horizontal_rules(axis_lines[:2], 5.0) == []


def test_nearest_rule_pair_excludes_header_line_from_table_bbox() -> None:
    """验证页首横线与表格上边界间没有多单元行时，不扩张表格 bbox。"""

    rows, lines, axis_lines = _rule_table_fixture(rule_count=2)
    axis_lines.insert(0, _axis_line("horizontal", (0.0, 0.0, 110.0, 0.1)))

    candidates = tables._build_rule_table_candidates(
        rows,
        lines,
        (150.0, 100.0),
        0,
        5.0,
        axis_lines,
    )

    assert len(candidates) == 1
    assert candidates[0].core_bbox is not None
    assert candidates[0].core_bbox[1] == pytest.approx(8.0)


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


def test_long_sparse_rule_interval_cannot_bridge_two_low_column_tables() -> None:
    """验证低列数表格之间仅有一行稀疏文本时不能跨长区间合并。"""

    rows: list[models._VisualRow] = []
    source_index = 0
    for row_index, (top, anchors) in enumerate(
        (
            (10.0, (10.0, 50.0, 90.0)),
            (20.0, (10.0, 50.0, 90.0)),
            (30.0, (10.0, 50.0, 90.0)),
            (80.0, (10.0, 90.0)),
            (130.0, (10.0, 50.0, 90.0)),
            (140.0, (10.0, 50.0, 90.0)),
        )
    ):
        fragments = []
        for anchor in anchors:
            bbox = (anchor, top, anchor + 5.0, top + 5.0)
            fragments.append(
                models._Fragment(
                    text=f"cell-{source_index}",
                    bbox=bbox,
                    local_bbox=bbox,
                    line_index=source_index,
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
    rules = [
        _axis_line("horizontal", (0.0, top, 110.0, top + 0.1))
        for top in (0.0, 40.0, 120.0, 150.0)
    ]

    assert not tables._rule_intervals_are_column_compatible(rows, rules, 5.0)


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
    assert not tables._is_table_note_text("1 Numeric table footnote")


def _table_note_reference_fixture(
    marker: str,
    reference_mode: str,
    *,
    angle: int = 0,
    note_height: float = 8.0,
) -> tuple[
    list[models._VisualRow],
    list[models._LineItem],
    tuple[float, float, float, float],
    set[int],
    tuple[float, float],
]:
    """构造含中性表内引用、表注首行和正文高度样本的局部坐标夹具。"""

    page_size = (200.0, 200.0)
    rule_bbox = (0.0, 20.0, 100.0, 50.0)
    core_local_bbox = (10.0, 25.0, 70.0, 35.0)
    core_chars: list[dict[str, object]] = []
    if reference_mode == "superscript":
        core_text = f"cell{marker}"
        x_position = 10.0
        for raw_char in "cell":
            local_bbox = (x_position, 25.0, x_position + 7.0, 35.0)
            core_chars.append(
                {
                    "char": raw_char,
                    "bbox": geometry._rotate_bbox_from_upright(local_bbox, page_size, angle),
                }
            )
            x_position += 8.0
        for raw_char in marker:
            local_bbox = (x_position, 22.0, x_position + 5.0, 28.0)
            core_chars.append(
                {
                    "char": raw_char,
                    "bbox": geometry._rotate_bbox_from_upright(local_bbox, page_size, angle),
                }
            )
            x_position += 5.5
    elif reference_mode == "compact":
        core_text = marker
    else:
        core_text = "neutral value"

    core_line = models._LineItem(
        text=core_text,
        bbox=geometry._rotate_bbox_from_upright(core_local_bbox, page_size, angle),
        angle=angle,
        source_index=0,
        chars=core_chars,  # type: ignore[arg-type]
        effective_height=10.0,
        font_signature=("Table", 0),
        font_coverage=1.0,
    )
    note_local_bbox = (0.0, 51.0, 80.0, 51.0 + note_height)
    note_line = models._LineItem(
        text=f"{marker} neutral explanation",
        bbox=geometry._rotate_bbox_from_upright(note_local_bbox, page_size, angle),
        angle=angle,
        source_index=1,
        effective_height=note_height,
        font_signature=("Note", 0),
        font_coverage=1.0,
    )
    body_lines = [
        models._LineItem(
            text=f"body-{source_index}",
            bbox=geometry._rotate_bbox_from_upright(
                (110.0, top, 190.0, top + 10.0),
                page_size,
                angle,
            ),
            angle=angle,
            source_index=source_index,
            effective_height=10.0,
            font_signature=("Body", 0),
            font_coverage=1.0,
        )
        for source_index, top in enumerate((140.0, 152.0, 164.0, 176.0), start=2)
    ]
    rows = [
        models._VisualRow(
            fragments=[models._Fragment(core_text, core_local_bbox, core_local_bbox, 0, 0)],
            center_y=geometry._bbox_center_y(core_local_bbox),
            bbox=core_local_bbox,
            visual_row_id=0,
        ),
        models._VisualRow(
            fragments=[models._Fragment(note_line.text, note_local_bbox, note_local_bbox, 1, 1)],
            center_y=geometry._bbox_center_y(note_local_bbox),
            bbox=note_local_bbox,
            visual_row_id=1,
        ),
    ]
    return rows, [core_line, note_line, *body_lines], rule_bbox, {0}, page_size


@pytest.mark.parametrize(
    ("text", "expected"),
    (("7 neutral", "7"), ("(q) neutral", "q"), ("xy: neutral", "xy")),
)
def test_auxiliary_table_note_marker_is_unicode_generic(text: str, expected: str) -> None:
    """验证辅助标记仅受通用 Unicode 形态约束，具体字符变化不影响提取。"""

    assert tables._extract_auxiliary_table_note_marker(text) == expected


@pytest.mark.parametrize(
    ("marker", "reference_mode"),
    (("7", "superscript"), ("q", "compact"), ("xy", "compact")),
)
def test_auxiliary_table_note_requires_neutral_core_reference(
    marker: str,
    reference_mode: str,
) -> None:
    """验证中性标记经上标或紧凑单元格确认后可以启动表注链。"""

    rows, lines, rule_bbox, core_indices, page_size = _table_note_reference_fixture(
        marker,
        reference_mode,
    )

    selected = tables._collect_footnote_rows(
        rows,
        lines,
        rule_bbox,
        8.0,
        core_indices,
        page_size,
        0,
    )

    assert [tables._visual_row_text(row) for row in selected] == [
        f"{marker} neutral explanation"
    ]


def test_auxiliary_table_note_rejects_marker_without_core_reference() -> None:
    """验证紧邻表格的短标记正文在缺少表内引用时不能启动表注链。"""

    rows, lines, rule_bbox, core_indices, page_size = _table_note_reference_fixture(
        "7",
        "none",
    )

    assert tables._collect_footnote_rows(
        rows,
        lines,
        rule_bbox,
        8.0,
        core_indices,
        page_size,
        0,
    ) == []


def test_superscript_reference_requires_smaller_raised_glyph() -> None:
    """验证普通基线上的同字符不能被当成表内上标引用。"""

    _rows, lines, _rule_bbox, _core_indices, page_size = _table_note_reference_fixture(
        "7",
        "superscript",
    )
    core_line = lines[0]
    for char in core_line.chars:
        if str(char.get("char")) == "7":
            char["bbox"] = (42.0, 25.0, 49.0, 35.0)

    assert not tables._line_has_superscript_marker(core_line, "7", page_size, 0)


@pytest.mark.parametrize("note_height", [10.0, 14.0])
def test_auxiliary_table_note_rejects_body_or_title_sized_first_row(
    note_height: float,
) -> None:
    """验证具有表内引用的普通正文或标题字号行仍不能启动表注链。"""

    rows, lines, rule_bbox, core_indices, page_size = _table_note_reference_fixture(
        "q",
        "compact",
        note_height=note_height,
    )

    assert tables._collect_footnote_rows(
        rows,
        lines,
        rule_bbox,
        8.0,
        core_indices,
        page_size,
        0,
    ) == []


def test_auxiliary_table_note_rejects_loose_first_gap() -> None:
    """验证辅助标记首行距超过四分之三局部行高时立即停止扩张。"""

    rows, lines, rule_bbox, core_indices, page_size = _table_note_reference_fixture(
        "q",
        "compact",
    )
    late_bbox = (0.0, 57.0, 80.0, 65.0)
    lines[1].bbox = late_bbox
    rows[1] = models._VisualRow(
        fragments=[models._Fragment(lines[1].text, late_bbox, late_bbox, 1, 1)],
        center_y=geometry._bbox_center_y(late_bbox),
        bbox=late_bbox,
        visual_row_id=1,
    )

    assert tables._collect_footnote_rows(
        rows,
        lines,
        rule_bbox,
        8.0,
        core_indices,
        page_size,
        0,
    ) == []


def test_auxiliary_table_note_requires_smaller_than_body_reference() -> None:
    """验证辅助标记首行还必须显著小于同方向正文参考高度。"""

    rows, lines, rule_bbox, core_indices, page_size = _table_note_reference_fixture(
        "q",
        "compact",
    )
    for line in lines[2:]:
        line.effective_height = 8.5

    assert tables._collect_footnote_rows(
        rows,
        lines,
        rule_bbox,
        8.0,
        core_indices,
        page_size,
        0,
    ) == []


def test_auxiliary_table_note_uses_clipped_corridor_projection() -> None:
    """验证另一栏短标记不能借表格栏内片段制造虚假的整行投影证据。"""

    rows, lines, rule_bbox, core_indices, page_size = _table_note_reference_fixture(
        "q",
        "compact",
    )
    outside_bbox = (140.0, 51.0, 190.0, 59.0)
    inside_bbox = (10.0, 51.0, 70.0, 59.0)
    rows[1] = models._VisualRow(
        fragments=[
            models._Fragment("q outside", outside_bbox, outside_bbox, 1, 1),
            models._Fragment("inside continuation", inside_bbox, inside_bbox, 6, 1),
        ],
        center_y=55.0,
        bbox=geometry._bbox_union(outside_bbox, inside_bbox),
        visual_row_id=1,
    )
    lines.append(
        models._LineItem(
            text="inside continuation",
            bbox=inside_bbox,
            angle=0,
            source_index=6,
            effective_height=8.0,
            font_signature=("Note", 0),
            font_coverage=1.0,
        )
    )

    assert tables._collect_footnote_rows(
        rows,
        lines,
        rule_bbox,
        8.0,
        core_indices,
        page_size,
        0,
    ) == []


def test_rotated_auxiliary_table_note_uses_local_superscript_geometry() -> None:
    """验证旋转表格先转入局部正向坐标后仍可确认上标引用。"""

    rows, lines, rule_bbox, core_indices, page_size = _table_note_reference_fixture(
        "7",
        "superscript",
        angle=90,
    )

    selected = tables._collect_footnote_rows(
        rows,
        lines,
        rule_bbox,
        8.0,
        core_indices,
        page_size,
        90,
    )

    assert [tables._visual_row_text(row) for row in selected] == [
        "7 neutral explanation"
    ]


def test_table_note_chain_cannot_expand_beyond_ten_line_heights() -> None:
    """验证表注续行链即使字体和行距稳定也不能无限向页面底部扩张。"""

    lines: list[models._LineItem] = []
    rows: list[models._VisualRow] = []
    specs = [("Note: neutral", (0.0, 51.0, 80.0, 59.0))]
    specs.extend(
        (f"continuation-{index}", (0.0, 59.5 + 8.5 * index, 80.0, 67.5 + 8.5 * index))
        for index in range(12)
    )
    for source_index, (text, bbox) in enumerate(specs):
        lines.append(
            models._LineItem(
                text=text,
                bbox=bbox,
                angle=0,
                source_index=source_index,
                effective_height=8.0,
                font_signature=("Note", 0),
                font_coverage=1.0,
            )
        )
        rows.append(
            models._VisualRow(
                fragments=[models._Fragment(text, bbox, bbox, source_index, source_index)],
                center_y=geometry._bbox_center_y(bbox),
                bbox=bbox,
                visual_row_id=source_index,
            )
        )

    selected = tables._collect_footnote_rows(
        rows,
        lines,
        (0.0, 0.0, 100.0, 50.0),
        8.0,
        set(),
        (200.0, 200.0),
        0,
    )

    assert selected
    assert len(selected) < len(rows)
    assert selected[-1].bbox[3] <= 130.0


def test_table_note_chain_stops_at_font_and_size_transition() -> None:
    """验证表注续行不能跨越字体字号突变的标题或后续正文。"""

    specs = [
        ("Note: neutral marker", (0.0, 52.0, 60.0, 60.0), ("Note", 0), 8.0),
        ("neutral continuation", (0.0, 61.0, 65.0, 69.0), ("Note", 0), 8.0),
        ("section barrier", (0.0, 70.0, 45.0, 84.0), ("Heading", 0), 14.0),
        ("ordinary body", (10.0, 85.0, 100.0, 95.0), ("Body", 0), 10.0),
    ]
    lines: list[models._LineItem] = []
    rows: list[models._VisualRow] = []
    for source_index, (text, bbox, font, height) in enumerate(specs):
        lines.append(
            models._LineItem(
                text=text,
                bbox=bbox,
                angle=0,
                source_index=source_index,
                effective_height=height,
                font_signature=font,
                font_coverage=1.0,
            )
        )
        fragment = models._Fragment(
            text=text,
            bbox=bbox,
            local_bbox=bbox,
            line_index=source_index,
            visual_row_id=source_index,
        )
        rows.append(
            models._VisualRow(
                fragments=[fragment],
                center_y=geometry._bbox_center_y(bbox),
                bbox=bbox,
                visual_row_id=source_index,
            )
        )

    selected = tables._collect_footnote_rows(
        rows,
        lines,
        (0.0, 0.0, 100.0, 50.0),
        10.0,
        set(),
        (100.0, 100.0),
        0,
    )

    assert [tables._visual_row_text(row) for row in selected] == [
        "Note: neutral marker",
        "neutral continuation",
    ]


def test_numeric_body_row_cannot_start_table_note_chain() -> None:
    """验证数字开头正文即使紧邻表格下边界也不能独立启动表注扩张。"""

    bbox = (0.0, 52.0, 80.0, 62.0)
    line = models._LineItem(
        text="5 ordinary numbered body",
        bbox=bbox,
        angle=0,
        source_index=0,
        effective_height=10.0,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )
    row = models._VisualRow(
        fragments=[models._Fragment(line.text, bbox, bbox, 0, 0)],
        center_y=geometry._bbox_center_y(bbox),
        bbox=bbox,
        visual_row_id=0,
    )

    assert tables._collect_footnote_rows(
        [row],
        [line],
        (0.0, 0.0, 100.0, 50.0),
        10.0,
        set(),
        (100.0, 100.0),
        0,
    ) == []


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
