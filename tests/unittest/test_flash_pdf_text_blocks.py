from __future__ import annotations


import pytest

from mineru.backend.flash.native_pdf import (
    line_layout,
    line_merging,
    models,
    native_text,
    pipeline,
    text_blocks,
)


from _flash_pdf_test_utils import (
    _text_line,
)


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
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lines],
    )

    group_map = text_blocks._build_hanging_indent_group_map(lane, [], [])
    blocks = text_blocks._build_text_blocks(lines, [], (120.0, 120.0))

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
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lines],
    )

    group_map = text_blocks._build_hanging_indent_group_map(lane, [], [])
    blocks = text_blocks._build_text_blocks(lines, [], (120.0, 120.0))

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
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lines],
    )

    assert text_blocks._build_hanging_indent_group_map(lane, [], []) == {}


def test_hanging_indent_keeps_confirmed_entries_before_plain_trailing_paragraph() -> None:
    """验证尾随普通左对齐段落不会让此前已确认的悬挂缩进条目整体失效。"""

    lines = [
        _text_line("entry one", (0.0, 0.0, 100.0, 10.0), 0),
        _text_line("entry one tail", (15.0, 10.0, 100.0, 20.0), 1),
        _text_line("entry two", (0.0, 20.0, 100.0, 30.0), 2),
        _text_line("entry two tail", (15.0, 30.0, 100.0, 40.0), 3),
        _text_line("entry three", (0.0, 40.0, 100.0, 50.0), 4),
        _text_line("entry three tail", (15.0, 50.0, 100.0, 60.0), 5),
        _text_line("plain trailing paragraph", (0.0, 60.0, 100.0, 70.0), 6),
        _text_line("plain continuation", (0.0, 70.0, 100.0, 80.0), 7),
    ]
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lines],
    )

    group_map = text_blocks._build_hanging_indent_group_map(lane, [], [])

    assert [group_map[index] for index in range(6)] == [0, 0, 1, 1, 2, 2]
    assert 6 not in group_map
    assert 7 not in group_map


def test_twelve_point_gutter_keeps_two_text_lanes_and_paragraphs_separate() -> None:
    """验证约 12pt 行高和栏沟仍识别为双栏，左右正文不会交叉拼接。"""

    lines: list[models._LineItem] = []
    for row_index, top in enumerate((100.0, 112.0, 124.0)):
        lines.extend(
            [
                _text_line(f"left-{row_index}", (49.0, top, 300.0, top + 12.0), row_index * 2),
                _text_line(f"right-{row_index}", (312.0, top, 563.0, top + 12.0), row_index * 2 + 1),
            ]
        )

    lanes = line_layout._infer_text_lanes(
        [(line, line.bbox) for line in lines],
        612.0,
        12.0,
    )
    blocks = text_blocks._build_text_blocks(lines, [], (612.0, 792.0))

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

    lines: list[models._LineItem] = [
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

    lanes = line_layout._infer_text_lanes(
        [(line, line.bbox) for line in lines],
        200.0,
        10.0,
    )
    blocks = text_blocks._build_text_blocks(lines, [], (200.0, 180.0))

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
    lane = models._TextLane(
        left=312.0,
        right=563.0,
        lines=[*((line, line.bbox) for line in body_lines), (legend, legend.bbox), (caption, caption.bbox)],
    )

    regular_gap, gap_mad = line_layout._estimate_lane_gap(lane)

    assert (regular_gap, gap_mad) == (0.0, 0.0)
    assert not line_layout._should_connect_text_rows(
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
    lane = models._TextLane(
        left=20.0,
        right=100.0,
        lines=[(previous, previous.bbox), (current, current.bbox)],
    )

    assert not line_layout._should_connect_text_rows(
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
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(previous, previous.bbox), (current, current.bbox)],
    )

    assert not line_layout._should_connect_text_rows(
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
    lane = models._TextLane(
        left=312.0,
        right=563.0,
        lines=[(previous, previous.bbox), (current, current.bbox)],
    )

    assert current.bbox[1] - previous.bbox[3] == pytest.approx(-6.82)
    assert line_layout._effective_text_row_gap(
        (previous, previous.bbox),
        (current, current.bbox),
    ) == pytest.approx(0.0)
    assert line_layout._should_connect_text_rows(
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

    merged_scripts = native_text._merge_native_inline_scripts(script_lines, (200.0, 100.0))

    assert [line.text for line in merged_scripts] == ["O(ω2", "Dip"]
    assert not any(line.restored_inline_cluster for line in merged_scripts)

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

    assert line_merging._can_merge_same_baseline_pair(
        caption_prefix,
        caption_prefix.bbox,
        caption_suffix,
        caption_suffix.bbox,
        [],
    )
    assert not line_merging._can_merge_same_baseline_pair(
        formula_body,
        formula_body.bbox,
        formula_number,
        formula_number.bbox,
        [],
    )


def test_low_overlap_detached_script_preserves_following_row_gap() -> None:
    """验证低重叠外置字形按几何归入主体，且抬高的顶边不会拆断后继行。"""

    base = _text_line(
        "Alpha",
        (10.0, 10.0, 70.0, 20.8),
        0,
        visual_row_id=0,
        effective_height=10.8,
    )
    detached_script = _text_line(
        "◇",
        (70.0, 3.6, 74.2, 11.1),
        1,
        visual_row_id=1,
        effective_height=7.5,
    )
    following = _text_line(
        "Beta",
        (35.0, 22.0, 80.0, 34.0),
        2,
        visual_row_id=2,
        effective_height=10.8,
    )

    merged = native_text._merge_native_inline_scripts(
        [base, detached_script, following],
        (120.0, 100.0),
    )

    assert [line.text for line in merged] == ["Alpha◇", "Beta"]
    assert merged[0].bbox == (10.0, 3.6, 74.2, 20.8)
    assert merged[0].restored_inline_cluster
    assert line_layout._effective_text_row_gap(
        (merged[0], merged[0].bbox),
        (following, following.bbox),
    ) == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("small_text", "small_bbox"),
    [
        pytest.param("wide", (50.0, 3.5, 62.0, 11.0), id="wide-token"),
        pytest.param("◇", (53.0, 3.5, 57.2, 11.0), id="horizontal-gap"),
        pytest.param("◇", (50.0, 0.0, 54.2, 7.5), id="vertical-gap"),
        pytest.param("◇", (50.0, 12.0, 54.2, 19.5), id="same-baseline-cell"),
    ],
)
def test_detached_script_geometry_rejects_ambiguous_small_runs(
    small_text: str,
    small_bbox: tuple[float, float, float, float],
) -> None:
    """验证宽文本、远距文本和同行小单元格不能仅凭小字号并入主体。"""

    base = _text_line(
        "Base",
        (0.0, 10.0, 50.0, 22.0),
        0,
        visual_row_id=0,
        effective_height=12.0,
    )
    small = _text_line(
        small_text,
        small_bbox,
        1,
        visual_row_id=1,
        effective_height=7.5,
    )

    merged = native_text._merge_native_inline_scripts([base, small], (120.0, 100.0))

    assert [line.text for line in merged] == ["Base", small_text]
    assert not any(line.restored_inline_cluster for line in merged)


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
    lane = models._TextLane(left=0.0, right=100.0)

    assert line_layout._should_connect_text_rows(
        (previous, previous.bbox),
        (full_width_url, full_width_url.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )
    assert not line_layout._should_connect_text_rows(
        (previous, previous.bbox),
        (short_formula, short_formula.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )
    assert not line_layout._should_connect_text_rows(
        (previous, previous.bbox),
        (styled_reference, styled_reference.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )


def test_pdf_subset_font_variants_keep_full_rows_and_short_tail_connected() -> None:
    """验证同字体族的 PDF 子集签名差异不会拆断满栏正文及其短尾行。"""

    first = _text_line(
        "first full row",
        (0.0, 0.0, 100.0, 10.0),
        0,
        font_signature=("ABCDEF+TimesNewRomanPSMT", 1),
        font_coverage=1.0,
    )
    second = _text_line(
        "second full row",
        (0.0, 10.0, 100.0, 20.0),
        1,
        font_signature=("TimesNewRomanPSMT", 29),
        font_coverage=1.0,
    )
    tail = _text_line(
        "short tail",
        (0.0, 20.0, 45.0, 30.0),
        2,
        font_signature=("GHIJKL+TimesNewRomanPSMT", 17),
        font_coverage=1.0,
    )
    lane = models._TextLane(left=0.0, right=100.0)

    assert line_layout._should_connect_text_rows(
        (first, first.bbox),
        (second, second.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )
    assert line_layout._should_connect_text_rows(
        (second, second.bbox),
        (tail, tail.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )


def test_pdf_subset_font_short_tail_keeps_significant_weight_barrier() -> None:
    """验证字体族相同但字重显著变化时，短尾续行仍保持硬分段。"""

    body = _text_line(
        "full body row",
        (0.0, 0.0, 100.0, 10.0),
        0,
        font_signature=("ABCDEF+Body", 1),
        font_coverage=1.0,
        dominant_font_weight=400.0,
    )
    emphasized_tail = _text_line(
        "bold short row",
        (0.0, 10.0, 45.0, 20.0),
        1,
        font_signature=("GHIJKL+Body", 29),
        font_coverage=1.0,
        dominant_font_weight=700.0,
    )
    lane = models._TextLane(left=0.0, right=100.0)

    assert not line_layout._should_connect_text_rows(
        (body, body.bbox),
        (emphasized_tail, emphasized_tail.bbox),
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
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(body, body.bbox), (footnote, footnote.bbox)],
    )

    assert not line_layout._should_connect_text_rows(
        (body, body.bbox),
        (footnote, footnote.bbox),
        lane,
        3.0,
        0.0,
        [],
        [],
    )


def test_overlapping_fraction_fragments_merge_with_ha_and_nb_body_hosts() -> None:
    """验证 Ha、Nb 的上下分式碎片并回正文宿主，且普通后续行保持独立。"""

    body_font = ("Body", 0)
    math_font = ("Math", 1)
    lines = [
        _text_line(
            "Ha =",
            (0.0, 10.0, 20.0, 20.0),
            0,
            visual_row_id=0,
            split_from_row=True,
            effective_height=10.0,
            font_signature=math_font,
            font_coverage=0.67,
        ),
        _text_line(
            "root",
            (30.0, 0.0, 40.0, 12.0),
            1,
            visual_row_id=0,
            split_from_row=True,
            effective_height=9.0,
            font_signature=math_font,
            font_coverage=0.57,
        ),
        _text_line(
            "sigma",
            (30.0, 9.0, 34.0, 16.0),
            2,
            visual_row_id=1,
            effective_height=7.0,
            font_signature=math_font,
            font_coverage=1.0,
        ),
        _text_line(
            "mu",
            (30.0, 16.0, 34.0, 23.0),
            3,
            visual_row_id=2,
            effective_height=7.0,
            font_signature=math_font,
            font_coverage=1.0,
        ),
        _text_line(
            "denotes the Hartmann number",
            (22.0, 3.0, 100.0, 25.0),
            4,
            visual_row_id=3,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=0.9,
        ),
        _text_line(
            "Nb = numerator",
            (0.0, 40.0, 42.0, 53.0),
            5,
            visual_row_id=4,
            effective_height=8.0,
            font_signature=math_font,
            font_coverage=0.6,
        ),
        _text_line(
            "mu",
            (30.0, 48.0, 34.0, 55.0),
            6,
            visual_row_id=5,
            effective_height=7.0,
            font_signature=math_font,
            font_coverage=1.0,
        ),
        _text_line(
            "denotes the Brownian parameter",
            (46.0, 43.0, 100.0, 53.0),
            7,
            visual_row_id=6,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "ordinary following row",
            (0.0, 70.0, 100.0, 80.0),
            8,
            visual_row_id=7,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
    ]

    merged = line_merging._merge_overlapping_inline_text_clusters(
        lines,
        (120.0, 100.0),
        [],
    )

    assert [line.source_index for line in merged] == [0, 5, 8]
    assert merged[0].text == "Ha = root sigma mu denotes the Hartmann number"
    assert merged[1].text == "Nb = numerator mu denotes the Brownian parameter"
    assert all(line.restored_inline_cluster for line in merged[:2])
    assert not any(line.text in {"sigma", "mu"} for line in merged)


def test_overlapping_delta_fraction_and_tail_form_one_text_block() -> None:
    """验证同一物理行的分子分母恢复后可与下一行幅值说明组成单块。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            "where, delta = numerator",
            (0.0, 0.0, 32.0, 13.0),
            0,
            visual_row_id=0,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=0.73,
        ),
        _text_line(
            "denominator displays the amplitude",
            (28.0, 3.0, 100.0, 16.0),
            1,
            visual_row_id=1,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=0.86,
        ),
        _text_line(
            "ratio.",
            (0.0, 18.0, 20.0, 28.0),
            2,
            visual_row_id=2,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
    ]

    merged = line_merging._merge_overlapping_inline_text_clusters(
        lines,
        (120.0, 100.0),
        [],
    )
    blocks = text_blocks._build_text_blocks(merged, [], (120.0, 100.0))

    assert len(merged) == 2
    assert [block["content"] for block in blocks] == ["where, delta = numerator denominator displays the amplitude ratio."]


def test_overlapping_inline_pair_respects_table_and_physical_row_gap() -> None:
    """验证二维碎片连接不会跨表格，也不会连接普通上下相邻正文行。"""

    first = _text_line("left", (0.0, 0.0, 40.0, 10.0), 0, effective_height=10.0)
    same_row = _text_line("right", (60.0, 0.0, 100.0, 10.0), 1, effective_height=10.0)
    next_row = _text_line("next", (0.0, 12.0, 40.0, 22.0), 2, effective_height=10.0)

    assert not line_merging._overlapping_inline_cluster_pair_is_connected(
        (first, first.bbox),
        (same_row, same_row.bbox),
        10.0,
        [(45.0, -5.0, 55.0, 15.0)],
    )
    assert not line_merging._overlapping_inline_cluster_pair_is_connected(
        (first, first.bbox),
        (next_row, next_row.bbox),
        10.0,
        [],
    )


def test_hyphen_continuation_cannot_start_false_hanging_indent_entry() -> None:
    """验证断词续行优先归前段，后续首行缩进说明和紧凑公式仍各自分块。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            "linear equations in the trans-",
            (0.0, 0.0, 100.0, 10.0),
            0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "verse direction.",
            (0.0, 12.0, 40.0, 22.0),
            1,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "The expression starts here",
            (12.0, 24.0, 100.0, 34.0),
            2,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "the stream function is given below:",
            (0.0, 36.0, 75.0, 46.0),
            3,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "F = fraction",
            (12.0, 48.0, 45.0, 55.0),
            4,
            effective_height=7.0,
            font_signature=("Math", 1),
            font_coverage=0.5,
        ),
    ]
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lines],
    )

    group_map = text_blocks._build_hanging_indent_group_map(lane, [], [])
    blocks = text_blocks._build_text_blocks(lines, [], (120.0, 100.0))

    assert 1 not in group_map
    assert [block["content"] for block in blocks] == [
        "linear equations in the transverse direction.",
        "The expression starts here the stream function is given below:",
        "F = fraction",
    ]


def test_local_two_column_band_survives_full_width_body_below() -> None:
    """验证页面顶部局部双栏不会被下方通栏正文覆盖。"""

    lines: list[models._LineItem] = []
    for row_index, top in enumerate((0.0, 12.0, 24.0, 36.0)):
        lines.extend(
            [
                _text_line(f"left-{row_index}", (10.0, top, 90.0, top + 10.0), row_index * 2),
                _text_line(f"right-{row_index}", (110.0, top, 190.0, top + 10.0), row_index * 2 + 1),
            ]
        )
    lines.extend(
        _text_line(f"wide-{row_index}", (10.0, top, 190.0, top + 10.0), 20 + row_index)
        for row_index, top in enumerate((70.0, 82.0, 94.0, 106.0))
    )

    lanes = line_layout._infer_text_lanes(
        [(line, line.bbox) for line in lines],
        200.0,
        10.0,
    )

    lane_texts = [{line.text for line, _bbox in lane.lines} for lane in lanes]
    assert {f"left-{index}" for index in range(4)} in lane_texts
    assert {f"right-{index}" for index in range(4)} in lane_texts
    assert {f"wide-{index}" for index in range(4)} in lane_texts


def test_nested_lane_accepts_wider_one_sided_rows_and_expands_interval() -> None:
    """验证窄图注推断的局部栏仍可接纳未进入另一栏的宽正文行。"""

    lines: list[models._LineItem] = []
    for row_index, top in enumerate((0.0, 10.0, 20.0, 30.0, 40.0)):
        lines.extend(
            [
                _text_line(
                    f"narrow-left-{row_index}",
                    (10.0, top, 55.0, top + 8.0),
                    row_index * 2,
                ),
                _text_line(
                    f"right-{row_index}",
                    (115.0, top, 190.0, top + 8.0),
                    row_index * 2 + 1,
                ),
            ]
        )
    lines.extend(
        [
            _text_line("wide-left-one", (10.0, 32.0, 105.0, 40.0), 20),
            _text_line("wide-left-two", (10.0, 42.0, 105.0, 50.0), 21),
        ]
    )
    lines.extend(
        _text_line(
            f"outer-{row_index}",
            (10.0, top, 190.0, top + 8.0),
            30 + row_index,
        )
        for row_index, top in enumerate(
            (90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0)
        )
    )

    lanes = line_layout._infer_text_lanes(
        [(line, line.bbox) for line in lines],
        200.0,
        8.0,
    )

    left_lane = next(
        lane
        for lane in lanes
        if any(line.text == "wide-left-one" for line, _bbox in lane.lines)
    )
    assert not left_lane.is_span
    assert left_lane.right == pytest.approx(105.0)
    assert {
        line.text
        for line, _bbox in left_lane.lines
        if line.text.startswith("wide-left")
    } == {"wide-left-one", "wide-left-two"}


def test_regular_lanes_accept_wider_one_sided_rows_but_keep_gutter_crossing_span() -> None:
    """验证普通双栏会扩展单侧栏宽，同时不吸收越过栏沟的通栏行。"""

    lines: list[models._LineItem] = []
    for row_index, top in enumerate((0.0, 10.0, 20.0, 30.0, 40.0)):
        lines.extend(
            [
                _text_line(
                    f"narrow-left-{row_index}",
                    (10.0, top, 55.0, top + 8.0),
                    row_index * 2,
                ),
                _text_line(
                    f"right-{row_index}",
                    (115.0, top, 190.0, top + 8.0),
                    row_index * 2 + 1,
                ),
            ]
        )
    lines.extend(
        [
            _text_line("wide-left-one", (10.0, 52.0, 105.0, 60.0), 20),
            _text_line("wide-left-two", (10.0, 62.0, 105.0, 70.0), 21),
            _text_line("crosses-gutter", (10.0, 74.0, 130.0, 82.0), 22),
        ]
    )

    lanes = line_layout._infer_text_lanes(
        [(line, line.bbox) for line in lines],
        200.0,
        8.0,
    )

    left_lane = next(
        lane
        for lane in lanes
        if any(line.text == "wide-left-one" for line, _bbox in lane.lines)
    )
    span_lane = next(
        lane
        for lane in lanes
        if any(line.text == "crosses-gutter" for line, _bbox in lane.lines)
    )
    assert not left_lane.is_span
    assert left_lane.right == pytest.approx(105.0)
    assert span_lane.is_span


def test_short_span_tail_reattaches_before_later_cross_layout_region() -> None:
    """验证局部通栏段的短尾行不会因页面后续另有通栏行而留在单栏。"""

    first = _text_line("wide one", (10.0, 0.0, 190.0, 10.0), 0)
    second = _text_line("wide two", (10.0, 10.0, 190.0, 20.0), 1)
    tail = _text_line("tail", (10.0, 20.0, 50.0, 30.0), 2)
    later = _text_line("later wide", (10.0, 80.0, 190.0, 90.0), 3)
    parallel = _text_line("parallel", (110.0, 50.0, 190.0, 60.0), 4)
    left_lane = models._TextLane(left=10.0, right=90.0, lines=[(tail, tail.bbox)])
    right_lane = models._TextLane(left=110.0, right=190.0, lines=[(parallel, parallel.bbox)])
    span_lane = models._TextLane(
        left=10.0,
        right=190.0,
        lines=[(first, first.bbox), (second, second.bbox), (later, later.bbox)],
        is_span=True,
    )

    line_layout._reattach_span_lane_continuations(
        [left_lane, right_lane, span_lane],
        10.0,
    )

    assert tail not in [line for line, _bbox in left_lane.lines]
    assert tail in [line for line, _bbox in span_lane.lines]


def test_spatial_post_merge_connects_short_opener_wide_body_and_tail() -> None:
    """验证跨栏拆开的短首行、满宽正文和紧邻尾行仅按空间关系重新连接。"""

    blocks = [
        {
            "type": "text",
            "bbox": (10.0, 10.0, 40.0, 20.0),
            "angle": 0,
            "content": "section",
            "_visual_row_ids": {0},
            "_local_line_bboxes": [(10.0, 10.0, 40.0, 20.0)],
            "_line_heights": [10.0],
        },
        {
            "type": "text",
            "bbox": (10.0, 22.0, 190.0, 42.0),
            "angle": 0,
            "content": "wide body",
            "_visual_row_ids": {1, 2},
            "_local_line_bboxes": [
                (10.0, 22.0, 190.0, 32.0),
                (10.0, 32.0, 190.0, 42.0),
            ],
            "_line_heights": [10.0, 10.0],
        },
        {
            "type": "text",
            "bbox": (10.0, 44.0, 100.0, 54.0),
            "angle": 0,
            "content": "tail",
            "_visual_row_ids": {3},
            "_local_line_bboxes": [(10.0, 44.0, 100.0, 54.0)],
            "_line_heights": [10.0],
        },
    ]

    merged = text_blocks._merge_spatial_text_components(
        blocks,
        (200.0, 100.0),
    )

    assert len(merged) == 1
    assert merged[0]["bbox"] == (10.0, 10.0, 190.0, 54.0)
    assert merged[0]["content"] == "section wide body tail"
    assert merged[0]["_visual_row_ids"] == {0, 1, 2, 3}


def test_spatial_post_merge_uses_compatible_local_lane_width() -> None:
    """验证半页栏内的短首行可连接多段正文，并在下一分组起点停止。"""

    lane_metadata = {
        "_lane_interval": (40.0, 370.0),
        "_lane_is_span": False,
    }
    blocks = [
        {
            "type": "text",
            "bbox": (40.0, 10.0, 80.0, 20.0),
            "angle": 0,
            "content": "opener",
            "_visual_row_ids": {0},
            "_local_line_bboxes": [(40.0, 10.0, 80.0, 20.0)],
            "_line_heights": [10.0],
            **lane_metadata,
        },
        {
            "type": "text",
            "bbox": (40.0, 22.0, 365.0, 42.0),
            "angle": 0,
            "content": "body one",
            "_visual_row_ids": {1, 2},
            "_local_line_bboxes": [
                (40.0, 22.0, 365.0, 32.0),
                (40.0, 32.0, 365.0, 42.0),
            ],
            "_line_heights": [10.0, 10.0],
            **lane_metadata,
        },
        {
            "type": "text",
            "bbox": (40.0, 44.0, 365.0, 54.0),
            "angle": 0,
            "content": "body two",
            "_visual_row_ids": {3},
            "_local_line_bboxes": [(40.0, 44.0, 365.0, 54.0)],
            "_line_heights": [10.0],
            **lane_metadata,
        },
        {
            "type": "text",
            "bbox": (40.0, 56.0, 365.0, 76.0),
            "angle": 0,
            "content": "body three",
            "_visual_row_ids": {4, 5},
            "_local_line_bboxes": [
                (40.0, 56.0, 365.0, 66.0),
                (40.0, 66.0, 365.0, 76.0),
            ],
            "_line_heights": [10.0, 10.0],
            **lane_metadata,
        },
        {
            "type": "text",
            "bbox": (40.0, 78.0, 230.0, 88.0),
            "angle": 0,
            "content": "body tail",
            "_visual_row_ids": {6},
            "_local_line_bboxes": [(40.0, 78.0, 230.0, 88.0)],
            "_line_heights": [10.0],
            **lane_metadata,
        },
        {
            "type": "text",
            "bbox": (40.0, 90.0, 365.0, 112.0),
            "angle": 0,
            "content": "next section body",
            "_visual_row_ids": {7, 8},
            "_local_line_bboxes": [
                (40.0, 90.0, 90.0, 100.0),
                (40.0, 102.0, 365.0, 112.0),
            ],
            "_line_heights": [10.0, 10.0],
            **lane_metadata,
        },
    ]

    merged = text_blocks._merge_spatial_text_components(
        blocks,
        (600.0, 200.0),
    )

    assert [block["content"] for block in merged] == [
        "opener body one body two body three body tail",
        "next section body",
    ]
    assert merged[0]["bbox"] == (40.0, 10.0, 365.0, 88.0)


def test_spatial_post_merge_does_not_share_incompatible_lane_width() -> None:
    """验证左右栏区间不兼容时仍使用页面宽度，不能放宽首段合并。"""

    blocks = [
        {
            "type": "text",
            "bbox": (40.0, 10.0, 80.0, 20.0),
            "angle": 0,
            "content": "left opener",
            "_visual_row_ids": {0},
            "_local_line_bboxes": [(40.0, 10.0, 80.0, 20.0)],
            "_line_heights": [10.0],
            "_lane_interval": (40.0, 370.0),
            "_lane_is_span": False,
        },
        {
            "type": "text",
            "bbox": (270.0, 22.0, 590.0, 42.0),
            "angle": 0,
            "content": "right body",
            "_visual_row_ids": {1, 2},
            "_local_line_bboxes": [
                (270.0, 22.0, 590.0, 32.0),
                (270.0, 32.0, 590.0, 42.0),
            ],
            "_line_heights": [10.0, 10.0],
            "_lane_interval": (270.0, 590.0),
            "_lane_is_span": False,
        },
    ]

    merged = text_blocks._merge_spatial_text_components(
        blocks,
        (600.0, 100.0),
    )

    assert [block["content"] for block in merged] == [
        "left opener",
        "right body",
    ]


def test_spatial_post_merge_does_not_join_span_caption_to_column_body() -> None:
    """验证跨栏图注即使紧邻单栏正文也不会在二次阶段合并。"""

    blocks = [
        {
            "type": "text",
            "bbox": (40.0, 10.0, 560.0, 30.0),
            "angle": 0,
            "content": "wide caption",
            "_visual_row_ids": {0, 1},
            "_local_line_bboxes": [
                (200.0, 10.0, 360.0, 20.0),
                (40.0, 20.0, 560.0, 30.0),
            ],
            "_line_heights": [10.0, 10.0],
            "_lane_interval": (40.0, 560.0),
            "_lane_is_span": True,
        },
        {
            "type": "text",
            "bbox": (40.0, 35.0, 290.0, 65.0),
            "angle": 0,
            "content": "left column body",
            "_visual_row_ids": {2, 3, 4},
            "_local_line_bboxes": [
                (40.0, 35.0, 290.0, 45.0),
                (40.0, 45.0, 290.0, 55.0),
                (40.0, 55.0, 290.0, 65.0),
            ],
            "_line_heights": [10.0, 10.0, 10.0],
            "_lane_interval": (40.0, 290.0),
            "_lane_is_span": False,
        },
    ]

    merged = text_blocks._merge_spatial_text_components(blocks, (600.0, 100.0))

    assert [block["content"] for block in merged] == [
        "wide caption",
        "left column body",
    ]


def test_image_caption_marker_only_attaches_same_font_spatial_tail() -> None:
    """验证通用图注标记仅确认图像邻近候选，并把同字体续行接回对应图注。"""

    common = {
        "type": "text",
        "angle": 0,
        "_single_run_row_id": None,
        "_lane_interval": (10.0, 90.0),
        "_lane_is_span": True,
    }
    blocks = [
        {
            **common,
            "bbox": (30.0, 50.0, 70.0, 60.0),
            "content": "图1 neutral caption",
            "_visual_row_ids": {1},
            "_local_line_bboxes": [(30.0, 50.0, 70.0, 60.0)],
            "_line_heights": [10.0],
            "_font_signatures": {("Chinese", 0)},
        },
        {
            **common,
            "bbox": (10.0, 58.0, 90.0, 68.0),
            "content": "Fig. 1 neutral caption;",
            "_visual_row_ids": {2},
            "_local_line_bboxes": [(10.0, 58.0, 90.0, 68.0)],
            "_line_heights": [10.0],
            "_font_signatures": {("English", 0)},
        },
        {
            **common,
            "bbox": (20.0, 67.0, 80.0, 77.0),
            "content": "caption continuation",
            "_visual_row_ids": {3},
            "_local_line_bboxes": [(20.0, 67.0, 80.0, 77.0)],
            "_line_heights": [10.0],
            "_font_signatures": {("English", 0)},
        },
    ]

    merged = text_blocks._merge_image_caption_text_blocks(
        blocks,
        [(20.0, 10.0, 80.0, 50.0)],
    )

    assert len(merged) == 2
    chinese = next(block for block in merged if block["content"].startswith("图1"))
    english = next(block for block in merged if block["content"].startswith("Fig. 1"))
    assert "continuation" not in chinese["content"]
    assert "caption continuation" in english["content"]
    assert english["bbox"] == (10.0, 58.0, 90.0, 77.0)


def test_fragmented_center_header_merges_but_remote_volume_stays_separate() -> None:
    """验证等距窄页眉字形形成一个逻辑块，远端卷期块不被吸收。"""

    blocks = []
    for index, left in enumerate((40.0, 60.0, 80.0, 100.0)):
        blocks.append(
            {
                "type": "header",
                "bbox": (left, 5.0, left + 6.0, 15.0),
                "angle": 0,
                "content": f"h{index}",
                "_visual_row_ids": {0},
                "_single_run_row_id": 0,
                "_local_line_bboxes": [(left, 5.0, left + 6.0, 15.0)],
                "_line_heights": [10.0],
                "_font_signatures": {("Header", 0)},
                "_lane_interval": (0.0, 200.0),
                "_lane_is_span": True,
            }
        )
    blocks.append(
        {
            "type": "header",
            "bbox": (170.0, 5.0, 190.0, 15.0),
            "angle": 0,
            "content": "volume",
            "_visual_row_ids": {0},
            "_single_run_row_id": 0,
            "_local_line_bboxes": [(170.0, 5.0, 190.0, 15.0)],
            "_line_heights": [10.0],
            "_font_signatures": {("Header", 0)},
            "_lane_interval": (0.0, 200.0),
            "_lane_is_span": True,
        }
    )

    merged = text_blocks._merge_fragmented_header_blocks(blocks)

    assert len(merged) == 2
    center = next(block for block in merged if block["content"].startswith("h0"))
    assert center["content"] == "h0 h1 h2 h3"
    assert center["bbox"] == (40.0, 5.0, 106.0, 15.0)
    assert next(block for block in merged if block["content"] == "volume")


def test_spatial_post_merge_cannot_skip_intervening_title_block() -> None:
    """验证二次组件合并不能越过同一水平流中的中间标题块。"""

    common = {
        "angle": 0,
        "_single_run_row_id": None,
        "_lane_interval": (0.0, 100.0),
        "_lane_is_span": False,
        "_font_signatures": {("Body", 0)},
    }
    blocks = [
        {
            **common,
            "type": "text",
            "bbox": (0.0, 0.0, 25.0, 10.0),
            "content": "short opener",
            "_visual_row_ids": {0},
            "_local_line_bboxes": [(0.0, 0.0, 25.0, 10.0)],
            "_line_heights": [10.0],
        },
        {
            **common,
            "type": "paragraph_title",
            "bbox": (0.0, 9.0, 45.0, 19.0),
            "content": "intervening title",
            "_visual_row_ids": {1},
            "_local_line_bboxes": [(0.0, 9.0, 45.0, 19.0)],
            "_line_heights": [10.0],
        },
        {
            **common,
            "type": "text",
            "bbox": (0.0, 12.0, 100.0, 22.0),
            "content": "wide body",
            "_visual_row_ids": {2},
            "_local_line_bboxes": [(0.0, 12.0, 100.0, 22.0)],
            "_line_heights": [10.0],
        },
    ]

    merged = text_blocks._merge_spatial_text_components(blocks, (100.0, 50.0))

    assert [block["content"] for block in merged] == [
        "short opener",
        "intervening title",
        "wide body",
    ]


def test_exaggerated_bbox_short_tail_stays_with_full_width_previous_row() -> None:
    """验证同左边界短尾行不会因异常字体框高度而从正文段落脱落。"""

    previous = _text_line(
        "full width previous row",
        (0.0, 0.0, 100.0, 10.0),
        0,
        effective_height=10.0,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )
    current = _text_line(
        "short tail",
        (0.0, 12.0, 25.0, 30.0),
        1,
        effective_height=18.0,
        font_signature=("Number", 0),
        font_coverage=1.0,
    )
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(previous, previous.bbox), (current, current.bbox)],
    )

    assert line_layout._should_connect_text_rows(
        (previous, previous.bbox),
        (current, current.bbox),
        lane,
        regular_gap=2.0,
        gap_mad=0.5,
        table_bboxes=[],
        axis_lines=[],
    )


def test_indented_list_row_can_return_to_lane_left_for_short_tail() -> None:
    """验证列表首行轻度缩进时，回到栏左边的同字体短尾行仍可续接。"""

    body_font = ("Body", 0)
    previous = _text_line(
        "indented list row",
        (15.0, 0.0, 100.0, 10.0),
        0,
        font_signature=body_font,
        font_coverage=1.0,
    )
    current = _text_line(
        "short tail",
        (0.0, 10.0, 65.0, 20.0),
        1,
        font_signature=body_font,
        font_coverage=1.0,
    )
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(previous, previous.bbox), (current, current.bbox)],
    )

    assert line_layout._should_connect_text_rows(
        (previous, previous.bbox),
        (current, current.bbox),
        lane,
        regular_gap=0.0,
        gap_mad=0.0,
        table_bboxes=[],
        axis_lines=[],
    )


def test_outdented_reference_number_starts_new_structural_entry() -> None:
    """验证通用参考文献编号仅在左突几何成立时切开上一条续行。"""

    previous = _text_line("previous continuation", (25.0, 0.0, 100.0, 10.0), 0)
    current = _text_line("［23］ next reference", (0.0, 10.0, 100.0, 20.0), 1)
    aligned = _text_line("［23］ inline marker", (25.0, 10.0, 100.0, 20.0), 2)

    assert text_blocks._starts_structural_reference_entry(
        (previous, previous.bbox),
        (current, current.bbox),
    )
    assert not text_blocks._starts_structural_reference_entry(
        (previous, previous.bbox),
        (aligned, aligned.bbox),
    )


def test_spatial_post_merge_limits_tapered_tail_to_parallel_information_grid() -> None:
    """验证略大间距的递减尾行只在并列信息网格中并回其左对齐主体。"""

    blocks = [
        {
            "type": "text",
            "bbox": (10.0, 10.0, 90.0, 40.0),
            "angle": 0,
            "content": "left body",
            "_visual_row_ids": {0},
            "_local_line_bboxes": [(10.0, 30.0, 90.0, 40.0)],
            "_line_heights": [10.0],
        },
        {
            "type": "text",
            "bbox": (110.0, 10.0, 190.0, 40.0),
            "angle": 0,
            "content": "right body",
            "_visual_row_ids": {1},
            "_local_line_bboxes": [(110.0, 30.0, 190.0, 40.0)],
            "_line_heights": [10.0],
        },
        {
            "type": "text",
            "bbox": (10.0, 50.0, 60.0, 60.0),
            "angle": 0,
            "content": "left tail",
            "_visual_row_ids": {2},
            "_local_line_bboxes": [(10.0, 50.0, 60.0, 60.0)],
            "_line_heights": [10.0],
        },
    ]

    merged = text_blocks._merge_spatial_text_components(
        blocks,
        (200.0, 100.0),
    )

    assert [block["content"] for block in merged] == [
        "left body left tail",
        "right body",
    ]

    isolated = text_blocks._merge_spatial_text_components(
        [blocks[0], blocks[2]],
        (200.0, 100.0),
    )
    assert [block["content"] for block in isolated] == [
        "left body",
        "left tail",
    ]


def test_repeated_leading_emphasis_and_tail_gap_split_structured_text() -> None:
    """验证重复行首强调只在前行留白充分时形成结构化正文边界。"""

    lines = [
        _text_line(
            "Purpose first section opener",
            (0.0, 0.0, 100.0, 10.0),
            0,
            leading_emphasis_width=12.0,
        ),
        _text_line("purpose tail.", (0.0, 10.0, 78.0, 20.0), 1),
        _text_line(
            "Methods second section opener",
            (0.0, 20.0, 100.0, 30.0),
            2,
            leading_emphasis_width=12.0,
        ),
        _text_line("methods tail.", (0.0, 30.0, 60.0, 40.0), 3),
        _text_line(
            "Results third section opener",
            (0.0, 40.0, 100.0, 50.0),
            4,
            leading_emphasis_width=12.0,
        ),
        _text_line("results tail.", (0.0, 50.0, 35.0, 60.0), 5),
        _text_line(
            "Conclusions fourth section opener",
            (0.0, 60.0, 100.0, 70.0),
            6,
            leading_emphasis_width=15.0,
        ),
        _text_line("conclusions tail.", (0.0, 70.0, 70.0, 80.0), 7),
    ]

    blocks = text_blocks._build_text_blocks(lines, [], (100.0, 100.0))

    assert [block["content"] for block in blocks] == [
        "Purpose first section opener purpose tail.",
        "Methods second section opener methods tail.",
        "Results third section opener results tail.",
        "Conclusions fourth section opener conclusions tail.",
    ]


def test_native_typography_caches_leading_emphasis_before_chars_are_cleared() -> None:
    """验证字符释放前缓存短粗体前缀宽度，后续文本分块仍可读取。"""

    chars = []
    for index, char in enumerate("Lead"):
        chars.append(
            {
                "char": char,
                "bbox": (float(index * 5), 0.0, float(index * 5 + 5), 10.0),
                "font": {"name": "LeadFont", "flags": 0, "weight": 600},
            }
        )
    for index, char in enumerate("body", start=5):
        chars.append(
            {
                "char": char,
                "bbox": (float(index * 5), 0.0, float(index * 5 + 5), 10.0),
                "font": {"name": "BodyFont", "flags": 0, "weight": 400},
            }
        )
    line = models._LineItem(
        text="Lead body",
        bbox=(0.0, 0.0, 45.0, 10.0),
        angle=0,
        source_index=0,
        chars=chars,
    )

    native_text._fill_native_typography(line, (100.0, 100.0))
    pipeline._compact_prepared_lines([line], (100.0, 100.0))

    assert line.leading_emphasis_width == 20.0
    assert not line.chars


def test_two_leading_emphasis_rows_do_not_activate_structured_text_split() -> None:
    """验证不足三个重复强调首行时不会把普通混排正文切碎。"""

    lines = [
        _text_line(
            "first emphasized opener",
            (0.0, 0.0, 100.0, 10.0),
            0,
            leading_emphasis_width=15.0,
        ),
        _text_line("first short tail", (0.0, 10.0, 60.0, 20.0), 1),
        _text_line(
            "second emphasized opener",
            (0.0, 20.0, 100.0, 30.0),
            2,
            leading_emphasis_width=15.0,
        ),
        _text_line("second short tail", (0.0, 30.0, 60.0, 40.0), 3),
    ]

    blocks = text_blocks._build_text_blocks(lines, [], (100.0, 100.0))

    assert len(blocks) == 1


def test_leading_emphasis_count_does_not_cross_semantic_title_boundary() -> None:
    """验证不同语义区段的强调首行数量不能合并触发结构化正文模式。"""

    lines = [
        _text_line("first opener", (0.0, 0.0, 100.0, 10.0), 0, leading_emphasis_width=10.0),
        _text_line("first tail", (0.0, 10.0, 60.0, 20.0), 1),
        _text_line("second opener", (0.0, 20.0, 100.0, 30.0), 2, leading_emphasis_width=10.0),
        _text_line("second tail", (0.0, 30.0, 60.0, 40.0), 3),
        _text_line(
            "semantic title",
            (0.0, 50.0, 40.0, 60.0),
            4,
            semantic_type="paragraph_title",
        ),
        _text_line("third opener", (0.0, 70.0, 100.0, 80.0), 5, leading_emphasis_width=10.0),
        _text_line("third tail", (0.0, 80.0, 60.0, 90.0), 6),
    ]

    blocks = text_blocks._build_text_blocks(lines, [], (100.0, 100.0))

    first_region = next(block for block in blocks if "first opener" in block["content"])
    assert "second opener" in first_region["content"]


def test_page_footnote_visual_markers_attach_six_affiliation_entries() -> None:
    """验证同行窄编号与右侧单位首行、续行分别组成独立脚注条目。"""

    lines = [
        _text_line("contact name", (10.0, 0.0, 48.0, 10.0), 0, semantic_type="page_footnote"),
        _text_line("contact email", (10.0, 10.0, 55.0, 20.0), 1, semantic_type="page_footnote"),
    ]
    source_index = 2
    for marker in range(1, 7):
        top = 20.0 + (marker - 1) * 20.0
        visual_row_id = 100 + marker
        lines.extend(
            [
                _text_line(
                    str(marker),
                    (0.0, top, 3.0, top + 8.0),
                    source_index,
                    visual_row_id=visual_row_id,
                    run_index=0,
                    split_from_row=True,
                    median_glyph_width=3.0,
                    semantic_type="page_footnote",
                ),
                _text_line(
                    f"affiliation {marker} first row",
                    (10.0, top, 90.0, top + 10.0),
                    source_index + 1,
                    visual_row_id=visual_row_id,
                    run_index=1,
                    split_from_row=True,
                    median_glyph_width=4.0,
                    semantic_type="page_footnote",
                ),
                _text_line(
                    f"affiliation {marker} continuation",
                    (10.0, top + 10.0, 75.0, top + 20.0),
                    source_index + 2,
                    visual_row_id=visual_row_id + 100,
                    median_glyph_width=4.0,
                    semantic_type="page_footnote",
                ),
            ]
        )
        source_index += 3

    blocks = text_blocks._build_text_blocks(
        lines,
        [],
        (100.0, 160.0),
        page_footnote_groups=[{line.source_index for line in lines}],
    )

    assert [block["content"] for block in blocks[:2]] == [
        "contact name",
        "contact email",
    ]
    assert len(blocks[2:]) == 6
    for marker, block in enumerate(blocks[2:], start=1):
        assert block["content"].startswith(f"{marker} affiliation {marker} first row")
        assert f"affiliation {marker} continuation" in block["content"]


def test_local_aligned_column_replaces_polluted_full_width_lane() -> None:
    """验证跨栏上文不会使后续稳定单栏正文被首行缩进规则拆开。"""

    lines = [
        _text_line("wide row one", (0.0, 0.0, 200.0, 10.0), 0),
        _text_line("wide row two", (0.0, 10.0, 200.0, 20.0), 1),
        _text_line("wide short tail", (0.0, 20.0, 150.0, 30.0), 2),
        _text_line(
            "Introduction",
            (110.0, 40.0, 150.0, 50.0),
            3,
            semantic_type="paragraph_title",
        ),
        _text_line("local first sentence.", (110.0, 55.0, 200.0, 65.0), 4),
        _text_line("local second sentence", (110.0, 65.0, 200.0, 75.0), 5),
        _text_line("local third sentence", (110.0, 75.0, 200.0, 85.0), 6),
        _text_line("local final sentence.", (110.0, 85.0, 195.0, 95.0), 7),
    ]

    blocks = text_blocks._build_text_blocks(lines, [], (200.0, 120.0))
    introduction_body = next(
        block for block in blocks if "local first sentence" in block["content"]
    )

    assert introduction_body["content"] == (
        "local first sentence. local second sentence local third sentence "
        "local final sentence."
    )
    assert introduction_body["_lane_interval"] == (110.0, 200.0)


def test_typographic_gap_splits_caption_body_without_image_geometry() -> None:
    """验证异常净空与行高层级变化可独立于图片位置形成段落硬边界。"""

    caption = _text_line(
        "caption ends without punctuation",
        (0.0, 0.0, 100.0, 10.0),
        0,
        effective_height=10.0,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )
    body = _text_line(
        "body starts",
        (0.0, 19.0, 100.0, 30.5),
        1,
        effective_height=11.5,
        font_signature=("Body", 0),
        font_coverage=1.0,
    )
    lane = models._TextLane(left=0.0, right=100.0)

    assert not line_layout._should_connect_text_rows(
        (caption, caption.bbox),
        (body, body.bbox),
        lane,
        regular_gap=0.0,
        gap_mad=0.0,
        table_bboxes=[],
        axis_lines=[],
    )


def test_typographic_gap_requires_both_spacing_and_typography_evidence() -> None:
    """验证单独字号变化或单独中等净空都不会触发新增段落屏障。"""

    base = _text_line("base", (0.0, 0.0, 100.0, 10.0), 0, effective_height=10.0)
    larger_nearby = _text_line(
        "larger nearby",
        (0.0, 10.0, 100.0, 21.5),
        1,
        effective_height=11.5,
    )
    same_size_spaced = _text_line(
        "same size spaced",
        (0.0, 18.0, 100.0, 28.0),
        2,
        effective_height=10.0,
    )
    lane = models._TextLane(left=0.0, right=100.0)

    assert line_layout._should_connect_text_rows(
        (base, base.bbox),
        (larger_nearby, larger_nearby.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )
    assert line_layout._should_connect_text_rows(
        (base, base.bbox),
        (same_size_spaced, same_size_spaced.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )


def test_hyphenated_row_bypasses_typographic_gap_barrier() -> None:
    """验证排版断词仍可在现有近距离上限内跨过字号和净空变化。"""

    previous = _text_line("hyphen-", (0.0, 0.0, 100.0, 10.0), 0, effective_height=10.0)
    current = _text_line("ated", (0.0, 19.0, 100.0, 30.5), 1, effective_height=11.5)
    lane = models._TextLane(left=0.0, right=100.0)

    assert line_layout._should_connect_text_rows(
        (previous, previous.bbox),
        (current, current.bbox),
        lane,
        0.0,
        0.0,
        [],
        [],
    )


def test_caption_post_merge_respects_typographic_gap_barrier() -> None:
    """验证图片邻接后处理不能跨过已确认的图注正文排版边界。"""

    common = {
        "type": "text",
        "angle": 0,
        "_single_run_row_id": None,
        "_lane_interval": (0.0, 100.0),
        "_lane_is_span": False,
        "_font_signatures": {("Body", 0)},
    }
    caption = {
        **common,
        "bbox": (0.0, 50.0, 100.0, 60.0),
        "content": "Fig. 1 caption",
        "_visual_row_ids": {1},
        "_local_line_bboxes": [(0.0, 50.0, 100.0, 60.0)],
        "_line_heights": [10.0],
    }
    body = {
        **common,
        "bbox": (0.0, 69.0, 100.0, 80.5),
        "content": "body starts",
        "_visual_row_ids": {2},
        "_local_line_bboxes": [(0.0, 69.0, 100.0, 80.5)],
        "_line_heights": [11.5],
    }

    merged = text_blocks._merge_image_caption_text_blocks(
        [caption, body],
        [(0.0, 10.0, 100.0, 50.0)],
    )

    assert [block["content"] for block in merged] == [
        "Fig. 1 caption",
        "body starts",
    ]
