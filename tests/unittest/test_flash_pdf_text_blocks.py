from __future__ import annotations


import pytest

from mineru.backend.flash.native_pdf import (
    line_layout,
    line_merging,
    models,
    native_text,
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


