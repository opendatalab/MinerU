from __future__ import annotations

import inspect

import pytest

from mineru.backend.flash.native_pdf import (
    line_layout,
    line_merging,
    text_blocks,
    titles,
)


from _flash_pdf_test_utils import (
    _text_line,
)


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

    titles._classify_page_titles(lines, (100.0, 150.0), page_index=1, container_bboxes=[])

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

    titles._classify_page_titles(lines, (100.0, 120.0), page_index=1, container_bboxes=[])

    assert lines[2].semantic_type is None


def test_physical_title_gap_ignores_disjoint_column_and_keeps_overlapping_row() -> None:
    """验证物理邻行只在水平投影相交时参与标题留白，避免另一栏正文压缩间距。"""

    target = _text_line("target", (0.0, 40.0, 40.0, 50.0), 0)
    disjoint = _text_line("other column", (60.0, 39.0, 100.0, 49.0), 1)
    overlapping = _text_line("wide row", (0.0, 25.0, 100.0, 35.0), 2)

    gaps = titles._build_physical_title_gap_map(
        [(target, target.bbox), (disjoint, disjoint.bbox), (overlapping, overlapping.bbox)]
    )

    assert gaps[target.source_index] == (5.0, None)


def test_grid_title_suppression_requires_two_distinct_parallel_bands() -> None:
    """验证单个三栏短首行带不足以抑制标题，重复两带才形成信息网格证据。"""

    lanes = []
    source_index = 0
    for lane_index in range(3):
        left = 110.0 * lane_index
        first_opener = _text_line(
            f"opener-{lane_index}-0",
            (left, 10.0, left + 20.0, 20.0),
            source_index,
        )
        source_index += 1
        first_body = _text_line(
            f"body-{lane_index}-0",
            (left, 20.0, left + 80.0, 30.0),
            source_index,
        )
        source_index += 1
        lanes.append(
            titles._TextLane(
                left=left,
                right=left + 100.0,
                lines=[
                    (first_opener, first_opener.bbox),
                    (first_body, first_body.bbox),
                ],
            )
        )

    assert titles._find_repeated_grid_title_suppressions(lanes, 10.0) == set()

    second_band_sources = set()
    for lane in lanes[:2]:
        second_opener = _text_line(
            "second opener",
            (lane.left, 40.0, lane.left + 20.0, 50.0),
            source_index,
        )
        second_band_sources.add(source_index)
        source_index += 1
        second_body = _text_line(
            "second body",
            (lane.left, 50.0, lane.left + 80.0, 60.0),
            source_index,
        )
        source_index += 1
        lane.lines.extend(
            [(second_opener, second_opener.bbox), (second_body, second_body.bbox)]
        )

    suppressions = titles._find_repeated_grid_title_suppressions(lanes, 10.0)

    assert second_band_sources <= suppressions
    assert len(suppressions) == 5


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

    titles._classify_page_titles(lines, (100.0, 150.0), page_index=1, container_bboxes=[])

    assert lines[2].semantic_type == "paragraph_title"


def test_centered_smaller_paragraph_title_accepts_compact_text_section() -> None:
    """验证小字号居中标题可由紧随其后的连续小字号正文区段支撑。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            f"body {index}",
            (0.0, 12.0 * index, 100.0, 12.0 * index + 10.0),
            index,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index in range(5)
    ]
    lines.extend(
        [
            _text_line(
                "neutral label",
                (30.0, 70.0, 70.0, 78.0),
                5,
                effective_height=8.0,
                font_signature=body_font,
                font_coverage=1.0,
            ),
            _text_line("compact row one", (0.0, 84.0, 100.0, 92.0), 6, effective_height=8.0),
            _text_line("compact row two", (0.0, 94.0, 100.0, 102.0), 7, effective_height=8.0),
            _text_line("compact row three", (0.0, 104.0, 100.0, 112.0), 8, effective_height=8.0),
        ]
    )

    titles._classify_page_titles(lines, (100.0, 140.0), page_index=1, container_bboxes=[])

    assert lines[5].semantic_type == "paragraph_title"


@pytest.mark.parametrize("failure_mode", ["too_few", "narrow", "large_gap"])
def test_centered_smaller_paragraph_title_rejects_incomplete_compact_section(
    failure_mode: str,
) -> None:
    """验证不足三行、行宽不足或行距中断的小字号区段不能放宽标题。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            f"body {index}",
            (0.0, 12.0 * index, 100.0, 12.0 * index + 10.0),
            index,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index in range(5)
    ]
    compact_rows = [
        _text_line("compact row one", (0.0, 84.0, 100.0, 92.0), 6, effective_height=8.0),
        _text_line("compact row two", (0.0, 94.0, 100.0, 102.0), 7, effective_height=8.0),
        _text_line("compact row three", (0.0, 104.0, 100.0, 112.0), 8, effective_height=8.0),
    ]
    if failure_mode == "too_few":
        compact_rows.pop()
    elif failure_mode == "narrow":
        compact_rows[-1].bbox = (0.0, 104.0, 40.0, 112.0)
    else:
        compact_rows[-1].bbox = (0.0, 114.0, 100.0, 122.0)
    lines.append(
        _text_line(
            "neutral label",
            (30.0, 70.0, 70.0, 78.0),
            5,
            effective_height=8.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
    )
    lines.extend(compact_rows)

    titles._classify_page_titles(lines, (100.0, 140.0), page_index=1, container_bboxes=[])

    assert lines[5].semantic_type is None


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

    titles._classify_page_titles(lines, (100.0, 200.0), page_index=0, container_bboxes=[])

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

    titles._classify_page_titles(
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

    titles._classify_page_titles(
        lines,
        (100.0, 140.0),
        page_index=1,
        container_bboxes=[],
    )
    merged = line_merging._merge_title_resolved_visual_rows(lines, (100.0, 140.0))

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

    titles._classify_page_titles(
        lines,
        (100.0, 100.0),
        page_index=1,
        container_bboxes=[],
    )
    merged = line_merging._merge_title_resolved_visual_rows(lines, (100.0, 100.0))

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

    titles._classify_page_titles(
        lines,
        (100.0, 120.0),
        page_index=1,
        container_bboxes=[],
    )
    blocks = text_blocks._build_text_blocks(lines, [], (100.0, 120.0))

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

    assert line_merging._is_dense_same_font_two_run_row(dense_members, (100.0, 100.0))
    assert not line_merging._is_dense_same_font_two_run_row(sparse_members, (100.0, 100.0))
    assert not line_merging._is_dense_same_font_two_run_row(partial_formula_row, (100.0, 100.0))
    assert not line_merging._is_dense_same_font_two_run_row(different_font_members, (100.0, 100.0))

    merged = line_merging._merge_title_resolved_visual_rows(
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
            titles._classify_page_titles,
            titles._build_physical_title_gap_map,
            titles._find_repeated_grid_title_suppressions,
            titles._infer_lane_body_profile,
            titles._classify_document_title,
            titles._document_title_uses_page_fallback,
            titles._classify_paragraph_titles_in_lane,
            titles._visual_row_has_body_style_sibling,
            titles._is_near_full_mixed_inline_row,
            titles._is_full_width_inline_heading,
            titles._has_following_body_row,
            titles._has_following_compact_text_section,
            titles._unify_visual_row_title_types,
            titles._protect_front_matter_title_types,
            titles._infer_front_matter_boundary,
            titles._normalized_title_gap,
            titles._line_near_visual_container,
            line_layout._title_fonts_compatible,
            titles._expand_paragraph_title_neighbors,
        )
    )

    assert ".text" not in source
