from __future__ import annotations

import inspect

import pytest

from mineru.backend.flash.native_pdf import (
    auxiliary_text,
    models,
    pipeline,
    text_blocks,
)


from _flash_pdf_test_utils import (
    _prepared_text_page,
    _text_line,
)


def test_page_footnote_uses_separator_and_stops_before_distant_footer_text() -> None:
    """验证单栏页脚注由页底横线触发，并在较大行间隙前停止扩展。"""

    lines = [
        _text_line("body one", (100.0, 100.0, 900.0, 110.0), 0),
        _text_line("body two", (100.0, 130.0, 900.0, 140.0), 1),
        _text_line("body three", (100.0, 160.0, 900.0, 170.0), 2),
        _text_line("note one", (100.0, 770.0, 500.0, 780.0), 3),
        _text_line("note two", (100.0, 788.0, 500.0, 798.0), 4),
        _text_line("note three", (100.0, 806.0, 500.0, 816.0), 5),
        _text_line("footer text", (100.0, 850.0, 600.0, 860.0), 6),
    ]
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))
    page.drawing_lines = [
        models._AxisLine(
            bbox=(100.0, 750.0, 260.0, 752.0),
            width=1.0,
            orientation="horizontal",
        ),
        models._AxisLine(
            bbox=(100.0, 750.5, 260.0, 752.5),
            width=1.0,
            orientation="horizontal",
        ),
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert [line.semantic_type for line in lines] == [
        None,
        None,
        None,
        "page_footnote",
        "page_footnote",
        "page_footnote",
        None,
    ]
    assert page.page_footnote_groups == [{3, 4, 5}]


def test_image_footnote_requires_image_rule_and_smaller_text() -> None:
    """验证图表脚注必须同时具备图片、下缘长横线和字号收缩证据。"""

    lines = [
        _text_line(
            f"body {index}",
            (50.0, 40.0 + 20.0 * index, 450.0, 50.0 + 20.0 * index),
            index,
            effective_height=10.0,
        )
        for index in range(4)
    ]
    note = _text_line(
        "chart source",
        (50.0, 325.0, 180.0, 333.0),
        4,
        effective_height=8.0,
    )
    lines.append(note)
    page = _prepared_text_page(*lines, page_size=(500.0, 500.0))
    page.fixed_blocks = [
        {
            "type": "image",
            "bbox": (50.0, 140.0, 300.0, 315.0),
            "angle": 0,
            "content": "",
        }
    ]
    page.drawing_lines = [
        models._AxisLine((50.0, 318.0, 300.0, 319.0), 1.0, "horizontal")
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert note.semantic_type == "footnote"
    assert page.page_footnote_groups == []


def test_sparse_image_footnotes_retry_with_document_body_height() -> None:
    """验证稀疏图片页可用全文正文尺度恢复两个被自身字号污染的图表脚注。"""

    caption = _text_line(
        "chart caption",
        (50.0, 40.0, 300.0, 48.0),
        0,
        effective_height=8.0,
    )
    notes = [
        _text_line(
            "left source",
            (50.0, 255.0, 300.0, 262.0),
            1,
            effective_height=7.0,
        ),
        _text_line(
            "right source",
            (550.0, 255.0, 800.0, 262.0),
            2,
            effective_height=7.0,
        ),
    ]
    page = _prepared_text_page(caption, *notes, page_size=(1000.0, 500.0))
    page.fixed_blocks = [
        {"type": "image", "bbox": bbox, "angle": 0, "content": ""}
        for bbox in [
            (50.0, 80.0, 450.0, 250.0),
            (550.0, 80.0, 950.0, 250.0),
        ]
    ]
    page.drawing_lines = [
        models._AxisLine((50.0, 252.0, 450.0, 253.0), 1.0, "horizontal"),
        models._AxisLine((550.0, 252.0, 950.0, 253.0), 1.0, "horizontal"),
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert [line.semantic_type for line in notes] == [None, None]

    auxiliary_text._classify_deferred_image_footnotes([page], body_height=10.0)

    assert [line.semantic_type for line in notes] == ["footnote", "footnote"]
    assert page.page_footnote_groups == []


@pytest.mark.parametrize(
    "missing_evidence",
    ["image", "rule", "small_text", "table_rule"],
)
def test_image_footnote_rejects_incomplete_visual_evidence(
    missing_evidence: str,
) -> None:
    """验证缺少任一联合证据时普通图下文字不会晋升为图表脚注。"""

    body = [
        _text_line(
            f"body {index}",
            (50.0, 40.0 + 20.0 * index, 450.0, 50.0 + 20.0 * index),
            index,
            effective_height=10.0,
        )
        for index in range(4)
    ]
    note = _text_line(
        "candidate",
        (50.0, 325.0, 180.0, 333.0),
        4,
        effective_height=10.0 if missing_evidence == "small_text" else 8.0,
    )
    page = _prepared_text_page(*body, note, page_size=(500.0, 500.0))
    if missing_evidence != "image":
        page.fixed_blocks = [
            {
                "type": "image",
                "bbox": (50.0, 140.0, 300.0, 315.0),
                "angle": 0,
                "content": "",
            }
        ]
    if missing_evidence != "rule":
        page.drawing_lines = [
            models._AxisLine((50.0, 318.0, 300.0, 319.0), 1.0, "horizontal")
        ]
    if missing_evidence == "table_rule":
        page.table_bboxes = [(40.0, 300.0, 310.0, 340.0)]

    auxiliary_text._classify_page_auxiliary_text(page)
    auxiliary_text._classify_deferred_image_footnotes([page], body_height=10.0)

    assert note.semantic_type is None


def test_page_footnote_supports_independent_column_rules() -> None:
    """验证左右栏各自的短横线只认领本栏连续页脚注。"""

    lines = [
        *[
            _text_line(f"left body {index}", (80.0, 100.0 + 30.0 * index, 460.0, 110.0 + 30.0 * index), index)
            for index in range(3)
        ],
        *[
            _text_line(
                f"right body {index}",
                (540.0, 100.0 + 30.0 * index, 920.0, 110.0 + 30.0 * index),
                index + 3,
            )
            for index in range(3)
        ],
        _text_line("left note one", (80.0, 820.0, 400.0, 830.0), 6),
        _text_line("left note two", (80.0, 838.0, 400.0, 848.0), 7),
        _text_line("right note one", (540.0, 820.0, 860.0, 830.0), 8),
        _text_line("right note two", (540.0, 838.0, 860.0, 848.0), 9),
    ]
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))
    page.drawing_lines = [
        models._AxisLine((80.0, 800.0, 220.0, 802.0), 1.0, "horizontal"),
        models._AxisLine((540.0, 800.0, 680.0, 802.0), 1.0, "horizontal"),
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert all(line.semantic_type is None for line in lines[:6])
    assert all(line.semantic_type == "page_footnote" for line in lines[6:])
    assert page.page_footnote_groups == [{6, 7}, {8, 9}]


def test_page_footnote_accepts_lower_half_column_width_rule_with_smaller_text() -> None:
    """验证页面下半部的栏宽横线可凭字号收缩识别单栏脚注。"""

    lines = [
        _text_line(
            f"body {index}",
            (100.0, 180.0 + 24.0 * index, 900.0, 190.0 + 24.0 * index),
            index,
            effective_height=10.0,
        )
        for index in range(4)
    ]
    lines.extend(
        [
            _text_line(
                "note one",
                (100.0, 620.0, 650.0, 628.0),
                4,
                effective_height=8.0,
            ),
            _text_line(
                "note two",
                (100.0, 632.0, 620.0, 640.0),
                5,
                effective_height=8.0,
            ),
        ]
    )
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))
    page.drawing_lines = [
        models._AxisLine((100.0, 600.0, 900.0, 602.0), 1.0, "horizontal")
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert all(line.semantic_type is None for line in lines[:4])
    assert [line.semantic_type for line in lines[4:]] == [
        "page_footnote",
        "page_footnote",
    ]
    assert page.page_footnote_groups == [{4, 5}]


def test_page_footnote_stops_at_next_aligned_column_rule() -> None:
    """验证同栏第二条分隔线会把贡献说明和作者单位拆成两个脚注组。"""

    lines = [
        *[
            _text_line(
                f"body {index}",
                (100.0, 150.0 + 30.0 * index, 500.0, 160.0 + 30.0 * index),
                index,
                effective_height=10.0,
            )
            for index in range(4)
        ],
        _text_line(
            "contribution one",
            (100.0, 590.0, 430.0, 598.0),
            4,
            effective_height=8.0,
        ),
        _text_line(
            "contribution two",
            (100.0, 602.0, 240.0, 610.0),
            5,
            effective_height=8.0,
        ),
        _text_line(
            "affiliation one",
            (120.0, 640.0, 460.0, 648.0),
            6,
            effective_height=8.0,
        ),
        _text_line(
            "affiliation two",
            (120.0, 652.0, 400.0, 660.0),
            7,
            effective_height=8.0,
        ),
    ]
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))
    page.drawing_lines = [
        models._AxisLine((100.0, 570.0, 500.0, 572.0), 1.0, "horizontal"),
        models._AxisLine((100.0, 620.0, 500.0, 622.0), 1.0, "horizontal"),
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert page.page_footnote_groups == [{4, 5}, {6, 7}]
    assert all(line.semantic_type == "page_footnote" for line in lines[4:])


def test_page_footnote_rejects_column_width_rule_without_size_contraction() -> None:
    """验证栏宽横线下的正常字号正文不会仅凭下半页位置变成脚注。"""

    lines = [
        _text_line(
            f"body {index}",
            (100.0, 180.0 + 24.0 * index, 900.0, 190.0 + 24.0 * index),
            index,
            effective_height=10.0,
        )
        for index in range(4)
    ]
    lines.append(
        _text_line(
            "continued body",
            (100.0, 620.0, 700.0, 630.0),
            4,
            effective_height=10.0,
        )
    )
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))
    page.drawing_lines = [
        models._AxisLine((100.0, 600.0, 900.0, 602.0), 1.0, "horizontal")
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert all(line.semantic_type is None for line in lines)


def test_page_footnote_unions_aligned_regular_and_span_lanes_only() -> None:
    """验证同左缘 regular/span 栏可联合认领，但不会吞掉右侧正文栏。"""

    lines = [
        *[
            _text_line(f"left body {index}", (100.0, 100.0 + 30.0 * index, 480.0, 110.0 + 30.0 * index), index)
            for index in range(3)
        ],
        *[
            _text_line(
                f"right body {index}",
                (520.0, 100.0 + 30.0 * index, 900.0, 110.0 + 30.0 * index),
                index + 3,
            )
            for index in range(3)
        ],
        _text_line("span note one", (100.0, 810.0, 900.0, 820.0), 6),
        _text_line("span note two", (100.0, 830.0, 900.0, 840.0), 7),
        _text_line("span note three", (100.0, 850.0, 900.0, 860.0), 8),
        _text_line("left note one", (100.0, 812.0, 480.0, 822.0), 9),
        _text_line("left note two", (100.0, 832.0, 480.0, 842.0), 10),
        _text_line("right body below rule", (520.0, 812.0, 900.0, 822.0), 11),
    ]
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))
    page.drawing_lines = [
        models._AxisLine((100.0, 800.0, 305.0, 802.0), 1.0, "horizontal")
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert {line.source_index for line in lines if line.semantic_type == "page_footnote"} == {
        6,
        7,
        8,
        9,
        10,
    }
    assert page.page_footnote_groups == [{6, 7, 8, 9, 10}]
    assert lines[11].semantic_type is None


def test_page_footnote_entries_split_first_line_indent_without_text() -> None:
    """验证首行缩进模式把两个脚注起始行拆开，并吸收左对齐续行。"""

    lines = [
        _text_line("first", (87.0, 732.0, 245.0, 742.0), 0, effective_height=8.0, median_glyph_width=4.5),
        _text_line("second", (83.0, 743.0, 293.0, 753.0), 1, effective_height=8.0, median_glyph_width=4.5),
        _text_line("continuation", (70.0, 755.0, 276.0, 763.0), 2, effective_height=7.0, median_glyph_width=5.0),
        _text_line("tail", (70.0, 766.0, 146.0, 773.0), 3, effective_height=7.0, median_glyph_width=5.0),
    ]

    entries = text_blocks._split_page_footnote_entries(
        lines,
        (595.0, 842.0),
    )

    assert [[line.source_index for line in entry] for entry in entries] == [
        [0],
        [1, 2, 3],
    ]


def test_page_footnote_entries_split_hanging_indent_and_tighten_boxes() -> None:
    """验证 Boosting 型续行缩进拆成四条脚注，且异常高字符框不再互相覆盖。"""

    page_size = (612.2833862304688, 858.8975830078125)
    lines = [
        _text_line(
            "receipt",
            (81.2767, 740.0383, 238.2958, 758.6347),
            66,
            effective_height=18.5964,
            median_glyph_width=3.92,
            semantic_type="page_footnote",
        ),
        _text_line(
            "fund start",
            (81.2767, 750.5583, 546.9993, 769.1547),
            67,
            effective_height=9.0,
            median_glyph_width=7.84,
            semantic_type="page_footnote",
        ),
        _text_line(
            "fund continuation",
            (119.6767, 761.0783, 364.8372, 779.6747),
            68,
            effective_height=10.38,
            median_glyph_width=4.78,
            semantic_type="page_footnote",
        ),
        _text_line(
            "author",
            (81.2767, 771.5983, 385.9195, 790.1947),
            69,
            effective_height=9.0,
            median_glyph_width=7.84,
            semantic_type="page_footnote",
        ),
        _text_line(
            "corresponding",
            (81.2767, 782.1183, 439.2661, 800.7147),
            70,
            effective_height=9.0,
            median_glyph_width=7.84,
            semantic_type="page_footnote",
        ),
    ]
    page = _prepared_text_page(*lines, page_size=page_size)
    page.page_footnote_groups = [{line.source_index for line in lines}]

    blocks = [
        block
        for block in pipeline._finalize_prepared_page(page, page_index=0)
        if block["type"] == "page_footnote"
    ]

    assert [block["content"] for block in blocks] == [
        "receipt",
        "fund start fund continuation",
        "author",
        "corresponding",
    ]
    assert [block["bbox"] for block in blocks] == [
        [0.133, 0.867, 0.389, 0.878],
        [0.133, 0.879, 0.893, 0.902],
        [0.133, 0.904, 0.63, 0.914],
        [0.133, 0.916, 0.717, 0.927],
    ]
    assert all(
        previous["bbox"][3] < current["bbox"][1]
        for previous, current in zip(blocks, blocks[1:])
    )


@pytest.mark.parametrize(
    ("rule_bbox", "table_bboxes"),
    [
        ((100.0, 750.0, 125.0, 752.0), []),
        ((400.0, 750.0, 550.0, 752.0), []),
        ((100.0, 750.0, 260.0, 752.0), [(90.0, 730.0, 500.0, 850.0)]),
        (None, []),
    ],
)
def test_page_footnote_rejects_decorative_formula_table_and_missing_rules(
    rule_bbox: tuple[float, float, float, float] | None,
    table_bboxes: list[tuple[float, float, float, float]],
) -> None:
    """验证装饰短线、居中公式线、表格线和无横线页底正文均不触发脚注。"""

    lines = [
        _text_line("body one", (100.0, 100.0, 900.0, 110.0), 0),
        _text_line("body two", (100.0, 130.0, 900.0, 140.0), 1),
        _text_line("body three", (100.0, 160.0, 900.0, 170.0), 2),
        _text_line("bottom body", (100.0, 770.0, 500.0, 780.0), 3),
    ]
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))
    page.table_bboxes = table_bboxes
    if rule_bbox is not None:
        page.drawing_lines = [models._AxisLine(rule_bbox, 1.0, "horizontal")]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert all(line.semantic_type is None for line in lines)


def test_page_footnote_rejects_collinear_rule_segment_next_to_table() -> None:
    """验证表格框外的同高近邻断裂横线不会独立触发页脚注。"""

    lines = [
        _text_line("body one", (100.0, 100.0, 900.0, 110.0), 0),
        _text_line("body two", (100.0, 130.0, 900.0, 140.0), 1),
        _text_line("body three", (100.0, 160.0, 900.0, 170.0), 2),
        _text_line("bottom table row", (100.0, 770.0, 500.0, 780.0), 3),
    ]
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))
    page.table_bboxes = [(280.0, 730.0, 900.0, 850.0)]
    page.drawing_lines = [
        models._AxisLine((100.0, 750.0, 260.0, 752.0), 1.0, "horizontal"),
        models._AxisLine((280.0, 750.0, 900.0, 752.0), 1.0, "horizontal"),
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert all(line.semantic_type is None for line in lines)
    assert page.page_footnote_groups == []


@pytest.mark.parametrize(
    ("angle", "bbox"),
    [
        (270, (20.0, 250.0, 50.0, 650.0)),
        (90, (930.0, 250.0, 960.0, 650.0)),
    ],
)
def test_aside_text_accepts_tall_vertical_text_in_either_edge_band(
    angle: int,
    bbox: tuple[float, float, float, float],
) -> None:
    """验证横排正文占主导时，左右边缘的高占比垂直文字均标为侧栏。"""

    lines = [
        *[
            _text_line(f"body {index}", (100.0, 100.0 + 30.0 * index, 900.0, 110.0 + 30.0 * index), index)
            for index in range(6)
        ],
        _text_line("aside", bbox, 6, angle=angle, effective_height=20.0),
    ]
    page = _prepared_text_page(*lines, page_size=(1000.0, 1000.0))

    auxiliary_text._classify_page_auxiliary_text(page)

    assert lines[-1].semantic_type == "aside_text"
    assert all(line.semantic_type is None for line in lines[:-1])


def test_aside_text_rejects_short_internal_wide_and_non_dominant_rotated_text() -> None:
    """验证短旋转行、页内旋转行、过宽边缘行及旋转正文均不误报侧栏。"""

    upright_lines = [
        _text_line(f"body {index}", (100.0, 100.0 + 30.0 * index, 900.0, 110.0 + 30.0 * index), index)
        for index in range(10)
    ]
    rejected = [
        _text_line("short", (20.0, 250.0, 50.0, 350.0), 10, angle=270, effective_height=20.0),
        _text_line("internal", (400.0, 200.0, 430.0, 600.0), 11, angle=270, effective_height=20.0),
        _text_line("wide", (0.0, 200.0, 100.0, 600.0), 12, angle=270, effective_height=20.0),
    ]
    page = _prepared_text_page(*upright_lines, *rejected, page_size=(1000.0, 1000.0))
    auxiliary_text._classify_page_auxiliary_text(page)
    assert all(line.semantic_type is None for line in rejected)

    rotated_body = [
        _text_line(f"upright {index}", (100.0, 100.0 + 20.0 * index, 300.0, 110.0 + 20.0 * index), index)
        for index in range(4)
    ]
    rotated_body.append(
        _text_line("edge but not aside", (20.0, 200.0, 50.0, 700.0), 4, angle=270, effective_height=20.0)
    )
    rotated_page = _prepared_text_page(*rotated_body, page_size=(1000.0, 1000.0))
    auxiliary_text._classify_page_auxiliary_text(rotated_page)
    assert rotated_body[-1].semantic_type is None


def test_auxiliary_text_classification_is_content_independent() -> None:
    """验证替换全部行文本不会改变页脚注空间分类结果。"""

    geometries = [
        (100.0, 100.0, 900.0, 110.0),
        (100.0, 130.0, 900.0, 140.0),
        (100.0, 160.0, 900.0, 170.0),
        (100.0, 770.0, 500.0, 780.0),
        (100.0, 788.0, 500.0, 798.0),
    ]
    first_lines = [_text_line(f"alpha {index}", bbox, index) for index, bbox in enumerate(geometries)]
    second_lines = [_text_line(f"完全不同 {index}", bbox, index) for index, bbox in enumerate(geometries)]
    first_page = _prepared_text_page(*first_lines, page_size=(1000.0, 1000.0))
    second_page = _prepared_text_page(*second_lines, page_size=(1000.0, 1000.0))
    for page in (first_page, second_page):
        page.drawing_lines = [
            models._AxisLine((100.0, 750.0, 260.0, 752.0), 1.0, "horizontal")
        ]
        auxiliary_text._classify_page_auxiliary_text(page)

    assert [line.semantic_type for line in first_lines] == [
        line.semantic_type for line in second_lines
    ]
    assert first_page.page_footnote_groups == second_page.page_footnote_groups


def test_auxiliary_text_classifiers_do_not_read_line_text() -> None:
    """静态守卫侧栏和页脚注分类函数不访问文本内容。"""

    source = "\n".join(
        inspect.getsource(function)
        for function in (
            auxiliary_text._classify_page_auxiliary_text,
            auxiliary_text._classify_aside_text,
            auxiliary_text._geometric_text_support_by_angle,
            auxiliary_text._classify_image_footnotes,
            auxiliary_text._classify_deferred_image_footnotes,
            auxiliary_text._image_footnote_members,
            auxiliary_text._classify_page_footnotes,
            auxiliary_text._augment_footnote_groups_with_edge_markers,
            auxiliary_text._classify_rule_delimited_headers,
            auxiliary_text._classify_page_number_outer_companions,
            auxiliary_text._rule_belongs_to_confirmed_table,
            auxiliary_text._merge_overlapping_source_groups,
            auxiliary_text._footnote_lane_members,
            text_blocks._split_page_footnote_entries,
            text_blocks._tight_page_footnote_bboxes,
        )
    )

    assert ".text" not in source


def test_finalize_preserves_preclassified_auxiliary_text_types() -> None:
    """验证单页终结阶段不会丢失已标注的侧栏和页脚注。"""

    page = _prepared_text_page(
        _text_line("note", (10.0, 80.0, 40.0, 90.0), 0, semantic_type="page_footnote"),
        _text_line("chart note", (10.0, 65.0, 40.0, 75.0), 2, semantic_type="footnote"),
        _text_line(
            "aside",
            (2.0, 20.0, 5.0, 60.0),
            1,
            angle=270,
            effective_height=3.0,
            semantic_type="aside_text",
        ),
    )

    blocks = pipeline._finalize_prepared_page(page, page_index=0)

    assert {block["type"] for block in blocks} == {
        "footnote",
        "page_footnote",
        "aside_text",
    }


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

    auxiliary_text._classify_repeated_page_marginals(pages)

    assert [[line.semantic_type for line in page.remaining_lines] for page in pages] == [
        ["header", None, None, None, "page_number", "footer"],
        ["header", None, None, "page_number", "footer"],
        ["header", None, None, "page_number", "footer"],
    ]


def test_top_rule_marks_only_unclassified_text_above_it_as_header() -> None:
    """验证页码先保留类型，长横线上方其余文本补标页眉且线下正文不变。"""

    page_number = _text_line(
        "7",
        (900.0, 30.0, 920.0, 40.0),
        0,
        semantic_type="page_number",
    )
    journal = _text_line("journal", (100.0, 30.0, 600.0, 40.0), 1)
    doi = _text_line("doi", (100.0, 44.0, 500.0, 54.0), 2)
    body = _text_line("body", (100.0, 80.0, 700.0, 90.0), 3)
    page = _prepared_text_page(
        page_number,
        journal,
        doi,
        body,
        page_size=(1000.0, 1000.0),
    )
    page.drawing_lines = [
        models._AxisLine((80.0, 60.0, 920.0, 62.0), 1.0, "horizontal")
    ]

    auxiliary_text._classify_rule_delimited_headers([page])

    assert [line.semantic_type for line in page.remaining_lines] == [
        "page_number",
        "header",
        "header",
        None,
    ]


def test_top_rule_inside_graphic_does_not_mark_header() -> None:
    """验证图形容器内的页首长横线不会把其上方普通文本误标为页眉。"""

    upper_text = _text_line("upper text", (100.0, 30.0, 600.0, 40.0), 0)
    body = _text_line("body", (100.0, 80.0, 700.0, 90.0), 1)
    page = _prepared_text_page(
        upper_text,
        body,
        page_size=(1000.0, 1000.0),
    )
    page.fixed_blocks = [
        {
            "type": "image",
            "bbox": (50.0, 50.0, 950.0, 70.0),
            "angle": 0,
            "content": "",
        }
    ]
    page.drawing_lines = [
        models._AxisLine((80.0, 60.0, 920.0, 62.0), 1.0, "horizontal")
    ]

    auxiliary_text._classify_rule_delimited_headers([page])

    assert upper_text.semantic_type is None
    assert body.semantic_type is None


def test_page_number_outer_companions_classify_top_and_bottom_text_and_images() -> None:
    """验证上下页码外侧的文本和空内容图片对称转换为页眉页脚。"""

    top_page = _prepared_text_page(
        _text_line("top visual", (20.0, 1.0, 80.0, 5.0), 0),
        _text_line(
            "1",
            (90.0, 8.0, 95.0, 13.0),
            1,
            semantic_type="page_number",
        ),
        _text_line("top body", (20.0, 18.0, 80.0, 28.0), 2),
    )
    top_page.fixed_blocks = [
        {
            "type": "image",
            "bbox": (40.0, 0.0, 60.0, 5.0),
            "angle": 0,
            "content": "",
        }
    ]
    bottom_page = _prepared_text_page(
        _text_line("bottom body", (20.0, 70.0, 80.0, 80.0), 0),
        _text_line(
            "2",
            (90.0, 82.0, 95.0, 87.0),
            1,
            semantic_type="page_number",
        ),
        _text_line("bottom visual", (20.0, 91.0, 80.0, 96.0), 2),
    )
    bottom_page.fixed_blocks = [
        {
            "type": "image",
            "bbox": (40.0, 90.0, 60.0, 99.0),
            "angle": 0,
            "content": "",
        }
    ]

    auxiliary_text._classify_page_number_outer_companions(
        [top_page, bottom_page]
    )

    assert [line.semantic_type for line in top_page.remaining_lines] == [
        "header",
        "page_number",
        None,
    ]
    assert top_page.fixed_blocks[0]["type"] == "header"
    assert [line.semantic_type for line in bottom_page.remaining_lines] == [
        None,
        "page_number",
        "footer",
    ]
    assert bottom_page.fixed_blocks[0]["type"] == "footer"

    blocks = pipeline._finalize_prepared_page(bottom_page, page_index=1)
    assert any(
        block["type"] == "footer" and block["content"] == ""
        for block in blocks
    )


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

    auxiliary_text._classify_repeated_page_marginals(pages)

    assert [page.remaining_lines[0].semantic_type for page in pages] == ["page_number"] * 3


def test_single_page_marginal_content_remains_text() -> None:
    """验证单页顶部和底部文字不会仅凭位置被猜成页眉、页脚或页码。"""

    page = _prepared_text_page(
        _text_line("Page 7", (45.0, 3.0, 55.0, 8.0), 0),
        _text_line("Copyright notice", (20.0, 92.0, 80.0, 98.0), 1),
    )

    auxiliary_text._classify_repeated_page_marginals([page])

    assert [line.semantic_type for line in page.remaining_lines] == [None, None]


def test_extreme_page_footnotes_can_be_overridden_by_repeated_marginals() -> None:
    """验证只有极底脚注可凭跨页重复和递增证据改判为页脚、页码。"""

    pages = []
    for page_index in range(3):
        pages.append(
            _prepared_text_page(
                _text_line(
                    "stable footer",
                    (20.0, 95.0, 70.0, 99.0),
                    0,
                    effective_height=4.0,
                    font_signature=("Margin", 0),
                    font_coverage=1.0,
                    semantic_type="page_footnote",
                ),
                _text_line(
                    str(page_index + 10),
                    (90.0, 95.0, 95.0, 99.0),
                    1,
                    effective_height=4.0,
                    font_signature=("Margin", 0),
                    font_coverage=1.0,
                    semantic_type="page_footnote",
                ),
                _text_line(
                    "local note",
                    (20.0, 86.0, 60.0, 90.0),
                    2,
                    semantic_type="page_footnote",
                ),
            )
        )

    auxiliary_text._classify_repeated_page_marginals(pages)

    assert [
        [line.semantic_type for line in page.remaining_lines]
        for page in pages
    ] == [["footer", "page_number", "page_footnote"]] * 3


def test_single_page_compound_header_requires_small_split_row_and_body_edge() -> None:
    """验证单页顶部的小字号拆分同行可由正文栏右缘确认为页眉。"""

    header_name = _text_line(
        "journal",
        (60.0, 2.0, 85.0, 4.0),
        0,
        visual_row_id=7,
        run_index=0,
        split_from_row=True,
        effective_height=2.0,
    )
    header_number = _text_line(
        "5",
        (96.0, 2.0, 98.0, 4.0),
        1,
        visual_row_id=7,
        run_index=1,
        split_from_row=True,
        effective_height=2.0,
    )
    page = _prepared_text_page(
        header_name,
        header_number,
        *[
            _text_line(
                f"body-{index}",
                (50.0, 10.0 + 12.0 * index, 98.0, 20.0 + 12.0 * index),
                2 + index,
                effective_height=10.0,
            )
            for index in range(4)
        ],
    )

    auxiliary_text._classify_single_page_compound_headers([page])

    assert header_name.semantic_type == "header"
    assert header_number.semantic_type == "header"


def test_isolated_first_page_footer_uses_multi_page_geometry_only() -> None:
    """验证多页首页唯一极底短行可补标页脚，而单页相同布局保持正文。"""

    def build_page() -> tuple[models._PreparedPage, models._LineItem]:
        """构造带四行下延正文和一个孤立极底候选的页面。"""

        footer = _text_line(
            "neutral notice",
            (37.5, 98.0, 62.5, 100.0),
            4,
            effective_height=2.0,
        )
        return (
            _prepared_text_page(
                *[
                    _text_line(
                        f"body-{index}",
                        (0.0, 55.0 + 10.0 * index, 100.0, 60.0 + 10.0 * index),
                        index,
                        effective_height=5.0,
                    )
                    for index in range(4)
                ],
                footer,
            ),
            footer,
        )

    multi_page, multi_footer = build_page()
    single_page, single_footer = build_page()

    auxiliary_text._classify_isolated_first_page_footer(
        [multi_page, _prepared_text_page()]
    )
    auxiliary_text._classify_isolated_first_page_footer([single_page])

    assert multi_footer.semantic_type == "footer"
    assert single_footer.semantic_type is None


def test_repeated_visual_headers_use_geometry_and_skip_first_page() -> None:
    """验证重复页首图片仅按几何重标，空 content 也可保留为 header。"""

    pages = [_prepared_text_page(page_size=(100.0, 100.0)) for _ in range(5)]
    contents = ["cover", "", "beta", "", "delta"]
    for page, content in zip(pages, contents, strict=True):
        page.fixed_blocks = [
            {
                "type": "image",
                "bbox": (10.0, 0.0, 90.0, 10.0),
                "angle": 0,
                "content": content,
            }
        ]

    auxiliary_text._classify_repeated_visual_headers(pages)

    assert pages[0].fixed_blocks[0]["type"] == "image"
    assert [page.fixed_blocks[0]["type"] for page in pages[1:]] == ["header"] * 4


def test_repeated_visual_headers_require_three_top_geometry_matches() -> None:
    """验证两页重复、非页首图片和明显漂移的 bbox 均不形成视觉页眉。"""

    pages = [_prepared_text_page(page_size=(100.0, 100.0)) for _ in range(6)]
    bboxes = [
        (10.0, 0.0, 90.0, 10.0),
        (10.0, 0.0, 90.0, 10.0),
        (10.0, 0.0, 90.0, 10.0),
        (10.0, 20.0, 90.0, 30.0),
        (30.0, 0.0, 90.0, 10.0),
        (10.0, 0.0, 90.0, 10.0),
    ]
    for page, bbox in zip(pages, bboxes, strict=True):
        page.fixed_blocks = [
            {
                "type": "image",
                "bbox": bbox,
                "angle": 0,
                "content": "value",
            }
        ]

    auxiliary_text._classify_repeated_visual_headers(pages)

    assert [page.fixed_blocks[0]["type"] for page in pages] == ["image"] * 6


def test_repeated_visual_headers_support_alternating_pages() -> None:
    """验证同奇偶页间隔为二的重复图片仍可形成视觉页眉簇。"""

    pages = [_prepared_text_page(page_size=(100.0, 100.0)) for _ in range(7)]
    for page_index, page in enumerate(pages):
        if page_index in {2, 4, 6}:
            page.fixed_blocks = [
                {
                    "type": "image",
                    "bbox": (10.0, 0.0, 90.0, 10.0),
                    "angle": 0,
                    "content": f"page-{page_index}",
                }
            ]

    auxiliary_text._classify_repeated_visual_headers(pages)

    assert [pages[index].fixed_blocks[0]["type"] for index in (2, 4, 6)] == ["header"] * 3


def test_repeated_visual_headers_require_matching_orientation() -> None:
    """验证 bbox 相同但方向不一致的三页图片不会形成同一视觉页眉簇。"""

    pages = [_prepared_text_page(page_size=(100.0, 100.0)) for _ in range(4)]
    for page_index, page in enumerate(pages[1:], start=1):
        page.fixed_blocks = [
            {
                "type": "image",
                "bbox": (10.0, 0.0, 90.0, 10.0),
                "angle": 90 if page_index == 3 else 0,
                "content": f"page-{page_index}",
            }
        ]

    auxiliary_text._classify_repeated_visual_headers(pages)

    assert [page.fixed_blocks[0]["type"] for page in pages[1:]] == ["image"] * 3


def test_image_bottom_border_does_not_create_image_or_page_footnote() -> None:
    """验证图片下沿自身的坐标轴横线不会把紧邻图中文字判成脚注。"""

    body = [
        _text_line(
            f"body {index}",
            (50.0, 40.0 + 20.0 * index, 450.0, 50.0 + 20.0 * index),
            index,
            effective_height=10.0,
        )
        for index in range(4)
    ]
    axis_label = _text_line(
        "T -1/4",
        (150.0, 317.0, 230.0, 325.0),
        4,
        effective_height=8.0,
    )
    page = _prepared_text_page(
        *body,
        axis_label,
        page_size=(500.0, 500.0),
    )
    page.fixed_blocks = [
        {
            "type": "image",
            "bbox": (50.0, 140.0, 300.0, 315.0),
            "angle": 0,
            "content": "",
        }
    ]
    page.drawing_lines = [
        models._AxisLine(
            (50.0, 314.0, 300.0, 315.0),
            1.0,
            "horizontal",
        )
    ]

    auxiliary_text._classify_page_auxiliary_text(page)

    assert axis_label.semantic_type is None
    assert page.page_footnote_groups == []


def test_two_bottom_rules_classify_centered_url_as_footer() -> None:
    """验证页面底部两条同跨度横线把其间居中网址标为页脚。"""

    body = [
        _text_line(
            f"body {index}",
            (100.0, 100.0 + 30.0 * index, 900.0, 110.0 + 30.0 * index),
            index,
            effective_height=10.0,
        )
        for index in range(4)
    ]
    url = _text_line(
        "https://example.test/journal",
        (350.0, 920.0, 650.0, 930.0),
        10,
        effective_height=10.0,
    )
    page = _prepared_text_page(
        *body,
        url,
        page_size=(1000.0, 1000.0),
    )
    page.drawing_lines = [
        models._AxisLine((300.0, 900.0, 700.0, 901.0), 1.0, "horizontal"),
        models._AxisLine((300.0, 950.0, 700.0, 951.0), 1.0, "horizontal"),
    ]

    auxiliary_text._classify_rule_delimited_footers([page])

    assert url.semantic_type == "footer"


def test_split_footer_row_fragments_inherit_stable_anchor_type() -> None:
    """验证同一页脚视觉行的左右碎片从稳定锚点继承 footer 类型。"""

    fragments = [
        _text_line(
            "journal",
            (100.0, 930.0, 300.0, 940.0),
            0,
            visual_row_id=8,
            split_from_row=True,
        ),
        _text_line(
            "Vol. 1",
            (420.0, 930.0, 500.0, 940.0),
            1,
            visual_row_id=8,
            split_from_row=True,
            semantic_type="footer",
        ),
        _text_line(
            "No. 2",
            (650.0, 930.0, 750.0, 940.0),
            2,
            visual_row_id=8,
            split_from_row=True,
        ),
    ]
    page = _prepared_text_page(
        *fragments,
        page_size=(1000.0, 1000.0),
    )

    auxiliary_text._classify_split_marginal_row_companions([page])

    assert [line.semantic_type for line in fragments] == ["footer"] * 3


def test_distant_same_row_text_inherits_top_page_number_header_type() -> None:
    """验证页码与远距运行标题同基线时不依赖水平距离也能标为页眉。"""

    page_number = _text_line(
        "8",
        (10.0, 10.0, 15.0, 15.0),
        0,
        semantic_type="page_number",
    )
    running_title = _text_line(
        "running title",
        (60.0, 10.0, 90.0, 15.0),
        1,
    )
    page = _prepared_text_page(
        page_number,
        running_title,
        page_size=(100.0, 100.0),
    )

    auxiliary_text._classify_page_number_outer_companions([page])

    assert page_number.semantic_type == "page_number"
    assert running_title.semantic_type == "header"
