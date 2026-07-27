from __future__ import annotations


import pytest

from mineru.backend.flash.native_pdf import (
    formulas,
    geometry,
    line_merging,
    models,
)


from _flash_pdf_test_utils import (
    _text_line,
)


def _formula_member(
    text: str,
    bbox: tuple[float, float, float, float],
    source_index: int,
) -> tuple[models._LineItem, tuple[float, float, float, float]]:
    """构造公式块序列化测试使用的文本行及其局部几何。"""

    return (
        models._LineItem(
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

    block = formulas._formula_members_to_block(
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

    block = formulas._formula_members_to_block(
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

    block = formulas._formula_members_to_block(
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

    block = formulas._formula_members_to_block(
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

    block = formulas._formula_members_to_block(
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

    block = formulas._formula_members_to_block(
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

    block = formulas._formula_members_to_block(
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

    block = formulas._formula_members_to_block(
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
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[(line, line.bbox) for line in lane_lines],
    )

    anchors = formulas._find_formula_spatial_anchors(lane, 10.0)

    assert len(anchors) == 1
    assert anchors[0].line is number
    assert anchors[0].detached_below_body

    anchor_center = geometry._bbox_center_y(anchors[0].bbox)
    dominant_font = formulas._infer_formula_body_font(lane, 10.0)
    members = formulas._grow_formula_spatial_component(
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


def test_overlapping_denominator_cannot_become_short_formula_anchor() -> None:
    """验证与正文横向重叠的右缘分母字符不会成为非编号公式锚点。"""

    body_font = ("Body", 0)
    body_lines = [
        _text_line(
            f"body-{index}",
            (0.0, top, 92.0, top + 10.0),
            index,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index, top in enumerate((0.0, 12.0, 24.0))
    ]
    formula_body = _text_line(
        "Pr = cp mu",
        (25.0, 36.0, 90.0, 46.0),
        3,
        effective_height=10.0,
        font_signature=body_font,
        font_coverage=0.8,
    )
    denominator = _text_line(
        "k",
        (88.0, 38.0, 92.0, 45.0),
        4,
        effective_height=7.0,
        font_signature=("Math", 1),
        font_coverage=1.0,
    )
    lane = models._TextLane(
        left=0.0,
        right=92.0,
        lines=[
            *((line, line.bbox) for line in body_lines),
            (formula_body, formula_body.bbox),
            (denominator, denominator.bbox),
        ],
    )

    assert formulas._find_formula_spatial_anchors(lane, 10.0, body_font) == []


def test_compact_multiline_cluster_becomes_one_isolated_equation() -> None:
    """验证可提取的紧凑 F/G 多行簇形成单个公式块且不进入前后正文。"""

    body_font = ("Body", 0)
    body_lines = [
        _text_line(
            f"body-{index}",
            (0.0, top, 100.0, top + 10.0),
            index,
            visual_row_id=index,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index, top in enumerate((0.0, 12.0, 24.0))
    ]
    fragments = [
        _text_line(
            "F = numerator",
            (10.0, 36.0, 25.0, 49.0),
            3,
            visual_row_id=3,
            effective_height=8.0,
            font_signature=("Math", 1),
            font_coverage=0.5,
        ),
        _text_line(
            "denominator, G =",
            (20.0, 39.0, 40.0, 52.0),
            4,
            visual_row_id=4,
            split_from_row=True,
            effective_height=9.5,
            font_signature=("Math", 1),
            font_coverage=0.4,
        ),
        _text_line(
            "numerator",
            (46.0, 36.0, 52.0, 43.0),
            5,
            visual_row_id=4,
            split_from_row=True,
            effective_height=7.0,
            font_signature=("Math", 1),
            font_coverage=0.5,
        ),
        _text_line(
            "denominator",
            (46.0, 39.0, 53.0, 52.0),
            6,
            visual_row_id=5,
            effective_height=7.0,
            font_signature=("Math", 1),
            font_coverage=0.33,
        ),
    ]
    following = _text_line(
        "following body",
        (0.0, 54.0, 100.0, 64.0),
        7,
        visual_row_id=6,
        effective_height=10.0,
        font_signature=body_font,
        font_coverage=1.0,
    )

    merged = line_merging._merge_overlapping_inline_text_clusters(
        [*body_lines, *fragments, following],
        (120.0, 100.0),
        [],
    )
    compact_cluster = next(line for line in merged if line.source_index == 3)
    blocks, remaining = formulas._build_formula_like_blocks(
        merged,
        [],
        (120.0, 100.0),
    )

    assert compact_cluster.compact_formula_cluster
    assert blocks == [
        {
            "type": "equation",
            "bbox": (10.0, 36.0, 53.0, 52.0),
            "angle": 0,
            "content": "F = numerator denominator, G = numerator denominator",
        }
    ]
    assert compact_cluster not in remaining
