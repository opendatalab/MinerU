from __future__ import annotations


from dataclasses import replace

import pytest

from mineru.backend.flash.native_pdf import (
    formulas,
    geometry,
    line_merging,
    models,
)
from mineru.utils.pdf_document import PDFPathInfo


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


def _vector_path(
    bbox: tuple[float, float, float, float],
    source_index: int,
    *,
    segment_count: int = 16,
    fill_visible: bool = True,
    stroke_visible: bool = False,
    form_depth: int = 0,
) -> PDFPathInfo:
    """构造矢量公式检测测试使用的 Path 信息。"""

    return PDFPathInfo(
        bbox=bbox,
        segment_count=segment_count,
        fill_visible=fill_visible,
        stroke_visible=stroke_visible,
        form_depth=form_depth,
        source_index=source_index,
    )


def _vector_formula_body_paths(
    *,
    left: float = 20.0,
    top: float = 50.0,
    source_start: int = 0,
) -> list[PDFPathInfo]:
    """构造满足复杂度和尺寸约束的六字形矢量公式主体。"""

    return [
        _vector_path(
            (left + index * 6.0, top, left + index * 6.0 + 4.0, top + 12.0),
            source_start + index,
        )
        for index in range(6)
    ]


def _vector_formula_source(
    *path_infos: PDFPathInfo,
    page_size: tuple[float, float] = (100.0, 120.0),
    extra_lines: list[models._LineItem] | None = None,
) -> models._PageSource:
    """构造具有稳定正文栏带且公式所在高度留白的页面源。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            f"body-{index}",
            (0.0, top, 100.0, top + 10.0),
            index,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index, top in enumerate((0.0, 20.0, 80.0, 100.0))
    ]
    lines.extend(extra_lines or [])
    return models._PageSource(
        page_size=page_size,
        lines=lines,
        chars=[],
        drawing_lines=[],
        path_infos=list(path_infos),
    )


def test_vector_formula_paths_and_detached_path_number_form_one_empty_equation() -> None:
    """验证矢量主体与远距栏右缘路径编号形成一个空内容公式。"""

    body_paths = _vector_formula_body_paths()
    number_paths = [
        _vector_path((91.0 + index * 3.0, 51.0, 93.0 + index * 3.0, 60.0), 10 + index)
        for index in range(3)
    ]
    blocks, claimed = formulas._build_vector_formula_blocks(
        _vector_formula_source(*body_paths, *number_paths),
        [],
        set(),
    )

    assert claimed == set()
    assert blocks == [
        {
            "type": "equation",
            "bbox": (19.0, 49.0, 100.0, 63.0),
            "angle": 0,
            "content": "",
        }
    ]


def test_vector_formula_claims_text_number_but_keeps_content_empty() -> None:
    """验证可提取的独立编号并入路径公式并唯一认领，但不充当公式正文。"""

    number = _text_line("(12)", (91.0, 51.0, 99.0, 60.0), 20, effective_height=9.0)
    blocks, claimed = formulas._build_vector_formula_blocks(
        _vector_formula_source(*_vector_formula_body_paths(), extra_lines=[number]),
        [],
        set(),
    )

    assert claimed == {20}
    assert blocks[0]["type"] == "equation"
    assert blocks[0]["content"] == ""
    assert blocks[0]["bbox"] == pytest.approx((19.0, 49.0, 100.0, 63.0))


def test_vector_formula_rejects_unmatched_number_rules_strokes_forms_and_inline_paths() -> None:
    """验证无主体编号、细规则、描边、Form 图标和正文同行路径均不会误报。"""

    unmatched_number = [
        _vector_path((91.0 + index * 3.0, 51.0, 93.0 + index * 3.0, 60.0), index)
        for index in range(3)
    ]
    rules = [
        _vector_path((10.0 + index * 12.0, 70.0, 20.0 + index * 12.0, 70.5), 10 + index, segment_count=5)
        for index in range(6)
    ]
    excluded = [
        _vector_path((20.0, 50.0, 24.0, 62.0), 30, stroke_visible=True),
        _vector_path((26.0, 50.0, 30.0, 62.0), 31, form_depth=1),
    ]
    inline_paths = _vector_formula_body_paths(top=20.0, source_start=40)
    blocks, claimed = formulas._build_vector_formula_blocks(
        _vector_formula_source(*unmatched_number, *rules, *excluded, *inline_paths),
        [],
        set(),
    )

    assert blocks == []
    assert claimed == set()


def test_vector_formula_respects_columns_and_existing_containers() -> None:
    """验证同高双栏主体不互连，且高优先级容器覆盖的主体被排除。"""

    left_lines = [
        _text_line(f"left-{index}", (0.0, top, 100.0, top + 10.0), index, effective_height=10.0)
        for index, top in enumerate((0.0, 20.0, 80.0, 100.0))
    ]
    right_lines = [
        _text_line(
            f"right-{index}",
            (120.0, top, 220.0, top + 10.0),
            10 + index,
            effective_height=10.0,
        )
        for index, top in enumerate((0.0, 20.0, 80.0, 100.0))
    ]
    source = models._PageSource(
        page_size=(220.0, 120.0),
        lines=[*left_lines, *right_lines],
        chars=[],
        drawing_lines=[],
        path_infos=[
            *_vector_formula_body_paths(left=20.0, source_start=0),
            *_vector_formula_body_paths(left=140.0, source_start=20),
        ],
    )

    column_blocks, column_claimed = formulas._build_vector_formula_blocks(
        source,
        [],
        set(),
    )
    blocks, claimed = formulas._build_vector_formula_blocks(
        source,
        [{"type": "image", "bbox": (130.0, 45.0, 180.0, 67.0), "content": ""}],
        set(),
    )

    assert column_claimed == set()
    assert len(column_blocks) == 2
    assert column_blocks[0]["bbox"] == pytest.approx((19.0, 49.0, 55.0, 63.0))
    assert column_blocks[1]["bbox"] == pytest.approx((139.0, 49.0, 175.0, 63.0))
    assert claimed == set()
    assert len(blocks) == 1
    assert blocks[0]["bbox"] == pytest.approx((19.0, 49.0, 55.0, 63.0))


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


def test_formula_above_dense_body_collects_math_but_stops_at_title_barrier() -> None:
    """验证正文密集区上方的公式可聚合，且不会吸收紧邻的章节标题。"""

    body_font = ("Body", 0)
    formula_lines = [
        _text_line(
            "formula numerator",
            (20.0, 8.0, 58.0, 18.0),
            0,
            effective_height=10.0,
            font_signature=("Math", 0),
            font_coverage=0.6,
        ),
        _text_line(
            "formula denominator",
            (25.0, 19.0, 55.0, 29.0),
            1,
            effective_height=10.0,
            font_signature=("Math", 0),
            font_coverage=0.6,
        ),
    ]
    number = _text_line("(5)", (91.0, 17.0, 100.0, 27.0), 2, effective_height=10.0)
    heading = _text_line(
        "neutral section heading",
        (0.0, 28.0, 40.0, 42.0),
        3,
        effective_height=14.0,
        font_signature=("Heading", 0),
        font_coverage=1.0,
    )
    body_lines = [
        _text_line(
            f"body-{index}",
            (0.0, top, 100.0, top + 10.0),
            4 + index,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index, top in enumerate((44.0, 56.0, 68.0, 80.0))
    ]
    lane = models._TextLane(
        left=0.0,
        right=100.0,
        lines=[
            *((line, line.bbox) for line in formula_lines),
            (number, number.bbox),
            (heading, heading.bbox),
            *((line, line.bbox) for line in body_lines),
        ],
    )

    anchors = formulas._find_formula_spatial_anchors(lane, 10.0, body_font)

    assert len(anchors) == 1
    assert anchors[0].detached_above_body
    members = formulas._grow_formula_spatial_component(
        lane,
        anchors[0],
        0.0,
        100.0,
        set(),
        [],
        body_font,
        10.0,
    )

    assert {line.text for line, _bbox in members} == {
        "formula numerator",
        "formula denominator",
        "(5)",
    }


def test_formula_number_cannot_upgrade_ordinary_body_row() -> None:
    """验证括号编号缺少独立公式主体时不能把普通正文升级为公式。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            f"body-{index}",
            (0.0, top, 100.0, top + 10.0),
            index,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index, top in enumerate((0.0, 12.0, 48.0, 60.0))
    ]
    ordinary = _text_line(
        "ordinary body row",
        (0.0, 30.0, 70.0, 40.0),
        4,
        effective_height=10.0,
        font_signature=body_font,
        font_coverage=1.0,
    )
    number = _text_line("(9)", (91.0, 30.0, 100.0, 40.0), 5, effective_height=10.0)

    blocks, remaining = formulas._build_formula_like_blocks(
        [*lines, ordinary, number],
        [],
        (100.0, 80.0),
    )

    assert blocks == []
    assert ordinary in remaining
    assert number in remaining


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


def _build_compact_margin_case(
    candidate_bbox: tuple[float, float, float, float],
    above_bbox: tuple[float, float, float, float],
    below_bbox: tuple[float, float, float, float],
) -> tuple[
    models._LineItem,
    list[dict[str, object]],
    list[models._LineItem],
]:
    """构造上下都有稳定正文行的紧凑公式页边用例。"""

    body_font = ("Body", 0)
    above = _text_line(
        "body above",
        above_bbox,
        0,
        effective_height=10.0,
        font_signature=body_font,
        font_coverage=1.0,
    )
    candidate = replace(
        _text_line(
            "formula cluster",
            candidate_bbox,
            1,
            effective_height=10.0,
            font_signature=("Math", 1),
            font_coverage=0.5,
        ),
        compact_formula_cluster=True,
    )
    below = _text_line(
        "body below",
        below_bbox,
        2,
        effective_height=10.0,
        font_signature=body_font,
        font_coverage=1.0,
    )
    blocks, remaining = formulas._build_formula_like_blocks(
        [above, candidate, below],
        [],
        (100.0, 1000.0),
    )
    return candidate, blocks, remaining


def _build_text_formula_margin_case(
    formula_bboxes: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ],
    body_tops: tuple[float, float, float, float],
) -> tuple[list[dict[str, object]], list[models._LineItem]]:
    """构造带右缘编号和稳定正文栏带的文本公式页边用例。"""

    body_font = ("Body", 0)
    formula_lines = [
        _text_line(
            "formula numerator",
            formula_bboxes[0],
            0,
            effective_height=10.0,
            font_signature=("Math", 0),
            font_coverage=0.6,
        ),
        _text_line(
            "formula denominator",
            formula_bboxes[1],
            1,
            effective_height=10.0,
            font_signature=("Math", 0),
            font_coverage=0.6,
        ),
        _text_line("(5)", formula_bboxes[2], 2, effective_height=10.0),
    ]
    body_lines = [
        _text_line(
            f"body-{index}",
            (0.0, top, 100.0, top + 10.0),
            3 + index,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index, top in enumerate(body_tops)
    ]
    return formulas._build_formula_like_blocks(
        [*formula_lines, *body_lines],
        [],
        (100.0, 1000.0),
    )


@pytest.mark.parametrize(
    ("candidate_bbox", "above_bbox", "below_bbox"),
    [
        ((25.0, 20.0, 65.0, 30.0), (0.0, 0.0, 100.0, 10.0), (0.0, 40.0, 100.0, 50.0)),
        (
            (25.0, 970.0, 65.0, 980.0),
            (0.0, 940.0, 100.0, 950.0),
            (0.0, 990.0, 100.0, 1000.0),
        ),
    ],
    ids=["top", "bottom"],
)
def test_compact_formula_fully_in_page_margin_remains_text(
    candidate_bbox: tuple[float, float, float, float],
    above_bbox: tuple[float, float, float, float],
    below_bbox: tuple[float, float, float, float],
) -> None:
    """验证紧凑公式簇整体落入顶部或底部 5% 时不升级为公式。"""

    candidate, blocks, remaining = _build_compact_margin_case(
        candidate_bbox,
        above_bbox,
        below_bbox,
    )

    assert blocks == []
    assert candidate in remaining


def test_compact_formula_crossing_page_margin_boundary_remains_equation() -> None:
    """验证跨过顶部 5% 分界的紧凑真公式仍能输出公式块。"""

    candidate, blocks, remaining = _build_compact_margin_case(
        (25.0, 45.0, 65.0, 55.0),
        (0.0, 30.0, 100.0, 40.0),
        (0.0, 60.0, 100.0, 70.0),
    )

    assert blocks == [
        {
            "type": "equation",
            "bbox": candidate.bbox,
            "angle": 0,
            "content": "formula cluster",
        }
    ]
    assert candidate not in remaining


@pytest.mark.parametrize(
    ("formula_bboxes", "body_tops"),
    [
        (
            (
                (20.0, 8.0, 58.0, 18.0),
                (25.0, 19.0, 55.0, 29.0),
                (91.0, 17.0, 100.0, 27.0),
            ),
            (44.0, 56.0, 68.0, 80.0),
        ),
        (
            (
                (20.0, 955.0, 58.0, 965.0),
                (25.0, 966.0, 55.0, 976.0),
                (91.0, 973.0, 100.0, 983.0),
            ),
            (900.0, 912.0, 924.0, 936.0),
        ),
    ],
    ids=["top", "bottom"],
)
def test_text_formula_fully_in_page_margin_remains_text(
    formula_bboxes: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ],
    body_tops: tuple[float, float, float, float],
) -> None:
    """验证文本公式空间分量整体落入页边 5% 时不认领原文本行。"""

    blocks, remaining = _build_text_formula_margin_case(
        formula_bboxes,
        body_tops,
    )

    assert blocks == []
    assert {line.source_index for line in remaining} == set(range(7))


@pytest.mark.parametrize(
    ("formula_bboxes", "body_tops", "expected_bbox"),
    [
        (
            (
                (20.0, 40.0, 58.0, 50.0),
                (25.0, 51.0, 55.0, 61.0),
                (91.0, 49.0, 100.0, 59.0),
            ),
            (76.0, 88.0, 100.0, 112.0),
            (20.0, 40.0, 100.0, 61.0),
        ),
        (
            (
                (20.0, 940.0, 58.0, 950.0),
                (25.0, 951.0, 55.0, 961.0),
                (91.0, 958.0, 100.0, 968.0),
            ),
            (885.0, 897.0, 909.0, 921.0),
            (20.0, 940.0, 100.0, 968.0),
        ),
    ],
    ids=["top", "bottom"],
)
def test_text_formula_crossing_page_margin_boundary_remains_equation(
    formula_bboxes: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ],
    body_tops: tuple[float, float, float, float],
    expected_bbox: tuple[float, float, float, float],
) -> None:
    """验证跨过顶部或底部 5% 分界的文本真公式仍能输出。"""

    blocks, remaining = _build_text_formula_margin_case(
        formula_bboxes,
        body_tops,
    )

    assert len(blocks) == 1
    assert blocks[0]["type"] == "equation"
    assert blocks[0]["bbox"] == expected_bbox
    assert {line.source_index for line in remaining} == {3, 4, 5, 6}


def test_justified_mixed_font_visual_row_before_body_is_not_formula_anchor() -> None:
    """验证粗体短语与常规字体混排的满栏同行不会因右缘短词误判公式。"""

    body_font = ("Body", 0)
    heading_font = ("Heading", 1)
    lines = [
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
            "label",
            (0.0, 36.0, 20.0, 46.0),
            10,
            visual_row_id=10,
            split_from_row=True,
            effective_height=10.0,
            font_signature=heading_font,
            font_coverage=1.0,
        ),
        _text_line(
            "one",
            (24.0, 36.0, 38.0, 46.0),
            11,
            visual_row_id=10,
            split_from_row=True,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "two",
            (43.0, 36.0, 57.0, 46.0),
            12,
            visual_row_id=10,
            split_from_row=True,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "three",
            (63.0, 36.0, 80.0, 46.0),
            13,
            visual_row_id=10,
            split_from_row=True,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "contains",
            (86.0, 36.0, 100.0, 46.0),
            14,
            visual_row_id=10,
            split_from_row=True,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
    ]
    continuation = _text_line(
        "continuation body row",
        (0.0, 46.0, 100.0, 56.0),
        15,
        visual_row_id=11,
        effective_height=10.0,
        font_signature=body_font,
        font_coverage=1.0,
    )

    blocks, remaining = formulas._build_formula_like_blocks(
        [*lines, *fragments, continuation],
        [],
        (100.0, 100.0),
    )

    assert blocks == []
    assert {line.source_index for line in remaining} == {
        0,
        1,
        2,
        10,
        11,
        12,
        13,
        14,
        15,
    }
