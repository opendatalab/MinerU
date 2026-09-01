from __future__ import annotations

import math
from typing import Any

import pytest

from mineru.model.flash.pdf import native_text


def _span(
    text: str,
    bbox: tuple[float, float, float, float],
    angle_degrees: float,
    *,
    chars: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造只包含方向、文本和 bbox 的最小 pdftext span。"""

    return {
        "text": text,
        "bbox": bbox,
        "rotation": math.radians(angle_degrees),
        "chars": chars or [],
    }


def _char(
    value: str,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    """构造字符基线方向测试使用的最小 pdftext 字符。"""

    return {"char": value, "bbox": bbox}


def _line(
    spans: list[dict[str, Any]],
    angle_degrees: float,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
) -> dict[str, Any]:
    """构造测试混合方向拆分使用的最小 pdftext line。"""

    return {
        "spans": spans,
        "bbox": bbox,
        "rotation": math.radians(angle_degrees),
    }


def test_mixed_footer_and_315_degree_watermark_are_split_before_filtering() -> None:
    """验证接近 315 度的 span 不会继续借用父行 0 度方向与巨型 bbox。"""

    pdf_line = _line(
        [
            _span("文档问题反馈: aw-document@example.com", (30.0, 90.0, 80.0, 98.0), 0.0),
            _span("通用测试文档", (5.0, 10.0, 25.0, 95.0), 315.000019),
            _span("\r\n", (26.0, 20.0, 26.0, 20.0), 0.0),
        ],
        0.0,
    )

    children = native_text._split_pdftext_line_by_rotation(pdf_line)
    items = native_text._build_native_line_items([pdf_line], (100.0, 100.0))

    assert [round(math.degrees(child["rotation"]), 6) for child in children] == [0.0, 315.000019]
    assert children[0]["bbox"] == (30.0, 90.0, 80.0, 98.0)
    assert children[1]["bbox"] == (5.0, 10.0, 25.0, 95.0)
    assert [(item.text, item.bbox, item.angle) for item in items] == [
        ("文档问题反馈: aw-document@example.com", (30.0, 90.0, 80.0, 98.0), 0)
    ]


def test_small_oblique_span_stays_inside_standard_direction_line() -> None:
    """验证约 19 度的仿斜体 span 不触发方向拆分且沿用父行 0 度。"""

    pdf_line = _line(
        [
            _span("normal ", (10.0, 20.0, 40.0, 30.0), 0.0),
            _span("oblique", (40.0, 18.0, 75.0, 30.0), 19.0),
        ],
        0.0,
    )

    children = native_text._split_pdftext_line_by_rotation(pdf_line)
    items = native_text._build_native_line_items([pdf_line], (100.0, 100.0))

    assert len(children) == 1
    assert math.degrees(children[0]["rotation"]) == pytest.approx(0.0)
    assert [(item.text, item.angle) for item in items] == [("normal oblique", 0)]


def test_sheared_horizontal_line_uses_char_baseline_and_splits_far_sidecar() -> None:
    """验证仿斜体粗行可由水平字符基线恢复，并继续拆开远端页码。"""

    chars = [_char(value, (10.0 + index * 5.0, 80.0, 14.0 + index * 5.0, 88.0)) for index, value in enumerate("NOTICE")]
    chars.append(_char("2", (90.0, 80.2, 94.0, 88.2)))
    pdf_line = _line(
        [
            _span(
                "NOTICE 2",
                (10.0, 80.0, 94.0, 88.2),
                18.433,
                chars=chars,
            )
        ],
        18.433,
        bbox=(10.0, 80.0, 94.0, 88.2),
    )

    items = native_text._build_native_line_items([pdf_line], (100.0, 100.0))

    assert [(item.text, item.angle) for item in items] == [
        ("NOTICE", 0),
        ("2", 0),
    ]
    assert all(item.split_from_row for item in items)


def test_true_diagonal_char_baseline_is_not_recovered_as_horizontal() -> None:
    """验证真实斜向字符中心不会因字符数量和长宽比被恢复为水平正文。"""

    chars = [
        _char(value, (10.0 + index * 8.0, 70.0 - index * 6.0, 14.0 + index * 8.0, 78.0 - index * 6.0))
        for index, value in enumerate("WATERMARK")
    ]
    pdf_line = _line(
        [
            _span(
                "WATERMARK",
                (10.0, 22.0, 78.0, 78.0),
                315.0,
                chars=chars,
            )
        ],
        315.0,
        bbox=(10.0, 22.0, 78.0, 78.0),
    )

    assert native_text._build_native_line_items([pdf_line], (100.0, 100.0)) == []


def test_small_shear_formula_line_is_retained_as_formula_only() -> None:
    """验证带数学运算符的小角度多基线粗行只进入 formula-only 流。"""
    chars = [
        _char("x", (10.0, 20.0, 15.0, 28.0)),
        _char("=", (18.0, 24.0, 24.0, 30.0)),
    ]
    pdf_line = _line(
        [_span("x=", (10.0, 20.0, 24.0, 30.0), 10.0, chars=chars)],
        10.0,
        bbox=(10.0, 20.0, 24.0, 30.0),
    )

    items = native_text._build_native_line_items([pdf_line], (100.0, 100.0))

    assert [(item.text, item.angle, item.formula_candidate_only) for item in items] == [("x=", 0, True)]


def test_formula_only_rows_do_not_shift_existing_source_indices() -> None:
    """验证新增公式候选使用尾部 source index，不改变既有自然文本身份。"""
    first = _line([_span("first", (10.0, 10.0, 35.0, 18.0), 0.0)], 0.0)
    formula = _line([_span("x=", (10.0, 20.0, 24.0, 30.0), 10.0)], 10.0)
    second = _line([_span("second", (10.0, 32.0, 40.0, 40.0), 0.0)], 0.0)

    items = native_text._build_native_line_items([first, formula, second], (100.0, 100.0))

    assert {item.text: item.source_index for item in items} == {"first": 0, "second": 1, "x=": 2}


def test_small_true_diagonal_without_formula_evidence_stays_rejected() -> None:
    """验证缺少公式证据的短斜向文字不会借 formula-only 流回到正文。"""
    chars = [
        _char(value, (10.0 + index * 5.0, 30.0 - index * 2.0, 14.0 + index * 5.0, 38.0 - index * 2.0))
        for index, value in enumerate("mark")
    ]
    pdf_line = _line(
        [_span("mark", (10.0, 24.0, 29.0, 38.0), 10.0, chars=chars)],
        10.0,
        bbox=(10.0, 24.0, 29.0, 38.0),
    )

    assert native_text._build_native_line_items([pdf_line], (100.0, 100.0)) == []


@pytest.mark.parametrize(
    ("line_angle", "page_rotation", "expected_angle"),
    [
        (0.0, 0, 0),
        (90.0, 0, 90),
        (270.0, 0, 270),
        (180.0, 0, None),
        (315.000019, 0, None),
        (180.0, 90, 270),
        (90.0, 90, None),
    ],
)
def test_only_supported_visual_line_directions_are_retained(
    line_angle: float,
    page_rotation: int,
    expected_angle: int | None,
) -> None:
    """验证方向白名单在页面旋转后生效，且不会先把斜向行归一为 0 度。"""

    pdf_line = _line([_span("value", (10.0, 20.0, 40.0, 30.0), line_angle)], line_angle)

    items = native_text._build_native_line_items(
        [pdf_line],
        (100.0, 100.0),
        page_rotation=page_rotation,
    )

    assert [item.angle for item in items] == ([] if expected_angle is None else [expected_angle])


def test_native_line_builder_can_opt_in_to_180_degree_visual_runs() -> None:
    """验证 Low/TXT 可显式扩展方向白名单且不改变 Flash 默认行为。"""

    pdf_line = _line(
        [_span("upside down", (10.0, 20.0, 60.0, 30.0), 180.0)],
        180.0,
        bbox=(10.0, 20.0, 60.0, 30.0),
    )

    default_items = native_text._build_native_line_items(
        [pdf_line],
        (100.0, 100.0),
    )
    low_txt_items = native_text._build_native_line_items(
        [pdf_line],
        (100.0, 100.0),
        supported_angles=(0.0, 90.0, 180.0, 270.0),
    )

    assert default_items == []
    assert [(item.text, item.angle) for item in low_txt_items] == [("upside down", 180)]


@pytest.mark.parametrize(
    "separator",
    [
        "\u00a0",
        "\u1680",
        "\u2000",
        "\u2001",
        "\u2002",
        "\u2003",
        "\u2004",
        "\u2005",
        "\u2006",
        "\u2007",
        "\u2008",
        "\u2009",
        "\u200a",
        "\u202f",
        "\u205f",
        "\u3000",
    ],
)
def test_pdf_unicode_separator_spaces_are_normalized_to_ascii(separator: str) -> None:
    """验证所有 Unicode Zs 排版空格均转换为普通 ASCII 空格。"""

    assert (
        native_text._sanitize_pdf_control_text(
            f"left{separator}right",
            preserve_newlines=True,
        )
        == "left right"
    )


def test_pdf_unicode_line_separators_follow_newline_policy() -> None:
    """验证 NEXT LINE、行分隔符和段分隔符统一遵循物理换行保留策略。"""

    content = "first\u0085second\u2028third\u2029fourth"

    assert (
        native_text._sanitize_pdf_control_text(
            content,
            preserve_newlines=True,
        )
        == "first\nsecond\nthird\nfourth"
    )
    assert (
        native_text._sanitize_pdf_control_text(
            content,
            preserve_newlines=False,
        )
        == "firstsecondthirdfourth"
    )


def test_pdf_safe_invisible_and_control_characters_are_removed_idempotently() -> None:
    """验证无正文语义的零宽字符和 C0/C1 控制字符被稳定删除。"""

    content = "A\u200bB\u2060C\ufeffD\x00E\x07F\x7fG\x80H\x9fI"
    normalized = native_text._sanitize_pdf_control_text(
        content,
        preserve_newlines=True,
    )

    assert normalized == "ABCDEFGHI"
    assert (
        native_text._sanitize_pdf_control_text(
            normalized,
            preserve_newlines=True,
        )
        == normalized
    )


def test_pdf_soft_hyphens_keep_only_latin_line_end_breaks() -> None:
    """验证两类 PDF 软断词仅在拉丁字母行末转成 ASCII hyphen。"""

    assert native_text._normalize_native_run_text("inter\u00ad") == "inter-"
    assert native_text._normalize_native_run_text("co\u00adoperate") == "cooperate"
    assert native_text._normalize_native_run_text("word\x02") == "word-"
    assert native_text._normalize_native_run_text("A\x02B") == "AB"


def test_pdf_semantic_joiners_and_decode_markers_are_preserved() -> None:
    """验证语言连接符、私用区字形和解码占位符不会被通用清理静默删除。"""

    content = "a\u200cb\u200dc\uf8f1d\ufffde"

    assert (
        native_text._sanitize_pdf_control_text(
            content,
            preserve_newlines=True,
        )
        == content
    )
