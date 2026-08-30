from __future__ import annotations
from _span_test_utils import inline as _inline

import pytest

from mineru.backend.postprocess.inline import inline_plain_text
from mineru.render._internal.common.index import looks_like_index_page_token, strip_index_page_tail
from mineru.render._internal.common.list_items import (
    has_markdown_unordered_marker,
    parse_list_item_marker,
    reference_list_needs_bullets,
)
from mineru.types import ListBlock, RefTextBlock, TextBlock


@pytest.mark.parametrize(
    ("content", "marker", "body", "kind", "value", "ordered_style"),
    [
        ("- item", "-", "item", "unordered", None, None),
        ("* item", "*", "item", "unordered", None, None),
        ("+ item", "+", "item", "unordered", None, None),
        ("12. item", "12.", "item", "ordered", 12, "decimal"),
        ("a. item", "a.", "item", "ordered", 1, "lower-alpha"),
        ("A. item", "A.", "item", "ordered", 1, "upper-alpha"),
        ("i. item", "i.", "item", "ordered", 1, "lower-roman"),
        ("I. item", "I.", "item", "ordered", 1, "upper-roman"),
        ("iv. item", "iv.", "item", "ordered", 4, "lower-roman"),
        ("XII. item", "XII.", "item", "ordered", 12, "upper-roman"),
        ("12) item", "12)", "item", "explicit", None, None),
        ("(12) item", "(12)", "item", "explicit", None, None),
        ("(12. item", "(12.", "item", "explicit", None, None),
        ("a) item", "a)", "item", "explicit", None, None),
        ("[12] item", "[12]", "item", "explicit", None, None),
        ("[x] done", "[x]", "done", "explicit", None, None),
        ("[ ] pending", "[ ]", "pending", "explicit", None, None),
        ("plain item", None, "plain item", "none", None, None),
        ("-without-space", None, "-without-space", "none", None, None),
    ],
)
def test_parse_list_item_marker_classifies_supported_styles(
    content: str,
    marker: str | None,
    body: str,
    kind: str,
    value: int | None,
    ordered_style: str | None,
) -> None:
    """验证共享 parser 区分原生列表与需要显式 marker 的编号风格。"""
    item = parse_list_item_marker(_inline(content))

    assert item.marker == marker
    assert inline_plain_text(item.body) == body
    assert item.kind == kind
    assert item.value == value
    assert item.ordered_style == ordered_style


@pytest.mark.parametrize("content", ["  2. item", "\t- item", "- \ncontinued", "  plain item"])
def test_parse_list_item_marker_preserves_reconstructable_whitespace(content: str) -> None:
    """验证 parser 单独保存前导与分隔空白，并可无损重建原内容。"""
    item = parse_list_item_marker(_inline(content))
    reconstructed = item.leading
    if item.marker is not None:
        reconstructed += item.marker + item.separator
    reconstructed += inline_plain_text(item.body)

    assert reconstructed == content


def test_single_roman_letters_use_stable_roman_precedence() -> None:
    """验证与字母编号同形的罗马字符固定优先解释为罗马数字。"""
    assert parse_list_item_marker(_inline("v. item")).ordered_style == "lower-roman"
    assert parse_list_item_marker(_inline("v. item")).value == 5
    assert parse_list_item_marker(_inline("c. item")).ordered_style == "lower-roman"
    assert parse_list_item_marker(_inline("c. item")).value == 100
    assert parse_list_item_marker(_inline("z. item")).ordered_style == "lower-alpha"
    assert parse_list_item_marker(_inline("z. item")).value == 26


@pytest.mark.parametrize(
    "content",
    [
        f"{'9' * 5000}. item",
        "1000001. item",
        "01. item",
        "００１. item",
        "IIII. item",
        f"{'M' * 5000}. item",
    ],
)
def test_unbounded_or_invalid_ordered_markers_degrade_to_explicit(content: str) -> None:
    """验证超长、超界或非规范编号不会触发大整数异常，并原样保留 marker。"""
    item = parse_list_item_marker(_inline(content))

    assert item.kind == "explicit"
    assert item.marker is not None
    assert item.value is None
    assert item.ordered_style is None


def test_markdown_existing_bullet_detection_remains_hyphen_only() -> None:
    """验证参考文献补 bullet 只避开既有短横线，保持 Markdown 历史输出。"""
    assert has_markdown_unordered_marker(_inline("  - existing"))
    assert not has_markdown_unordered_marker(_inline("* existing"))
    assert not has_markdown_unordered_marker(_inline("+ existing"))
    assert not has_markdown_unordered_marker(_inline("-without-space"))
    assert not has_markdown_unordered_marker(_inline("-\ncontinued"))


def _reference_list(*children: TextBlock | RefTextBlock | ListBlock) -> ListBlock:
    """构造用于严格多数判定的参考文献列表。"""
    return ListBlock(type="list", sub_type="ref_text", content=list(children))


def test_reference_list_bullet_rule_uses_visible_direct_item_strict_majority() -> None:
    """验证富文本可见数字、空项和嵌套列表遵守既有严格多数规则。"""
    nested = _reference_list(RefTextBlock(type="ref_text", content=_inline("Author nested")))
    numbered_majority = _reference_list(
        RefTextBlock(
            type="ref_text",
            content=[
                {"type": "text", "content": "[1]", "styles": ["bold"]},
                {"type": "text", "content": " first"},
            ],
        ),
        RefTextBlock(type="ref_text", content=_inline("missing marker")),
        TextBlock(type="text", content=_inline("3) third")),
        RefTextBlock(type="ref_text", content=[]),
        nested,
    )
    tied = _reference_list(
        RefTextBlock(type="ref_text", content=_inline("[1] first")),
        RefTextBlock(type="ref_text", content=_inline("Author A")),
    )

    assert not reference_list_needs_bullets(numbered_majority)
    assert reference_list_needs_bullets(tied)
    assert not reference_list_needs_bullets(ListBlock(type="list", sub_type="text", content=tied.content))


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Title", "Title"),
        ("Title\t12", "Title"),
        ('Title\t<text style="bold">IV</text>', "Title"),
        ("Part\tSection\tA", "Part Section"),
        ("Title\tAppendix", "Title Appendix"),
        ("Title\tA2", "Title A2"),
    ],
)
def test_strip_index_page_tail_uses_visible_tail_token(content: str, expected: str) -> None:
    """验证目录仅删除可信的末尾页码，并把保留 tab 转为空格。"""
    spans = (
        [
            {"type": "text", "content": "Title\t"},
            {"type": "text", "content": "IV", "styles": ["bold"]},
        ]
        if "<text" in content
        else _inline(content)
    )
    assert inline_plain_text(strip_index_page_tail(spans)) == expected


@pytest.mark.parametrize("content", ["1", "１２", "iv", "XII", "a", "Z"])
def test_index_page_token_accepts_supported_forms(content: str) -> None:
    """验证数字、罗马数字与单字母可作为目录页码 token。"""
    assert looks_like_index_page_token(content)


@pytest.mark.parametrize("content", ["", "AB", "A2", "appendix", "1234567890123"])
def test_index_page_token_rejects_ambiguous_forms(content: str) -> None:
    """验证空值、长值和普通词不会被误删为目录页码。"""
    assert not looks_like_index_page_token(content)
