# Copyright (c) Opendatalab. All rights reserved.
"""各格式共用的列表 marker 解析与参考文献判定。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, TypeAlias

from mineru.render._internal.common.inline import inline_plain_text, parse_inline_content
from mineru.types import BlockType, ListBlock

ListItemKind: TypeAlias = Literal["unordered", "ordered", "explicit", "none"]
OrderedListStyle: TypeAlias = Literal["decimal", "lower-alpha", "upper-alpha", "lower-roman", "upper-roman"]

_LIST_ITEM_MARKER_RE = re.compile(
    r"^(?P<leading>\s*)(?P<marker>"
    r"(?P<unordered>[-*+])"
    r"|(?P<ordered>\d+\.|[A-Za-z]\.|[IVXLCDMivxlcdm]{2,}\.)"
    r"|(?P<explicit>\d+\)|\(\d+[.)]|[A-Za-z]\)|[IVXLCDMivxlcdm]{2,}\)|\[[^\]\n]+\])"
    r")(?P<separator>\s+)(?P<body>.*)$",
    re.DOTALL,
)
_LEADING_WHITESPACE_RE = re.compile(r"^[ \t]*")
# 去除首部空白后，前五个可见字符内出现 Unicode 数字即视为单项命中。
_REFERENCE_NUMBER_PREFIX_RE = re.compile(r"^\D{0,4}\d")
_MARKDOWN_UNORDERED_MARKER_RE = re.compile(r"^[ \t]*-[ \t]+")
_ROMAN_MARKER_RE = re.compile(r"[IVXLCDM]+", re.IGNORECASE)
_CANONICAL_ROMAN_RE = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_MAX_NATIVE_ORDERED_VALUE = 1_000_000


@dataclass(frozen=True, slots=True)
class ListItem:
    """保存一个列表条目的原始标记、正文与 HTML 所需分类。"""

    marker: str | None
    body: str
    kind: ListItemKind
    value: int | None
    ordered_style: OrderedListStyle | None
    leading: str
    separator: str


def parse_list_item_marker(content: str) -> ListItem:
    """解析列表行首 marker；无法识别时仍拆出前导水平空白。"""
    match = _LIST_ITEM_MARKER_RE.match(content)
    if match is None:
        leading_match = _LEADING_WHITESPACE_RE.match(content)
        leading = leading_match.group(0) if leading_match is not None else ""
        return ListItem(
            marker=None,
            body=content[len(leading) :],
            kind="none",
            value=None,
            ordered_style=None,
            leading=leading,
            separator="",
        )

    marker = match.group("marker")
    if match.group("unordered") is not None:
        kind: ListItemKind = "unordered"
        value = None
        ordered_style = None
    elif match.group("ordered") is not None:
        ordered = _ordered_marker_value(marker)
        if ordered is None:
            kind = "explicit"
            value = None
            ordered_style = None
        else:
            kind = "ordered"
            value, ordered_style = ordered
    else:
        kind = "explicit"
        value = None
        ordered_style = None
    return ListItem(
        marker=marker,
        body=match.group("body"),
        kind=kind,
        value=value,
        ordered_style=ordered_style,
        leading=match.group("leading"),
        separator=match.group("separator"),
    )


def _ordered_marker_value(marker: str) -> tuple[int, OrderedListStyle] | None:
    """把有界且规范的点号 marker 转成序号；超限或畸形时返回 None。"""
    stem = marker[:-1]
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", stem):
        if len(stem) > 7:
            return None
        value = int(stem)
        return (value, "decimal") if value <= _MAX_NATIVE_ORDERED_VALUE else None
    # 单字符 i/v/x/l/c/d/m 固定按罗马数字解释，消除与字母序号的歧义。
    if _ROMAN_MARKER_RE.fullmatch(stem):
        if _CANONICAL_ROMAN_RE.fullmatch(stem.upper()) is None:
            return None
        style: OrderedListStyle = "upper-roman" if stem.isupper() else "lower-roman"
        return _roman_marker_value(stem), style
    if re.fullmatch(r"[A-Za-z]", stem) is None:
        return None
    style = "upper-alpha" if stem.isupper() else "lower-alpha"
    return ord(stem.lower()) - ord("a") + 1, style


def _roman_marker_value(marker: str) -> int:
    """按减法记数规则计算罗马 marker 的数值，兼容 producer 的宽松组合。"""
    total = 0
    previous = 0
    for character in reversed(marker.upper()):
        current = _ROMAN_VALUES[character]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def has_markdown_unordered_marker(content: str) -> bool:
    """判断条目是否已有 Markdown 短横线 marker，保持既有补 bullet 规则。"""
    return _MARKDOWN_UNORDERED_MARKER_RE.match(content) is not None


def reference_list_needs_bullets(block: ListBlock) -> bool:
    """按直属非空条目的数字前缀严格多数规则判断是否补无序 marker。"""
    if block.sub_type != BlockType.REF_TEXT:
        return False

    item_count = 0
    numbered_count = 0
    for child in block.content:
        if isinstance(child, ListBlock):
            continue
        visible_text = inline_plain_text(parse_inline_content(child.content)).lstrip()
        if not visible_text:
            continue
        item_count += 1
        if _REFERENCE_NUMBER_PREFIX_RE.match(visible_text):
            numbered_count += 1
    return item_count > 0 and numbered_count * 2 <= item_count


__all__ = [
    "ListItem",
    "ListItemKind",
    "OrderedListStyle",
    "has_markdown_unordered_marker",
    "parse_list_item_marker",
    "reference_list_needs_bullets",
]
