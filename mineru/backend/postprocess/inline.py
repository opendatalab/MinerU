# Copyright (c) Opendatalab. All rights reserved.
"""各格式共用的 MiddleJson 行内 AST、解析与段落边界合并。"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import TypeAlias, Union

from ...utils.language import detect_lang
from ...utils.text import CJK_LANGS, resolve_text_line_boundary

_INLINE_START_RE = re.compile(
    r"<(?P<tag>eq|text|hyperlink|sup|sub|strong|b|em|i|s|u)(?P<attrs>\s[^<>]*?)?>",
    re.IGNORECASE,
)
_STYLE_ATTR_RE = re.compile(r"\bstyle\s*=\s*([\"'])(?P<style>.*?)\1", re.IGNORECASE | re.DOTALL)
_URL_RE = re.compile(r"<url>(?P<url>.*?)</url>", re.IGNORECASE | re.DOTALL)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_KNOWN_STYLES = {
    "bold",
    "italic",
    "underline",
    "emphasis",
    "strikethrough",
    "superscript",
    "subscript",
}
_DIRECT_TAG_STYLES = {
    "strong": "bold",
    "b": "bold",
    "em": "italic",
    "i": "italic",
    "s": "strikethrough",
    "u": "underline",
    "sup": "superscript",
    "sub": "subscript",
}


@dataclass(slots=True)
class InlineText:
    """保存普通行内文本。"""

    content: str


@dataclass(slots=True)
class InlineEquation:
    """保存不含外层定界符的行内 LaTeX。"""

    latex: str


@dataclass(slots=True)
class InlineStyled:
    """保存带 Office 字体样式的行内子节点。"""

    children: list[InlineNode]
    styles: tuple[str, ...]


@dataclass(slots=True)
class InlineLink:
    """保存超链接标签节点及目标地址。"""

    children: list[InlineNode]
    url: str


InlineNode: TypeAlias = Union[InlineText, InlineEquation, InlineStyled, InlineLink]


def parse_inline_content(content: str) -> list[InlineNode]:
    """把 MiddleJson 字符串解析为白名单行内语义节点。"""
    return _parse_inline_range(content or "")


def _parse_inline_range(content: str) -> list[InlineNode]:
    """解析当前字符串范围，未知标签始终保留为普通文本。"""
    nodes: list[InlineNode] = []
    cursor = 0
    while match := _INLINE_START_RE.search(content, cursor):
        _append_text(nodes, content[cursor : match.start()])
        tag = match.group("tag").lower()
        close = _find_matching_close(content, tag, match.end())
        if close is None:
            _append_text(nodes, match.group(0))
            cursor = match.end()
            continue

        inner = content[match.end() : close[0]]
        original = content[match.start() : close[1]]
        parsed = _parse_inline_element(tag, match.group("attrs") or "", inner)
        if parsed is None:
            _append_text(nodes, original)
        elif isinstance(parsed, list):
            nodes.extend(parsed)
        else:
            nodes.append(parsed)
        cursor = close[1]
    _append_text(nodes, content[cursor:])
    return nodes


def _find_matching_close(content: str, tag: str, start: int) -> tuple[int, int] | None:
    """查找同名结束标签，并兼容极少出现的同名嵌套。"""
    token_re = re.compile(rf"<(?P<close>/)?{re.escape(tag)}(?:\s[^<>]*?)?>", re.IGNORECASE)
    depth = 1
    for match in token_re.finditer(content, start):
        depth += -1 if match.group("close") else 1
        if depth == 0:
            return match.start(), match.end()
    return None


def _parse_inline_element(
    tag: str,
    attrs: str,
    inner: str,
) -> InlineNode | list[InlineNode] | None:
    """把一个已闭合白名单元素转换为对应行内节点。"""
    if tag == "eq":
        latex = html.unescape(inner).strip()
        return InlineEquation(latex) if latex else []
    if tag == "hyperlink":
        return _parse_hyperlink(inner)

    children = _parse_inline_range(inner)
    if tag == "text":
        styles = _parse_styles(attrs)
        return InlineStyled(children, styles) if styles else children
    style = _DIRECT_TAG_STYLES[tag]
    return InlineStyled(children, (style,))


def _parse_hyperlink(inner: str) -> InlineLink | None:
    """解析 hyperlink 的标签与 URL，结构损坏时交由调用方保留原文。"""
    match = _URL_RE.search(inner)
    if match is None:
        return None
    label_content = f"{inner[: match.start()]}{inner[match.end() :]}"
    url = html.unescape(match.group("url")).strip()
    if not url:
        return None
    children = _parse_inline_range(label_content)
    if not children:
        return None
    return InlineLink(children=children, url=url)


def _parse_styles(attrs: str) -> tuple[str, ...]:
    """从 text 标签属性中提取去重后的已知样式。"""
    match = _STYLE_ATTR_RE.search(attrs)
    if match is None:
        return ()
    styles: list[str] = []
    for value in match.group("style").split(","):
        style = value.strip().lower()
        if style in _KNOWN_STYLES and style not in styles:
            styles.append(style)
    return tuple(styles)


def _append_text(nodes: list[InlineNode], content: str) -> None:
    """追加普通文本，并合并相邻文本节点。"""
    if not content:
        return
    if nodes and isinstance(nodes[-1], InlineText):
        nodes[-1].content += content
    else:
        nodes.append(InlineText(content))


def join_inline_contents(contents: list[str]) -> list[InlineNode]:
    """按物理段落边界规则合并多段 content，并保留中性的行内节点。"""
    merged: list[InlineNode] = []
    for content in contents:
        current = parse_inline_content(content)
        if not current:
            continue
        if merged:
            _join_inline_node_sequences(merged, current)
        merged.extend(current)
    return merged


def _join_inline_node_sequences(previous: list[InlineNode], current: list[InlineNode]) -> None:
    """在两组行内节点之间插入安全的语言相关边界。"""
    previous_visible = inline_plain_text(previous).rstrip()
    current_visible = inline_plain_text(current).lstrip()
    if not previous_visible or not current_visible:
        return

    last_text = _last_text_node(previous)
    first_text = _first_text_node(current)
    if last_text is not None:
        last_text.content = last_text.content.rstrip()
    if first_text is not None:
        first_text.content = first_text.content.lstrip()

    language = _detect_boundary_language(f"{previous_visible}{current_visible}")
    if last_text is not None:
        processed, separator = resolve_text_line_boundary(
            last_text.content,
            block_language=language,
            next_content=current_visible,
        )
        last_text.content = processed
    else:
        separator = "" if language in CJK_LANGS else " "
    if separator:
        previous.append(InlineText(separator))


def _detect_boundary_language(content: str) -> str:
    """检测段落边界语言，短 CJK 文本优先使用字符范围兜底。"""
    if _CJK_RE.search(content):
        return "zh"
    try:
        return detect_lang(content)
    except Exception:
        return ""


def _first_text_node(nodes: list[InlineNode]) -> InlineText | None:
    """仅当首个可见行内叶子是普通文本时返回该节点。"""
    for node in nodes:
        if not inline_plain_text([node]):
            continue
        if isinstance(node, InlineText):
            return node
        if isinstance(node, (InlineStyled, InlineLink)):
            return _first_text_node(node.children)
        return None
    return None


def _last_text_node(nodes: list[InlineNode]) -> InlineText | None:
    """仅当末个可见行内叶子是普通文本时返回该节点。"""
    for node in reversed(nodes):
        if not inline_plain_text([node]):
            continue
        if isinstance(node, InlineText):
            return node
        if isinstance(node, (InlineStyled, InlineLink)):
            return _last_text_node(node.children)
        return None
    return None


def inline_plain_text(nodes: list[InlineNode]) -> str:
    """提取行内节点的可见文本，供边界和目录页码判断使用。"""
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, InlineText):
            parts.append(node.content)
        elif isinstance(node, InlineEquation):
            parts.append(node.latex)
        elif isinstance(node, (InlineStyled, InlineLink)):
            parts.append(inline_plain_text(node.children))
    return "".join(parts)


__all__ = [
    "InlineEquation",
    "InlineLink",
    "InlineNode",
    "InlineStyled",
    "InlineText",
    "inline_plain_text",
    "join_inline_contents",
    "parse_inline_content",
]
