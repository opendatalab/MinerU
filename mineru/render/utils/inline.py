# Copyright (c) Opendatalab. All rights reserved.
"""MiddleJson 行内语义解析与 Markdown 序列化。"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import TypeAlias, Union

from mineru.config import LatexDelimitersConfig
from mineru.render.utils.markdown_utils import escape_conservative_markdown_text
from mineru.utils.language import detect_lang
from mineru.utils.text_utils import CJK_LANGS, resolve_text_line_boundary

_INLINE_START_RE = re.compile(
    r"<(?P<tag>eq|text|hyperlink|sup|sub|strong|b|em|i|s|u)(?P<attrs>\s[^<>]*?)?>",
    re.IGNORECASE,
)
_STYLE_ATTR_RE = re.compile(r"\bstyle\s*=\s*([\"'])(?P<style>.*?)\1", re.IGNORECASE | re.DOTALL)
_URL_RE = re.compile(r"<url>(?P<url>.*?)</url>", re.IGNORECASE | re.DOTALL)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_ANGLE_TEXT_RE = re.compile(r"<[^<>\n]*>")

_SIMPLE_STYLE_WRAPPERS = {
    frozenset({"bold"}): "**",
    frozenset({"italic"}): "*",
    frozenset({"strikethrough"}): "~~",
    frozenset({"bold", "italic"}): "***",
}
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
    label_content = f"{inner[:match.start()]}{inner[match.end():]}"
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


def render_inline_content(content: str, delimiters: LatexDelimitersConfig) -> str:
    """把一段 MiddleJson 行内内容渲染为 Markdown。"""
    return render_inline_nodes(parse_inline_content(content), delimiters)


def render_joined_inline_contents(contents: list[str], delimiters: LatexDelimitersConfig) -> str:
    """按物理段落边界规则合并多段 content 后渲染 Markdown。"""
    merged: list[InlineNode] = []
    for content in contents:
        current = parse_inline_content(content)
        if not current:
            continue
        if merged:
            _join_inline_node_sequences(merged, current)
        merged.extend(current)
    return render_inline_nodes(merged, delimiters)


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
    next_starts_lowercase = current_visible[0].islower()
    if last_text is not None:
        processed, separator = resolve_text_line_boundary(
            last_text.content,
            block_language=language,
            next_starts_with_lowercase=next_starts_lowercase,
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


def render_inline_nodes(nodes: list[InlineNode], delimiters: LatexDelimitersConfig) -> str:
    """把已解析的行内节点序列化为 Markdown 与安全 HTML。"""
    return "".join(_render_inline_node(node, delimiters) for node in nodes)


def _render_inline_node(node: InlineNode, delimiters: LatexDelimitersConfig) -> str:
    """渲染单个行内节点。"""
    if isinstance(node, InlineText):
        return _escape_plain_markdown_text(node.content)
    if isinstance(node, InlineEquation):
        return f"{delimiters.inline.left}{node.latex}{delimiters.inline.right}"
    if isinstance(node, InlineStyled):
        content = render_inline_nodes(node.children, delimiters)
        return _apply_styles(content, inline_plain_text(node.children), node.styles)
    if isinstance(node, InlineLink):
        label = render_inline_nodes(node.children, delimiters)
        return _render_link(label, node.url, _requires_html_link(node.children))
    raise TypeError(f"Unsupported inline node: {type(node).__name__}")


def _escape_plain_markdown_text(content: str) -> str:
    """转义普通 Markdown 符号，同时原样保留正文中的尖括号片段。"""
    parts: list[str] = []
    cursor = 0
    for match in _ANGLE_TEXT_RE.finditer(content):
        parts.append(escape_conservative_markdown_text(content[cursor : match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(escape_conservative_markdown_text(content[cursor:]))
    return "".join(parts)


def _apply_styles(content: str, plain_text: str, styles: tuple[str, ...]) -> str:
    """按样式复杂度选择 Markdown wrapper 或安全 HTML 标签。"""
    if not content or not styles:
        return content
    if plain_text and not plain_text.strip() and any(
        style in styles for style in ("underline", "strikethrough", "emphasis")
    ):
        return _render_visible_whitespace(plain_text, styles)

    leading = content[: len(content) - len(content.lstrip(" \t"))]
    trailing = content[len(content.rstrip(" \t")) :]
    core = content.strip(" \t")
    if not core:
        return content

    style_key = frozenset(styles)
    wrapper = _SIMPLE_STYLE_WRAPPERS.get(style_key)
    if wrapper is not None:
        return f"{leading}{wrapper}{core}{wrapper}{trailing}"
    return f"{leading}{_apply_html_styles(core, styles)}{trailing}"


def _render_visible_whitespace(content: str, styles: tuple[str, ...]) -> str:
    """使用不换行空格保留带可见样式的纯空白。"""
    visible = "".join("<br>" if char == "\n" else "&nbsp;" for char in content.expandtabs(4))
    return _apply_html_styles(visible, styles)


def _apply_html_styles(content: str, styles: tuple[str, ...]) -> str:
    """按稳定顺序给复杂样式添加 HTML wrapper。"""
    if "superscript" in styles:
        content = f"<sup>{content}</sup>"
    elif "subscript" in styles:
        content = f"<sub>{content}</sub>"
    if "underline" in styles:
        content = f"<u>{content}</u>"
    if "bold" in styles:
        content = f"<strong>{content}</strong>"
    if "italic" in styles:
        content = f"<em>{content}</em>"
    if "strikethrough" in styles:
        content = f"<s>{content}</s>"
    if "emphasis" in styles:
        content = f'<span style="text-emphasis: dot; text-emphasis-position: under;">{content}</span>'
    return content


def _requires_html_link(nodes: list[InlineNode]) -> bool:
    """判断链接标签是否含不适合嵌入 Markdown link 的复杂样式。"""
    for node in nodes:
        if isinstance(node, InlineLink):
            return True
        if isinstance(node, InlineStyled):
            if frozenset(node.styles) not in _SIMPLE_STYLE_WRAPPERS:
                return True
            if _requires_html_link(node.children):
                return True
    return False


def _render_link(label: str, url: str, use_html: bool) -> str:
    """按标签复杂度输出 Markdown 或 HTML 超链接。"""
    if not label:
        return ""
    if not url or url == ".":
        return label
    if use_html:
        return f'<a href="{html.escape(url, quote=True)}">{label}</a>'
    safe_url = url.replace("\\", "%5C").replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{_escape_markdown_link_label(label)}]({safe_url})"


def render_internal_link(label: str, anchor: str) -> str:
    """把已渲染目录标签包装为当前文档内锚点链接。"""
    safe_anchor = anchor.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{_escape_markdown_link_label(label)}](#{safe_anchor})"


def _escape_markdown_link_label(label: str) -> str:
    """转义 Markdown link 标签中的方括号并保留既有反斜杠。"""
    return re.sub(r"(?<!\\)([\[\]])", r"\\\1", label)


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
    "InlineNode",
    "inline_plain_text",
    "parse_inline_content",
    "render_inline_content",
    "render_inline_nodes",
    "render_internal_link",
    "render_joined_inline_contents",
]
