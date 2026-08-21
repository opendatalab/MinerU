# Copyright (c) Opendatalab. All rights reserved.
"""共享 inline AST 到 Markdown 与安全 HTML 的序列化。"""

from __future__ import annotations

import html
import re

from mineru.config import LatexDelimitersConfig
from mineru.render._internal.common.inline import (
    InlineEquation,
    InlineLink,
    InlineNode,
    InlineStyled,
    InlineText,
    inline_plain_text,
    join_inline_contents,
    parse_inline_content,
)
from mineru.render._internal.markdown.escaping import escape_conservative_markdown_text

_ANGLE_TEXT_RE = re.compile(r"<[^<>\n]*>")
_SIMPLE_STYLE_WRAPPERS = {
    frozenset({"bold"}): "**",
    frozenset({"italic"}): "*",
    frozenset({"strikethrough"}): "~~",
    frozenset({"bold", "italic"}): "***",
}


def render_inline_content(content: str, delimiters: LatexDelimitersConfig) -> str:
    """把一段 MiddleJson 行内内容渲染为 Markdown。"""
    return render_inline_nodes(parse_inline_content(content), delimiters)


def render_joined_inline_contents(contents: list[str], delimiters: LatexDelimitersConfig) -> str:
    """按物理段落边界规则合并多段 content 后渲染 Markdown。"""
    return render_inline_nodes(join_inline_contents(contents), delimiters)


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
    marker = _get_visible_space_marker(styles)
    if marker is not None:
        rendered_markers = _render_visible_space_marker_text(content, plain_text, styles, marker)
        if rendered_markers is not None:
            return rendered_markers
    if plain_text and not plain_text.strip() and any(style in styles for style in ("underline", "strikethrough", "emphasis")):
        return _render_visible_whitespace(plain_text, styles)

    return _apply_style_wrappers(content, styles)


def _get_visible_space_marker(styles: tuple[str, ...]) -> str | None:
    """按 dev 规则选择可见空格 marker，下划线优先于删除线。"""
    if "underline" in styles:
        return "_"
    if "strikethrough" in styles:
        return "-"
    return None


def _render_visible_space_marker_text(
    content: str,
    plain_text: str,
    styles: tuple[str, ...],
    marker: str,
) -> str | None:
    """把纯 ASCII 空格或非空文本首尾空格转换为可见 marker。"""
    if not plain_text:
        return None
    style_key = frozenset(styles)
    force_html = style_key not in _SIMPLE_STYLE_WRAPPERS
    if all(char == " " for char in plain_text):
        ignored_style = "underline" if marker == "_" else "strikethrough"
        remaining_styles = tuple(style for style in styles if style != ignored_style)
        return _apply_style_wrappers(
            marker * len(plain_text),
            remaining_styles,
            force_html=force_html,
        )

    leading_count = len(plain_text) - len(plain_text.lstrip(" "))
    trailing_count = len(plain_text) - len(plain_text.rstrip(" "))
    if leading_count == 0 and trailing_count == 0:
        return None
    if not content.startswith(" " * leading_count) or not content.endswith(" " * trailing_count):
        return None
    content_end = len(content) - trailing_count if trailing_count else len(content)
    core = content[leading_count:content_end]
    rendered = f"{marker * leading_count}{core}{marker * trailing_count}"
    return _apply_style_wrappers(rendered, styles, force_html=force_html)


def _apply_style_wrappers(
    content: str,
    styles: tuple[str, ...],
    *,
    force_html: bool = False,
) -> str:
    """给已处理空格的内容添加 Markdown 或 HTML 样式 wrapper。"""
    if not content or not styles:
        return content

    leading = content[: len(content) - len(content.lstrip(" \t"))]
    trailing = content[len(content.rstrip(" \t")) :]
    core = content.strip(" \t")
    if not core:
        return content

    style_key = frozenset(styles)
    wrapper = _SIMPLE_STYLE_WRAPPERS.get(style_key)
    if wrapper is not None and not force_html:
        return f"{leading}{wrapper}{core}{wrapper}{trailing}"
    return f"{leading}{_apply_html_styles(core, styles)}{trailing}"


def _render_visible_whitespace(content: str, styles: tuple[str, ...]) -> str:
    """使用原 HTML 规则保留非 ASCII marker 场景的可见空白。"""
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


__all__ = [
    "render_inline_content",
    "render_inline_nodes",
    "render_internal_link",
    "render_joined_inline_contents",
]
