# Copyright (c) Opendatalab. All rights reserved.
"""Middle JSON 3.0 行内 Span 到 Markdown 的安全序列化。"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

from ....backend.postprocess.inline import (
    join_inline_spans,
)
from ....config import LatexDelimitersConfig
from ....types import CodeInlineSpan, EquationInlineSpan, HyperlinkSpan, InlineSpan, TextSpan
from .escaping import escape_conservative_markdown_text

_HTML_LIKE_TEXT_RE = re.compile(r"</?[A-Za-z][^<>\n]*>|<!--.*?-->|<![A-Za-z][^<>\n]*>|<\?[^<>\n]*\?>")
_ENTITY_LIKE_RE = re.compile(r"&(?:#[xX][0-9A-Fa-f]+|#[0-9]+|[A-Za-z][A-Za-z0-9]+);?")
_SIMPLE_STYLE_WRAPPERS = {
    frozenset({"bold"}): "**",
    frozenset({"italic"}): "*",
    frozenset({"strikethrough"}): "~~",
    frozenset({"bold", "italic"}): "***",
}


def render_inline_content(content: list[InlineSpan], delimiters: LatexDelimitersConfig) -> str:
    """把一段 MiddleJson 行内内容渲染为 Markdown。"""
    return render_inline_spans(content, delimiters)


def render_joined_inline_contents(contents: list[list[InlineSpan]], delimiters: LatexDelimitersConfig) -> str:
    """按物理段落边界规则合并多段 content 后渲染 Markdown。"""
    return render_inline_spans(join_inline_spans(contents), delimiters)


def render_inline_spans(spans: list[InlineSpan], delimiters: LatexDelimitersConfig) -> str:
    """把行内 Span 序列化为 Markdown 与必要的安全 HTML。"""
    return "".join(_render_inline_span(span, delimiters) for span in spans)


def render_inline_spans_in_html_context(spans: list[InlineSpan], delimiters: LatexDelimitersConfig) -> str:
    """把 Markdown raw HTML 容器内的 Span 全部序列化为安全 HTML 行内语法。"""
    return "".join(_render_inline_span_in_html_context(span, delimiters) for span in spans)


def _render_inline_span(span: InlineSpan, delimiters: LatexDelimitersConfig) -> str:
    """渲染单个结构化行内 Span。"""
    if isinstance(span, TextSpan):
        content = _escape_plain_markdown_text(span.content)
        return _apply_styles(content, span.content, span.styles)
    if isinstance(span, CodeInlineSpan):
        return _render_inline_code(span.content)
    if isinstance(span, EquationInlineSpan):
        return f"{delimiters.inline.left}{span.content}{delimiters.inline.right}"
    if isinstance(span, HyperlinkSpan):
        label = render_inline_spans(list(span.content), delimiters)
        return _render_link(label, span.url, _requires_html_link(list(span.content)))
    raise TypeError(f"Unsupported inline span: {type(span).__name__}")


def _render_inline_span_in_html_context(span: InlineSpan, delimiters: LatexDelimitersConfig) -> str:
    """渲染一个嵌入 Markdown raw HTML block 的结构化行内 Span。"""
    if isinstance(span, TextSpan):
        return _apply_html_styles(html.escape(span.content, quote=False), span.styles)
    if isinstance(span, CodeInlineSpan):
        return f"<code>{html.escape(span.content, quote=False)}</code>"
    if isinstance(span, EquationInlineSpan):
        return html.escape(f"{delimiters.inline.left}{span.content}{delimiters.inline.right}", quote=False)
    if isinstance(span, HyperlinkSpan):
        label = render_inline_spans_in_html_context(list(span.content), delimiters)
        return _render_link(label, span.url, True)
    raise TypeError(f"Unsupported inline span: {type(span).__name__}")


def _render_inline_code(content: str) -> str:
    """选择长于内容中反引号游程的 fence，稳定输出 Markdown 行内代码。"""
    normalized = content.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", normalized)), default=0)
    fence = "`" * (longest + 1)
    if normalized.startswith(("`", " ")) or normalized.endswith(("`", " ")):
        return f"{fence} {normalized} {fence}"
    return f"{fence}{normalized}{fence}"


def _escape_plain_markdown_text(content: str) -> str:
    """转义 Markdown 符号，并把普通文字中的标签外观保持为惰性实体。"""
    parts: list[str] = []
    cursor = 0
    for match in _HTML_LIKE_TEXT_RE.finditer(content):
        parts.append(_escape_entity_like_text(escape_conservative_markdown_text(content[cursor : match.start()])))
        parts.append(html.escape(match.group(0), quote=False))
        cursor = match.end()
    parts.append(_escape_entity_like_text(escape_conservative_markdown_text(content[cursor:])))
    return "".join(parts)


def _escape_entity_like_text(content: str) -> str:
    """保护会被下游 Markdown 解析器当作 HTML 实体的字面量文本。"""

    def replace(match: re.Match[str]) -> str:
        """只保护确实会被 HTML 实体解码器改写的候选。"""
        candidate = match.group(0)
        return f"&amp;{candidate[1:]}" if html.unescape(candidate) != candidate else candidate

    return _ENTITY_LIKE_RE.sub(replace, content)


def _apply_styles(content: str, plain_text: str, styles: Sequence[str]) -> str:
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


def _get_visible_space_marker(styles: Sequence[str]) -> str | None:
    """按 dev 规则选择可见空格 marker，下划线优先于删除线。"""
    if "underline" in styles:
        return "_"
    if "strikethrough" in styles:
        return "-"
    return None


def _render_visible_space_marker_text(
    content: str,
    plain_text: str,
    styles: Sequence[str],
    marker: str,
) -> str | None:
    """把纯 ASCII 空格或非空文本首尾空格转换为可见 marker。"""
    if not plain_text:
        return None
    style_key = frozenset(styles)
    force_html = style_key not in _SIMPLE_STYLE_WRAPPERS
    if all(char == " " for char in plain_text):
        ignored_style = "underline" if marker == "_" else "strikethrough"
        remaining_styles = [style for style in styles if style != ignored_style]
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
    styles: Sequence[str],
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


def _render_visible_whitespace(content: str, styles: Sequence[str]) -> str:
    """使用原 HTML 规则保留非 ASCII marker 场景的可见空白。"""
    visible = "".join("<br>" if char == "\n" else "&nbsp;" for char in content.expandtabs(4))
    return _apply_html_styles(visible, styles)


def _apply_html_styles(content: str, styles: Sequence[str]) -> str:
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


def _requires_html_link(spans: list[InlineSpan]) -> bool:
    """判断链接标签是否含不适合嵌入 Markdown link 的复杂样式。"""
    for span in spans:
        if isinstance(span, TextSpan):
            if span.styles and frozenset(span.styles) not in _SIMPLE_STYLE_WRAPPERS:
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
    "render_inline_spans",
    "render_inline_spans_in_html_context",
    "render_internal_link",
    "render_joined_inline_contents",
]
