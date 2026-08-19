# Copyright (c) Opendatalab. All rights reserved.
"""MiddleJson 共享行内语义到安全 HTML 的序列化。"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

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
from mineru.render._internal.html.sanitizer import sanitize_link_url


@dataclass(frozen=True, slots=True)
class HtmlInlineResult:
    """保存一段行内 HTML 及其是否包含需由 MathJax 处理的公式。"""

    html: str
    has_math: bool = False


def render_inline_content_html(content: str) -> HtmlInlineResult:
    """把一段 MiddleJson 行内内容渲染为安全 HTML。"""
    return render_inline_nodes_html(parse_inline_content(content))


def render_joined_inline_contents_html(contents: list[str]) -> HtmlInlineResult:
    """按共享物理段落边界规则合并多段内容后渲染 HTML。"""
    return render_inline_nodes_html(join_inline_contents(contents))


def render_inline_nodes_html(
    nodes: list[InlineNode],
    *,
    separate_adjacent_math: bool = False,
    preserve_newlines: bool = False,
) -> HtmlInlineResult:
    """渲染行内节点，可分隔相邻公式或给 pre-wrap 容器保留原始换行。"""
    parts: list[str] = []
    has_math = False
    previous_was_math = False
    for node in nodes:
        rendered = _render_inline_node_html(node, preserve_newlines=preserve_newlines)
        if not rendered.html:
            continue
        current_is_math = isinstance(node, InlineEquation)
        if separate_adjacent_math and previous_was_math and current_is_math:
            parts.append(" ")
        parts.append(rendered.html)
        has_math = rendered.has_math or has_math
        previous_was_math = current_is_math
    return HtmlInlineResult("".join(parts), has_math)


def render_math_html(latex: str, *, display: bool) -> HtmlInlineResult:
    """把裸 LaTeX 放入只由 MathJax 扫描的行内或行间公式载体。"""
    normalized = latex.strip()
    if not normalized:
        return HtmlInlineResult("")
    normalized = _neutralize_math_closing_delimiter(normalized, "]" if display else ")")
    escaped = _escape_text(normalized)
    if display:
        return HtmlInlineResult(
            f'<div class="mineru-math mineru-math--block">\\[\n{escaped}\n\\]</div>',
            has_math=True,
        )
    return HtmlInlineResult(
        f'<span class="mineru-math mineru-math--inline">\\({escaped}\\)</span>',
        has_math=True,
    )


def _render_inline_node_html(node: InlineNode, *, preserve_newlines: bool) -> HtmlInlineResult:
    """把一个共享行内节点映射为 HTML。"""
    if isinstance(node, InlineText):
        escaped = _escape_text(node.content)
        return HtmlInlineResult(escaped if preserve_newlines else escaped.replace("\n", "<br>\n"))
    if isinstance(node, InlineEquation):
        return render_math_html(node.latex, display=False)
    if isinstance(node, InlineStyled):
        children = render_inline_nodes_html(node.children, preserve_newlines=preserve_newlines)
        rendered = _apply_html_styles(children.html, node.styles)
        plain_text = inline_plain_text(node.children)
        if _needs_whitespace_preservation(plain_text):
            rendered = f'<span class="mineru-preserve-whitespace">{rendered}</span>'
        return HtmlInlineResult(rendered, children.has_math)
    if isinstance(node, InlineLink):
        children = render_inline_nodes_html(node.children, preserve_newlines=preserve_newlines)
        safe_url = sanitize_link_url(node.url)
        if not safe_url or safe_url == ".":
            return children
        href = html.escape(safe_url, quote=True)
        return HtmlInlineResult(
            f'<a href="{href}" rel="noopener noreferrer">{children.html}</a>',
            children.has_math,
        )
    raise TypeError(f"Unsupported inline node: {type(node).__name__}")


def _apply_html_styles(content: str, styles: tuple[str, ...]) -> str:
    """按 Markdown renderer 的稳定顺序应用共享富文本样式。"""
    if not content:
        return content
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
        content = f'<span class="mineru-text-emphasis">{content}</span>'
    return content


def _escape_text(content: str) -> str:
    """转义普通文本，并替换 HTML 不允许的 C0 控制字符。"""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]", "\ufffd", normalized)
    return html.escape(normalized, quote=False)


def _needs_whitespace_preservation(content: str) -> bool:
    """判断富文本是否包含浏览器默认会折叠的有效空白。"""
    return bool(content and (content != content.strip(" \t\n") or "  " in content or "\t" in content or "\n" in content))


def _neutralize_math_closing_delimiter(latex: str, closing: str) -> str:
    """把公式体内奇数反斜杠引出的结束定界符改写为等价 TeX，防止提前闭合。"""
    token = re.escape(closing)

    def _replace(match: re.Match[str]) -> str:
        """保留成对反斜杠，并把最后一个定界反斜杠改为 mathclose。"""
        slashes = match.group("slashes")
        if len(slashes) % 2 == 0:
            return match.group(0)
        prefix = "\\" * (len(slashes) - 1)
        return f"{prefix}\\mathclose{{{closing}}}"

    return re.sub(rf"(?P<slashes>\\+){token}", _replace, latex)


__all__ = [
    "HtmlInlineResult",
    "render_inline_content_html",
    "render_inline_nodes_html",
    "render_joined_inline_contents_html",
    "render_math_html",
]
