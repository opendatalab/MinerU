# Copyright (c) Opendatalab. All rights reserved.
"""MiddleJson 共享行内语义到安全 HTML 的序列化。"""

from __future__ import annotations

import html
import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

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


_AUTOLINK_TLD_PATTERN = r"(?:[A-Za-z]{2,63}|xn--[A-Za-z0-9-]{2,59})"
_AUTOLINK_DOMAIN_PATTERN = (
    rf"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{{0,61}}[A-Za-z0-9])?\.){{1,10}}{_AUTOLINK_TLD_PATTERN}"
)
_AUTOLINK_EMAIL_EDGE_PATTERN = r"[A-Za-z0-9!#$%&'*+/=?^_{|}~-]"
_AUTOLINK_EMAIL_PATTERN = (
    rf"{_AUTOLINK_EMAIL_EDGE_PATTERN}"
    rf"(?:[A-Za-z0-9.!#$%&'*+/=?^_{{|}}~-]{{0,62}}{_AUTOLINK_EMAIL_EDGE_PATTERN})?"
    rf"@{_AUTOLINK_DOMAIN_PATTERN}"
)
_AUTOLINK_EXPLICIT_HOST_PATTERN = (
    rf"(?:{_AUTOLINK_DOMAIN_PATTERN}|localhost|(?:\d{{1,3}}\.){{3}}\d{{1,3}}|\[[0-9A-Fa-f:.]+\])"
)
_AUTOLINK_URL_TAIL_PATTERN = (
    r"(?::\d{1,5})?(?:[/?#](?:(?!https?://)[A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-])*)?"
)
_AUTOLINK_RE = re.compile(
    rf"(?:(?P<email>(?<![A-Za-z0-9._%+@-]){_AUTOLINK_EMAIL_PATTERN})"
    rf"|(?P<explicit>https?://{_AUTOLINK_EXPLICIT_HOST_PATTERN}{_AUTOLINK_URL_TAIL_PATTERN})"
    rf"|(?P<www>(?<![A-Za-z0-9._%+@-])www\.{_AUTOLINK_DOMAIN_PATTERN}{_AUTOLINK_URL_TAIL_PATTERN})"
    rf"|(?P<domain>(?<![A-Za-z0-9._%+@-]){_AUTOLINK_DOMAIN_PATTERN}{_AUTOLINK_URL_TAIL_PATTERN}))",
    re.IGNORECASE,
)
_AUTOLINK_UNSUPPORTED_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:[^\s]*$")
_AUTOLINK_TRAILING_PUNCTUATION = ".,;:!?，。；：！？、"
_AUTOLINK_CLOSING_PAIRS = {")": "(", "]": "[", "}": "{", "）": "（", "】": "【", "》": "《"}
_AUTOLINK_COMMON_BARE_TLDS = frozenset(
    {
        "ai",
        "app",
        "au",
        "biz",
        "br",
        "ca",
        "cloud",
        "cn",
        "com",
        "de",
        "dev",
        "edu",
        "eu",
        "fr",
        "gov",
        "hk",
        "in",
        "info",
        "io",
        "jp",
        "kr",
        "net",
        "nl",
        "nz",
        "online",
        "org",
        "ru",
        "sg",
        "shop",
        "site",
        "store",
        "tech",
        "tw",
        "uk",
        "xyz",
    }
)


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
    linkify_text: bool = True,
    separate_adjacent_math: bool = False,
    preserve_newlines: bool = False,
) -> HtmlInlineResult:
    """渲染行内节点，可分隔相邻公式或给 pre-wrap 容器保留原始换行。"""
    parts: list[str] = []
    has_math = False
    previous_was_math = False
    for node in nodes:
        rendered = _render_inline_node_html(
            node,
            linkify_text=linkify_text,
            preserve_newlines=preserve_newlines,
        )
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


def _render_inline_node_html(
    node: InlineNode,
    *,
    linkify_text: bool,
    preserve_newlines: bool,
) -> HtmlInlineResult:
    """把一个共享行内节点映射为 HTML。"""
    if isinstance(node, InlineText):
        return HtmlInlineResult(
            _render_text_html(
                node.content,
                linkify_text=linkify_text,
                preserve_newlines=preserve_newlines,
            )
        )
    if isinstance(node, InlineEquation):
        return render_math_html(node.latex, display=False)
    if isinstance(node, InlineStyled):
        children = render_inline_nodes_html(
            node.children,
            linkify_text=linkify_text,
            preserve_newlines=preserve_newlines,
        )
        rendered = _apply_html_styles(children.html, node.styles)
        plain_text = inline_plain_text(node.children)
        if _needs_whitespace_preservation(plain_text):
            rendered = f'<span class="mineru-preserve-whitespace">{rendered}</span>'
        return HtmlInlineResult(rendered, children.has_math)
    if isinstance(node, InlineLink):
        children = render_inline_nodes_html(
            node.children,
            linkify_text=False,
            preserve_newlines=preserve_newlines,
        )
        safe_url = sanitize_link_url(node.url)
        if not safe_url or safe_url == ".":
            return children
        href = html.escape(safe_url, quote=True)
        return HtmlInlineResult(
            f'<a href="{href}" rel="noopener noreferrer">{children.html}</a>',
            children.has_math,
        )
    raise TypeError(f"Unsupported inline node: {type(node).__name__}")


def _render_text_html(
    content: str,
    *,
    linkify_text: bool,
    preserve_newlines: bool,
) -> str:
    """转义普通文本，并按需把安全 URL-like 候选转换为链接。"""
    if not linkify_text:
        return _escape_text_with_breaks(content, preserve_newlines=preserve_newlines)

    parts: list[str] = []
    cursor = 0
    for match in _AUTOLINK_RE.finditer(content):
        candidate = match.group(0)
        candidate, trailing = _trim_autolink_candidate(candidate)
        if not candidate or _is_embedded_autolink_candidate(content, match, candidate):
            continue
        href = _autolink_href(match, candidate)
        if href is None:
            continue
        parts.append(_escape_text_with_breaks(content[cursor : match.start()], preserve_newlines=preserve_newlines))
        escaped_href = html.escape(href, quote=True)
        escaped_label = _escape_text_with_breaks(candidate, preserve_newlines=preserve_newlines)
        parts.append(f'<a href="{escaped_href}" rel="noopener noreferrer">{escaped_label}</a>')
        cursor = match.end() - len(trailing)
    parts.append(_escape_text_with_breaks(content[cursor:], preserve_newlines=preserve_newlines))
    return "".join(parts)


def _trim_autolink_candidate(candidate: str) -> tuple[str, str]:
    """去除 URL 尾部句读和不平衡右括号，并返回应保留的原文后缀。"""
    end = len(candidate)
    while end and candidate[end - 1] in _AUTOLINK_TRAILING_PUNCTUATION:
        end -= 1
    changed = True
    while end and changed:
        changed = False
        closing = candidate[end - 1]
        opening = _AUTOLINK_CLOSING_PAIRS.get(closing)
        if opening is not None and candidate[:end].count(closing) > candidate[:end].count(opening):
            end -= 1
            changed = True
    return candidate[:end], candidate[end:]


def _is_embedded_autolink_candidate(
    content: str,
    match: re.Match[str],
    candidate: str,
) -> bool:
    """拒绝未知协议、相对路径和裸 IP 内部的模糊候选。"""
    if match.group("explicit") is not None:
        return False
    prefix = content[max(0, match.start() - 32) : match.start()]
    if _AUTOLINK_UNSUPPORTED_SCHEME_RE.search(prefix) is not None:
        return True
    if match.group("email") is not None:
        return False
    if match.start() and content[match.start() - 1] in "/\\":
        return True
    host = candidate.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].split(":", 1)[0]
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _autolink_href(match: re.Match[str], candidate: str) -> str | None:
    """把候选转换为显式安全 href，并验证端口和显式 IP 地址。"""
    if match.group("email") is not None:
        if ".." in candidate.rsplit("@", 1)[0]:
            return None
        href = f"mailto:{candidate}"
    elif match.group("explicit") is not None:
        href = candidate
    else:
        if match.group("domain") is not None and not _is_allowed_bare_domain(candidate):
            return None
        href = f"https://{candidate}"
    try:
        parsed = urlsplit(href)
        _ = parsed.port
        if parsed.hostname and re.fullmatch(r"\d+(?:\.\d+){3}", parsed.hostname):
            ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return None
    return sanitize_link_url(href)


def _is_allowed_bare_domain(candidate: str) -> bool:
    """仅允许工程常用 TLD 的无协议裸域名，减少文件名和股票代码误判。"""
    authority = re.split(r"[/?#]", candidate, maxsplit=1)[0]
    host = authority.rsplit(":", 1)[0]
    tld = host.rsplit(".", 1)[-1].lower()
    return tld in _AUTOLINK_COMMON_BARE_TLDS


def _escape_text_with_breaks(content: str, *, preserve_newlines: bool) -> str:
    """转义文本，并按调用方要求把换行转换为 HTML br。"""
    escaped = _escape_text(content)
    return escaped if preserve_newlines else escaped.replace("\n", "<br>\n")


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
