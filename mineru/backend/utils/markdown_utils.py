# Copyright (c) Opendatalab. All rights reserved.

import re
from html import escape

CONSERVATIVE_MARKDOWN_SPECIAL_CHARS = ("*", "_", "`", "~", "$")
TEXT_BLOCK_MARKDOWN_PREFIX_RE = re.compile(
    r"^(?P<indent>[ \t]{0,3})(?P<marker>#{1,6}|[+-])(?=[ \t])"
)
BARE_HTTP_URL_RE = re.compile(
    r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+"
)
TRAILING_URL_PUNCTUATION = ".,;:!?"
TRAILING_URL_BRACKETS = {
    ")": "(",
    "]": "[",
    "}": "{",
}


def escape_conservative_markdown_text(
    content: str,
    protect_bare_urls: bool = True,
) -> str:
    """Escape plain-text characters that carry inline Markdown semantics."""
    if not content:
        return content

    escaped_chars = []
    preceding_backslashes = 0

    for char in content:
        if char == "\\":
            escaped_chars.append(char)
            preceding_backslashes += 1
            continue

        if (
            char in CONSERVATIVE_MARKDOWN_SPECIAL_CHARS
            and preceding_backslashes % 2 == 0
        ):
            escaped_chars.append("\\")

        escaped_chars.append(char)
        preceding_backslashes = 0

    escaped_content = "".join(escaped_chars)
    if protect_bare_urls:
        return protect_bare_urls_in_markdown_text(escaped_content)
    return escaped_content


def protect_bare_urls_in_markdown_text(content: str) -> str:
    """将普通文本中的裸 URL 包成显式 autolink，避免预览器误吞后续正文。"""
    if not content:
        return content

    return BARE_HTTP_URL_RE.sub(
        lambda match: _render_protected_bare_url(content, match),
        content,
    )


def _render_protected_bare_url(content: str, match: re.Match) -> str:
    """渲染一个裸 URL 命中；已有 Markdown/HTML 链接上下文保持原样。"""
    if _is_existing_link_url_context(content, match.start()):
        return match.group(0)

    url, trailing = _split_trailing_url_punctuation(match.group(0))
    if not url:
        return match.group(0)

    return f"<{url}>{trailing}"


def _is_existing_link_url_context(content: str, url_start: int) -> bool:
    """判断 URL 是否已经处在 autolink、Markdown 链接或 HTML 属性中。"""
    if url_start <= 0:
        return False

    previous_char = content[url_start - 1]
    prefix = content[max(0, url_start - 10):url_start].lower()

    if previous_char == "<":
        return True
    if previous_char == "(" and url_start >= 2 and content[url_start - 2] == "]":
        return True
    return prefix.endswith(('href="', "href='", 'src="', "src='", "href=", "src="))


def _split_trailing_url_punctuation(url: str) -> tuple[str, str]:
    """拆出 URL 末尾不属于地址本体的标点，尤其是正文括号和句读符号。"""
    trailing = []
    while url:
        last_char = url[-1]
        if last_char in TRAILING_URL_PUNCTUATION:
            trailing.append(last_char)
            url = url[:-1]
            continue
        if _is_unbalanced_trailing_bracket(url, last_char):
            trailing.append(last_char)
            url = url[:-1]
            continue
        break

    return url, "".join(reversed(trailing))


def _is_unbalanced_trailing_bracket(url: str, last_char: str) -> bool:
    """判断 URL 末尾闭合括号是否更像正文标点，而不是 URL path 的一部分。"""
    open_char = TRAILING_URL_BRACKETS.get(last_char)
    if open_char is None:
        return False
    return url.count(last_char) > url.count(open_char)


def escape_text_block_markdown_prefix(content: str) -> str:
    """Escape a leading Markdown block marker in an assembled text block."""
    if not content:
        return content

    match = TEXT_BLOCK_MARKDOWN_PREFIX_RE.match(content)
    if not match:
        return content

    marker_start = match.start("marker")
    return f"{content[:marker_start]}\\{content[marker_start:]}"


def render_algorithm_html_from_lines(
    lines: list[dict],
    inline_left_delimiter: str,
    inline_right_delimiter: str,
    text_normalizer=None,
) -> str:
    """将 algorithm 的行内 span 渲染为 HTML，以同时保留缩进和公式渲染能力。"""
    html_parts = []
    previous_span_type = None
    for line in lines or []:
        for span in line.get("spans", []):
            span_type = span.get("type")
            content = span.get("content", "")
            if content is None:
                content = ""

            if span_type == "text":
                if text_normalizer is not None:
                    content = text_normalizer(content)
                html_parts.append(escape(str(content), quote=False))
                if content:
                    previous_span_type = span_type
            elif span_type == "inline_equation":
                if str(content).strip():
                    if (
                        previous_span_type == "inline_equation"
                        and html_parts
                        and not html_parts[-1].endswith((" ", "\n", "\t"))
                    ):
                        html_parts.append(" ")
                    html_parts.append(
                        f"{inline_left_delimiter}"
                        f"{escape(str(content), quote=False)}"
                        f"{inline_right_delimiter}"
                    )
                    previous_span_type = span_type

    html_body = "".join(html_parts)
    if not html_body.strip():
        return ""

    return (
        '<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">\n'
        f"{html_body}\n"
        "</div>"
    )
