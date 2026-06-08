# Copyright (c) Opendatalab. All rights reserved.

import re
from html import escape

CONSERVATIVE_MARKDOWN_SPECIAL_CHARS = ("*", "_", "`", "~", "$")
TEXT_BLOCK_MARKDOWN_PREFIX_RE = re.compile(
    r"^(?P<indent>[ \t]{0,3})(?P<marker>#{1,6}|[+-])(?=[ \t])"
)


def escape_conservative_markdown_text(content: str) -> str:
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

    return "".join(escaped_chars)


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
