# Copyright (c) Opendatalab. All rights reserved.

import re

CONSERVATIVE_MARKDOWN_SPECIAL_CHARS = ("*", "_", "`", "~", "$")
TEXT_BLOCK_MARKDOWN_PREFIX_RE = re.compile(
    r"^(?P<indent>[ \t]{0,3})(?P<marker>#{1,6}|[+-])(?=[ \t])"
)


def escape_conservative_markdown_text(content: str) -> str:
    """Escape plain-text characters that carry inline Markdown semantics."""
    if not content:
        return content

    for char in CONSERVATIVE_MARKDOWN_SPECIAL_CHARS:
        content = content.replace(char, "\\" + char)

    return content


def escape_text_block_markdown_prefix(content: str) -> str:
    """Escape a leading Markdown block marker in an assembled text block."""
    if not content:
        return content

    match = TEXT_BLOCK_MARKDOWN_PREFIX_RE.match(content)
    if not match:
        return content

    marker_start = match.start("marker")
    return f"{content[:marker_start]}\\{content[marker_start:]}"
