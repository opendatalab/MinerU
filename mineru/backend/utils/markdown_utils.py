# Copyright (c) Opendatalab. All rights reserved.

CONSERVATIVE_MARKDOWN_SPECIAL_CHARS = ("*", "_", "`", "~", "$")


def escape_conservative_markdown_text(content: str) -> str:
    """Escape plain-text characters that carry inline Markdown semantics."""
    if not content:
        return content

    for char in CONSERVATIVE_MARKDOWN_SPECIAL_CHARS:
        content = content.replace(char, "\\" + char)

    return content
