# Copyright (c) Opendatalab. All rights reserved.
"""Unified paragraph-level text merging for Pipeline and VLM backends.

Produces a single Markdown string from a Block's line/spans, handling
language detection, CJK / western spacing, hyphen de-duplication, and
Markdown escaping.
"""

from __future__ import annotations

from ..backend.utils.char_utils import full_to_half_exclude_marks, is_hyphen_at_line_end
from ..types import Block, Line, Span
from ..utils.config_reader import get_latex_delimiter_config
from ..utils.enum_class import BlockType, ContentType
from ..utils.language import detect_lang

CJK_LANGS = frozenset({"zh", "ja", "ko"})

_latex_config = get_latex_delimiter_config()

if isinstance(_latex_config, dict):
    inline_left = _latex_config.get("inline_left", "$")
    inline_right = _latex_config.get("inline_right", "$")
    display_left = _latex_config.get("display_left", "$$")
    display_right = _latex_config.get("display_right", "$$")
else:
    inline_left = "$"
    inline_right = "$"
    display_left = "$$"
    display_right = "$$"

del _latex_config


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #


def merge_para_text(para_block: Block) -> str:
    """Convert a block's lines/spans into a single Markdown string.

    Accepted block types:
      TEXT, TITLE, INDEX, ABSTRACT, REF_TEXT, LIST, INTERLINE_EQUATION, PHONETIC
      {IMAGE/TABLE/CHART/CODE)_(CAPTION/FOOTNOTE), CODE_BODY

    Handles fenced code blocks, Markdown escaping, and TEXT-block prefix
    escape automatically.
    """
    if _is_fenced_code_block(para_block):
        code_text = _merge_para_text(para_block, escape_markdown=False, list_line_break="\n")
        if not code_text:
            return ""
        code_text = "\n".join(line.rstrip() for line in code_text.split("\n"))
        guess_lang = para_block.guess_lang or "txt"
        return f"```{guess_lang}\n{code_text}\n```"

    if para_block.type == BlockType.LIST and len(para_block.blocks):
        # (VLM) list block
        list_text = ""
        for block in para_block.blocks:
            item_text = _merge_para_text(block)
            list_text += f"{item_text}  \n"
        return list_text

    para_text = _merge_para_text(para_block)
    if para_block.type == BlockType.TEXT:  # (VLM) list do not escape
        para_text = _escape_text_block_markdown_prefix(para_text)
    return para_text


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

import re  # noqa: E402

_TEXT_BLOCK_MD_PREFIX_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<marker>#{1,6}|[+-])(?=[ \t])")


def _escape_text_block_markdown_prefix(content: str) -> str:
    """Escape a leading Markdown block marker in an assembled text block."""
    if not content:
        return content
    match = _TEXT_BLOCK_MD_PREFIX_RE.match(content)
    if not match:
        return content
    marker_start = match.start("marker")
    return f"{content[:marker_start]}\\{content[marker_start:]}"


def _is_fenced_code_block(para_block: Block) -> bool:
    return para_block.type == BlockType.CODE_BODY and para_block.sub_type == BlockType.CODE


def _normalize_text_content(content: str) -> str:
    return full_to_half_exclude_marks(content or "")


def _collect_text_for_lang_detection(para_block: Block) -> str:
    parts: list[str] = []
    for line in para_block.lines:
        for span in line.spans:
            if span.type == ContentType.TEXT:
                parts.append(_normalize_text_content(span.content))
    return "".join(parts)


def _line_prefix(line_idx: int, line: Line, list_line_break: str = "  \n") -> str:
    if line_idx >= 1 and line._is_list_start:
        return list_line_break
    return ""


_MARKDOWN_SPECIAL = frozenset({"*", "_", "`", "~", "$"})


def _escape_conservative_markdown_text(content: str) -> str:
    """Escape plain-text characters that carry inline Markdown semantics."""
    if not content:
        return content
    escaped: list[str] = []
    backslashes = 0
    for char in content:
        if char == "\\":
            escaped.append(char)
            backslashes += 1
            continue
        if char in _MARKDOWN_SPECIAL and backslashes % 2 == 0:
            escaped.append("\\")
        escaped.append(char)
        backslashes = 0
    return "".join(escaped)


def _render_span(span: Span, *, escape_markdown: bool = True) -> tuple[str, str] | None:
    span_type = span.type
    content = ""

    if span_type == ContentType.TEXT:
        content = _normalize_text_content(span.content)  # (VLM) SKIP
        if escape_markdown:
            content = _escape_conservative_markdown_text(content)
    elif span_type == ContentType.INLINE_EQUATION:
        if span.content:
            content = f"{inline_left}{span.content}{inline_right}"
    elif span_type == ContentType.INTERLINE_EQUATION:
        if span.content:
            content = f"\n{display_left}\n{span.content}\n{display_right}\n"
        # (VLM) if not(formula_enable) use image link
    else:
        return None

    content = content.strip()
    if not content:
        return None
    return span_type, content


def _next_line_starts_with_lowercase_text(para_block: Block, line_idx: int) -> bool:
    if line_idx + 1 >= len(para_block.lines):
        return False
    next_spans = para_block.lines[line_idx + 1].spans
    if not next_spans:
        return False
    first = next_spans[0]
    if first.type != ContentType.TEXT:
        return False
    first_content = _normalize_text_content(first.content)
    return bool(first_content) and first_content[0].islower()


def _join_rendered_span(
    para_block: Block, block_lang: str, line: Line, line_idx: int, span_idx: int, span_type: str, content: str
) -> tuple[str, str]:
    if span_type == ContentType.INTERLINE_EQUATION:
        return content, ""

    is_last_span = bool(span_idx == len(line.spans) - 1)

    # 中文/日语/韩文语境下，换行不需要空格分隔,但是如果是行内公式结尾，还是要加空格
    if block_lang in CJK_LANGS:
        if is_last_span and span_type != ContentType.INLINE_EQUATION:
            return content, ""
        # (VLM) use _has_following_joinable_span()
        return content, " "

    if span_type not in (ContentType.TEXT, ContentType.INLINE_EQUATION):
        return content, ""

    if is_last_span and span_type == ContentType.TEXT and is_hyphen_at_line_end(content):
        if _next_line_starts_with_lowercase_text(para_block, line_idx):
            return content[:-1], ""
        return content, ""
    # 西方文本语境下 content间需要空格分隔
    return content, " "


# ------------------------------------------------------------------ #
#  Core merge
# ------------------------------------------------------------------ #


def _merge_para_text(para_block: Block, *, escape_markdown: bool = True, list_line_break: str = "  \n") -> str:
    """Core paragraph rendering shared by Pipeline and VLM backends."""
    block_lang = detect_lang(_collect_text_for_lang_detection(para_block))
    para_parts: list[str] = []
    # (VLM) escape_markdown_text = para_block.type != BlockType.CODE_BODY

    for line_idx, line in enumerate(para_block.lines):
        # (PIPELINE ONLY)
        if prefix := _line_prefix(line_idx, line, list_line_break):
            para_parts.append(prefix)

        for span_idx, span in enumerate(line.spans):
            rendered = _render_span(span, escape_markdown=escape_markdown)
            if rendered is None:
                continue

            span_type, content = rendered
            content, suffix = _join_rendered_span(para_block, block_lang, line, line_idx, span_idx, span_type, content)
            para_parts.append(content)
            if suffix:
                para_parts.append(suffix)

    return "".join(para_parts).rstrip()
