# Copyright (c) Opendatalab. All rights reserved.
"""各格式共用的目录页码尾部识别与清理。"""

from __future__ import annotations

import re

from ....backend.postprocess.inline import inline_plain_text, map_text_span_content, normalize_inline_spans, slice_inline_spans
from ....types import InlineSpan

_INDEX_ROMAN_RE = re.compile(r"[ivxlcdm]+", re.IGNORECASE)


def strip_index_page_tail(content: list[InlineSpan]) -> list[InlineSpan]:
    """删除目录末尾可信页码，并把其余 tab 转换为普通空格。"""
    content = normalize_inline_spans(content)
    visible_text = inline_plain_text(content)
    if "\t" not in visible_text:
        return content
    tab_offset = visible_text.rfind("\t")
    tail_text = visible_text[tab_offset + 1 :].strip()
    if looks_like_index_page_token(tail_text):
        content = slice_inline_spans(content, 0, tab_offset)
    return map_text_span_content(content, lambda value: value.replace("\t", " "))


def looks_like_index_page_token(content: str) -> bool:
    """判断目录 tab 后缀是否为数字、罗马数字或单字母页码。"""
    if not content or len(content) > 12:
        return False
    return bool(content.isdigit() or _INDEX_ROMAN_RE.fullmatch(content) or re.fullmatch(r"[A-Za-z]", content))


__all__ = ["looks_like_index_page_token", "strip_index_page_tail"]
