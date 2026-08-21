# Copyright (c) Opendatalab. All rights reserved.
"""各格式共用的目录页码尾部识别与清理。"""

from __future__ import annotations

import re

from mineru.render._internal.common.inline import inline_plain_text, parse_inline_content

_INDEX_ROMAN_RE = re.compile(r"[ivxlcdm]+", re.IGNORECASE)


def strip_index_page_tail(content: str) -> str:
    """删除目录末尾可信页码，并把其余 tab 转换为普通空格。"""
    if "\t" not in content:
        return content
    head, tail = content.rsplit("\t", 1)
    tail_text = inline_plain_text(parse_inline_content(tail)).strip()
    if looks_like_index_page_token(tail_text):
        content = head
    return content.replace("\t", " ")


def looks_like_index_page_token(content: str) -> bool:
    """判断目录 tab 后缀是否为数字、罗马数字或单字母页码。"""
    if not content or len(content) > 12:
        return False
    return bool(content.isdigit() or _INDEX_ROMAN_RE.fullmatch(content) or re.fullmatch(r"[A-Za-z]", content))


__all__ = ["looks_like_index_page_token", "strip_index_page_tail"]
