# Copyright (c) Opendatalab. All rights reserved.
"""Flash 各格式复用的超链接目标校验与内部协议转义。"""

from __future__ import annotations

import html
from collections.abc import Collection
from typing import Any, Final
from urllib.parse import urlsplit


DEFAULT_EXTERNAL_HYPERLINK_SCHEMES: Final = frozenset({"http", "https", "mailto", "tel"})
OFFICE_EXTERNAL_HYPERLINK_SCHEMES: Final = DEFAULT_EXTERNAL_HYPERLINK_SCHEMES | {"ftp"}


def escape_inline_protocol_text(value: Any) -> str:
    """把不可信字面文本转义为不会重建 MinerU 行内协议的内容。"""
    return html.escape(str(value), quote=False)


def sanitize_hyperlink_target(
    value: Any,
    *,
    allowed_schemes: Collection[str] = DEFAULT_EXTERNAL_HYPERLINK_SCHEMES,
    allow_relative: bool = False,
    allow_fragment: bool = False,
    allow_root_relative: bool = False,
) -> str | None:
    """按格式策略保留安全外链、相对链接或文档内 fragment。"""
    if value is None:
        return None
    normalized = str(value).strip()
    if (
        not normalized
        or normalized == "."
        or any(ord(char) < 0x20 or ord(char) == 0x7F or 0xD800 <= ord(char) <= 0xDFFF for char in normalized)
        or any(char in normalized for char in ("<", ">", "\\"))
        or normalized.startswith("//")
    ):
        return None
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return None
    scheme = parsed.scheme.casefold()
    allowed = {item.casefold() for item in allowed_schemes}
    if scheme:
        if scheme not in allowed:
            return None
        if scheme in {"http", "https", "ftp"}:
            try:
                if not parsed.netloc or parsed.hostname is None:
                    return None
            except ValueError:
                return None
        elif not parsed.path:
            return None
        return normalized
    if parsed.netloc:
        return None
    if normalized.startswith("#"):
        return normalized if allow_fragment and normalized[1:].strip() else None
    if normalized.startswith("/") and not allow_root_relative:
        return None
    return normalized if allow_relative else None


def render_inline_hyperlink(label_markup: str, target: str) -> str:
    """把已安全构造的标签内容和目标写入统一 hyperlink 行内协议。"""
    if not label_markup or not target:
        return label_markup
    escaped_target = escape_inline_protocol_text(target)
    return f"<hyperlink>{label_markup}<url>{escaped_target}</url></hyperlink>"


__all__ = [
    "DEFAULT_EXTERNAL_HYPERLINK_SCHEMES",
    "OFFICE_EXTERNAL_HYPERLINK_SCHEMES",
    "escape_inline_protocol_text",
    "render_inline_hyperlink",
    "sanitize_hyperlink_target",
]
