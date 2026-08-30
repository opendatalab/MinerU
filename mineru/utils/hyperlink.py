# Copyright (c) Opendatalab. All rights reserved.
"""不依赖解析器层的安全超链接目标校验。"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Final
from urllib.parse import urlsplit


DEFAULT_EXTERNAL_HYPERLINK_SCHEMES: Final = frozenset({"http", "https", "mailto", "tel"})
OFFICE_EXTERNAL_HYPERLINK_SCHEMES: Final = DEFAULT_EXTERNAL_HYPERLINK_SCHEMES | {"ftp"}


def sanitize_hyperlink_target(
    value: Any,
    *,
    allowed_schemes: Collection[str] = DEFAULT_EXTERNAL_HYPERLINK_SCHEMES,
    allow_relative: bool = False,
    allow_fragment: bool = False,
    allow_root_relative: bool = False,
) -> str | None:
    """按调用方策略保留安全外链、相对链接或文档内 fragment。"""
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


__all__ = [
    "DEFAULT_EXTERNAL_HYPERLINK_SCHEMES",
    "OFFICE_EXTERNAL_HYPERLINK_SCHEMES",
    "sanitize_hyperlink_target",
]
