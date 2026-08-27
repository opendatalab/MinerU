# Copyright (c) Opendatalab. All rights reserved.
"""安全归一化 XML namespace 与 legacy HTML 前缀标签名。"""

from __future__ import annotations

from lxml import etree  # type: ignore[reportMissingImports]


def local_name(element: etree._Element) -> str:
    """返回 Clark notation、普通或冒号前缀标签的小写本地名。"""
    tag = element.tag
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{") and "}" in tag:
        tag = tag.split("}", 1)[1]
    elif ":" in tag:
        tag = tag.rsplit(":", 1)[1]
    return tag.casefold()


__all__ = ["local_name"]
