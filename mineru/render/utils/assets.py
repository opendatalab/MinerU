# Copyright (c) Opendatalab. All rights reserved.
"""Markdown 与嵌入 HTML 共用的图片资源解析。"""

from __future__ import annotations

import re
from urllib.parse import quote

from mineru.types import ImagePayloadBlock

_HTML_IMAGE_SRC_RE = re.compile(
    r"(?P<prefix>\bsrc\s*=\s*)(?P<quote>[\"'])(?P<src>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)


def resolve_image_source(block: ImagePayloadBlock, asset_base_url: str = "") -> str | None:
    """按 image_path 优先、image_base64 兜底解析图片来源。"""
    if block.image_path:
        return join_asset_base_url(asset_base_url, block.image_path)
    if block.image_base64:
        return block.image_base64
    return None


def join_asset_base_url(asset_base_url: str, relative_path: str) -> str:
    """使用 POSIX 语义拼接资源根地址与安全相对路径。"""
    if not asset_base_url:
        return relative_path
    return f"{asset_base_url.rstrip('/')}/{relative_path.lstrip('/')}"


def build_markdown_image(source: str, alt: str = "") -> str:
    """构造不会被空格或括号截断的 Markdown 图片语法。"""
    if not source:
        return ""
    safe_alt = alt.replace("[", r"\[").replace("]", r"\]")
    if source.startswith("data:"):
        destination = source
    else:
        destination = quote(source, safe="/:#?&=%@+~,;!$'*-._")
    return f"![{safe_alt}]({destination})"


def prefix_html_image_sources(markup: str, asset_base_url: str = "") -> str:
    """给 HTML 中的相对图片地址添加资源根地址。"""
    if not markup or not asset_base_url:
        return markup

    def _replace(match: re.Match[str]) -> str:
        """只重写相对 src，保留 data URI、绝对 URL 与根路径。"""
        source = match.group("src")
        if _is_absolute_image_source(source):
            return match.group(0)
        resolved = join_asset_base_url(asset_base_url, source)
        return f"{match.group('prefix')}{match.group('quote')}{resolved}{match.group('quote')}"

    return _HTML_IMAGE_SRC_RE.sub(_replace, markup)


def _is_absolute_image_source(source: str) -> bool:
    """判断 HTML 图片地址是否已经是绝对或不可前缀化的来源。"""
    normalized = source.strip().lower()
    return normalized.startswith(("data:", "http://", "https://", "//", "/", "#"))


__all__ = [
    "build_markdown_image",
    "join_asset_base_url",
    "prefix_html_image_sources",
    "resolve_image_source",
]
