# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 EPUB 3.3 的轻量公共门面。"""

from __future__ import annotations

from datetime import datetime

from ..types import MiddleJson
from .contracts import AssetResolver


def render_epub(
    middle_json: MiddleJson,
    *,
    title: str | None = None,
    authors: tuple[str, ...] = (),
    language: str = "und",
    identifier: str | None = None,
    modified_at: datetime | None = None,
    asset_resolver: AssetResolver | None = None,
) -> bytes:
    """惰性加载 EPUB 实现并返回完整 EPUB 3.3 容器字节。"""
    from ._internal.epub.renderer import render_epub as _render_epub

    return _render_epub(
        middle_json,
        title=title,
        authors=authors,
        language=language,
        identifier=identifier,
        modified_at=modified_at,
        asset_resolver=asset_resolver,
    )


__all__ = ["render_epub"]
