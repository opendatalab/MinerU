# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 PDF bytes 的轻量公共门面。"""

from __future__ import annotations

from ..types import MiddleJson
from .contracts import AssetResolver


def render_pdf(
    middle_json: MiddleJson,
    *,
    asset_resolver: AssetResolver | None = None,
    document_title: str | None = None,
) -> bytes:
    """惰性加载 PDF 实现并渲染严格 MiddleJson。"""
    from ._internal.pdf.renderer import render_pdf as _render_pdf

    return _render_pdf(
        middle_json,
        asset_resolver=asset_resolver,
        document_title=document_title,
    )


__all__ = ["render_pdf"]
