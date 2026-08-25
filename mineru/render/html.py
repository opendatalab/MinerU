# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 HTML 的轻量公共门面。"""

from __future__ import annotations

from ..types import MiddleJson
from .contracts import RenderMode


def render_html(
    middle_json: MiddleJson,
    *,
    mode: RenderMode = RenderMode.DEFAULT,
    asset_base_url: str = "",
    standalone: bool = True,
    document_title: str | None = None,
) -> str:
    """惰性加载 HTML 实现并渲染严格 MiddleJson。"""
    from ._internal.html.renderer import render_html as _render_html

    return _render_html(
        middle_json,
        mode=mode,
        asset_base_url=asset_base_url,
        standalone=standalone,
        document_title=document_title,
    )


__all__ = ["render_html"]
