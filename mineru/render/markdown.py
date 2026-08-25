# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 Markdown 的轻量公共门面。"""

from __future__ import annotations

from ..types import MiddleJson
from .contracts import ImageRenderer, RenderMode


def render_markdown(
    middle_json: MiddleJson,
    *,
    mode: RenderMode = RenderMode.DEFAULT,
    asset_base_url: str = "",
    image_renderer: ImageRenderer | None = None,
) -> str:
    """惰性加载 Markdown 实现并渲染严格 MiddleJson。"""
    from ._internal.markdown.renderer import render_markdown as _render_markdown

    return _render_markdown(
        middle_json,
        mode=mode,
        asset_base_url=asset_base_url,
        image_renderer=image_renderer,
    )


__all__ = ["render_markdown"]
