# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 Structured Content 的轻量公共门面。"""

from __future__ import annotations

from typing import Any

from ..types import MiddleJson


def render_structured_content(
    middle_json: MiddleJson,
    *,
    asset_base_url: str = "",
) -> dict[str, Any]:
    """惰性加载 Structured Content 实现并渲染严格 MiddleJson。"""
    from ._internal.structured_content.renderer import render_structured_content as _render_structured_content

    return _render_structured_content(middle_json, asset_base_url=asset_base_url)


__all__ = ["render_structured_content"]
