# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 Content List V1 的轻量公共门面。"""

from __future__ import annotations

from typing import Any

from ..types import MiddleJson


def render_content_list(
    middle_json: MiddleJson,
    *,
    asset_base_url: str = "",
) -> list[dict[str, Any]]:
    """惰性加载 Content List V1 实现并渲染严格 MiddleJson。"""
    from ._internal.content_list.v1 import render_content_list as _render_content_list

    return _render_content_list(middle_json, asset_base_url=asset_base_url)


__all__ = ["render_content_list"]
