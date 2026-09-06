# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 DOCX 的轻量公共门面与稳定异常。"""

from __future__ import annotations

from ..types import MiddleJson
from .contracts import AssetResolver


from docvortex.render.docx import DocxRenderError


def render_docx(
    middle_json: MiddleJson,
    *,
    asset_resolver: AssetResolver | None = None,
) -> bytes:
    """惰性加载 DOCX 实现并渲染严格 MiddleJson。"""
    from docvortex.render.docx import render_docx as _render_docx

    return _render_docx(middle_json, asset_resolver=asset_resolver)


__all__ = ["DocxRenderError", "render_docx"]
