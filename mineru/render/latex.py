# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 TeX Live LaTeX 源码的轻量公共门面。"""

from __future__ import annotations

from ..types import MiddleJson


def render_latex(
    middle_json: MiddleJson,
    *,
    asset_base_path: str = "",
    document_title: str | None = None,
) -> str:
    """惰性加载 LaTeX 实现并返回完整 UTF-8 文档源码。"""
    from ._internal.latex.renderer import render_latex as _render_latex

    return _render_latex(
        middle_json,
        asset_base_path=asset_base_path,
        document_title=document_title,
    )


__all__ = ["render_latex"]
