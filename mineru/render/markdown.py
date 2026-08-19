# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 Markdown 的公共渲染实现。"""

from __future__ import annotations

from mineru.config import LatexDelimitersConfig, config
from mineru.render.contracts import MarkdownRenderMode
from mineru.render._internal.common.planner import PlannedBlock, build_render_plan
from mineru.render._internal.markdown.blocks import render_planned_block
from mineru.types import PAGE_AUXILIARY_BLOCK_TYPES, MiddleJson

_PAGE_SEPARATOR = "\n\n---\n\n"


def render_markdown(
    middle_json: MiddleJson,
    *,
    mode: MarkdownRenderMode = MarkdownRenderMode.DEFAULT,
    asset_base_url: str = "",
) -> str:
    """把严格 MiddleJson 纯函数式渲染为 Markdown 字符串。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render_markdown expects a MiddleJson instance")
    if not isinstance(mode, MarkdownRenderMode):
        raise TypeError("mode must be a MarkdownRenderMode value")

    delimiters = config.render.latex_delimiters
    planned_pages = build_render_plan(middle_json, mode)
    rendered_pages = [
        _render_page(
            page,
            mode=mode,
            delimiters=delimiters,
            asset_base_url=asset_base_url,
        )
        for page in planned_pages
    ]
    if mode is MarkdownRenderMode.FULL:
        return _PAGE_SEPARATOR.join(rendered_pages)
    return "\n\n".join(page for page in rendered_pages if page)


def _render_page(
    planned_blocks: list[PlannedBlock],
    *,
    mode: MarkdownRenderMode,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """渲染单页逻辑块，并在默认模式中过滤重复页元素。"""
    rendered: list[str] = []
    for planned in planned_blocks:
        if planned.removed:
            continue
        if mode is MarkdownRenderMode.DEFAULT and planned.block.type in PAGE_AUXILIARY_BLOCK_TYPES:
            continue
        text = render_planned_block(planned, delimiters=delimiters, asset_base_url=asset_base_url)
        if text and text.strip():
            rendered.append(text.strip("\n"))
    return "\n\n".join(rendered)


__all__ = ["MarkdownRenderMode", "render_markdown"]
