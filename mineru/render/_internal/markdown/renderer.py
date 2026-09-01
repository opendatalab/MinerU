# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 Markdown 的公共渲染实现。"""

from __future__ import annotations

from ....config import LatexDelimitersConfig, config
from ....backend.postprocess.inline import inline_plain_text
from ....types import PAGE_AUXILIARY_BLOCK_TYPES, MiddleJson, PageFootnoteBlock, TextBlock, TitleBlockBase
from ...contracts import ImageRenderer, RenderMode
from ..common.planner import PlannedBlock, build_render_plan
from .blocks import render_planned_block

_PAGE_SEPARATOR = "\n\n---\n\n"


def render_markdown(
    middle_json: MiddleJson,
    *,
    mode: RenderMode = RenderMode.DEFAULT,
    asset_base_url: str = "",
    image_renderer: ImageRenderer | None = None,
) -> str:
    """把严格 MiddleJson 纯函数式渲染为 Markdown 字符串。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render_markdown expects a MiddleJson instance")
    if not isinstance(mode, RenderMode):
        raise TypeError("mode must be a RenderMode value")

    delimiters = config.render.latex_delimiters
    planned_pages = build_render_plan(middle_json, mode)
    anchor_targets = _collect_markdown_anchor_targets(middle_json)
    emitted_anchors: set[str] = set()
    rendered_pages = [
        _render_page(
            page,
            mode=mode,
            delimiters=delimiters,
            asset_base_url=asset_base_url,
            image_renderer=image_renderer,
            anchor_targets=anchor_targets,
            emitted_anchors=emitted_anchors,
        )
        for page in planned_pages
    ]
    if mode is RenderMode.FULL:
        return _PAGE_SEPARATOR.join(rendered_pages)
    return "\n\n".join(page for page in rendered_pages if page)


def _render_page(
    planned_blocks: list[PlannedBlock],
    *,
    mode: RenderMode,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
    image_renderer: ImageRenderer | None = None,
    anchor_targets: set[str] | None = None,
    emitted_anchors: set[str] | None = None,
) -> str:
    """渲染单页逻辑块，并在默认模式中过滤重复页元素。"""
    rendered: list[str] = []
    for planned in planned_blocks:
        if planned.removed:
            continue
        if mode is RenderMode.DEFAULT and planned.block.type in PAGE_AUXILIARY_BLOCK_TYPES:
            continue
        text = render_planned_block(
            planned,
            delimiters=delimiters,
            asset_base_url=asset_base_url,
            image_renderer=image_renderer,
            anchor_targets=anchor_targets,
            emitted_anchors=emitted_anchors,
        )
        if text and text.strip():
            rendered.append(text.strip("\n"))
    return "\n\n".join(rendered)


def _collect_markdown_anchor_targets(middle_json: MiddleJson) -> set[str]:
    """收集真实可见的顶层正文、标题和页面脚注 anchor，供目录链接判定。"""
    targets: set[str] = set()
    for page in middle_json.pages:
        for block in page.blocks:
            if not isinstance(block, (TextBlock, TitleBlockBase, PageFootnoteBlock)):
                continue
            anchor = (block.anchor or "").strip()
            if anchor and inline_plain_text(block.content).strip():
                targets.add(anchor)
    return targets


__all__ = ["render_markdown"]
