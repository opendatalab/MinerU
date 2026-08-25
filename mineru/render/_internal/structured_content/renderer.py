# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到树形 Markdown structured_content 的公共渲染实现。"""

from __future__ import annotations

from typing import Any, TypeAlias

from ....config import LatexDelimitersConfig, config
from ..markdown.blocks import (
    render_single_block,
    render_title_inline_content,
    render_visual_annotation,
    render_visual_body_content,
)
from ..markdown.assets import normalize_image_source, resolve_image_source
from ..markdown.escaping import escape_standalone_marker_rule, escape_text_block_markdown_prefix
from ....types import (
    BlockType,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    DocTitleBlock,
    EquationBlock,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    ImagePayloadBlock,
    MiddleJson,
    PageBlock,
    ParagraphTitleBlock,
    TableAnnotationBlock,
    TableBlock,
    TableBodyBlock,
)

VisualBlock: TypeAlias = ImageBlock | TableBlock | ChartBlock | CodeBlock
VisualAnnotationBlock: TypeAlias = ImageAnnotationBlock | TableAnnotationBlock | ChartAnnotationBlock | CodeAnnotationBlock

_CAPTION_TYPES = {
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.CHART_CAPTION,
    BlockType.CODE_CAPTION,
}
_FOOTNOTE_TYPES = {
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_FOOTNOTE,
    BlockType.CODE_FOOTNOTE,
}
_REMOVED_BLOCK_FIELDS = {"content", "index", "guess_lang", "image_path", "image_base64"}


def render_structured_content(
    middle_json: MiddleJson,
    *,
    asset_base_url: str = "",
) -> dict[str, Any]:
    """把严格 MiddleJson 无副作用地渲染为树形 Markdown structured_content。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render_structured_content expects a MiddleJson instance")

    delimiters = config.render.latex_delimiters
    document_fields = middle_json.model_dump(
        mode="json",
        exclude={"pages"},
        exclude_defaults=True,
    )
    pages = [
        {
            "page_idx": page.page_idx,
            "blocks": [
                _render_content_block(
                    block,
                    delimiters=delimiters,
                    asset_base_url=asset_base_url,
                )
                for block in page.blocks
            ],
        }
        for page in middle_json.pages
    ]
    return {"pages": pages, **document_fields}


def _render_content_block(
    block: PageBlock,
    *,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> dict[str, Any]:
    """保留父块元数据，并把 block 内容收敛为 Markdown 字符串。"""
    payload = block.model_dump(
        mode="json",
        exclude=_REMOVED_BLOCK_FIELDS,
        exclude_defaults=True,
    )
    if isinstance(block, (DocTitleBlock, ParagraphTitleBlock)):
        content = render_title_inline_content(block, delimiters)
        payload["content"] = escape_standalone_marker_rule(escape_text_block_markdown_prefix(content))
        return payload
    if isinstance(block, EquationBlock):
        payload["content"] = block.content.strip()
        image_source = _resolve_content_image_source(block, asset_base_url)
        if image_source is not None:
            payload["image_source"] = image_source
        return payload
    if not isinstance(block, (ImageBlock, TableBlock, ChartBlock, CodeBlock)):
        payload["content"] = render_single_block(
            block,
            delimiters=delimiters,
            asset_base_url=asset_base_url,
        )
        return payload

    payload["content"] = render_visual_body_content(
        block,
        delimiters=delimiters,
        asset_base_url=asset_base_url,
    )
    payload["captions"] = _render_annotation_group(block, _CAPTION_TYPES, delimiters)
    payload["footnotes"] = _render_annotation_group(block, _FOOTNOTE_TYPES, delimiters)
    image_source = _resolve_visual_image_source(block, asset_base_url)
    if image_source is not None:
        payload["image_source"] = image_source
    return payload


def _render_annotation_group(
    block: VisualBlock,
    accepted_types: set[str],
    delimiters: LatexDelimitersConfig,
) -> list[dict[str, Any]]:
    """按源 index 稳定排序视觉说明，并保留可用 bbox 与 Markdown 内容。"""
    annotations: list[tuple[int, VisualAnnotationBlock]] = []
    for position, child in enumerate(block.content):
        if (
            isinstance(
                child,
                (ImageAnnotationBlock, TableAnnotationBlock, ChartAnnotationBlock, CodeAnnotationBlock),
            )
            and child.type in accepted_types
        ):
            annotations.append((position, child))
    annotations.sort(key=_annotation_sort_key)
    rendered_annotations: list[dict[str, Any]] = []
    for _, child in annotations:
        annotation_payload: dict[str, Any] = {}
        if child.bbox is not None:
            annotation_payload["bbox"] = list(child.bbox)
        annotation_payload["content"] = render_visual_annotation(child, delimiters)
        rendered_annotations.append(annotation_payload)
    return rendered_annotations


def _annotation_sort_key(
    item: tuple[int, VisualAnnotationBlock],
) -> tuple[bool, int, int]:
    """让有 index 的说明升序优先，缺失 index 的说明稳定排在末尾。"""
    position, block = item
    return block.index is None, block.index if block.index is not None else 0, position


def _resolve_visual_image_source(block: VisualBlock, asset_base_url: str) -> str | None:
    """解析视觉 body 实际选择的图片来源，并返回安全的 Markdown 地址。"""
    for child in block.content:
        if not isinstance(child, (ImageBodyBlock, TableBodyBlock, ChartBodyBlock)):
            continue
        return _resolve_content_image_source(child, asset_base_url)
    return None


def _resolve_content_image_source(block: ImagePayloadBlock, asset_base_url: str) -> str | None:
    """把图片载荷收敛为 structured_content 中唯一且安全的 image_source。"""
    source = resolve_image_source(block, asset_base_url)
    return normalize_image_source(source) if source else None


__all__ = ["render_structured_content"]
