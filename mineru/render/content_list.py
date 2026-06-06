# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from ..types import EMPTY_BBOX, BBox, Block, ContentItem, IntBBox
from ..utils.enum_class import BlockType, ContentType
from .markdown import _build_media_path, _get_title_level
from .merge import _merge_para_text, merge_para_text
from .merge_visual import _format_embedded_html, _inherit_parent_code_render_metadata, _normalize_visual_content


def block_to_content_list(
    para_block: Block,
    img_bucket_path: str,
    page_idx: int,
    page_size: tuple[int, int] | None,
) -> ContentItem | None:
    para_type = para_block.type
    if para_type in [
        BlockType.TEXT,
        BlockType.PHONETIC,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_FOOTNOTE,
    ]:
        item = ContentItem(
            type=para_type,
            text=merge_para_text(para_block),
        )
    elif para_type in [BlockType.INDEX, BlockType.ABSTRACT]:
        item = ContentItem(
            type=ContentType.TEXT,
            text=merge_para_text(para_block),
        )
    elif para_type == BlockType.LIST:
        if para_block.blocks:
            # Some LayoutModel use LIST as a container.
            item = ContentItem(type=para_type, sub_type=para_block.sub_type or None)
            for block in para_block.blocks:
                if (item_text := _merge_para_text(block)).strip():
                    item.list_items.append(item_text)
        else:
            item = ContentItem(
                type=ContentType.TEXT,
                text=merge_para_text(para_block),
            )
    elif para_type == BlockType.REF_TEXT:
        if para_block.blocks:
            item = ContentItem(type=BlockType.LIST, sub_type=BlockType.REF_TEXT)
            for block in para_block.blocks:
                if (item_text := _merge_para_text(block)).strip():
                    item.list_items.append(item_text)
        else:
            item = ContentItem(
                type=para_type,
                text=merge_para_text(para_block),
            )
    elif para_type == BlockType.TITLE:
        title_level = _get_title_level(para_block)
        item = ContentItem(
            type=ContentType.TEXT,
            text=merge_para_text(para_block),
            text_level=title_level if title_level != 0 else None,
        )
    elif para_type == BlockType.INTERLINE_EQUATION:
        item = ContentItem(
            type=ContentType.EQUATION,
            img_path=_body_data(para_block, img_bucket_path)[0],
            text=merge_para_text(para_block),
            text_format="latex",
        )
    elif para_type == BlockType.IMAGE:
        item = ContentItem(
            type=ContentType.IMAGE,
            sub_type=para_block.sub_type or None,
        )
        for block in para_block.blocks:
            if block.type == BlockType.IMAGE_BODY:
                item.img_path, item.content = _body_data(block, img_bucket_path)
            if block.type == BlockType.IMAGE_CAPTION:
                item.image_caption.append(merge_para_text(block))
            if block.type == BlockType.IMAGE_FOOTNOTE:
                item.image_footnote.append(merge_para_text(block))
    elif para_type == BlockType.TABLE:
        item = ContentItem(type=ContentType.TABLE)
        for block in para_block.blocks:
            if block.type == BlockType.TABLE_BODY:
                item.img_path, item.table_body = _body_data(block, img_bucket_path)
            if block.type == BlockType.TABLE_CAPTION:
                item.table_caption.append(merge_para_text(block))
            if block.type == BlockType.TABLE_FOOTNOTE:
                item.table_footnote.append(merge_para_text(block))
    elif para_type == BlockType.CHART:
        item = ContentItem(
            type=ContentType.CHART,
            sub_type=para_block.sub_type or None,
        )
        for block in para_block.blocks:
            if block.type == BlockType.CHART_BODY:
                item.img_path, item.content = _body_data(block, img_bucket_path)
            if block.type == BlockType.CHART_CAPTION:
                item.chart_caption.append(merge_para_text(block))
            if block.type == BlockType.CHART_FOOTNOTE:
                item.chart_footnote.append(merge_para_text(block))
    elif para_type == BlockType.CODE:
        item = ContentItem(
            type=BlockType.CODE,
            sub_type=para_block.sub_type or None,
        )
        for block in para_block.blocks:
            if block.type == BlockType.CODE_BODY:
                render_block = _inherit_parent_code_render_metadata(block, para_block)
                item.code_body = merge_para_text(render_block)
            if block.type == BlockType.CODE_CAPTION:
                item.code_caption.append(merge_para_text(block))
            if block.type == BlockType.CODE_FOOTNOTE:
                item.code_footnote.append(merge_para_text(block))
    else:
        return None

    item.page_idx = page_idx
    if bbox := _norm1k_bbox(para_block.bbox, page_size):
        item.bbox = bbox
    return item


def _body_data(para_block: Block, img_bucket_path: str) -> tuple[str, str]:
    image_path, content = "", ""
    for span in (
        span
        for block in para_block.blocks or [para_block]
        if block.type
        in [
            BlockType.IMAGE_BODY,
            BlockType.TABLE_BODY,
            BlockType.CHART_BODY,
            BlockType.INTERLINE_EQUATION,
        ]
        for line in block.lines
        for span in line.spans
    ):
        if span.type == ContentType.IMAGE:
            image_path, content = span.image_path, _normalize_visual_content(span.content)
        elif span.type == ContentType.TABLE:
            image_path, content = span.image_path, _format_embedded_html(span.html, img_bucket_path)
        elif span.type == ContentType.CHART:
            image_path, content = span.image_path, span.content
        elif span.type == ContentType.INTERLINE_EQUATION:
            image_path, content = span.image_path, span.content
        if image_path or content:
            break
    return _build_media_path(img_bucket_path, image_path), content


def _norm1k_bbox(para_bbox: BBox, page_size: tuple[int, int] | None) -> IntBBox | None:
    if para_bbox == EMPTY_BBOX or not page_size:
        return None
    page_width, page_height = page_size
    x0, y0, x1, y1 = para_bbox
    return (
        int(x0 * 1000 / page_width),
        int(y0 * 1000 / page_height),
        int(x1 * 1000 / page_width),
        int(y1 * 1000 / page_height),
    )
