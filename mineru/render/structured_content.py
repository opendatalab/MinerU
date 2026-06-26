# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Any

from ..backend.utils.char_utils import is_hyphen_at_line_end
from ..types import BBox, Block, BlockType, ContentType, ContentTypeV2, IntBBox, Line
from ..utils.language import detect_lang
from .markdown import _get_title_level
from .merge import (
    CJK_LANGS,
    _collect_text_for_lang_detection,
    _next_line_starts_with_lowercase_text,
    _normalize_text_content,
)
from .merge_visual import _build_media_path, _format_embedded_html, _normalize_visual_content


def merge_adjacent_ref_text_blocks_for_content(para_blocks: list[Block]) -> list[Block]:
    """将连续参考文献块合并为临时父块，统一 content_list 与 structured_content 的列表输出。"""
    merged_blocks: list[Block] = []
    ref_group: list[Block] = []

    def flush_ref_group() -> None:
        """把当前连续 ref_text 分组写入结果，单个块保持原样避免无意义包装。"""
        nonlocal ref_group
        if not ref_group:
            return
        if len(ref_group) == 1:
            merged_blocks.append(ref_group[0])
        else:
            merged_blocks.append(
                Block(
                    index=ref_group[0].index,
                    type=BlockType.REF_TEXT,
                    bbox=ref_group[0].bbox,
                    blocks=list(ref_group),
                )
            )
        ref_group = []

    for para_block in para_blocks or []:
        if para_block.type == BlockType.REF_TEXT:
            ref_group.append(para_block)
            continue

        flush_ref_group()
        merged_blocks.append(para_block)

    flush_ref_group()
    return merged_blocks


def block_to_structured_content(
    para_block: Block,
    img_bucket_path: str,
    page_size: tuple[int, int] | list[int] | None,
) -> dict[str, Any] | None:
    """将 PDF middle_json Block 转换为 structured_content 单项。"""
    para_type = para_block.type
    para_content: dict[str, Any] | None = None

    if para_type in [
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_NUMBER,
        BlockType.PAGE_FOOTNOTE,
    ]:
        content_type = _page_content_type(para_type)
        para_content = {
            "type": content_type,
            "content": {
                f"{content_type}_content": merge_para_with_structured_spans(para_block),
            },
        }
    elif para_type == BlockType.TITLE:
        title_level = _get_title_level(para_block)
        if title_level != 0:
            para_content = {
                "type": ContentTypeV2.TITLE,
                "content": {
                    "title_content": merge_para_with_structured_spans(para_block),
                    "level": title_level,
                },
            }
        else:
            para_content = _paragraph_content(para_block)
    elif para_type in [BlockType.TEXT, BlockType.ABSTRACT, BlockType.PHONETIC]:
        para_content = _paragraph_content(para_block)
    elif para_type == BlockType.INTERLINE_EQUATION:
        image_path, math_content = _get_body_data(para_block)
        para_content = {
            "type": ContentTypeV2.EQUATION_INTERLINE,
            "content": {
                "math_content": math_content,
                "math_type": "latex",
                "image_source": {"path": _build_media_path(img_bucket_path, image_path)},
            },
        }
    elif para_type == BlockType.IMAGE:
        para_content = _image_content(para_block, img_bucket_path)
    elif para_type == BlockType.TABLE:
        para_content = _table_content(para_block, img_bucket_path)
    elif para_type == BlockType.CHART:
        para_content = _chart_content(para_block, img_bucket_path)
    elif para_type == BlockType.CODE:
        para_content = _code_content(para_block)
    elif para_type == BlockType.REF_TEXT:
        para_content = _ref_text_content(para_block)
    elif para_type == BlockType.LIST:
        para_content = _list_content(para_block)
    elif para_type == BlockType.INDEX:
        para_content = _index_content(para_block)

    if not para_content:
        return None

    bbox = _build_bbox(para_block.bbox, page_size)
    if bbox:
        para_content["bbox"] = bbox
    return para_content


def merge_para_with_structured_spans(para_block: Block) -> list[dict[str, Any]]:
    """将段落文本和行内公式转换为 structured_content span，且不修改原始 span。"""
    block_lang = detect_lang(_collect_text_for_lang_detection(para_block))
    para_content: list[dict[str, Any]] = []
    para_type = para_block.type

    for line_idx, line in enumerate(para_block.lines):
        for span_idx, span in enumerate(line.spans):
            span_type = span.type

            if span_type == ContentType.TEXT:
                content = _normalize_text_content(span.content)
                if not content.strip():
                    continue

                output_type = ContentTypeV2.SPAN_PHONETIC if para_type == BlockType.PHONETIC else ContentTypeV2.SPAN_TEXT
                rendered_content = _render_text_span_content(
                    para_block,
                    block_lang,
                    line,
                    line_idx,
                    span_idx,
                    content,
                )
                if para_content and para_content[-1]["type"] == output_type:
                    para_content[-1]["content"] += rendered_content
                else:
                    para_content.append(
                        {
                            "type": output_type,
                            "content": rendered_content,
                        }
                    )
            elif span_type == ContentType.INLINE_EQUATION:
                content = span.content.strip()
                if content:
                    para_content.append(
                        {
                            "type": ContentTypeV2.SPAN_EQUATION_INLINE,
                            "content": content,
                        }
                    )

    if para_content and para_content[-1]["type"] in [
        ContentTypeV2.SPAN_TEXT,
        ContentTypeV2.SPAN_PHONETIC,
    ]:
        para_content[-1]["content"] = para_content[-1]["content"].rstrip()

    return para_content


def _page_content_type(para_type: str) -> str:
    """将页面附属块类型映射到 structured_content 顶层类型。"""
    if para_type == BlockType.HEADER:
        return ContentTypeV2.PAGE_HEADER
    if para_type == BlockType.FOOTER:
        return ContentTypeV2.PAGE_FOOTER
    if para_type == BlockType.ASIDE_TEXT:
        return ContentTypeV2.PAGE_ASIDE_TEXT
    if para_type == BlockType.PAGE_NUMBER:
        return ContentTypeV2.PAGE_NUMBER
    if para_type == BlockType.PAGE_FOOTNOTE:
        return ContentTypeV2.PAGE_FOOTNOTE
    raise ValueError(f"Unknown para_type: {para_type}")


def _paragraph_content(para_block: Block) -> dict[str, Any]:
    """构造普通段落 structured_content 项。"""
    return {
        "type": ContentTypeV2.PARAGRAPH,
        "content": {
            "paragraph_content": merge_para_with_structured_spans(para_block),
        },
    }


def _image_content(para_block: Block, img_bucket_path: str) -> dict[str, Any]:
    """构造图片 structured_content 项，保留 caption、footnote、sub_type 与视觉文本。"""
    image_caption: list[dict[str, Any]] = []
    image_footnote: list[dict[str, Any]] = []
    image_path, image_content = _get_body_data(para_block)
    for block in para_block.blocks:
        if block.type == BlockType.IMAGE_CAPTION:
            image_caption.extend(merge_para_with_structured_spans(block))
        if block.type == BlockType.IMAGE_FOOTNOTE:
            image_footnote.extend(merge_para_with_structured_spans(block))
    para_content: dict[str, Any] = {
        "type": ContentTypeV2.IMAGE,
        "content": {
            "image_source": {"path": _build_media_path(img_bucket_path, image_path)},
            "image_caption": image_caption,
            "image_footnote": image_footnote,
        },
    }
    if image_content or para_block.sub_type:
        para_content["content"]["content"] = image_content
    _apply_visual_sub_type(para_content, para_block)
    return para_content


def _table_content(para_block: Block, img_bucket_path: str) -> dict[str, Any]:
    """构造表格 structured_content 项，并计算简单/复杂表格标记。"""
    table_caption: list[dict[str, Any]] = []
    table_footnote: list[dict[str, Any]] = []
    image_path, html = _get_body_data(para_block)
    table_html = _format_embedded_html(html, img_bucket_path)
    table_nest_level = 2 if table_html.count("<table") > 1 else 1
    if "colspan" in table_html or "rowspan" in table_html or table_nest_level > 1:
        table_type = ContentTypeV2.TABLE_COMPLEX
    else:
        table_type = ContentTypeV2.TABLE_SIMPLE
    for block in para_block.blocks:
        if block.type == BlockType.TABLE_CAPTION:
            table_caption.extend(merge_para_with_structured_spans(block))
        if block.type == BlockType.TABLE_FOOTNOTE:
            table_footnote.extend(merge_para_with_structured_spans(block))
    return {
        "type": ContentTypeV2.TABLE,
        "content": {
            "image_source": {"path": _build_media_path(img_bucket_path, image_path)},
            "table_caption": table_caption,
            "table_footnote": table_footnote,
            "html": table_html,
            "table_type": table_type,
            "table_nest_level": table_nest_level,
        },
    }


def _chart_content(para_block: Block, img_bucket_path: str) -> dict[str, Any]:
    """构造图表 structured_content 项，兼容仅有截图或仅有 HTML 内容的情况。"""
    chart_caption: list[dict[str, Any]] = []
    chart_footnote: list[dict[str, Any]] = []
    image_path, chart_content = _get_body_data(para_block)
    for block in para_block.blocks:
        if block.type == BlockType.CHART_CAPTION:
            chart_caption.extend(merge_para_with_structured_spans(block))
        if block.type == BlockType.CHART_FOOTNOTE:
            chart_footnote.extend(merge_para_with_structured_spans(block))
    para_content = {
        "type": ContentTypeV2.CHART,
        "content": {
            "image_source": {
                "path": _build_media_path(img_bucket_path, image_path),
            },
            "content": chart_content if chart_content else "",
            "chart_caption": chart_caption,
            "chart_footnote": chart_footnote,
        },
    }
    _apply_visual_sub_type(para_content, para_block)
    return para_content


def _code_content(para_block: Block) -> dict[str, Any]:
    """构造代码或算法 structured_content 项，统一保留 caption 与 footnote。"""
    code_caption: list[dict[str, Any]] = []
    code_footnote: list[dict[str, Any]] = []
    code_content: list[dict[str, Any]] = []
    for block in para_block.blocks:
        if block.type == BlockType.CODE_CAPTION:
            code_caption.extend(merge_para_with_structured_spans(block))
        if block.type == BlockType.CODE_FOOTNOTE:
            code_footnote.extend(merge_para_with_structured_spans(block))
        if block.type == BlockType.CODE_BODY:
            code_content = merge_para_with_structured_spans(block)

    sub_type = para_block.sub_type
    if sub_type == BlockType.CODE:
        return {
            "type": ContentTypeV2.CODE,
            "content": {
                "code_caption": code_caption,
                "code_content": code_content,
                "code_footnote": code_footnote,
                "code_language": para_block.guess_lang or "txt",
            },
        }
    if sub_type == BlockType.ALGORITHM:
        return {
            "type": ContentTypeV2.ALGORITHM,
            "content": {
                "algorithm_caption": code_caption,
                "algorithm_content": code_content,
                "algorithm_footnote": code_footnote,
            },
        }
    raise ValueError(f"Unknown code sub_type: {sub_type}")


def _ref_text_content(para_block: Block) -> dict[str, Any]:
    """构造参考文献列表项，支持已经合并的临时父 ref_text 块。"""
    list_items: list[dict[str, Any]] = []
    for block in _get_ref_text_item_blocks(para_block):
        item_content = merge_para_with_structured_spans(block)
        if item_content:
            list_items.append(
                {
                    "item_type": "text",
                    "item_content": item_content,
                }
            )
    return {
        "type": ContentTypeV2.LIST,
        "content": {
            "list_type": ContentTypeV2.LIST_REF,
            "list_items": list_items,
        },
    }


def _list_content(para_block: Block) -> dict[str, Any]:
    """构造普通列表 structured_content 项，兼容子 block 列表和按行拆分列表。"""
    if para_block.sub_type:
        if para_block.sub_type == BlockType.REF_TEXT:
            list_type = ContentTypeV2.LIST_REF
        elif para_block.sub_type == BlockType.TEXT:
            list_type = ContentTypeV2.LIST_TEXT
        else:
            raise ValueError(f"Unknown list sub_type: {para_block.sub_type}")
    else:
        list_type = ContentTypeV2.LIST_TEXT

    list_items = _block_list_items(para_block)
    content: dict[str, Any] = {
        "list_type": list_type,
        "list_items": list_items,
    }
    if list_type == ContentTypeV2.LIST_TEXT:
        content["attribute"] = para_block._list_attribute or "unordered"
    return {
        "type": ContentTypeV2.LIST,
        "content": content,
    }


def _index_content(para_block: Block) -> dict[str, Any]:
    """构造目录索引 structured_content 项。"""
    return {
        "type": ContentTypeV2.INDEX,
        "content": {
            "list_type": ContentTypeV2.LIST_TEXT,
            "list_items": _block_list_items(para_block),
        },
    }


def _block_list_items(para_block: Block) -> list[dict[str, Any]]:
    """将列表容器中的子块或行拆分为 structured_content list_items。"""
    item_blocks = para_block.blocks if para_block.blocks else _split_list_item_blocks(para_block)
    list_items: list[dict[str, Any]] = []
    for block in item_blocks:
        item_content = merge_para_with_structured_spans(block)
        if item_content:
            list_items.append(
                {
                    "item_type": "text",
                    "item_content": item_content,
                }
            )
    return list_items


def _apply_visual_sub_type(para_content: dict[str, Any], para_block: Block) -> None:
    """将视觉块 sub_type 透传到 structured_content 顶层。"""
    sub_type = para_block.sub_type
    if sub_type:
        para_content["sub_type"] = sub_type


def _build_bbox(para_bbox: BBox, page_size: tuple[int, int] | list[int] | None) -> IntBBox | None:
    """按千分位归一化 bbox，缺少页面尺寸时跳过 bbox 输出。"""
    if not para_bbox or not page_size:
        return None
    page_width, page_height = page_size
    if not page_width or not page_height:
        return None
    x0, y0, x1, y1 = para_bbox
    return [
        int(x0 * 1000 / page_width),
        int(y0 * 1000 / page_height),
        int(x1 * 1000 / page_width),
        int(y1 * 1000 / page_height),
    ]


def _get_ref_text_item_blocks(para_block: Block) -> list[Block]:
    """返回参考文献列表项块，未合并时使用当前块作为单项。"""
    return para_block.blocks or [para_block]


def _split_list_item_blocks(para_block: Block) -> list[Block]:
    """按 line._is_list_start 将单个列表块拆成临时列表项块。"""
    item_blocks = []
    current_lines = []

    for line_idx, line in enumerate(para_block.lines):
        if line_idx > 0 and line._is_list_start and current_lines:
            item_blocks.append(
                Block(
                    index=para_block.index,
                    type=BlockType.TEXT,
                    bbox=para_block.bbox,
                    lines=current_lines,
                )
            )
            current_lines = []
        current_lines.append(line)

    if current_lines:
        item_blocks.append(
            Block(
                index=para_block.index,
                type=BlockType.TEXT,
                bbox=para_block.bbox,
                lines=current_lines,
            )
        )

    return item_blocks


def _get_body_data(para_block: Block) -> tuple[str, str]:
    """从视觉类 block 中提取图片路径和主体内容。"""

    def get_data_from_spans(lines: list[Line]) -> tuple[str, str]:
        """按 span 顺序读取第一个可渲染的视觉或文本载荷。"""
        for line in lines:
            for span in line.spans:
                span_type = span.type
                if span_type == ContentType.TABLE:
                    return span.image_path, span.content
                if span_type == ContentType.CHART:
                    return span.image_path, span.content
                if span_type == ContentType.IMAGE:
                    return span.image_path, _normalize_visual_content(span.content)
                if span_type == ContentType.INTERLINE_EQUATION:
                    return span.image_path, span.content
                if span_type == ContentType.TEXT:
                    return "", span.content
        return "", ""

    if para_block.blocks:
        for block in para_block.blocks:
            block_type = block.type
            if block_type in [
                BlockType.IMAGE_BODY,
                BlockType.TABLE_BODY,
                BlockType.CHART_BODY,
                BlockType.CODE_BODY,
            ]:
                result = get_data_from_spans(block.lines)
                if result != ("", "") or block_type == BlockType.CHART_BODY:
                    return result
        return "", ""

    return get_data_from_spans(para_block.lines)


def _has_following_joinable_span(para_block: Block, line_idx: int, span_idx: int) -> bool:
    """判断当前 span 后面是否还有可拼接内容，避免多余尾随空格。"""
    for next_line_idx in range(line_idx, len(para_block.lines)):
        next_line = para_block.lines[next_line_idx]
        start_span_idx = span_idx + 1 if next_line_idx == line_idx else 0
        for next_span in next_line.spans[start_span_idx:]:
            next_span_type = next_span.type
            if next_span_type == ContentType.TEXT:
                if _normalize_text_content(next_span.content).strip():
                    return True
            elif next_span_type == ContentType.INLINE_EQUATION:
                if str(next_span.content).strip():
                    return True
    return False


def _render_text_span_content(
    para_block: Block,
    block_lang: str,
    line: Line,
    line_idx: int,
    span_idx: int,
    content: str,
) -> str:
    """按语言和跨行断词规则渲染 structured_content 文本 span 内容。"""
    is_last_span = span_idx == len(line.spans) - 1
    has_following_joinable_span = _has_following_joinable_span(para_block, line_idx, span_idx)

    if block_lang in CJK_LANGS:
        if has_following_joinable_span and not is_last_span:
            return f"{content} "
        return content

    if is_last_span and is_hyphen_at_line_end(content):
        if _next_line_starts_with_lowercase_text(para_block, line_idx):
            return content[:-1]
        return content
    if has_following_joinable_span:
        return f"{content} "
    return content
