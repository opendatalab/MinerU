# Copyright (c) Opendatalab. All rights reserved.
from typing import Any

from loguru import logger

from ...render.markdown import _get_title_level
from ...render.merge import _normalize_text_content
from ...render.merge_visual import _build_media_path, _format_embedded_html
from ...types import Block, BlockType, ContentType, ContentTypeV2, Line
from ...utils.language import detect_lang
from ..utils.char_utils import is_hyphen_at_line_end


def _has_following_joinable_span(para_block: Block, line_idx: int, span_idx: int) -> bool:
    """判断当前 span 后面是否还有可拼接文本，避免把分隔空格落到段落末尾。"""
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


def _apply_visual_sub_type(para_content: dict[str, Any], para_block: Block) -> None:
    sub_type = para_block.sub_type
    if sub_type:
        para_content["sub_type"] = sub_type


def make_blocks_to_content_list_v2(para_block: Block, img_bucket_path: str, page_size: tuple[int, int]) -> dict:
    para_type = para_block.type
    para_content = {}
    if para_type in [
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_NUMBER,
        BlockType.PAGE_FOOTNOTE,
    ]:
        if para_type == BlockType.HEADER:
            content_type = ContentTypeV2.PAGE_HEADER
        elif para_type == BlockType.FOOTER:
            content_type = ContentTypeV2.PAGE_FOOTER
        elif para_type == BlockType.ASIDE_TEXT:
            content_type = ContentTypeV2.PAGE_ASIDE_TEXT
        elif para_type == BlockType.PAGE_NUMBER:
            content_type = ContentTypeV2.PAGE_NUMBER
        elif para_type == BlockType.PAGE_FOOTNOTE:
            content_type = ContentTypeV2.PAGE_FOOTNOTE
        else:
            raise ValueError(f"Unknown para_type: {para_type}")
        para_content = {
            "type": content_type,
            "content": {
                f"{content_type}_content": merge_para_with_text_v2(para_block),
            },
        }
    elif para_type == BlockType.TITLE:
        title_level = _get_title_level(para_block)
        if title_level != 0:
            para_content = {
                "type": ContentTypeV2.TITLE,
                "content": {"title_content": merge_para_with_text_v2(para_block), "level": title_level},
            }
        else:
            para_content = {
                "type": ContentTypeV2.PARAGRAPH,
                "content": {
                    "paragraph_content": merge_para_with_text_v2(para_block),
                },
            }
    elif para_type in [BlockType.TEXT, BlockType.PHONETIC]:
        para_content = {
            "type": ContentTypeV2.PARAGRAPH,
            "content": {
                "paragraph_content": merge_para_with_text_v2(para_block),
            },
        }
    elif para_type == BlockType.INTERLINE_EQUATION:
        image_path, math_content = get_body_data(para_block)
        para_content = {
            "type": ContentTypeV2.EQUATION_INTERLINE,
            "content": {
                "math_content": math_content,
                "math_type": "latex",
                "image_source": {"path": f"{img_bucket_path}/{image_path}"},
            },
        }
    elif para_type == BlockType.IMAGE:
        image_caption = []
        image_footnote = []
        image_path, image_content = get_body_data(para_block)
        image_source = {
            "path": _build_media_path(img_bucket_path, image_path),
        }
        for block in para_block.blocks:
            if block.type == BlockType.IMAGE_CAPTION:
                image_caption.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.IMAGE_FOOTNOTE:
                image_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            "type": ContentTypeV2.IMAGE,
            "content": {
                "image_source": image_source,
                "content": image_content if image_content else "",
                "image_caption": image_caption,
                "image_footnote": image_footnote,
            },
        }
        _apply_visual_sub_type(para_content, para_block)
    elif para_type == BlockType.TABLE:
        table_caption = []
        table_footnote = []
        image_path, html = get_body_data(para_block)
        table_html = _format_embedded_html(html, img_bucket_path)
        image_source = {
            "path": f"{img_bucket_path}/{image_path}",
        }
        if table_html.count("<table") > 1:
            table_nest_level = 2
        else:
            table_nest_level = 1
        if "colspan" in table_html or "rowspan" in table_html or table_nest_level > 1:
            table_type = ContentTypeV2.TABLE_COMPLEX
        else:
            table_type = ContentTypeV2.TABLE_SIMPLE

        for block in para_block.blocks:
            if block.type == BlockType.TABLE_CAPTION:
                table_caption.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.TABLE_FOOTNOTE:
                table_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            "type": ContentTypeV2.TABLE,
            "content": {
                "image_source": image_source,
                "table_caption": table_caption,
                "table_footnote": table_footnote,
                "html": table_html,
                "table_type": table_type,
                "table_nest_level": table_nest_level,
            },
        }
    elif para_type == BlockType.CHART:
        chart_caption = []
        chart_footnote = []
        image_path, chart_content = get_body_data(para_block)
        for block in para_block.blocks:
            if block.type == BlockType.CHART_CAPTION:
                chart_caption.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.CHART_FOOTNOTE:
                chart_footnote.extend(merge_para_with_text_v2(block))
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
    elif para_type == BlockType.CODE:
        code_caption = []
        code_content = []
        for block in para_block.blocks:
            if block.type == BlockType.CODE_CAPTION:
                code_caption.extend(merge_para_with_text_v2(block))
            if block.type == BlockType.CODE_BODY:
                code_content = merge_para_with_text_v2(block)
        sub_type = para_block.sub_type
        if sub_type == BlockType.CODE:
            para_content = {
                "type": ContentTypeV2.CODE,
                "content": {
                    "code_caption": code_caption,
                    "code_content": code_content,
                    "code_language": para_block.guess_lang or "txt",
                },
            }
        elif sub_type == BlockType.ALGORITHM:
            para_content = {
                "type": ContentTypeV2.ALGORITHM,
                "content": {
                    "algorithm_caption": code_caption,
                    "algorithm_content": code_content,
                },
            }
        else:
            raise ValueError(f"Unknown code sub_type: {sub_type}")
    elif para_type == BlockType.REF_TEXT:
        para_content = {
            "type": ContentTypeV2.LIST,
            "content": {
                "list_type": ContentTypeV2.LIST_REF,
                "list_items": [
                    {
                        "item_type": "text",
                        "item_content": merge_para_with_text_v2(para_block),
                    }
                ],
            },
        }
    elif para_type == BlockType.LIST:
        if para_block.sub_type:
            if para_block.sub_type == BlockType.REF_TEXT:
                list_type = ContentTypeV2.LIST_REF
            elif para_block.sub_type == BlockType.TEXT:
                list_type = ContentTypeV2.LIST_TEXT
            else:
                raise ValueError(f"Unknown list sub_type: {para_block.sub_type}")
        else:
            list_type = ContentTypeV2.LIST_TEXT
        list_items = []
        for block in para_block.blocks:
            item_content = merge_para_with_text_v2(block)
            if item_content:
                list_items.append(
                    {
                        "item_type": "text",
                        "item_content": item_content,
                    }
                )
        para_content = {
            "type": ContentTypeV2.LIST,
            "content": {
                "list_type": list_type,
                "list_items": list_items,
            },
        }

    page_width, page_height = page_size
    para_bbox = para_block.bbox
    if para_bbox:
        x0, y0, x1, y1 = para_bbox
        para_content["bbox"] = [
            int(x0 * 1000 / page_width),
            int(y0 * 1000 / page_height),
            int(x1 * 1000 / page_width),
            int(y1 * 1000 / page_height),
        ]

    return para_content


def get_body_data(para_block: Block) -> tuple[str, str]:
    """
    Extract image_path and body content from para_block
    Returns:
        - For IMAGE: (image_path, content)
        - For TABLE: (image_path, html)
        - For CHART: (image_path, content)
        - For INTERLINE_EQUATION: (image_path, content)
        - Default: ('', '')
    """

    def get_data_from_spans(lines: list[Line]) -> tuple[str, str]:
        for line in lines:
            for span in line.spans:
                span_type = span.type
                if span_type == ContentType.TABLE:
                    return span.image_path, span.content
                elif span_type == ContentType.CHART:
                    return span.image_path, span.content
                elif span_type == ContentType.IMAGE:
                    return span.image_path, span.content
                elif span_type == ContentType.INTERLINE_EQUATION:
                    return span.image_path, span.content
                elif span_type == ContentType.TEXT:
                    return "", span.content
        return "", ""

    # 处理嵌套的 blocks 结构
    if para_block.blocks:
        for block in para_block.blocks:
            block_type = block.type
            if block_type in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.CHART_BODY, BlockType.CODE_BODY]:
                result = get_data_from_spans(block.lines)
                if result != ("", ""):
                    return result
        return "", ""

    # 处理直接包含 lines 的结构
    return get_data_from_spans(para_block.lines)


def merge_para_with_text_v2(para_block: Block) -> list[dict[str, Any]]:
    block_text = ""
    for line in para_block.lines:
        for span in line.spans:
            if span.type in [ContentType.TEXT]:
                span.content = _normalize_text_content(span.content)
                block_text += span.content
    block_lang = detect_lang(block_text)

    para_content: list[dict[str, Any]] = []
    para_type = para_block.type
    for i, line in enumerate(para_block.lines):
        for j, span in enumerate(line.spans):
            span_type = span.type
            if span.content.strip():
                if span_type == ContentType.TEXT:
                    if para_type == BlockType.PHONETIC:
                        span_type = ContentTypeV2.SPAN_PHONETIC
                    else:
                        span_type = ContentTypeV2.SPAN_TEXT
                if span_type == ContentType.INLINE_EQUATION:
                    span_type = ContentTypeV2.SPAN_EQUATION_INLINE
                if span_type in [
                    ContentTypeV2.SPAN_TEXT,
                ]:
                    # 定义CJK语言集合(中日韩)
                    cjk_langs = {"zh", "ja", "ko"}
                    # logger.info(f'block_lang: {block_lang}, content: {content}')

                    # 判断是否为行末span
                    is_last_span = j == len(line.spans) - 1
                    has_following_joinable_span = _has_following_joinable_span(para_block, i, j)

                    if block_lang in cjk_langs:  # 中文/日语/韩文语境下，换行不需要空格分隔,但是如果是行内公式结尾，还是要加空格
                        if has_following_joinable_span and not is_last_span:
                            span_content = f"{span.content} "
                        else:
                            span_content = span.content
                    else:
                        # 如果span是line的最后一个且末尾带有-连字符，那么末尾不应该加空格,同时应该把-删除
                        if is_last_span and is_hyphen_at_line_end(span.content):
                            # 如果下一行的第一个span是小写字母开头，删除连字符
                            if (
                                i + 1 < len(para_block.lines)
                                and para_block.lines[i + 1].spans
                                and para_block.lines[i + 1].spans[0].type == ContentType.TEXT
                                and para_block.lines[i + 1].spans[0].content
                                and para_block.lines[i + 1].spans[0].content[0].islower()
                            ):
                                span_content = span.content[:-1]
                            else:  # 如果没有下一行，或者下一行的第一个span不是小写字母开头，则保留连字符但不加空格
                                span_content = span.content
                        else:
                            # 西方文本语境下content间需要空格分隔
                            if has_following_joinable_span:
                                span_content = f"{span.content} "
                            else:
                                span_content = span.content

                    if para_content and para_content[-1]["type"] == span_type:
                        # 合并相同类型的span
                        para_content[-1]["content"] += span_content
                    else:
                        span_content = {"type": span_type, "content": span_content}
                        para_content.append(span_content)

                elif span_type in [ContentTypeV2.SPAN_PHONETIC, ContentTypeV2.SPAN_EQUATION_INLINE]:
                    span_content = {"type": span_type, "content": span.content}
                    para_content.append(span_content)
                else:
                    logger.warning(f"Unknown span type in merge_para_with_text_v2: {span_type}")
    return para_content
