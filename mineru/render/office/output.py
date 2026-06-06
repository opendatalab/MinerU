# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re
from collections.abc import Generator
from dataclasses import asdict
from typing import Any

from ...types import Block, BlockType, ContentType, ContentTypeV2, Line, Span
from .merge import (
    _append_hyperlink_part,
    _append_text_part,
    _apply_configured_style,
    _escape_office_inline_text,
    _join_rendered_parts,
    _make_rendered_part,
    _render_link,
    _select_block_inline_syntax,
    get_title_level,
    inline_left_delimiter,
    inline_right_delimiter,
    merge_para_with_text,
)


def _prefix_table_img_src(html: str, img_bucket_path: str) -> str:
    """给表格 HTML 内的本地图片 src 加上输出图片目录前缀。"""
    if not html or not img_bucket_path:
        return html
    return re.sub(
        r'src="(?!data:)([^"]+)"',
        lambda m: f'src="{img_bucket_path}/{m.group(1)}"',
        html,
    )


def _replace_eq_tags_in_table_html(html: str) -> str:
    """把表格或图表 HTML 中的 <eq> 标签替换为当前配置的行内公式定界符。"""
    if not html:
        return html
    return re.sub(
        r"<eq>(.*?)</eq>",
        lambda m: f" {inline_left_delimiter}{m.group(1)}{inline_right_delimiter} ",
        html,
        flags=re.DOTALL,
    )


def _format_embedded_html(html: str, img_bucket_path: str) -> str:
    """统一处理 Office 嵌入 HTML 的图片路径和行内公式标签。"""
    return _replace_eq_tags_in_table_html(_prefix_table_img_src(html, img_bucket_path))


def _build_media_path(img_bucket_path: str, image_path: str) -> str:
    """构造图片展示路径，空图片路径保持为空。"""
    if not image_path:
        return ""
    if not img_bucket_path:
        return image_path
    return f"{img_bucket_path}/{image_path}"


def _get_ordered_list_start(list_block: Block) -> int:
    """读取有序列表起始编号，兼容旧版数据缺少 start 字段的情况。"""
    try:
        start = list_block.start
        if start is None:
            start = 1
        return int(start)
    except (TypeError, ValueError):
        return 1


def _get_list_ilevel(list_block: Block) -> int:
    """安全读取 DOCX 列表原始 ilevel，异常值按顶层 0 处理。"""
    try:
        ilevel = list_block.ilevel
        if ilevel is None:
            ilevel = 0
        return int(ilevel)
    except (TypeError, ValueError):
        return 0


def _get_relative_list_ilevel(list_block: Block, root_ilevel: int) -> int:
    """将 DOCX 原始 ilevel 转为当前列表树内的相对缩进层级。"""
    return max(_get_list_ilevel(list_block) - root_ilevel, 0)


def _flatten_list_items(list_block: Block, root_ilevel: int | None = None) -> list[str]:
    """Recursively flatten nested list blocks into a list of prefixed item strings."""
    items = []
    if root_ilevel is None:
        root_ilevel = _get_list_ilevel(list_block)
    relative_ilevel = _get_relative_list_ilevel(list_block, root_ilevel)
    attribute = list_block._list_attribute or "unordered"
    indent = "    " * relative_ilevel
    ordered_counter = _get_ordered_list_start(list_block)

    for block in list_block.blocks:
        if block.type in [BlockType.LIST, BlockType.INDEX]:
            items.extend(_flatten_list_items(block, root_ilevel))
        else:
            item_text = merge_para_with_text(block, escape_text_block_prefix=False)
            if item_text.strip():
                if attribute == "ordered":
                    items.append(f"{indent}{ordered_counter}. {item_text}")
                    ordered_counter += 1
                else:
                    items.append(f"{indent}- {item_text}")

    return items


def _flatten_list_items_v2(list_block: Block, root_ilevel: int | None = None) -> list[dict[str, Any]]:
    """Recursively flatten nested list blocks into v2-structured item dicts."""
    items = []
    if root_ilevel is None:
        root_ilevel = _get_list_ilevel(list_block)
    relative_ilevel = _get_relative_list_ilevel(list_block, root_ilevel)
    attribute = list_block._list_attribute or "unordered"
    ordered_counter = _get_ordered_list_start(list_block)

    for block in list_block.blocks:
        if block.type in [BlockType.LIST, BlockType.INDEX]:
            items.extend(_flatten_list_items_v2(block, root_ilevel))
        else:
            item_content = merge_para_with_text_v2(block)
            if item_content:
                if attribute == "ordered":
                    prefix = f"{'    ' * relative_ilevel}{ordered_counter}."
                    ordered_counter += 1
                else:
                    prefix = f"{'    ' * relative_ilevel}-"
                item = {
                    "item_type": "text",
                    "ilevel": relative_ilevel,
                    "prefix": prefix,
                    "item_content": item_content,
                }
                anchor = block.anchor
                if isinstance(anchor, str) and anchor.strip():
                    item["anchor"] = anchor.strip()
                items.append(item)

    return items


def merge_list_to_markdown(list_block: Block) -> str:
    """Recursively convert a nested list block to markdown text."""
    return "\n".join(_flatten_list_items(list_block)) + "\n"


def _collect_index_span_items(text_block: Block) -> list[tuple[str, str, list[str]]]:
    """按原始顺序收集目录叶子节点中的 span 三元组。"""
    span_items = []
    for line in text_block.lines:
        for span in line.spans:
            span_items.append((span.content, span.type, span._style))
    return span_items


def _normalize_anchor(block: Block) -> str | None:
    """规范化 block 锚点，空值返回 None。"""
    anchor = block.anchor
    if not isinstance(anchor, str) or not anchor.strip():
        return None
    return anchor.strip()


def _looks_like_index_page_token(token: str) -> bool:
    """判断目录 tab 后缀是否像页码，避免误删正文内容。"""
    token = token.strip()
    if not token:
        return False
    if len(token) > 12:
        return False
    if re.search(r"[\u4e00-\u9fff]", token):
        return False
    if re.fullmatch(r"\d+", token):
        return True
    if re.fullmatch(r"[ivxlcdm]+", token.lower()):
        return True
    if re.fullmatch(r"[a-zA-Z]", token):
        return True
    return False


def _strip_index_page_tail(span_items: list[tuple[str, str, list[str]]]) -> list[tuple[str, str, list[str]]]:
    """去掉目录叶子末尾 tab+页码，并把剩余 tab 转成普通空格。"""
    last_tab_span_idx = -1
    for i, (content, span_type, _) in enumerate(span_items):
        if span_type != ContentType.INLINE_EQUATION and "\t" in content:
            last_tab_span_idx = i

    should_strip_page_tail = False
    if last_tab_span_idx != -1:
        last_tab_content = span_items[last_tab_span_idx][0]
        tab_tail = last_tab_content.rsplit("\t", 1)[1]
        should_strip_page_tail = _looks_like_index_page_token(tab_tail)

    stripped_span_items = []
    for i, (content, span_type, span_style) in enumerate(span_items):
        if span_type != ContentType.INLINE_EQUATION:
            if i == last_tab_span_idx and should_strip_page_tail:
                content = content.rsplit("\t", 1)[0]
            content = content.replace("\t", " ")
        stripped_span_items.append((content, span_type, span_style))
    return stripped_span_items


def _get_uniform_index_style(span_items: list[tuple[str, str, list[str]]]) -> list[str] | None:
    """如果目录叶子所有非公式 span 样式完全一致，则返回统一样式。"""
    non_eq_styles = [
        tuple(span_style)
        for content, span_type, span_style in span_items
        if content and span_type != ContentType.INLINE_EQUATION
    ]
    if not non_eq_styles:
        return None
    first_style = non_eq_styles[0]
    if first_style and all(style == first_style for style in non_eq_styles):
        return list(first_style)
    return None


def _render_uniform_index_item(
    span_items: list[tuple[str, str, list[str]]], uniform_style: list[str], inline_syntax: str
) -> str:
    """用统一样式渲染目录叶子，避免同样式片段产生碎片化 marker。"""
    raw_parts = []
    for content, span_type, _span_style in span_items:
        if not content:
            continue
        if span_type == ContentType.INLINE_EQUATION:
            raw_parts.append(f"{inline_left_delimiter}{content}{inline_right_delimiter}")
        else:
            raw_parts.append(_escape_office_inline_text(content, inline_syntax))
    item_text = "".join(raw_parts).strip()
    if not item_text:
        return ""
    return _apply_configured_style(item_text, uniform_style, inline_syntax)


def _render_mixed_index_item(span_items: list[tuple[str, str, list[str]]], inline_syntax: str) -> str:
    """逐 span 渲染目录叶子，超链接 span 在目录中仅保留可见文本。"""
    rendered_parts = []
    for content, span_type, span_style in span_items:
        if not content:
            continue
        if span_type == ContentType.INLINE_EQUATION:
            rendered_parts.append(
                _make_rendered_part(
                    span_type,
                    f"{inline_left_delimiter}{content}{inline_right_delimiter}",
                )
            )
        elif span_type == ContentType.HYPERLINK:
            _append_hyperlink_part(
                rendered_parts,
                content,
                span_style,
                inline_syntax,
                plain_text_only=True,
            )
        else:
            _append_text_part(
                rendered_parts,
                content,
                span_style,
                inline_syntax,
            )
    return _join_rendered_parts(rendered_parts).strip()


def _render_index_leaf_item(text_block: Block, indent: str) -> str | None:
    """渲染单个目录叶子节点，并在有锚点时挂载内部链接。"""
    inline_syntax = _select_block_inline_syntax(text_block)
    span_items = _collect_index_span_items(text_block)
    if not span_items:
        return None

    span_items = _strip_index_page_tail(span_items)
    uniform_style = _get_uniform_index_style(span_items)
    if uniform_style:
        item_text = _render_uniform_index_item(
            span_items,
            uniform_style,
            inline_syntax,
        )
    else:
        item_text = _render_mixed_index_item(span_items, inline_syntax)
    if not item_text:
        return None

    anchor = _normalize_anchor(text_block)
    if anchor is not None:
        item_text = _render_link(item_text, f"#{anchor}", inline_syntax)
    return f"{indent}- {item_text}"


def _flatten_index_items(index_block: Block) -> list[str]:
    """递归展平目录 block，保留目录锚点和目录项样式。"""
    items = []
    ilevel = index_block.ilevel
    if ilevel is None:
        ilevel = 0
    indent = "    " * ilevel
    for child in index_block.blocks:
        if child.type == BlockType.INDEX:
            items.extend(_flatten_index_items(child))
        elif child.type == BlockType.TEXT:
            item_text = _render_index_leaf_item(child, indent)
            if item_text:
                items.append(item_text)
    return items


def merge_index_to_markdown(index_block: Block) -> str:
    """Convert a nested index (TOC) block to markdown with hyperlinks."""
    return "\n".join(_flatten_index_items(index_block)) + "\n"


def _iter_child_blocks(para_block: Block, block_type: str) -> Generator[Block, None, None]:
    """按原始顺序遍历指定类型的子 block。"""
    for block in para_block.blocks:
        if block.type == block_type:
            yield block


def _iter_block_spans(block: Block) -> Generator[Span, None, None]:
    """按原始顺序遍历 block 下所有 span。"""
    for line in block.lines:
        for span in line.spans:
            yield span


def _iter_body_spans(para_block: Block, body_type: str, span_type: str) -> Generator[Span, None, None]:
    """遍历视觉类 block 的 body span，统一 image/table/chart 的查找方式。"""
    for block in _iter_child_blocks(para_block, body_type):
        for span in _iter_block_spans(block):
            if span.type == span_type:
                yield span


def _collect_caption_texts(para_block: Block, caption_type: str) -> list[str]:
    """收集 legacy markdown/content_list 使用的 caption 文本。"""
    return [merge_para_with_text(block) for block in _iter_child_blocks(para_block, caption_type)]


def _collect_caption_v2(para_block: Block, caption_type: str) -> list[dict[str, Any]]:
    """收集 content_list_v2 使用的结构化 caption spans。"""
    caption_content = []
    for block in _iter_child_blocks(para_block, caption_type):
        caption_content.extend(merge_para_with_text_v2(block))
    return caption_content


def blocks_to_markdown(
    para_blocks: list[Block],
    img_bucket_path: str = "",
    no_rich_content: bool = False,
) -> list[str]:
    page_markdown = []
    for para_block in para_blocks:
        para_text = ""
        para_type = para_block.type
        if para_type in [BlockType.TEXT, BlockType.INTERLINE_EQUATION]:
            para_text = merge_para_with_text(para_block)
            if para_type == BlockType.TEXT:
                bookmark_anchor = para_block.anchor
                if isinstance(bookmark_anchor, str) and bookmark_anchor.strip() and bookmark_anchor.strip().startswith("_Toc"):
                    para_text = f'<a id="{bookmark_anchor.strip()}"></a>\n{para_text}'
        elif para_type == BlockType.LIST:
            para_text = merge_list_to_markdown(para_block)
        elif para_type == BlockType.INDEX:
            para_text = merge_index_to_markdown(para_block)
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            title_text = merge_para_with_text(para_block)
            bookmark_anchor = para_block.anchor
            if isinstance(bookmark_anchor, str) and bookmark_anchor.strip():
                para_text = f'<a id="{bookmark_anchor.strip()}"></a>\n{"#" * title_level} {title_text}'
            else:
                para_text = f"{'#' * title_level} {title_text}"
        elif para_type == BlockType.IMAGE:
            if no_rich_content:
                continue
            for span in _iter_body_spans(para_block, BlockType.IMAGE_BODY, ContentType.IMAGE):
                if span:
                    para_text += f"![]({img_bucket_path}/{span.image_path})"
            for caption_text in _collect_caption_texts(para_block, BlockType.IMAGE_CAPTION):
                para_text += "  \n" + caption_text

        elif para_type == BlockType.TABLE:
            if no_rich_content:
                continue
            for span in _iter_body_spans(para_block, BlockType.TABLE_BODY, ContentType.TABLE):
                para_text += f"\n{_format_embedded_html(span.html, img_bucket_path)}\n"
            for caption_text in _collect_caption_texts(para_block, BlockType.TABLE_CAPTION):
                para_text += "  \n" + caption_text
        elif para_type == BlockType.CHART:
            if no_rich_content:
                continue
            image_path, chart_content = get_body_data(para_block)
            if chart_content:
                para_text += f"\n{_format_embedded_html(chart_content, img_bucket_path)}\n"
            elif image_path:
                para_text += f"![]({_build_media_path(img_bucket_path, image_path)})"
            else:
                continue
            for caption_text in _collect_caption_texts(para_block, BlockType.CHART_CAPTION):
                para_text += "  \n" + caption_text

        if para_text.strip() != "":
            page_markdown.append(para_text.strip("\r\n"))

    return page_markdown


def make_blocks_to_content_list(para_block: Block, img_bucket_path: str, page_idx: int | None) -> dict[str, Any]:
    para_type = para_block.type
    para_content: dict[str, Any] = {}
    if para_type in [
        BlockType.TEXT,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_FOOTNOTE,
    ]:
        para_content = {
            "type": para_type,
            "text": merge_para_with_text(para_block),
        }
    elif para_type == BlockType.LIST:
        para_content = {
            "type": para_type,
            "list_items": _flatten_list_items(para_block),
        }
    elif para_type == BlockType.INDEX:
        para_content = {
            "type": para_type,
            "list_items": _flatten_index_items(para_block),
        }
    elif para_type == BlockType.TITLE:
        title_level = get_title_level(para_block)
        para_content = {
            "type": ContentType.TEXT,
            "text": merge_para_with_text(para_block),
        }
        if title_level != 0:
            para_content["text_level"] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        para_content = {
            "type": ContentType.EQUATION,
            "text": merge_para_with_text(para_block),
            "text_format": "latex",
        }
    elif para_type == BlockType.IMAGE:
        para_content = {"type": ContentType.IMAGE, "img_path": "", BlockType.IMAGE_CAPTION: []}
        for span in _iter_body_spans(
            para_block,
            BlockType.IMAGE_BODY,
            ContentType.IMAGE,
        ):
            if span.image_path:
                para_content["img_path"] = f"{img_bucket_path}/{span.image_path}"
        para_content[BlockType.IMAGE_CAPTION].extend(_collect_caption_texts(para_block, BlockType.IMAGE_CAPTION))
    elif para_type == BlockType.TABLE:
        para_content = {"type": ContentType.TABLE, BlockType.TABLE_CAPTION: []}
        for span in _iter_body_spans(
            para_block,
            BlockType.TABLE_BODY,
            ContentType.TABLE,
        ):
            if span.html:
                para_content[BlockType.TABLE_BODY] = _format_embedded_html(span.html, img_bucket_path)
        para_content[BlockType.TABLE_CAPTION].extend(_collect_caption_texts(para_block, BlockType.TABLE_CAPTION))
    elif para_type == BlockType.CHART:
        para_content = {
            "type": ContentType.CHART,
            "img_path": "",
            "content": "",
            BlockType.CHART_CAPTION: [],
        }
        for span in _iter_body_spans(para_block, BlockType.CHART_BODY, ContentType.CHART):
            para_content["img_path"] = _build_media_path(img_bucket_path, span.image_path)
            if span.content:
                para_content["content"] = _format_embedded_html(span.content, img_bucket_path)
        para_content[BlockType.CHART_CAPTION].extend(_collect_caption_texts(para_block, BlockType.CHART_CAPTION))

    para_content["page_idx"] = page_idx
    anchor = para_block.anchor
    if isinstance(anchor, str) and anchor.strip():
        para_content["anchor"] = anchor.strip()

    return para_content


def make_blocks_to_content_list_v2(para_block: Block, img_bucket_path: str) -> dict[str, Any]:
    para_type = para_block.type
    para_content: dict[str, Any] = {}
    if para_type in [
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_FOOTNOTE,
    ]:
        if para_type == BlockType.HEADER:
            content_type = ContentTypeV2.PAGE_HEADER
        elif para_type == BlockType.FOOTER:
            content_type = ContentTypeV2.PAGE_FOOTER
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
        title_level = get_title_level(para_block)
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
    elif para_type in [
        BlockType.TEXT,
    ]:
        para_content = {
            "type": ContentTypeV2.PARAGRAPH,
            "content": {
                "paragraph_content": merge_para_with_text_v2(para_block),
            },
        }
    elif para_type == BlockType.INTERLINE_EQUATION:
        _, math_content = get_body_data(para_block)
        para_content = {
            "type": ContentTypeV2.EQUATION_INTERLINE,
            "content": {
                "math_content": math_content,
                "math_type": "latex",
            },
        }
    elif para_type == BlockType.IMAGE:
        image_path, _ = get_body_data(para_block)
        image_source = {
            "path": f"{img_bucket_path}/{image_path}",
        }
        para_content = {
            "type": ContentTypeV2.IMAGE,
            "content": {
                "image_source": image_source,
                "image_caption": _collect_caption_v2(
                    para_block,
                    BlockType.IMAGE_CAPTION,
                ),
            },
        }
    elif para_type == BlockType.TABLE:
        _, html = get_body_data(para_block)
        if html.count("<table") > 1:
            table_nest_level = 2
        else:
            table_nest_level = 1
        if "colspan" in html or "rowspan" in html or table_nest_level > 1:
            table_type = ContentTypeV2.TABLE_COMPLEX
        else:
            table_type = ContentTypeV2.TABLE_SIMPLE

        para_content = {
            "type": ContentTypeV2.TABLE,
            "content": {
                "table_caption": _collect_caption_v2(
                    para_block,
                    BlockType.TABLE_CAPTION,
                ),
                "html": _format_embedded_html(html, img_bucket_path),
                "table_type": table_type,
                "table_nest_level": table_nest_level,
            },
        }
    elif para_type == BlockType.CHART:
        image_path, chart_content = get_body_data(para_block)
        para_content = {
            "type": ContentTypeV2.CHART,
            "content": {
                "image_source": {
                    "path": _build_media_path(img_bucket_path, image_path),
                },
                "content": _format_embedded_html(chart_content, img_bucket_path),
                "chart_caption": _collect_caption_v2(
                    para_block,
                    BlockType.CHART_CAPTION,
                ),
            },
        }
    elif para_type == BlockType.LIST:
        list_type = ContentTypeV2.LIST_TEXT
        attribute = para_block._list_attribute or "unordered"
        para_content = {
            "type": ContentTypeV2.LIST,
            "content": {
                "list_type": list_type,
                "attribute": attribute,
                "list_items": _flatten_list_items_v2(para_block),
            },
        }
    elif para_type == BlockType.INDEX:
        para_content = {
            "type": ContentTypeV2.INDEX,
            "content": {
                "list_type": ContentTypeV2.LIST_TEXT,
                "list_items": _flatten_list_items_v2(para_block),
            },
        }

    anchor = para_block.anchor
    if isinstance(anchor, str) and anchor.strip():
        para_content["anchor"] = anchor.strip()

    return para_content


def get_body_data(para_block: Block) -> tuple[str, str]:
    """
    Extract image_path and body content from para_block
    Returns:
        - For IMAGE/INTERLINE_EQUATION: (image_path, '')
        - For TABLE: (image_path, html)
        - For CHART: (image_path, content)
        - Default: ('', '')
    """

    def get_data_from_spans(lines: list[Line]) -> tuple[str, str]:
        for line in lines:
            for span in line.spans:
                span_type = span.type
                if span_type == ContentType.TABLE:
                    return span.image_path, span.html
                elif span_type == ContentType.CHART:
                    return span.image_path, span.content
                elif span_type == ContentType.IMAGE:
                    return span.image_path, ""
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
                if block_type == BlockType.CHART_BODY:
                    return result
        return "", ""

    # 处理直接包含 lines 的结构
    return get_data_from_spans(para_block.lines)


def _span_has_content_for_v2(span: Span, visible_styles: set[str]) -> bool:
    """判断 V2 span 是否应保留，支持 hyperlink children 的可见空白样式。"""
    content = span.content
    span_style = span._style
    if content.strip():
        return True
    if content and span_style and any(s in visible_styles for s in span_style):
        return True
    for child in span._children:
        child_content = child.content
        child_style = child._style
        if child_content.strip():
            return True
        if child_content and child_style and any(s in visible_styles for s in child_style):
            return True
    return False


def merge_para_with_text_v2(para_block: Block) -> list[dict[str, Any]]:
    """将 Office 段落转换为 content_list_v2 spans，避免原地修改 middle_json。"""
    _visible_styles = {"underline", "strikethrough"}
    para_content: list[dict[str, Any]] = []
    if para_block.type == BlockType.TITLE:
        section_number = para_block.section_number
        if section_number:
            # v2 保持结构化 spans，同时补上 middle_json 已生成的自动标题编号。
            para_content.append(
                {
                    "type": ContentTypeV2.SPAN_TEXT,
                    "content": f"{section_number} ",
                }
            )
    for line in para_block.lines:
        for span in line.spans:
            if _span_has_content_for_v2(span, _visible_styles):
                rendered_span = asdict(span)
                if rendered_span["type"] == ContentType.INLINE_EQUATION:
                    rendered_span["type"] = ContentTypeV2.SPAN_EQUATION_INLINE
                para_content.append(rendered_span)
    return para_content
