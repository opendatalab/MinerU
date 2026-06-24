# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import html as html_lib
import re
from typing import Any, Literal
from urllib.parse import urlparse

from ...types import EMPTY_BBOX, Block, BlockType, ContentType, Line, Span
from ..utils.magic_model_utils import tie_up_category_by_index


class MagicModel:
    def __init__(self, page_blocks: list[dict[str, Any]]) -> None:
        blocks: list[Block] = []

        # 对caption块进行分类，将其分类为image_caption, table_caption, chart_caption
        page_blocks = classify_caption_blocks(page_blocks)

        # 解析每个块
        for index, block_info in enumerate(page_blocks):
            block_type = block_info["type"]
            block_content = block_info.get("content", "")
            if not block_content and block_type != BlockType.CHART:
                continue

            if block_type in [
                "text",
                "title",
                "image_caption",
                "table_caption",
                "chart_caption",
                "header",
                "footer",
                "page_footnote",
            ]:
                span = parse_text_block_spans(block_content)

            elif block_type in ["image"]:
                block_type = BlockType.IMAGE_BODY
                span = Span(type=ContentType.IMAGE, bbox=EMPTY_BBOX, image_base64=block_content)
            elif block_type in ["table"]:
                block_type = BlockType.TABLE_BODY
                span = Span(type=ContentType.TABLE, bbox=EMPTY_BBOX, html=clean_table_html(block_content))
            elif block_type in ["chart"]:
                block_type = BlockType.CHART_BODY
                span = Span(type=ContentType.CHART, bbox=EMPTY_BBOX, content=block_content)
                if block_info.get("image_base64"):
                    span.image_base64 = block_info["image_base64"]
            elif block_type in ["equation"]:
                block_type = BlockType.INTERLINE_EQUATION
                span = Span(type=ContentType.INTERLINE_EQUATION, bbox=EMPTY_BBOX, content=block_content)
            elif block_type in ["list"]:
                # 解析嵌套列表结构，生成与VLM一致的blocks结构
                parsed_list = parse_list_block(block_info)
                if parsed_list:
                    # 使用外层index作为列表block的index
                    parsed_list.index = index
                    blocks.append(parsed_list)
                continue
            elif block_type in ["index"]:
                # 解析嵌套索引结构（目录），生成与list一致的blocks结构
                parsed_index = parse_index_block(block_info)
                if parsed_index:
                    parsed_index.index = index
                    blocks.append(parsed_index)
                continue
            else:
                # 未知类型，跳过
                continue

            # 处理span类型并添加到all_spans
            if isinstance(span, Span):
                line = Line(bbox=EMPTY_BBOX, spans=[span])
            elif isinstance(span, list):
                line = Line(bbox=EMPTY_BBOX, spans=span)
            else:
                raise ValueError(f"Unsupported span type: {type(span)}")

            block = Block(type=block_type, bbox=EMPTY_BBOX, lines=[line], index=index)
            anchor = block_info.get("anchor")
            if (
                isinstance(anchor, str)
                and anchor.strip()
                and block_type in [BlockType.TITLE, BlockType.TEXT, BlockType.INTERLINE_EQUATION]
            ):
                block.anchor = anchor.strip()
            if block_type == BlockType.TITLE:
                block.is_numbered_style = block_info.get("is_numbered_style", False)
                block.level = block_info.get("level", 1)
            blocks.append(block)

        self.image_blocks: list[Block] = []
        self.table_blocks: list[Block] = []
        self.chart_blocks: list[Block] = []
        self.interline_equation_blocks: list[Block] = []
        self.text_blocks: list[Block] = []
        self.title_blocks: list[Block] = []
        self.discarded_blocks: list[Block] = []
        self.list_blocks: list[Block] = []
        self.index_blocks: list[Block] = []
        for block in blocks:
            if block.type in [BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]:
                self.image_blocks.append(block)
            elif block.type in [BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]:
                self.table_blocks.append(block)
            elif block.type in [BlockType.CHART_BODY, BlockType.CHART_CAPTION]:
                self.chart_blocks.append(block)
            elif block.type == BlockType.INTERLINE_EQUATION:
                self.interline_equation_blocks.append(block)
            elif block.type == BlockType.TEXT:
                self.text_blocks.append(block)
            elif block.type == BlockType.TITLE:
                self.title_blocks.append(block)
            # elif block.type in [BlockType.REF_TEXT]:
            #     self.ref_text_blocks.append(block)
            # elif block.type in [BlockType.PHONETIC]:
            #     self.phonetic_blocks.append(block)
            elif block.type in [
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
                BlockType.ASIDE_TEXT,
                BlockType.PAGE_FOOTNOTE,
            ]:
                self.discarded_blocks.append(block)
            elif block.type == BlockType.LIST:
                self.list_blocks.append(block)
            elif block.type == BlockType.INDEX:
                self.index_blocks.append(block)
            else:
                continue

        self.image_blocks, not_include_image_blocks = fix_two_layer_blocks(self.image_blocks, BlockType.IMAGE)
        self.table_blocks, not_include_table_blocks = fix_two_layer_blocks(self.table_blocks, BlockType.TABLE)
        self.chart_blocks, not_include_chart_blocks = fix_two_layer_blocks(self.chart_blocks, BlockType.CHART)

        for block in not_include_image_blocks + not_include_table_blocks + not_include_chart_blocks:
            block.type = BlockType.TEXT
            self.text_blocks.append(block)

    def get_list_blocks(self) -> list[Block]:
        return self.list_blocks

    def get_index_blocks(self) -> list[Block]:
        return self.index_blocks

    def get_image_blocks(self) -> list[Block]:
        return self.image_blocks

    def get_table_blocks(self) -> list[Block]:
        return self.table_blocks

    def get_chart_blocks(self) -> list[Block]:
        return self.chart_blocks

    def get_title_blocks(self) -> list[Block]:
        return self.title_blocks

    def get_text_blocks(self) -> list[Block]:
        return self.text_blocks

    def get_interline_equation_blocks(self) -> list[Block]:
        return self.interline_equation_blocks

    def get_discarded_blocks(self) -> list[Block]:
        return self.discarded_blocks


def _parse_style_list(style_str: str | None) -> list[str]:
    """解析逗号分隔的 Office inline style 字符串。"""
    if not style_str:
        return []
    return [style.strip() for style in style_str.split(",") if style.strip()]


def _parse_hyperlink_text_children(hyperlink_content: str, text_tag_re: re.Pattern[str]) -> tuple[list[Span], str]:
    """解析一个 hyperlink 内部的多个 text 子片段，并保留每段样式。"""
    url_start = hyperlink_content.find("<url>")
    url_end = hyperlink_content.find("</url>")
    if url_start == -1 or url_end == -1 or url_end < url_start:
        return [], ""

    children: list[Span] = []
    pos = 0
    while pos < url_start:
        text_match = text_tag_re.search(hyperlink_content, pos)
        if text_match is None or text_match.start() >= url_start:
            break

        text_end = hyperlink_content.find("</text>", text_match.end())
        if text_end == -1 or text_end > url_start:
            return [], ""

        child = Span(
            type=ContentType.TEXT,
            bbox=EMPTY_BBOX,
            content=hyperlink_content[text_match.end() : text_end],
            _style=_parse_style_list(text_match.group(1)),
        )
        children.append(child)
        pos = text_end + 7

    return children, hyperlink_content[url_start + 5 : url_end]


def parse_text_block_spans(content: str) -> list[Span]:
    """
    解析文本类block的content，提取其中的文本、行内公式、超链接和字体样式。

    支持的标签格式：
    - <eq>...</eq>: 行内公式
    - <hyperlink><text [style="..."]>...</text><url>...</url></hyperlink>: 超链接（支持样式）
    - <text style="...">...</text>: 带字体样式的普通文本

    字体样式值（逗号分隔）：bold, italic, underline, emphasis, strikethrough, superscript, subscript

    Args:
        content: 文本块的content字符串，可能包含特殊标签

    Returns:
        包含多个span的列表，每个span是一个字典，包含type和content等字段。
        带样式的文本span额外包含 style 字段（list类型）。
    """
    if not content:
        return []

    # 匹配 <text> 或 <text style="..."> 开始标签
    _text_tag_re = re.compile(r'<text(?:\s+style="([^"]*)")?>')

    spans = []
    last_end = 0
    pos = 0

    while pos < len(content):
        # 查找行内公式标签 <eq>...</eq>
        eq_start = content.find("<eq>", pos)
        # 查找超链接标签 <hyperlink>
        hyperlink_start = content.find("<hyperlink>", pos)
        # 查找带样式的文本标签 <text ...>（顶层，不在 hyperlink 内部）
        text_tag_match = _text_tag_re.search(content, pos)
        text_tag_start = text_tag_match.start() if text_tag_match else -1

        # 收集所有有效的标签位置
        candidates = []
        if eq_start != -1:
            candidates.append((eq_start, "eq"))
        if hyperlink_start != -1:
            candidates.append((hyperlink_start, "hyperlink"))
        if text_tag_start != -1:
            candidates.append((text_tag_start, "text"))

        # 没有找到任何标签，处理剩余文本
        if not candidates:
            remaining_text = content[last_end:]
            if remaining_text:
                spans.append(Span(type=ContentType.TEXT, bbox=EMPTY_BBOX, content=remaining_text))
            break

        # 取位置最小的标签
        next_tag_pos, next_tag_type = min(candidates, key=lambda x: x[0])

        # 处理标签前的文本
        if next_tag_pos > last_end:
            text_before = content[last_end:next_tag_pos]
            if text_before:
                spans.append(Span(type=ContentType.TEXT, bbox=EMPTY_BBOX, content=text_before))

        # 处理行内公式
        if next_tag_type == "eq":
            eq_end = content.find("</eq>", next_tag_pos)
            if eq_end != -1:
                formula_content = content[next_tag_pos + 4 : eq_end]
                spans.append(Span(type=ContentType.INLINE_EQUATION, bbox=EMPTY_BBOX, content=formula_content))
                pos = eq_end + 5
                last_end = pos
            else:
                # 未找到闭合标签，将<eq>作为普通文本处理
                spans.append(Span(type=ContentType.TEXT, bbox=EMPTY_BBOX, content=content[last_end:]))
                break

        # 处理带样式的文本标签
        elif next_tag_type == "text":
            text_end = content.find("</text>", next_tag_pos)
            if text_end != -1:
                # text_tag_match 对应当前 next_tag_pos 的匹配
                # 重新匹配确保位置对齐
                tag_open_end = content.find(">", next_tag_pos) + 1
                text_content = content[tag_open_end:text_end]
                style_str = text_tag_match.group(1) if text_tag_match and text_tag_match.start() == next_tag_pos else None
                span = Span(type=ContentType.TEXT, bbox=EMPTY_BBOX, content=text_content)
                if style_str:
                    span._style = [s.strip() for s in style_str.split(",") if s.strip()]
                spans.append(span)
                pos = text_end + 7
                last_end = pos
            else:
                # 未找到闭合标签，作为普通文本处理
                spans.append(Span(type=ContentType.TEXT, bbox=EMPTY_BBOX, content=content[last_end:]))
                break

        # 处理超链接
        elif next_tag_type == "hyperlink":
            hyperlink_end = content.find("</hyperlink>", next_tag_pos)
            if hyperlink_end != -1:
                # 提取超链接内容
                hyperlink_content = content[next_tag_pos + 11 : hyperlink_end]

                # 解析内部的一个或多个 <text [style="..."]> 和一个 <url> 标签
                children, link_url = _parse_hyperlink_text_children(
                    hyperlink_content,
                    _text_tag_re,
                )

                if children and link_url:
                    if len(children) == 1:
                        child = children[0]
                        span = Span(
                            type=ContentType.HYPERLINK,
                            bbox=EMPTY_BBOX,
                            content=child.content,
                            _url=link_url,
                            _style=child._style,
                        )
                    else:
                        span = Span(
                            type=ContentType.HYPERLINK,
                            bbox=EMPTY_BBOX,
                            content="".join(child.content for child in children),
                            _url=link_url,
                            _children=children,
                        )
                    spans.append(span)
                    pos = hyperlink_end + 12
                    last_end = pos
                else:
                    # 超链接格式不正确，作为普通文本处理
                    spans.append(Span(type=ContentType.TEXT, bbox=EMPTY_BBOX, content=content[last_end:]))
                    break
            else:
                # 未找到闭合标签，将<hyperlink>作为普通文本处理
                spans.append(Span(type=ContentType.TEXT, bbox=EMPTY_BBOX, content=content[last_end:]))
                break

    return spans


def parse_list_block(list_block: dict[str, Any]) -> Block | None:
    """
    递归解析嵌套列表结构，生成与VLM一致的blocks结构。

    Args:
        list_block: 列表块字典

    Returns:
        tuple: (解析后的列表block, 下一个可用索引)
    """
    content = list_block.get("content", [])
    if not content:
        return None

    blocks = []

    for item in content:
        item_type = item.get("type", "")

        if item_type == "text":
            # 解析文本项（可能包含行内公式和超链接）
            text_content = item.get("content", "")
            spans = parse_text_block_spans(text_content)
            text_block = Block(type=BlockType.TEXT, bbox=EMPTY_BBOX, index=0, lines=[Line(bbox=EMPTY_BBOX, spans=spans)])
            blocks.append(text_block)

        elif item_type == "list":
            # 递归解析嵌套列表
            nested_list = parse_list_block(item)
            if nested_list:
                blocks.append(nested_list)

    # 构建当前列表block
    result = Block(
        type=BlockType.LIST,
        bbox=EMPTY_BBOX,
        index=0,
        blocks=blocks,
        start=list_block.get("start"),
        ilevel=list_block.get("ilevel", 0),
        _list_attribute=list_block.get("attribute", "unordered"),
    )
    return result


def parse_index_block(index_block: dict[str, Any]) -> Block | None:
    """
    递归解析嵌套索引结构（目录），生成与list一致的blocks结构。

    Args:
        index_block: 索引块字典

    Returns:
        解析后的索引block字典，若内容为空则返回 None
    """
    content = index_block.get("content", [])
    if not content:
        return None

    blocks = []

    for item in content:
        item_type = item.get("type", "")

        if item_type == "text":
            text_content = item.get("content", "")
            spans = parse_text_block_spans(text_content)
            text_block = Block(type=BlockType.TEXT, bbox=EMPTY_BBOX, index=0, lines=[Line(bbox=EMPTY_BBOX, spans=spans)])
            anchor = item.get("anchor")
            if isinstance(anchor, str) and anchor.strip():
                text_block.anchor = anchor.strip()
            blocks.append(text_block)

        elif item_type == "index":
            nested_index = parse_index_block(item)
            if nested_index:
                blocks.append(nested_index)

    result = Block(
        type=BlockType.INDEX,
        bbox=EMPTY_BBOX,
        index=0,
        blocks=blocks,
        ilevel=index_block.get("ilevel", 0),
    )
    return result


def _sanitize_table_hyperlink_href(href: str) -> str:
    """清洗表格内超链接地址，仅保留安全协议或相对链接。"""
    normalized_href = html_lib.unescape(href).strip()
    if not normalized_href:
        return ""

    if normalized_href.lower().startswith(("javascript:", "data:", "vbscript:")):
        return ""

    parsed = urlparse(normalized_href)
    scheme = parsed.scheme.lower() if parsed.scheme else ""
    if scheme and scheme not in {"http", "https", "mailto", "ftp"}:
        return ""

    return html_lib.escape(normalized_href, quote=True)


def clean_table_html(html: str) -> str:
    """
    清洗表格HTML，只保留对表格结构表示有用的信息。

    保留的属性：
    - colspan: 列合并
    - rowspan: 行合并
    - a.href: 表格内超链接
    - img.src/alt/width/height: 表格内图片

    清洗的内容：
    - 移除所有style属性
    - 移除所有class属性
    - 移除border等其他属性
    - 保持表格结构标签（table, thead, tbody, tr, th, td等）

    Args:
        html: 原始表格HTML字符串

    Returns:
        清洗后的HTML字符串
    """
    if not html:
        return ""

    # 需要保留的属性（对表格结构有用）
    preserved_attrs = {"colspan", "rowspan"}
    # img 标签需要额外保留的属性（内联 base64 图片内容）
    img_preserved_attrs = {"src", "alt", "width", "height"}
    # a 标签只保留清洗后的 href，避免表格超链接在中间层被清掉。
    anchor_preserved_attrs = {"href"}

    def clean_tag(match: re.Match[str]) -> str:
        """清洗单个标签，只保留结构相关的属性"""
        full_tag = match.group(0)
        tag_name = match.group(1).lower()

        # 自闭合标签的处理
        is_self_closing = full_tag.rstrip().endswith("/>")

        # img 标签额外保留图片相关属性（如内联 base64 src）
        current_preserved = preserved_attrs | (img_preserved_attrs if tag_name == "img" else set())
        current_preserved |= anchor_preserved_attrs if tag_name == "a" else set()

        # 提取需要保留的属性
        kept_attrs = []

        # 匹配所有属性: attr="value" 或 attr='value' 或 attr=value 或单独的attr
        attr_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))|(\w+)(?=\s|>|/>)'
        for attr_match in re.finditer(attr_pattern, full_tag):
            if attr_match.group(5):
                # 单独的属性（如 disabled），跳过
                continue

            attr_name = attr_match.group(1)
            if attr_name is None:
                continue
            attr_name = attr_name.lower()
            attr_value = attr_match.group(2) or attr_match.group(3) or attr_match.group(4) or ""

            # 只保留指定属性（表格结构属性，img 标签还额外保留图片内容属性）
            if tag_name == "a" and attr_name == "href":
                attr_value = _sanitize_table_hyperlink_href(attr_value)
                if not attr_value:
                    continue

            if attr_name in current_preserved:
                kept_attrs.append(f'{attr_name}="{attr_value}"')

        # 重建标签
        if kept_attrs:
            attrs_str = " " + " ".join(kept_attrs)
        else:
            attrs_str = ""

        if is_self_closing:
            return f"<{tag_name}{attrs_str}/>"
        else:
            return f"<{tag_name}{attrs_str}>"

    # 匹配开始标签（包括自闭合标签），捕获标签名
    # 匹配 <tagname ...> 或 <tagname .../>
    tag_pattern = r"<(\w+)(?:\s+[^>]*)?\s*/?>"

    result = re.sub(tag_pattern, clean_tag, html)

    return result


def isolated_formula_clean(txt: str) -> str:
    latex = txt[:]
    if latex.startswith("\\["):
        latex = latex[2:]
    if latex.endswith("\\]"):
        latex = latex[:-2]
    latex = latex.strip()
    return latex


def code_content_clean(content: str) -> str:
    """清理代码内容，移除Markdown代码块的开始和结束标记"""
    if not content:
        return ""

    lines = content.splitlines()
    start_idx = 0
    end_idx = len(lines)

    # 处理开头的三个反引号
    if lines and lines[0].startswith("```"):
        start_idx = 1

    # 处理结尾的三个反引号
    if lines and end_idx > start_idx and lines[end_idx - 1].strip() == "```":
        end_idx -= 1

    # 只有在有内容时才进行join操作
    if start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx]).strip()
    return ""


def __tie_up_category_by_index(blocks: list[Block], subject_block_type: str, object_block_type: str) -> list[dict[str, Any]]:
    """基于index的主客体关联包装函数"""

    # 定义获取主体和客体对象的函数
    def get_subjects() -> list[Block]:
        return [block for block in blocks if block.type == subject_block_type]

    def get_objects() -> list[Block]:
        return [block for block in blocks if block.type == object_block_type]

    # 调用通用方法
    return tie_up_category_by_index(
        get_subjects,
        get_objects,
        include_bbox=False,
    )


def get_type_blocks(blocks: list[Block], block_type: Literal["image", "table", "chart"]) -> list[dict[str, Any]]:
    with_captions = __tie_up_category_by_index(blocks, f"{block_type}_body", f"{block_type}_caption")
    ret: list[dict[str, Any]] = []
    for v in with_captions:
        record = {
            f"{block_type}_body": v["sub_bbox"],
            f"{block_type}_caption_list": v["obj_bboxes"],
        }
        ret.append(record)
    return ret


def fix_two_layer_blocks(blocks: list[Block], fix_type: Literal["image", "table", "chart"]) -> tuple[list[Block], list[Block]]:
    """TODO: this function has many unknown types."""

    need_fix_blocks = get_type_blocks(blocks, fix_type)

    fixed_blocks: list[Block] = []
    not_include_blocks: list[Block] = []
    processed_indices: set[int] = set()

    # 将每个block的caption_list中不连续index的元素提出来作为普通block处理
    for block in need_fix_blocks:
        caption_list = block[f"{fix_type}_caption_list"]
        body_index = block[f"{fix_type}_body"].index

        # 处理caption_list (从body往前看,caption在body之前)
        if caption_list:
            # 按index降序排列,从最接近body的开始检查
            caption_list.sort(key=lambda x: x.index, reverse=True)
            filtered_captions = [caption_list[0]]
            for i in range(1, len(caption_list)):
                prev_index = caption_list[i - 1].index
                curr_index = caption_list[i].index

                # 检查是否连续
                if curr_index == prev_index - 1:
                    filtered_captions.append(caption_list[i])
                else:
                    # 检查gap中是否只有body_index
                    gap_indices = set(range(curr_index + 1, prev_index))
                    if gap_indices == {body_index}:
                        # gap中只有body_index,不算真正的gap
                        filtered_captions.append(caption_list[i])
                    else:
                        # 出现真正的gap,后续所有caption都作为普通block
                        not_include_blocks.extend(caption_list[i:])
                        break
            # 恢复升序
            filtered_captions.reverse()
            block[f"{fix_type}_caption_list"] = filtered_captions

    # 构建两层结构blocks
    for block in need_fix_blocks:
        body = block[f"{fix_type}_body"]
        caption_list = block[f"{fix_type}_caption_list"]

        body.type = f"{fix_type}_body"
        for caption in caption_list:
            caption.type = f"{fix_type}_caption"
            processed_indices.add(caption.index)

        processed_indices.add(body.index)

        two_layer_block = Block(
            index=body.index,
            type=fix_type,
            bbox=EMPTY_BBOX,
            blocks=sorted([body, *caption_list], key=lambda x: x.index),
        )
        fixed_blocks.append(two_layer_block)

    # 添加未处理的blocks
    for block in blocks:
        block.type = ""
        if block.index not in processed_indices and block not in not_include_blocks:
            not_include_blocks.append(block)

    return fixed_blocks, not_include_blocks


def classify_caption_blocks(page_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    对page_blocks中的caption块进行分类，将其分类为image_caption、table_caption或chart_caption。

    规则：
    1. 只有与type为table、image或chart相邻的caption可以作为caption
    2. caption块与table、image或chart中相隔的块全部是caption的情况视为该caption块与母块相邻
    3. caption的类型与他前置位相邻的母块type一致，如果没有前置位母块则检查是否有后置位母块
    4. 没有相邻母块的caption需要变更type为text
    5. 当一个block的type是table、image或chart时，其后续的第一个text块如果以特定前缀开头，则将其设置为相应的caption类型
       - table后的text块以["表", "table"]开头（不区分大小写）-> table_caption
       - image后的text块以["图", "fig"]开头（不区分大小写）-> image_caption
       - chart后的text块以["图", "fig", "chart"]开头（不区分大小写）-> chart_caption
    """
    if not page_blocks:
        return page_blocks

    available_types = ["table", "image", "chart"]

    # 定义caption前缀匹配规则
    table_caption_prefixes = ["表", "table"]
    image_caption_prefixes = ["图", "fig"]
    chart_caption_prefixes = ["图", "fig", "chart"]

    # 第一步：处理table/image/chart后续的text块，将符合条件的text块标记为caption
    preprocessed_blocks = []
    n = len(page_blocks)

    for i, block in enumerate(page_blocks):
        block_type = block.get("type")

        # 检查是否是table或image块
        if block_type in available_types:
            preprocessed_blocks.append(block)

            # 查找后续的第一个text块
            if i + 1 < n:
                next_block = page_blocks[i + 1]
                next_block_type = next_block.get("type")

                if next_block_type == "text":
                    content = next_block.get("content", "").strip().lower()

                    # 根据当前块类型检查是否匹配caption前缀
                    if block_type == "table":
                        if any(content.startswith(prefix.lower()) for prefix in table_caption_prefixes):
                            # 将text块标记为caption，后续会被处理为table_caption
                            next_block = next_block.copy()
                            next_block["type"] = "caption"
                            page_blocks[i + 1] = next_block
                    elif block_type == "image":
                        if any(content.startswith(prefix.lower()) for prefix in image_caption_prefixes):
                            # 将text块标记为caption，后续会被处理为image_caption
                            next_block = next_block.copy()
                            next_block["type"] = "caption"
                            page_blocks[i + 1] = next_block
                    elif block_type == "chart":
                        if any(content.startswith(prefix.lower()) for prefix in chart_caption_prefixes):
                            # 将text块标记为caption，后续会被处理为chart_caption
                            next_block = next_block.copy()
                            next_block["type"] = "caption"
                            page_blocks[i + 1] = next_block
        else:
            preprocessed_blocks.append(block)

    # 第二步：处理caption块的分类
    result_blocks = []

    for i, block in enumerate(page_blocks):
        if block.get("type") != "caption":
            result_blocks.append(block)
            continue

        # 查找前置位相邻的母块（table、image或chart）
        # 向前查找，跳过连续的caption块
        prev_parent_type = None
        j = i - 1
        while j >= 0:
            prev_block_type = page_blocks[j].get("type")
            if prev_block_type in available_types:
                prev_parent_type = prev_block_type
                break
            elif prev_block_type == "caption":
                # 继续向前查找
                j -= 1
            else:
                # 遇到非caption且非table/image/chart的块，停止查找
                break

        # 查找后置位相邻的母块（table、image或chart）
        # 向后查找，跳过连续的caption块
        next_parent_type = None
        k = i + 1
        while k < n:
            next_block_type = page_blocks[k].get("type")
            if next_block_type in available_types:
                next_parent_type = next_block_type
                break
            elif next_block_type == "caption":
                # 继续向后查找
                k += 1
            else:
                # 遇到非caption且非table/image/chart的块，停止查找
                break

        # 根据规则确定caption类型
        new_block = block.copy()
        if prev_parent_type:
            # 优先使用前置位母块的类型
            new_block["type"] = f"{prev_parent_type}_caption"
        elif next_parent_type:
            # 没有前置位母块，使用后置位母块的类型
            new_block["type"] = f"{next_parent_type}_caption"
        else:
            # 没有相邻母块，变更为text
            new_block["type"] = "text"

        result_blocks.append(new_block)

    return result_blocks
