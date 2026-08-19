# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 Markdown 的公共渲染实现。"""

from __future__ import annotations

import html
import re

from mineru.config import LatexDelimitersConfig, config
from mineru.render.utils.assets import build_markdown_image, resolve_image_source
from mineru.render.utils.inline import (
    inline_plain_text,
    parse_inline_content,
    render_inline_content,
    render_internal_link,
    render_joined_inline_contents,
)
from mineru.render.utils.logical_blocks import MarkdownRenderMode, PlannedBlock, build_render_plan
from mineru.render.utils.markdown_table import format_embedded_html, render_html_table
from mineru.render.utils.markdown_utils import escape_standalone_marker_rule, escape_text_block_markdown_prefix
from mineru.types import (
    RAW_ALGORITHM,
    BlockType,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageBlock,
    ParagraphTitleBlock,
    RefTextBlock,
    TableAnnotationBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
)

_DEFAULT_HIDDEN_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
}
_PAGE_SEPARATOR = "\n\n---\n\n"
_ALGORITHM_TOKEN_RE = re.compile(
    r"<eq>(?P<latex>.*?)</eq>|(?P<script_tag></?(?:sub|sup)>)",
    re.IGNORECASE | re.DOTALL,
)
_VALID_CODE_LANGUAGE_RE = re.compile(r"[A-Za-z0-9_.+#-]+")
_INDEX_ROMAN_RE = re.compile(r"[ivxlcdm]+", re.IGNORECASE)
# 去除首部空白后，前五个可见字符内出现 Unicode 数字即视为单项命中。
_REF_ITEM_NUMBER_PREFIX_RE = re.compile(r"^\D{0,4}\d")
_UNORDERED_LIST_ITEM_RE = re.compile(r"^[ \t]*-[ \t]+")


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
        if mode is MarkdownRenderMode.DEFAULT and planned.block.type in _DEFAULT_HIDDEN_TYPES:
            continue
        text = _render_planned_block(planned, delimiters=delimiters, asset_base_url=asset_base_url)
        if text and text.strip():
            rendered.append(text.strip("\n"))
    return "\n\n".join(rendered)


def _render_planned_block(
    planned: PlannedBlock,
    *,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """按具体 Pydantic block 类型分发 Markdown 渲染。"""
    block = planned.block
    if isinstance(block, TextBlock):
        content = render_joined_inline_contents(planned.text_contents or [block.content], delimiters)
        return escape_standalone_marker_rule(escape_text_block_markdown_prefix(content))
    if isinstance(block, RefTextBlock):
        content = render_joined_inline_contents(planned.text_contents or [block.content], delimiters)
        return escape_standalone_marker_rule(content)
    if isinstance(block, (DocTitleBlock, ParagraphTitleBlock)):
        return _render_title(block, delimiters)
    if isinstance(block, PageAuxTextBlock):
        content = escape_text_block_markdown_prefix(render_inline_content(block.content, delimiters))
        return escape_standalone_marker_rule(content)
    if isinstance(block, EquationBlock):
        return _render_equation(block, delimiters, asset_base_url)
    if isinstance(block, ListBlock):
        return _render_list(block, delimiters)
    if isinstance(block, IndexBlock):
        return _render_index(block, delimiters)
    if isinstance(block, ImageBlock):
        return _render_image_block(block, delimiters, asset_base_url)
    if isinstance(block, TableBlock):
        return _render_table_block(block, delimiters, asset_base_url)
    if isinstance(block, ChartBlock):
        return _render_chart_block(block, delimiters, asset_base_url)
    if isinstance(block, CodeBlock):
        return _render_code_block(block, delimiters)
    raise TypeError(f"Unsupported PageBlock type: {type(block).__name__}")


def _render_single_block(
    block: PageBlock,
    *,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """不执行续段合并或页面过滤，直接渲染一个顶层 block。"""
    text_contents = [block.content] if isinstance(block, (TextBlock, RefTextBlock)) else []
    planned = PlannedBlock(page_idx=0, block=block, text_contents=text_contents)
    return _render_planned_block(
        planned,
        delimiters=delimiters,
        asset_base_url=asset_base_url,
    )


def _render_title(
    block: DocTitleBlock | ParagraphTitleBlock,
    delimiters: LatexDelimitersConfig,
) -> str:
    """渲染带可选 HTML anchor 的 Markdown 标题。"""
    level = min(max(block.level, 1), 6)
    title = f"{'#' * level} {_render_title_inline_content(block, delimiters)}"
    if not block.anchor:
        return title
    anchor = html.escape(block.anchor.strip(), quote=True)
    return f'<a id="{anchor}"></a>\n{title}'


def _render_title_inline_content(
    block: DocTitleBlock | ParagraphTitleBlock,
    delimiters: LatexDelimitersConfig,
) -> str:
    """只渲染标题的行内语义，不添加 heading 标记或 HTML anchor。"""
    return render_inline_content(block.content, delimiters)


def _render_equation(
    block: EquationBlock,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """优先渲染行间 LaTeX，空公式内容回退到公式图片。"""
    latex = block.content.strip()
    if latex:
        return f"{delimiters.display.left}\n{latex}\n{delimiters.display.right}"
    source = resolve_image_source(block, asset_base_url)
    return build_markdown_image(source) if source else ""


def _render_list(block: ListBlock, delimiters: LatexDelimitersConfig, depth: int = 0) -> str:
    """递归渲染列表，并给多数条目无数字前缀的参考文献补无序标记。"""
    add_ref_bullets = _should_add_reference_list_bullets(block)
    lines: list[str] = []
    for child in block.content:
        if isinstance(child, ListBlock):
            nested = _render_list(child, delimiters, depth + 1)
            if nested:
                lines.extend(nested.splitlines())
            continue
        item = render_inline_content(child.content, delimiters)
        if not item:
            continue
        item = escape_standalone_marker_rule(item)
        indent = "    " * depth
        item_lines = item.splitlines() or [item]
        if add_ref_bullets and not _UNORDERED_LIST_ITEM_RE.match(item):
            lines.append(f"{indent}- {item_lines[0]}")
            lines.extend(f"{indent}  {line}" for line in item_lines[1:])
        else:
            lines.extend(f"{indent}{line}" for line in item_lines)
    return "\n".join(lines)


def _should_add_reference_list_bullets(block: ListBlock) -> bool:
    """统计直属可见条目的数字前缀，未达到严格多数时给参考文献补项目符号。"""
    if block.sub_type != BlockType.REF_TEXT:
        return False

    item_count = 0
    numbered_count = 0
    for child in block.content:
        if isinstance(child, ListBlock):
            continue
        visible_text = inline_plain_text(parse_inline_content(child.content)).lstrip()
        if not visible_text:
            continue
        item_count += 1
        if _REF_ITEM_NUMBER_PREFIX_RE.match(visible_text):
            numbered_count += 1
    return item_count > 0 and numbered_count * 2 <= item_count


def _render_index(block: IndexBlock, delimiters: LatexDelimitersConfig, depth: int = 0) -> str:
    """递归渲染目录列表，并给标题叶子添加内部 anchor 链接。"""
    lines: list[str] = []
    for child in block.content:
        if isinstance(child, IndexBlock):
            nested = _render_index(child, delimiters, depth + 1)
            if nested:
                lines.extend(nested.splitlines())
            continue
        content = _strip_index_page_tail(child.content)
        label = render_inline_content(content, delimiters).strip()
        if not label:
            continue
        if isinstance(child, (DocTitleBlock, ParagraphTitleBlock)) and child.anchor:
            label = render_internal_link(label, child.anchor)
        lines.append(f"{'    ' * depth}- {label}")
    return "\n".join(lines)


def _strip_index_page_tail(content: str) -> str:
    """删除目录末尾可信页码，并把其余 tab 转换为普通空格。"""
    if "\t" not in content:
        return content
    head, tail = content.rsplit("\t", 1)
    tail_text = inline_plain_text(parse_inline_content(tail)).strip()
    if _looks_like_index_page_token(tail_text):
        content = head
    return content.replace("\t", " ")


def _looks_like_index_page_token(content: str) -> bool:
    """判断目录 tab 后缀是否为数字、罗马数字或单字母页码。"""
    if not content or len(content) > 12:
        return False
    return bool(content.isdigit() or _INDEX_ROMAN_RE.fullmatch(content) or re.fullmatch(r"[A-Za-z]", content))


def _render_image_block(
    block: ImageBlock,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """按原始子块顺序渲染图片主体及说明文本。"""
    parts: list[str] = []
    for child in block.content:
        if isinstance(child, ImageBodyBlock):
            body = _render_visual_body_child(
                block,
                child,
                delimiters=delimiters,
                asset_base_url=asset_base_url,
            )
        elif isinstance(child, ImageAnnotationBlock):
            body = _render_visual_annotation(child, delimiters)
        else:
            raise TypeError(f"Unsupported image child: {type(child).__name__}")
        if body:
            parts.append(body)
    return _join_visual_parts(parts)


def _render_chart_block(
    block: ChartBlock,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """按原始子块顺序渲染图表图片、结构内容和说明文本。"""
    parts: list[str] = []
    for child in block.content:
        if isinstance(child, ChartBodyBlock):
            body = _render_visual_body_child(
                block,
                child,
                delimiters=delimiters,
                asset_base_url=asset_base_url,
            )
        elif isinstance(child, ChartAnnotationBlock):
            body = _render_visual_annotation(child, delimiters)
        else:
            raise TypeError(f"Unsupported chart child: {type(child).__name__}")
        if body:
            parts.append(body)
    return _join_visual_parts(parts)


def _render_media_body(
    block: ImageBodyBlock,
    *,
    summary: str,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """渲染图片载荷，并将识别内容放入折叠详情。"""
    source = resolve_image_source(block, asset_base_url)
    rendered_content = _render_media_content(block, delimiters, asset_base_url)
    if source:
        parts = [build_markdown_image(source)]
        if rendered_content:
            parts.append(_render_details(rendered_content, summary))
        return "\n\n".join(part for part in parts if part)
    return rendered_content


def _render_media_content(
    block: ImageBodyBlock,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """渲染图片识别内容，不添加图片语法或折叠详情包装。"""
    content = block.content.strip()
    if not content:
        return ""
    return format_embedded_html(content, asset_base_url=asset_base_url, delimiters=delimiters)


def _render_chart_body(
    block: ChartBodyBlock,
    *,
    summary: str,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """渲染 chart 图片，并统一转换其 HTML 表格内容。"""
    source = resolve_image_source(block, asset_base_url)
    rendered_content = _render_chart_content(block.content, delimiters, asset_base_url)
    if source:
        parts = [build_markdown_image(source)]
        if rendered_content:
            parts.append(_render_details(rendered_content, summary))
        return "\n\n".join(part for part in parts if part)
    return rendered_content


def _render_chart_content(
    content: str,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """把 chart 内容中的简单 HTML 表格转为 GFM，其他内容保持原表示。"""
    normalized = content.strip()
    if not normalized:
        return ""
    html_table = render_html_table(
        normalized,
        asset_base_url=asset_base_url,
        delimiters=delimiters,
    )
    if html_table is not None:
        return html_table
    return format_embedded_html(normalized, asset_base_url=asset_base_url, delimiters=delimiters)


def _render_details(content: str, summary: str) -> str:
    """构造保留已渲染视觉内容的折叠 HTML 详情块。"""
    safe_summary = html.escape(summary, quote=False)
    return f"<details>\n<summary>{safe_summary}</summary>\n\n{content.strip()}\n</details>"


def _render_table_block(
    block: TableBlock,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """按原始子块顺序渲染表格主体及说明文本。"""
    parts: list[str] = []
    for child in block.content:
        if isinstance(child, TableBodyBlock):
            body = _render_visual_body_child(
                block,
                child,
                delimiters=delimiters,
                asset_base_url=asset_base_url,
            )
        elif isinstance(child, TableAnnotationBlock):
            body = _render_visual_annotation(child, delimiters)
        else:
            raise TypeError(f"Unsupported table child: {type(child).__name__}")
        if body:
            parts.append(body)
    return _join_visual_parts(parts)


def _render_table_body(
    block: TableBodyBlock,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """按 HTML、空间投影文本、图片的优先级渲染表格主体。"""
    rendered_content = _render_table_content(block, delimiters, asset_base_url)
    if rendered_content:
        return rendered_content
    source = resolve_image_source(block, asset_base_url)
    return build_markdown_image(source) if source else ""


def _render_table_content(
    block: TableBodyBlock,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """渲染表格结构内容，不执行空内容时的图片回退。"""
    if not block.content:
        return ""
    html_table = render_html_table(
        block.content,
        asset_base_url=asset_base_url,
        delimiters=delimiters,
    )
    if html_table is not None:
        return html_table
    return _render_fenced_content(block.content)


def _render_code_block(block: CodeBlock, delimiters: LatexDelimitersConfig) -> str:
    """按父块 subtype 渲染普通代码或支持公式的算法。"""
    parts: list[str] = []
    for child in block.content:
        if isinstance(child, CodeBodyBlock):
            body = _render_visual_body_child(
                block,
                child,
                delimiters=delimiters,
                asset_base_url="",
            )
        elif isinstance(child, CodeAnnotationBlock):
            body = _render_visual_annotation(child, delimiters)
        else:
            raise TypeError(f"Unsupported code child: {type(child).__name__}")
        if body:
            parts.append(body)
    return _join_visual_parts(parts)


def _render_visual_body_content(
    block: ImageBlock | TableBlock | ChartBlock | CodeBlock,
    *,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """查找视觉父块唯一 body，并只渲染可结构化消费的语义内容。"""
    for child in block.content:
        if isinstance(block, ImageBlock) and isinstance(child, ImageBodyBlock):
            return _render_media_content(child, delimiters, asset_base_url)
        if isinstance(block, TableBlock) and isinstance(child, TableBodyBlock):
            return _render_table_content(child, delimiters, asset_base_url)
        if isinstance(block, ChartBlock) and isinstance(child, ChartBodyBlock):
            return _render_chart_content(child.content, delimiters, asset_base_url)
        if isinstance(block, CodeBlock) and isinstance(child, CodeBodyBlock):
            return _render_code_body(block, child, delimiters)
    raise ValueError(f"Missing visual body: {block.type}")


def _render_visual_body_child(
    block: ImageBlock | TableBlock | ChartBlock | CodeBlock,
    child: ImageBodyBlock | TableBodyBlock | ChartBodyBlock | CodeBodyBlock,
    *,
    delimiters: LatexDelimitersConfig,
    asset_base_url: str,
) -> str:
    """按视觉父子类型组合渲染一个 body 子块。"""
    if isinstance(block, ImageBlock) and isinstance(child, ImageBodyBlock):
        return _render_media_body(
            child,
            summary=block.sub_type or "image content",
            delimiters=delimiters,
            asset_base_url=asset_base_url,
        )
    if isinstance(block, TableBlock) and isinstance(child, TableBodyBlock):
        return _render_table_body(child, delimiters, asset_base_url)
    if isinstance(block, ChartBlock) and isinstance(child, ChartBodyBlock):
        return _render_chart_body(
            child,
            summary=block.sub_type or "chart content",
            delimiters=delimiters,
            asset_base_url=asset_base_url,
        )
    if isinstance(block, CodeBlock) and isinstance(child, CodeBodyBlock):
        return _render_code_body(block, child, delimiters)
    raise TypeError(f"Unsupported visual body pair: {block.type}/{child.type}")


def _render_visual_annotation(
    block: ImageAnnotationBlock | TableAnnotationBlock | ChartAnnotationBlock | CodeAnnotationBlock,
    delimiters: LatexDelimitersConfig,
) -> str:
    """把一个视觉说明子块渲染为独立 Markdown 字符串。"""
    return escape_standalone_marker_rule(render_inline_content(block.content, delimiters))


def _render_code_body(
    block: CodeBlock,
    child: CodeBodyBlock,
    delimiters: LatexDelimitersConfig,
) -> str:
    """依据父块 subtype 渲染代码或算法 body。"""
    if block.sub_type == BlockType.CODE:
        return _render_fenced_content(child.content, _normalize_code_language(block.guess_lang))
    if block.sub_type == RAW_ALGORITHM:
        return _render_algorithm_html(child.content, delimiters)
    raise ValueError(f"Unsupported code subtype: {block.sub_type}")


def _normalize_code_language(language: str | None) -> str:
    """校验 fenced code info string，非法值统一回退 txt。"""
    normalized = (language or "").strip()
    if not normalized or _VALID_CODE_LANGUAGE_RE.fullmatch(normalized) is None:
        return "txt"
    return normalized


def _render_fenced_content(content: str, language: str | None = None) -> str:
    """使用长于正文反引号游程的围栏包裹原始内容。"""
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    fence = "`" * max(3, longest + 1)
    opening = f"{fence}{language or ''}"
    closing_prefix = "" if content.endswith("\n") else "\n"
    return f"{opening}\n{content}{closing_prefix}{fence}"


def _render_algorithm_html(content: str, delimiters: LatexDelimitersConfig) -> str:
    """参考 dev 实现渲染保留空白、上下标和行内公式的算法 HTML。"""
    if not content.strip():
        return ""
    parts: list[str] = []
    cursor = 0
    previous_was_equation = False
    for match in _ALGORITHM_TOKEN_RE.finditer(content):
        plain = content[cursor : match.start()]
        if plain:
            parts.append(plain)
            previous_was_equation = False
        script_tag = match.group("script_tag")
        if script_tag is not None:
            parts.append(script_tag.lower())
            previous_was_equation = False
            cursor = match.end()
            continue
        latex = html.unescape(match.group("latex")).strip()
        if latex:
            if previous_was_equation and parts and not parts[-1].endswith((" ", "\n", "\t")):
                parts.append(" ")
            parts.append(f"{delimiters.inline.left}{latex}{delimiters.inline.right}")
            previous_was_equation = True
        cursor = match.end()
    parts.append(content[cursor:])
    body = "".join(parts)
    return f'<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">\n{body}\n</div>'


def _join_visual_parts(parts: list[str]) -> str:
    """使用安全空行连接同一视觉父块中的有序子块。"""
    return "\n\n".join(part.strip("\n") for part in parts if part and part.strip())


__all__ = ["MarkdownRenderMode", "render_markdown"]
