# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from copy import deepcopy
from html import escape, unescape
from typing import Any

from ..types import Block, BlockType, ContentType
from .merge import inline_left, inline_right, merge_para_text

# --------------------------------------------------------------------------- #
#  Merge IMAGE / TABLE / CHART / CODE composite blocks into a single
#  Markdown string.  Shared by Pipeline and VLM backends.
# --------------------------------------------------------------------------- #


def merge_visual_para_text(para_block: Block, img_bucket_path: str = "") -> str:
    rendered_segments: list[tuple[str, str]] = []

    for block in _blocks_in_index_order(para_block.blocks):
        render_block = _inherit_parent_code_render_metadata(block, para_block)
        rendered_segments.extend(_render_visual_block_segments(render_block, para_block, img_bucket_path))

    para_text = ""
    prev_kind: str | None = None
    for segment_text, segment_kind in rendered_segments:
        if para_text:
            para_text += _visual_block_separator(prev_kind, segment_kind)
        para_text += segment_text
        prev_kind = segment_kind

    return para_text


# --------------------------------------------------------------------------- #
#  Internal helpers — sorted by dependency order
# --------------------------------------------------------------------------- #

import re  # noqa: E402


def _blocks_in_index_order(blocks: list[Block]) -> list[Block]:
    return [b for _, b in sorted(enumerate(blocks), key=lambda item: (item[1].index, item[0]))]


def _inherit_parent_code_render_metadata(block: Block, parent_block: Block) -> Block:
    # 本地 Hybrid 模型会把 code_body 的 sub_type/guess_lang 提升到父 code block。
    # markdown 渲染 code_body 时需要把这两个字段临时透传回来，但不能修改原始输入。
    if block.type != BlockType.CODE_BODY:
        return block
    if parent_block.type != BlockType.CODE:
        return block

    needs_sub_type = not block.sub_type and parent_block.sub_type
    needs_guess_lang = not block.guess_lang and parent_block.guess_lang
    if not needs_sub_type and not needs_guess_lang:
        return block

    render_block = deepcopy(block)
    if needs_sub_type:
        render_block.sub_type = parent_block.sub_type
    if needs_guess_lang:
        render_block.guess_lang = parent_block.guess_lang
    return render_block


# -- HTML helpers ------------------------------------------------------------


def _prefix_table_img_src(html: str, img_bucket_path: str) -> str:
    if not html or not img_bucket_path:
        return html
    return re.sub(
        r'src="(?!data:)([^"]+)"',
        lambda m: f'src="{img_bucket_path}/{m.group(1)}"',
        html,
    )


def _replace_eq_tags_in_table_html(html: str) -> str:
    if not html:
        return html
    return re.sub(
        r"<eq>(.*?)</eq>",
        lambda m: f" {inline_left}{unescape(m.group(1))}{inline_right} ",
        html,
        flags=re.DOTALL,
    )


def _format_embedded_html(html: str, img_bucket_path: str) -> str:
    return _replace_eq_tags_in_table_html(_prefix_table_img_src(html, img_bucket_path))


def _build_media_path(img_bucket_path: str, image_path: str) -> str:
    if not image_path:
        return ""
    if not img_bucket_path:
        return image_path
    return f"{img_bucket_path}/{image_path}"


# -- Visual body / details ---------------------------------------------------


def _build_visual_details_block(content: Any, summary: str) -> str:
    normalized = _normalize_visual_content(content)
    if not normalized:
        return ""
    return f"<details>\n<summary>{summary}</summary>\n\n{normalized}\n</details>"


def _normalize_visual_content(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(item) for item in content if str(item).strip())
    if isinstance(content, str):
        return content.strip()
    return ""


def _build_visual_body_segments(
    *, image_path: str, content: str, img_bucket_path: str, details_summary: str
) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    media_path = _build_media_path(img_bucket_path, image_path)
    if media_path:
        segments.append((f"![]({media_path})", "markdown_line"))
    details = _build_visual_details_block(content, details_summary)
    if details:
        segments.append((details, "details_block"))
    return segments


def _render_algorithm_code_html(block: Block) -> str:
    """将 algorithm code body 渲染为 HTML pre，保留缩进并支持行内公式。"""
    rendered_lines: list[str] = []
    for line in block.lines:
        line_parts: list[str] = []
        for span in line.spans:
            if span.type == ContentType.TEXT:
                line_parts.append(escape(span.content or ""))
            elif span.type == ContentType.INLINE_EQUATION and span.content:
                line_parts.append(f"{inline_left}{escape(span.content)}{inline_right}")
        rendered_lines.append("".join(line_parts).rstrip())

    content = "\n".join(rendered_lines).strip("\n")
    if not content.strip():
        return ""
    return f'<pre class="mineru-algorithm-code"><code style="white-space: pre-wrap;">{content}</code></pre>'


# -- Per-block-type segment rendering ----------------------------------------


def _render_visual_block_segments(block: Block, para_block: Block, img_bucket_path: str) -> list[tuple[str, str]]:
    # 将单个视觉子 block 渲染成一个或多个 segment。
    # 文本类子块统一输出 markdown_line；
    # table 的 html 输出为 html_block，供后续决定是否需要空行隔开。

    block_type = block.type

    if block_type == BlockType.CODE_BODY and block.sub_type == BlockType.ALGORITHM:
        text = _render_algorithm_code_html(block)
        if text:
            return [(text, "html_block")]
        return []

    # text-like sub-blocks  (caption / footnote / code body)
    if block_type in (
        BlockType.IMAGE_CAPTION,
        BlockType.IMAGE_FOOTNOTE,
        BlockType.TABLE_CAPTION,
        BlockType.TABLE_FOOTNOTE,
        BlockType.CODE_BODY,
        BlockType.CODE_CAPTION,
        BlockType.CODE_FOOTNOTE,
        BlockType.CHART_CAPTION,
        BlockType.CHART_FOOTNOTE,
    ):
        text = merge_para_text(block)
        if text.strip():
            return [(text, "markdown_line")]
        return []

    # IMAGE_BODY
    if block_type == BlockType.IMAGE_BODY:
        image_segments: list[tuple[str, str]] = []
        for line in block.lines:
            for span in line.spans:
                if span.type != ContentType.IMAGE:
                    continue
                image_segments.extend(
                    _build_visual_body_segments(
                        image_path=span.image_path,
                        content=span.content,
                        img_bucket_path=img_bucket_path,
                        details_summary=para_block.sub_type or "image content",
                    )
                )
        return image_segments

    # CHART_BODY
    if block_type == BlockType.CHART_BODY:
        chart_segments: list[tuple[str, str]] = []
        for line in block.lines:
            for span in line.spans:
                if span.type != ContentType.CHART:
                    continue
                chart_segments.extend(
                    _build_visual_body_segments(
                        image_path=span.image_path,
                        content=span.content,
                        img_bucket_path=img_bucket_path,
                        details_summary=para_block.sub_type or "chart content",
                    )
                )
        return chart_segments

    # TABLE_BODY
    if block_type == BlockType.TABLE_BODY:
        table_segments: list[tuple[str, str]] = []
        for line in block.lines:
            for span in line.spans:
                if span.type != ContentType.TABLE:
                    continue
                if span.content:  # (VLM) also checks table_enable
                    table_segments.append((_format_embedded_html(span.content, img_bucket_path), "html_block"))
                elif span.image_path:
                    if media_path := _build_media_path(img_bucket_path, span.image_path):
                        table_segments.append((f"![]({media_path})", "markdown_line"))
        return table_segments

    return []


# -- Segment separator -------------------------------------------------------


def _visual_block_separator(prev_kind: str | None, current_kind: str) -> str:
    # 根据前后 segment 类型决定分隔符：
    # 1. 普通 markdown 行之间用 hard break（"  \\n"）
    # 2. 进入 html block 前只换一行
    # 3. html block 后必须留空行，否则后续文本仍会被当作 html 块内容
    if prev_kind == "html_block":
        # Raw HTML blocks need a blank line after them, otherwise the following
        # markdown text is still treated as part of the HTML block.
        return "\n\n"
    if prev_kind == "details_block" or current_kind == "details_block":
        return "\n\n"
    if current_kind == "html_block":
        return "\n"
    return "  \n"
