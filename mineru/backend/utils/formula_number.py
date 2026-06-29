# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Any

from ...types import Block, BlockType, ContentType, PageInfo, Span
from .char_utils import full_to_half


def extract_text_from_block(block: Block) -> str:
    """从 typed Block 中提取纯文本，用于公式编号归一化。"""
    text_parts = []
    for line in block.lines:
        for span in line.spans:
            if span.type == ContentType.TEXT:
                text_parts.append(span.content)
    return "".join(text_parts).strip()


def normalize_formula_tag_content(tag_content: str) -> str:
    """归一化公式编号文本，去掉全角字符和包裹括号后用于 \\tag{}。"""
    tag_content = full_to_half(str(tag_content or "").strip())
    if tag_content.startswith("("):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith(")"):
        tag_content = tag_content[:-1].strip()
    return tag_content


def get_interline_equation_span(block: Block) -> Span | None:
    """查找 typed 公式块中的行间公式 span。"""
    for line in block.lines:
        for span in line.spans:
            if span.type == ContentType.INTERLINE_EQUATION:
                return span
    return None


def append_formula_number_tag(equation_block: Block, formula_number_block: Block) -> None:
    """把 typed 公式编号合并进相邻行间公式的 LaTeX 内容。"""
    equation_span = get_interline_equation_span(equation_block)
    tag_content = normalize_formula_tag_content(extract_text_from_block(formula_number_block))
    if equation_span is not None and tag_content:
        equation_span.content = f"{equation_span.content}\\tag{{{tag_content}}}"


def optimize_pipeline_formula_number_blocks(pages: list[PageInfo]) -> None:
    """合并 Pipeline typed middle-json 中的公式编号块，无法合并时降级为文本块。"""
    for page_info in pages:
        optimized_blocks = []
        blocks = page_info.preproc_blocks
        for index, block in enumerate(blocks):
            if block.type != BlockType.FORMULA_NUMBER:
                optimized_blocks.append(block)
                continue

            prev_block = blocks[index - 1] if index > 0 else None
            if prev_block and prev_block.type == BlockType.INTERLINE_EQUATION:
                append_formula_number_tag(prev_block, block)
                continue

            next_block = blocks[index + 1] if index + 1 < len(blocks) else None
            next_next_block = blocks[index + 2] if index + 2 < len(blocks) else None
            if (
                next_block
                and next_block.type == BlockType.INTERLINE_EQUATION
                and (next_next_block is None or next_next_block.type != BlockType.FORMULA_NUMBER)
            ):
                append_formula_number_tag(next_block, block)
                continue

            block.type = BlockType.TEXT
            optimized_blocks.append(block)
        page_info.preproc_blocks = optimized_blocks


def _is_hybrid_equation_block(block: dict[str, Any]) -> bool:
    """判断 raw Hybrid/VLM block 是否表示可合并编号的行间公式。"""
    return str(block.get("type") or "").lower() in {"equation", "display_formula", BlockType.INTERLINE_EQUATION}


def _is_hybrid_formula_number_block(block: dict[str, Any]) -> bool:
    """判断 raw Hybrid/VLM block 是否表示公式编号。"""
    return str(block.get("type") or "").lower() == BlockType.FORMULA_NUMBER


def _append_hybrid_formula_number_tag(equation_block: dict[str, Any], number_block: dict[str, Any]) -> None:
    """把 raw Hybrid/VLM 公式编号合并到相邻公式 block 的 content/latex 字段。"""
    tag_content = normalize_formula_tag_content(str(number_block.get("content") or number_block.get("text") or ""))
    if not tag_content:
        return
    target_key = "latex" if equation_block.get("latex") else "content"
    formula = str(equation_block.get(target_key) or "")
    equation_block[target_key] = f"{formula}\\tag{{{tag_content}}}"


def optimize_hybrid_formula_number_blocks(page_model_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并 Hybrid raw model list 中的 formula_number，并返回新的 block 列表。"""
    optimized_blocks: list[dict[str, Any]] = []
    blocks = list(page_model_list)
    for index, block in enumerate(blocks):
        if not _is_hybrid_formula_number_block(block):
            optimized_blocks.append(block)
            continue

        prev_block = blocks[index - 1] if index > 0 else None
        if prev_block and _is_hybrid_equation_block(prev_block):
            _append_hybrid_formula_number_tag(prev_block, block)
            continue

        next_block = blocks[index + 1] if index + 1 < len(blocks) else None
        next_next_block = blocks[index + 2] if index + 2 < len(blocks) else None
        if (
            next_block
            and _is_hybrid_equation_block(next_block)
            and (next_next_block is None or not _is_hybrid_formula_number_block(next_next_block))
        ):
            _append_hybrid_formula_number_tag(next_block, block)
            continue

        fallback_block = dict(block)
        fallback_block["type"] = BlockType.TEXT
        optimized_blocks.append(fallback_block)
    return optimized_blocks
