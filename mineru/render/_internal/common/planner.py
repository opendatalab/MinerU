# Copyright (c) Opendatalab. All rights reserved.
"""各格式共用的逻辑块复制、延续合并与页面规划。"""

from __future__ import annotations

from dataclasses import dataclass, field

from mineru.backend.postprocess.table_merge import merge_table_content
from mineru.render.contracts import RenderMode
from mineru.types import (
    PAGE_AUXILIARY_BLOCK_TYPES,
    BlockType,
    ContinuableTextBlockBase,
    ListBlock,
    MiddleJson,
    PageBlock,
    RefTextBlock,
    TableBlock,
    TextBlock,
)


@dataclass(slots=True)
class PlannedBlock:
    """保存一个待渲染块及其来源页和文本延续片段。"""

    page_idx: int
    block: PageBlock
    text_contents: list[str] = field(default_factory=list)
    removed: bool = False


def build_render_plan(middle_json: MiddleJson, mode: RenderMode) -> list[list[PlannedBlock]]:
    """深拷贝 MiddleJson，并按模式生成不污染输入的逐页逻辑块计划。"""
    copied = middle_json.model_copy(deep=True)
    pages = [
        [
            PlannedBlock(
                page_idx=page.page_idx,
                block=block,
                text_contents=[block.content] if isinstance(block, ContinuableTextBlockBase) else [],
            )
            for block in page.blocks
        ]
        for page in copied.pages
    ]
    flattened = [planned for page in pages for planned in page]
    _merge_continued_text_blocks(flattened, mode)
    _merge_continued_list_blocks(flattened, mode)
    if mode is RenderMode.DEFAULT:
        _merge_continued_table_blocks(flattened)
    return pages


def _merge_continued_text_blocks(blocks: list[PlannedBlock], mode: RenderMode) -> None:
    """把 continues_prev 文本吸收到最近的前序文本逻辑块。"""
    for current_index, current in enumerate(blocks):
        if (
            current.removed
            or not isinstance(current.block, ContinuableTextBlockBase)
            or current.block.continues_prev is not True
        ):
            continue
        previous = _find_previous_planned_text(blocks, current_index)
        if previous is None:
            continue
        is_cross_page = previous.page_idx != current.page_idx
        if is_cross_page and mode is RenderMode.FULL:
            continue
        previous.text_contents.extend(current.text_contents)
        current.removed = True


def _find_previous_planned_text(blocks: list[PlannedBlock], current_index: int) -> PlannedBlock | None:
    """按 text/ref_text 各自的透明块规则查找仍参与输出的前序同类文本。"""
    current = blocks[current_index]
    if isinstance(current.block, RefTextBlock):
        previous_index = current_index - 1
        while previous_index >= 0:
            candidate = blocks[previous_index]
            if candidate.block.type in PAGE_AUXILIARY_BLOCK_TYPES:
                previous_index -= 1
                continue
            if not isinstance(candidate.block, RefTextBlock):
                return None
            if not candidate.removed:
                return candidate
            previous_index -= 1
        return None
    if not isinstance(current.block, TextBlock):
        return None
    for candidate in reversed(blocks[:current_index]):
        if not candidate.removed and isinstance(candidate.block, TextBlock):
            return candidate
    return None


def _merge_continued_list_blocks(blocks: list[PlannedBlock], mode: RenderMode) -> None:
    """把续接列表吸收到子类型一致的前序列表，参考文献可跨过页面辅助块。"""
    for current_index, current in enumerate(blocks):
        if current.removed or not isinstance(current.block, ListBlock) or current.block.continues_prev is not True:
            continue
        previous = _find_previous_planned_list(blocks, current_index)
        if previous is None or not isinstance(previous.block, ListBlock):
            continue
        if previous.block.sub_type != current.block.sub_type:
            continue
        is_cross_page = previous.page_idx != current.page_idx
        if is_cross_page and mode is RenderMode.FULL:
            continue
        previous.block.content.extend(current.block.content)
        current.removed = True


def _find_previous_planned_list(
    blocks: list[PlannedBlock],
    current_index: int,
) -> PlannedBlock | None:
    """查找前序有效列表，仅允许参考文献跳过页面辅助块。"""
    current = blocks[current_index]
    if not isinstance(current.block, ListBlock):
        return None
    can_skip_page_auxiliary = current.block.sub_type == BlockType.REF_TEXT
    previous_index = current_index - 1
    while previous_index >= 0:
        candidate = blocks[previous_index]
        if can_skip_page_auxiliary and candidate.block.type in PAGE_AUXILIARY_BLOCK_TYPES:
            previous_index -= 1
            continue
        if not isinstance(candidate.block, ListBlock):
            return None
        if not candidate.removed:
            return candidate
        previous_index -= 1
    return None


def _merge_continued_table_blocks(blocks: list[PlannedBlock]) -> None:
    """在默认模式中把跨页续表合并到最近的前序表格。"""
    for current_index, current in enumerate(blocks):
        if current.removed or not isinstance(current.block, TableBlock) or current.block.continues_prev is not True:
            continue
        previous = _find_previous_planned_table(blocks, current_index)
        if previous is None or previous.page_idx == current.page_idx:
            continue
        merged = merge_table_content(
            previous.block.model_dump(mode="python", exclude_none=True),
            current.block.model_dump(mode="python", exclude_none=True),
        )
        if merged is None:
            continue
        try:
            previous.block = TableBlock.model_validate(merged)
        except (TypeError, ValueError):
            continue
        current.removed = True


def _find_previous_planned_table(blocks: list[PlannedBlock], current_index: int) -> PlannedBlock | None:
    """查找最近且仍参与输出的前序表格。"""
    for candidate in reversed(blocks[:current_index]):
        if not candidate.removed and isinstance(candidate.block, TableBlock):
            return candidate
    return None


__all__ = ["PlannedBlock", "RenderMode", "build_render_plan"]
