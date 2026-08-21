# Copyright (c) Opendatalab. All rights reserved.
"""使用后置外部 LLM 优化严格 MiddleJson 的标题层级。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mineru.types import DocTitleBlock, PageInfo, ParagraphTitleBlock
from mineru.render._internal.common.inline import inline_plain_text, parse_inline_content

from .llm_client import LLMAidedClient

ParagraphTitleGroup = list[ParagraphTitleBlock]


def _collect_paragraph_title_groups(pages: list[PageInfo]) -> list[ParagraphTitleGroup]:
    """按文档标题边界把段落标题划分为保持阅读顺序的非空分组。"""
    groups: list[ParagraphTitleGroup] = []
    current_group: ParagraphTitleGroup = []
    for page in pages:
        for block in page.blocks:
            if isinstance(block, DocTitleBlock):
                block.level = 1
                if current_group:
                    groups.append(current_group)
                    current_group = []
            elif isinstance(block, ParagraphTitleBlock):
                current_group.append(block)
    if current_group:
        groups.append(current_group)
    return groups


def _plain_title_text(block: ParagraphTitleBlock) -> str:
    """提取标题可见文本供 LLM 使用，不改写原始行内样式内容。"""
    return inline_plain_text(parse_inline_content(block.content))


def _build_title_prompt(group: ParagraphTitleGroup) -> str:
    """只使用组内序号和段落标题纯文本构造分级提示。"""
    titles = {str(index): _plain_title_text(block) for index, block in enumerate(group)}
    return f"""请优化下列同一文档分组内章节和段落标题的层级，标题层级使用 2 到 6 级。
必须保留所有输入项，并返回键完全相同的 JSON 对象；值只能是 1 到 6 的整数，返回 1 时系统会归一为 2。
不要返回 Markdown、代码块或解释文字。

输入标题：
{json.dumps(titles, ensure_ascii=False)}

输出格式示例：
{{"0": 2, "1": 4, "2": 6}}
"""


def _validate_title_levels(value: Any, expected_count: int) -> dict[int, int] | None:
    """校验标题层级响应必须覆盖全部索引且值位于一至六级。"""
    if not isinstance(value, dict):
        return None

    levels: dict[int, int] = {}
    try:
        for raw_key, raw_level in value.items():
            if isinstance(raw_key, bool) or isinstance(raw_level, bool):
                return None
            key = int(raw_key)
            level = int(raw_level)
            if str(key) != str(raw_key) or type(raw_level) is not int or not 1 <= level <= 6:
                return None
            levels[key] = level
    except (TypeError, ValueError):
        return None

    if set(levels) != set(range(expected_count)):
        return None
    return levels


async def apply_llm_title_leveling(pages: list[PageInfo], client: LLMAidedClient) -> None:
    """并发优化各文档标题分组，单组失败时保留该组原始层级。"""
    groups = _collect_paragraph_title_groups(pages)
    if not groups:
        return

    group_levels = await asyncio.gather(
        *(
            client.request_validated_json(
                operation="title_leveling",
                prompt=_build_title_prompt(group),
                validator=lambda value, count=len(group): _validate_title_levels(value, count),
                temperature=0.7,
            )
            for group in groups
        )
    )
    for group, levels in zip(groups, group_levels):
        if levels is None:
            continue
        for index, block in enumerate(group):
            block.level = max(2, levels[index])


__all__ = ["apply_llm_title_leveling"]
