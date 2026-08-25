# Copyright (c) Opendatalab. All rights reserved.
"""使用后置外部 LLM 分析跨页表格边界单元格续接关系。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, cast

from ....types import PageInfo, TableBlock, TableBodyBlock

from .html import build_row_rendered_cell_segments, build_table_state_from_html
from .structure import _expand_header_count_by_rowspan, detect_table_headers
from ..llm_client import LLMAidedClient

CellMergeFlag = Literal[0, 1]


@dataclass(frozen=True, slots=True)
class _CellMergeCandidate:
    """保存一次后置单元格分析所需的目标块与视觉列映射。"""

    current_table: TableBlock
    prompt: str
    expected_segment_count: int
    expanded_col_count: int
    previous_segment_col_ranges: tuple[tuple[int, int], ...]


def _table_body_html(table: TableBlock) -> str | None:
    """读取严格 table 根块下唯一主体的 HTML 内容。"""
    for child in table.content:
        if isinstance(child, TableBodyBlock):
            return child.content or None
    return None


def _find_continued_table_pairs(pages: list[PageInfo]) -> list[tuple[TableBlock, TableBlock]]:
    """按相邻页面查找已由确定性规则标记的跨页续表对。"""
    pairs: list[tuple[TableBlock, TableBlock]] = []
    for previous_page, current_page in zip(pages, pages[1:]):
        if current_page.page_idx != previous_page.page_idx + 1:
            continue
        current_table = next(
            (
                block
                for block in current_page.blocks
                if isinstance(block, TableBlock) and block.continues_prev is True
            ),
            None,
        )
        if current_table is None:
            continue
        previous_table = next(
            (block for block in reversed(previous_page.blocks) if isinstance(block, TableBlock)),
            None,
        )
        if previous_table is not None:
            pairs.append((previous_table, current_table))
    return pairs


def _build_cell_merge_prompt(previous_texts: list[str], current_texts: list[str]) -> str:
    """构造逐渲染单元格返回零一状态的跨页续接提示。"""
    return f"""请判断跨页表格边界两行中的对应单元格是否属于同一个被分页截断的逻辑单元格。
对每一组对应单元格独立判断：1 表示需要把当前页单元格续接到上一页单元格，0 表示不续接。
必须返回与输入单元格数量相同的 JSON 整数数组；数组元素只能是 0 或 1。
全 0、全 1 和混合结果都允许。不要返回 Markdown、代码块或解释文字。

上一页表格最后一个数据行：
{json.dumps(previous_texts, ensure_ascii=False)}

当前页表格第一个数据行：
{json.dumps(current_texts, ensure_ascii=False)}

输出示例：
[0, 1, 0]
"""


def _prepare_cell_merge_candidate(
    previous_table: TableBlock,
    current_table: TableBlock,
) -> _CellMergeCandidate | None:
    """从跨页表格 HTML 中提取边界数据行及其渲染单元格段。"""
    previous_html = _table_body_html(previous_table)
    current_html = _table_body_html(current_table)
    if previous_html is None or current_html is None:
        return None

    previous_state = build_table_state_from_html(previous_html)
    current_state = build_table_state_from_html(current_html)
    if previous_state is None or current_state is None:
        return None

    header_count, _, _ = detect_table_headers(previous_state, current_state)
    header_count = _expand_header_count_by_rowspan(current_state.rows, header_count)
    previous_metrics = previous_state.last_data_row_metrics
    current_metrics = current_state.front_first_data_row_metrics.get(header_count)
    if previous_metrics is None or current_metrics is None:
        return None

    previous_segments = build_row_rendered_cell_segments(previous_state.rows, previous_metrics.row_idx)
    current_segments = build_row_rendered_cell_segments(current_state.rows, current_metrics.row_idx)
    if not previous_segments or len(previous_segments) != len(current_segments):
        return None

    expanded_col_count = max((segment.end_col for segment in previous_segments), default=0)
    if expanded_col_count <= 0:
        return None

    return _CellMergeCandidate(
        current_table=current_table,
        prompt=_build_cell_merge_prompt(
            [segment.text for segment in previous_segments],
            [segment.text for segment in current_segments],
        ),
        expected_segment_count=len(previous_segments),
        expanded_col_count=expanded_col_count,
        previous_segment_col_ranges=tuple((segment.start_col, segment.end_col) for segment in previous_segments),
    )


def _validate_cell_merge(value: object, expected_count: int) -> list[CellMergeFlag] | None:
    """校验 LLM 返回列表长度及每个独立单元格的零一状态。"""
    if not isinstance(value, list) or len(value) != expected_count:
        return None
    if any(type(flag) is not int or flag not in (0, 1) for flag in value):
        return None
    return cast(list[CellMergeFlag], list(value))


def _expand_cell_merge(
    candidate: _CellMergeCandidate,
    segment_flags: list[CellMergeFlag],
) -> list[CellMergeFlag]:
    """将逐渲染单元格状态按 colspan 覆盖范围展开到视觉列。"""
    expanded: list[CellMergeFlag] = [0] * candidate.expanded_col_count
    for flag, (start_col, end_col) in zip(segment_flags, candidate.previous_segment_col_ranges):
        if flag != 1:
            continue
        for col_idx in range(start_col, min(end_col, candidate.expanded_col_count)):
            expanded[col_idx] = 1
    return expanded


async def apply_llm_cross_page_cell_merge(pages: list[PageInfo], client: LLMAidedClient) -> None:
    """并发分析已确认的跨页续表，并按候选原始顺序写入 cell_merge。"""
    candidates = [
        candidate
        for previous_table, current_table in _find_continued_table_pairs(pages)
        if (candidate := _prepare_cell_merge_candidate(previous_table, current_table)) is not None
    ]
    if not candidates:
        return

    candidate_flags = await asyncio.gather(
        *(
            client.request_validated_json(
                operation="cross_page_table_cell_merge",
                prompt=candidate.prompt,
                validator=lambda value, count=candidate.expected_segment_count: _validate_cell_merge(value, count),
                temperature=0.1,
            )
            for candidate in candidates
        )
    )
    for candidate, segment_flags in zip(candidates, candidate_flags):
        if segment_flags is None:
            continue
        candidate.current_table.cell_merge = _expand_cell_merge(candidate, segment_flags)


__all__ = ["apply_llm_cross_page_cell_merge"]
