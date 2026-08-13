from __future__ import annotations

from copy import deepcopy
from typing import Any

from bs4 import BeautifulSoup

from mineru.backend.postprocess.table_merge import merge_table_content
from mineru.types import BlockType


def _build_table_block(index: int, html: str, cell_merge: list[int] | None = None) -> dict[str, Any]:
    """构造纯内容合并测试使用的两层 dict 表格块。"""
    body: dict[str, Any] = {
        "index": index,
        "type": BlockType.TABLE_BODY,
        "bbox": [0.1, 0.1, 0.9, 0.9],
        "content": html,
    }
    if cell_merge is not None:
        body["cell_merge"] = cell_merge
    return {
        "index": index,
        "type": BlockType.TABLE,
        "bbox": [0.1, 0.1, 0.9, 0.9],
        "content": [body],
    }


def _row_texts(table: dict[str, Any]) -> list[list[str]]:
    """提取合并后 table body 的逐行单元格文本。"""
    body = table["content"][0]
    soup = BeautifulSoup(body["content"], "html.parser")
    return [[cell.get_text() for cell in row.find_all(["td", "th"])] for row in soup.find_all("tr")]


def test_merge_table_content_applies_partial_cell_merge_from_table_body() -> None:
    """验证部分视觉列续接后保留当前首行并清空已迁移单元格。"""
    previous_table = _build_table_block(
        0,
        "<table><tr><td>A</td><td>X</td></tr></table>",
    )
    current_table = _build_table_block(
        1,
        "<table><tr><td>B</td><td>Y</td></tr></table>",
        [1, 0],
    )
    previous_original = deepcopy(previous_table)
    current_original = deepcopy(current_table)

    merged = merge_table_content(previous_table, current_table)

    assert merged is not None
    assert _row_texts(merged) == [["AB", "X"], ["", "Y"]]
    assert previous_table == previous_original
    assert current_table == current_original


def test_merge_table_content_applies_full_cell_merge_and_removes_consumed_row() -> None:
    """验证全部视觉列续接后删除已消费行，并继续追加后续数据行。"""
    previous_table = _build_table_block(
        0,
        "<table><tr><td>A</td><td>X</td></tr></table>",
    )
    current_table = _build_table_block(
        1,
        "<table><tr><td>B</td><td>Y</td></tr><tr><td>C</td><td>Z</td></tr></table>",
        [1, 1],
    )

    merged = merge_table_content(previous_table, current_table)

    assert merged is not None
    assert _row_texts(merged) == [["AB", "XY"], ["C", "Z"]]


def test_merge_table_content_supports_fully_consumed_only_current_row() -> None:
    """验证当前表仅有一行且被 cell_merge 全部消费时仍返回合并结果。"""
    previous_table = _build_table_block(
        0,
        "<table><tr><td>A</td><td>X</td></tr></table>",
    )
    current_table = _build_table_block(
        1,
        "<table><tr><td>B</td><td>Y</td></tr></table>",
        [1, 1],
    )

    merged = merge_table_content(previous_table, current_table)

    assert merged is not None
    assert _row_texts(merged) == [["AB", "XY"]]
