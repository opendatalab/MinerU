from __future__ import annotations

from mineru.backend.utils.table_merge import _apply_cell_merge, _build_table_state
from mineru.types import Block, BlockType, Line, Span


def _build_table_block(index: int, html: str, cell_merge: list[int]) -> Block:
    """构造 cell_merge 跨页合并测试使用的两层表格块。"""
    bbox = (0.0, 0.0, 100.0, 100.0)
    body_block = Block(
        index=index,
        type=BlockType.TABLE_BODY,
        bbox=bbox,
        lines=[
            Line(
                bbox=bbox,
                spans=[Span(type=BlockType.TABLE, bbox=bbox, content=html)],
            )
        ],
    )
    body_block._cell_merge = cell_merge
    return Block(
        index=index,
        type=BlockType.TABLE,
        bbox=bbox,
        blocks=[body_block],
    )


def test_apply_cell_merge_reads_metadata_from_table_body() -> None:
    """验证顶层无 cell_merge 时仍按 table body 元数据续接单元格。"""
    previous_table = _build_table_block(
        0,
        "<table><tr><td>A</td><td>X</td></tr></table>",
        [],
    )
    current_table = _build_table_block(
        1,
        "<table><tr><td>B</td><td>Y</td></tr></table>",
        [1, 0],
    )
    previous_state = _build_table_state(previous_table)
    current_state = _build_table_state(current_table)
    assert previous_state is not None
    assert current_state is not None
    assert current_table._cell_merge == []

    _apply_cell_merge(previous_state, current_state, header_count=0)

    previous_cells = previous_state.rows[-1].find_all(["td", "th"])
    current_cells = current_state.rows[0].find_all(["td", "th"])
    assert [cell.get_text() for cell in previous_cells] == ["AB", "X"]
    assert [cell.get_text() for cell in current_cells] == ["", "Y"]


def test_apply_cell_merge_skips_missing_or_empty_body_metadata() -> None:
    """验证 table body 缺失或 cell_merge 为空时安全跳过。"""
    for remove_body, cell_merge in [(False, []), (True, [1, 0])]:
        previous_table = _build_table_block(
            0,
            "<table><tr><td>A</td><td>X</td></tr></table>",
            [],
        )
        current_table = _build_table_block(
            1,
            "<table><tr><td>B</td><td>Y</td></tr></table>",
            cell_merge,
        )
        previous_state = _build_table_state(previous_table)
        current_state = _build_table_state(current_table)
        assert previous_state is not None
        assert current_state is not None
        if remove_body:
            current_table.blocks = []

        _apply_cell_merge(previous_state, current_state, header_count=0)

        previous_cells = previous_state.rows[-1].find_all(["td", "th"])
        current_cells = current_state.rows[0].find_all(["td", "th"])
        assert [cell.get_text() for cell in previous_cells] == ["A", "X"]
        assert [cell.get_text() for cell in current_cells] == ["B", "Y"]
