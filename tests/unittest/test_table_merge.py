from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from bs4 import BeautifulSoup

from mineru.backend.postprocess.table_merge import merge_table, merge_table_content
from mineru.types import BlockType


DEFAULT_HTML = "<table><tr><td>A</td><td>B</td></tr></table>"


def _table_body(
    html: str = DEFAULT_HTML,
    *,
    index: int = 0,
    bbox: list[float] | None = None,
    cell_merge: list[int] | None = None,
) -> dict[str, Any]:
    """构造归一化 bbox 的 table body 测试块。"""
    body: dict[str, Any] = {
        "type": BlockType.TABLE_BODY,
        "index": index,
        "bbox": deepcopy(bbox or [0.1, 0.1, 0.9, 0.9]),
        "content": html,
    }
    if cell_merge is not None:
        body["cell_merge"] = cell_merge
    return body


def _caption(
    text: str,
    *,
    index: int = 0,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    """构造 table caption 测试块。"""
    return {
        "type": BlockType.TABLE_CAPTION,
        "index": index,
        "bbox": deepcopy(bbox or [0.1, 0.05, 0.9, 0.08]),
        "content": text,
    }


def _footnote(text: str, *, index: int = 1) -> dict[str, Any]:
    """构造 table footnote 测试块。"""
    return {
        "type": BlockType.TABLE_FOOTNOTE,
        "index": index,
        "bbox": [0.1, 0.91, 0.9, 0.95],
        "content": text,
    }


def _table(
    index: int,
    html: str = DEFAULT_HTML,
    *,
    bbox: list[float] | None = None,
    children: list[dict[str, Any]] | None = None,
    cell_merge: list[int] | None = None,
) -> dict[str, Any]:
    """构造两层 dict table 测试块。"""
    table_bbox = deepcopy(bbox or [0.1, 0.1, 0.9, 0.9])
    body = _table_body(
        html,
        index=index,
        bbox=table_bbox,
        cell_merge=cell_merge,
    )
    return {
        "type": BlockType.TABLE,
        "index": index,
        "bbox": table_bbox,
        "content": children if children is not None else [body],
    }


def _page(page_idx: int, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """构造 model_list_to_pages 输出契约的页面。"""
    return {"page_idx": page_idx, "blocks": blocks}


def _noise(block_type: str) -> dict[str, Any]:
    """构造页边界可忽略的噪声块。"""
    return {
        "type": block_type,
        "index": 99,
        "bbox": [0.1, 0.01, 0.9, 0.03],
        "content": "noise",
    }


def _merged_soup(table: dict[str, Any]) -> BeautifulSoup:
    """读取合并结果中的 table body HTML。"""
    body = next(child for child in table["content"] if child["type"] == BlockType.TABLE_BODY)
    return BeautifulSoup(body["content"], "html.parser")


def _row_texts(table: dict[str, Any]) -> list[list[str]]:
    """提取合并 HTML 的逐行单元格文本。"""
    return [[cell.get_text() for cell in row.find_all(["td", "th"])] for row in _merged_soup(table).find_all("tr")]


def test_merge_table_marks_reverse_multi_page_chain_and_cleans_stale_markers() -> None:
    """验证多页链倒序识别，并清理根块及子块的过期标记。"""
    tables = [_table(index) for index in range(3)]
    tables[0]["continues_prev"] = True
    tables[0]["content"][0]["continues_prev"] = True
    pages = [_page(index, [table]) for index, table in enumerate(tables)]

    merge_table(pages)

    assert "continues_prev" not in tables[0]
    assert "continues_prev" not in tables[0]["content"][0]
    assert tables[1]["continues_prev"] is True
    assert tables[2]["continues_prev"] is True

    merge_table(pages)
    assert tables[1]["continues_prev"] is True
    assert tables[2]["continues_prev"] is True


@pytest.mark.parametrize("page_indices", [(0, 2), (2, 1)])
def test_merge_table_requires_consecutive_increasing_page_indices(
    page_indices: tuple[int, int],
) -> None:
    """验证跳页和逆序 page_idx 都不会建立跨页关系。"""
    previous_table = _table(0)
    current_table = _table(1)

    merge_table(
        [
            _page(page_indices[0], [previous_table]),
            _page(page_indices[1], [current_table]),
        ]
    )

    assert "continues_prev" not in current_table


@pytest.mark.parametrize(
    "block_type",
    [
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.PAGE_FOOTNOTE,
        BlockType.ASIDE_TEXT,
    ],
)
def test_merge_table_skips_boundary_noise(block_type: str) -> None:
    """验证指定页眉页脚类块不会阻断边界 table 扫描。"""
    previous_table = _table(0)
    current_table = _table(1)
    pages = [
        _page(0, [previous_table, _noise(block_type)]),
        _page(1, [_noise(block_type), current_table]),
    ]

    merge_table(pages)

    assert current_table["continues_prev"] is True


@pytest.mark.parametrize("barrier_side", ["previous", "current"])
def test_merge_table_is_blocked_by_boundary_semantic_block(barrier_side: str) -> None:
    """验证普通正文位于页边界时会阻断 table 关系。"""
    previous_table = _table(0)
    current_table = _table(1)
    text = _noise(BlockType.TEXT)
    previous_blocks = [previous_table, text] if barrier_side == "previous" else [previous_table]
    current_blocks = [text, current_table] if barrier_side == "current" else [current_table]

    merge_table([_page(0, previous_blocks), _page(1, current_blocks)])

    assert "continues_prev" not in current_table


@pytest.mark.parametrize(
    ("previous_footnotes", "current_caption", "expected"),
    [
        (0, None, True),
        (0, "Table 1", False),
        (0, "Table 1 (continued)", True),
        (1, None, False),
        (1, "Table 1 (continued)", True),
        (2, "Table 1 (continued)", False),
    ],
)
def test_merge_table_applies_caption_and_footnote_rules(
    previous_footnotes: int,
    current_caption: str | None,
    expected: bool,
) -> None:
    """验证续表 caption 与前表 footnote 的组合规则。"""
    previous_children = [_table_body()]
    previous_children.extend(_footnote(f"old-{index}", index=index + 1) for index in range(previous_footnotes))
    current_children = [_table_body(index=1)]
    if current_caption is not None:
        current_children.insert(0, _caption(current_caption))
    previous_table = _table(0, children=previous_children)
    current_table = _table(1, children=current_children)

    merge_table([_page(0, [previous_table]), _page(1, [current_table])])

    assert current_table.get("continues_prev", False) is expected


def test_post_table_non_continuation_caption_does_not_block_merge() -> None:
    """验证 table body 下方的非续表 caption 不参与阻断判断。"""
    current_children = [
        _table_body(index=1, bbox=[0.1, 0.1, 0.9, 0.7]),
        _caption("Next section", index=2, bbox=[0.1, 0.75, 0.9, 0.8]),
    ]
    current_table = _table(1, children=current_children)

    merge_table([_page(0, [_table(0)]), _page(1, [current_table])])

    assert current_table["continues_prev"] is True


def test_merge_table_enforces_strict_ten_percent_width_threshold() -> None:
    """验证宽度差恰好百分之十时拒绝，小于阈值时接受。"""
    previous_table = _table(0, bbox=[0.1, 0.1, 0.9, 0.9])
    exact_threshold = _table(1, bbox=[0.06, 0.1, 0.94, 0.9])
    pages = [_page(0, [previous_table]), _page(1, [exact_threshold])]

    merge_table(pages)
    assert "continues_prev" not in exact_threshold

    exact_threshold["bbox"] = [0.08, 0.1, 0.92, 0.9]
    exact_threshold["content"][0]["bbox"] = [0.08, 0.1, 0.92, 0.9]
    merge_table(pages)
    assert exact_threshold["continues_prev"] is True


def test_merge_table_uses_boundary_row_metrics_when_total_columns_differ() -> None:
    """验证总列数不同后按边界行 effective/actual/渲染段数回退判断。"""
    matching_previous = _table(
        0,
        "<table><tr><td colspan='3'>H</td></tr><tr><td>A</td><td>B</td></tr></table>",
    )
    matching_current = _table(1, "<table><tr><td>C</td><td>D</td></tr></table>")
    merge_table([_page(0, [matching_previous]), _page(1, [matching_current])])
    assert matching_current["continues_prev"] is True

    mismatching_previous = _table(
        0,
        "<table><tr><td colspan='4'>H</td></tr><tr><td colspan='2'>A</td><td>B</td></tr></table>",
    )
    mismatching_current = _table(1, "<table><tr><td colspan='2'>C</td></tr></table>")
    merge_table([_page(0, [mismatching_previous]), _page(1, [mismatching_current])])
    assert "continues_prev" not in mismatching_current


@pytest.mark.parametrize(
    "current_table",
    [
        _table(1, bbox=[10.0, 10.0, 90.0, 90.0]),
        {"type": BlockType.TABLE, "index": 1, "bbox": [0.1, 0.1, 0.9, 0.9], "content": []},
        _table(1, "<table></table>"),
        _table(1, "<table><tr><td colspan='bad'>X</td></tr></table>"),
    ],
)
def test_merge_table_safely_skips_invalid_bbox_body_or_html(current_table: dict[str, Any]) -> None:
    """验证非法 bbox、table body 和 HTML 均安全降级。"""
    current_table = deepcopy(current_table)
    merge_table([_page(0, [_table(0)]), _page(1, [current_table])])
    assert "continues_prev" not in current_table


def test_merge_table_only_changes_markers_and_preserves_normalized_bboxes() -> None:
    """验证千分位计算不会回写 bbox，检测也不修改任何表格内容。"""
    previous_table = _table(0)
    current_table = _table(1)
    pages = [_page(0, [previous_table]), _page(1, [current_table])]
    original_pages = deepcopy(pages)

    merge_table(pages)

    assert current_table.pop("continues_prev") is True
    assert pages == original_pages


def test_merge_table_content_removes_repeated_rowspan_header_and_appends_data() -> None:
    """验证重复表头按 rowspan 覆盖范围删除，并追加当前数据行。"""
    previous_html = (
        "<table><tr><th rowspan='2'>H1</th><th>H2</th></tr><tr><th>H3</th></tr><tr><td>A</td><td>B</td></tr></table>"
    )
    current_html = "<table><tr><th rowspan='2'>H1</th><th>H2</th></tr><tr><th>H3</th></tr><tr><td>C</td><td>D</td></tr></table>"

    merged = merge_table_content(_table(0, previous_html), _table(1, current_html))

    assert merged is not None
    assert _row_texts(merged) == [["H1", "H2"], ["H3"], ["A", "B"], ["C", "D"]]


def test_merge_table_content_repairs_colspan_on_narrower_current_rows() -> None:
    """验证较窄的当前页数据行会补齐末单元格 colspan。"""
    previous = _table(0, "<table><tr><td>A</td><td colspan='2'>B</td></tr></table>")
    current = _table(1, "<table><tr><td>C</td><td>D</td></tr></table>")

    merged = merge_table_content(previous, current)

    assert merged is not None
    appended_cells = _merged_soup(merged).find_all("tr")[-1].find_all(["td", "th"])
    assert [cell.get_text() for cell in appended_cells] == ["C", "D"]
    assert appended_cells[-1]["colspan"] == "2"


def test_merge_table_content_clips_overlapped_blank_rowspan_placeholder() -> None:
    """验证当前页重复空白 rowspan 占位会按上页剩余跨度裁剪。"""
    previous = _table(0, "<table><tr><td rowspan='2'>A</td><td>X</td></tr></table>")
    current = _table(
        1,
        "<table><tr><td rowspan='2'></td><td>Y</td></tr><tr><td>Z</td></tr></table>",
    )

    merged = merge_table_content(previous, current)

    assert merged is not None
    assert _row_texts(merged) == [["A", "X"], ["Y"], ["", "Z"]]
    moved_blank_cell = _merged_soup(merged).find_all("tr")[-1].find_all(["td", "th"])[0]
    assert "rowspan" not in moved_blank_cell.attrs


def test_merge_table_content_replaces_footnote_and_preserves_previous_payloads() -> None:
    """验证纯合并保留前表载荷、忽略当前 caption，并替换 footnote。"""
    previous_body = _table_body(
        "<table><tr><th>H</th></tr><tr><td>A</td></tr></table>",
        index=5,
    )
    previous_body["image_base64"] = "previous-image"
    previous = _table(
        5,
        children=[_caption("Original caption", index=4), previous_body, _footnote("old", index=6)],
    )
    previous["image_base64"] = "outer-image"
    current = _table(
        8,
        children=[
            _caption("Table 1 (continued)", index=7),
            _table_body("<table><tr><th>H</th></tr><tr><td>B</td></tr></table>", index=8),
            {**_footnote("new", index=9), "_cross_page": True},
        ],
    )
    previous_original = deepcopy(previous)
    current_original = deepcopy(current)

    merged = merge_table_content(previous, current)

    assert merged is not None
    assert previous == previous_original
    assert current == current_original
    assert merged["index"] == previous["index"]
    assert merged["bbox"] == previous["bbox"]
    assert merged["image_base64"] == "outer-image"
    merged_body = next(child for child in merged["content"] if child["type"] == BlockType.TABLE_BODY)
    assert merged_body["image_base64"] == "previous-image"
    assert _row_texts(merged) == [["H"], ["A"], ["B"]]
    captions = [child["content"] for child in merged["content"] if child["type"] == BlockType.TABLE_CAPTION]
    footnotes = [child for child in merged["content"] if child["type"] == BlockType.TABLE_FOOTNOTE]
    assert captions == ["Original caption"]
    assert [footnote["content"] for footnote in footnotes] == ["new"]
    assert "_cross_page" not in footnotes[0]


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (_table(0, "<table></table>"), _table(1)),
        (_table(0), _table(1, "<table><tr><td colspan='bad'>X</td></tr></table>")),
        (_table(0), {"type": BlockType.TABLE, "bbox": [0.1, 0.1, 0.9, 0.9], "content": []}),
        (_table(0), _table(1, bbox=[0.0, 0.0, 100.0, 100.0])),
    ],
)
def test_merge_table_content_returns_none_for_invalid_input(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """验证纯内容操作遇到非法 HTML、主体或 bbox 时返回空。"""
    previous_original = deepcopy(previous)
    current_original = deepcopy(current)

    assert merge_table_content(previous, current) is None
    assert previous == previous_original
    assert current == current_original
