"""验证按相同跨行范围补齐共享空白格，并保护新子项和独立设备属性。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup, Tag
from docvortex.content.table import build_row_rendered_cell_segments

from mineru.backend.postprocess.table_merge import llm_cell_merge
from mineru.render import render_html
from mineru.types import PageInfo

from test_llm_aided_postprocess import _cross_page_table_pages, _middle_json, _QueuedValidatedClient


def _shared_pages(
    *,
    previous_group: str = "Group A",
    current_group: str = "",
    current_remarks: str = "",
    previous_remarks_span: int = 2,
    current_remarks_span: int = 3,
    previous_group_columns: int = 1,
) -> list[PageInfo]:
    """构造共享字段跨两行接三行、明细每行独立的通用跨页表格。"""
    previous = (
        f'<table><tr><td rowspan="2" colspan="{previous_group_columns}">{previous_group}</td>'
        '<td>Old detail 1</td><td rowspan="2">Policy starts</td>'
        f'<td rowspan="{previous_remarks_span}"></td></tr><tr><td>Old detail 2</td>'
        + ("<td></td>" if previous_remarks_span == 1 else "")
        + "</tr></table>"
    )
    current = (
        f'<table><tr><td rowspan="3">{current_group}</td><td>New detail 1</td>'
        '<td rowspan="3">and continues</td>'
        f'<td rowspan="{current_remarks_span}">{current_remarks}</td></tr>'
    )
    for row in range(1, 3):
        current += f"<tr><td>New detail {row + 1}</td>"
        if row >= current_remarks_span:
            current += "<td></td>"
        current += "</tr>"
    return _cross_page_table_pages(previous_html=previous, current_html=current + "</table>")


def test_shared_blank_merge_joins_group_without_consuming_new_details() -> None:
    """模型确认共享文本后补齐同范围空白格，真实渲染跨五行并保留五个独立明细。"""
    pages = _shared_pages()
    original_html = [page.blocks[0].content[0].content for page in pages]
    client = _QueuedValidatedClient([[0, 0, 1, 0]])
    asyncio.run(llm_cell_merge.apply_llm_cross_page_cell_merge(pages, client))  # type: ignore[arg-type]
    assert pages[1].blocks[0].cell_merge == [1, 0, 1, 1]
    assert len(client.prompts) == 1
    assert "<table>" not in client.prompts[0]
    soup = BeautifulSoup(render_html(_middle_json(pages)), "html.parser")
    assert len(soup.find_all("table")) == 1
    rows = soup.find_all("tr")
    assert len(rows) == 5
    cells = rows[0].find_all("td", recursive=False)
    assert [cells[index]["rowspan"] for index in (0, 2, 3)] == ["5", "5", "5"]
    assert [row.find_all("td", recursive=False)[-1].get_text() for row in rows[1:]] == [
        "Old detail 2",
        "New detail 1",
        "New detail 2",
        "New detail 3",
    ]
    assert [page.blocks[0].content[0].content for page in pages] == original_html


@pytest.mark.parametrize(
    "raw,expected", [([0, 0, 0, 0], [0, 0, 0, 0]), ([0, 1, 0, 0], [1, 1, 0, 0]), ([1, 0, 0, 0], [1, 0, 0, 0])]
)
def test_shared_blank_merge_requires_original_text_signal_in_same_group(raw: list[int], expected: list[int]) -> None:
    """全零、组外续接或上有下空的标记都不能代替组内原始双侧文本续接依据。"""
    pages = _shared_pages()
    asyncio.run(llm_cell_merge.apply_llm_cross_page_cell_merge(pages, _QueuedValidatedClient([raw])))  # type: ignore[arg-type]
    assert pages[1].blocks[0].cell_merge == expected


@pytest.mark.parametrize("previous_group", ["Group A", ""])
def test_shared_blank_merge_preserves_conflicting_group_identity(previous_group: str) -> None:
    """组内存在未确认的新内容时保留共享空白边界，并覆盖前空后有内容的方向限制。"""
    pages = _shared_pages(previous_group=previous_group, current_group="Group B")
    asyncio.run(llm_cell_merge.apply_llm_cross_page_cell_merge(pages, _QueuedValidatedClient([[0, 0, 1, 0]])))  # type: ignore[arg-type]
    assert pages[1].blocks[0].cell_merge == [0, 0, 1, 0]


@pytest.mark.parametrize("content", ['<img src="image.png"/>', '<eq data-latex="x"></eq>'])
def test_shared_blank_merge_preserves_media_cells(content: str) -> None:
    """图片和公式即使没有可提取文字，也作为组内未确认的非空格保留。"""
    pages = _shared_pages(current_remarks=content)
    asyncio.run(llm_cell_merge.apply_llm_cross_page_cell_merge(pages, _QueuedValidatedClient([[0, 0, 1, 0]])))  # type: ignore[arg-type]
    assert pages[1].blocks[0].cell_merge == [1, 0, 1, 0]


@pytest.mark.parametrize("previous_span,current_span", [(1, 1), (2, 2), (2, 4)])
def test_shared_blank_merge_rejects_single_mismatched_or_overflowing_ranges(previous_span: int, current_span: int) -> None:
    """单行空格、与参考格不一致的范围及超出当前页的范围都保留边界。"""
    pages = _shared_pages(previous_remarks_span=previous_span, current_remarks_span=current_span)
    asyncio.run(llm_cell_merge.apply_llm_cross_page_cell_merge(pages, _QueuedValidatedClient([[0, 0, 1, 0]])))  # type: ignore[arg-type]
    assert pages[1].blocks[0].cell_merge == [1, 0, 1, 0]


def test_shared_blank_merge_rejects_unaligned_column_ranges() -> None:
    """上下视觉列区间不一致时只采用既有规则，禁止凭相同跨度推断共享归属。"""
    pages = _shared_pages(previous_group_columns=2)
    asyncio.run(llm_cell_merge.apply_llm_cross_page_cell_merge(pages, _QueuedValidatedClient([[0, 0, 1, 0]])))  # type: ignore[arg-type]
    assert pages[1].blocks[0].cell_merge == [1, 1, 0, 1, 0]


def test_missing_optional_row_ranges_preserves_existing_enhancement(monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟已发布包的旧片段形状，缺少跨度字段时仍能执行原有空白续接规则。"""

    def legacy_segments(rows: list[Tag], row_index: int) -> list[SimpleNamespace]:
        """仅暴露旧发布包已经提供的文本、视觉列与非文本内容字段。"""
        return [
            SimpleNamespace(text=s.text, start_col=s.start_col, end_col=s.end_col, has_non_text_content=s.has_non_text_content)
            for s in build_row_rendered_cell_segments(rows, row_index)
        ]

    monkeypatch.setattr(llm_cell_merge, "build_row_rendered_cell_segments", legacy_segments)
    pages = _shared_pages()
    client = _QueuedValidatedClient([[0, 0, 1, 0]])
    asyncio.run(llm_cell_merge.apply_llm_cross_page_cell_merge(pages, client))  # type: ignore[arg-type]
    assert len(client.prompts) == 1
    assert pages[1].blocks[0].cell_merge == [1, 0, 1, 0]
