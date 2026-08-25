from __future__ import annotations

import pytest

from mineru.model.flash.pdf import index_blocks

from _flash_pdf_test_utils import _text_line


def _directory_lines(
    numeric_rows: set[int],
    *,
    row_count: int = 10,
) -> list[index_blocks._LineItem]:
    """构造带居中标题、宽目录条目和右侧页码片段的视觉行。"""

    lines = [
        _text_line(
            "centered heading",
            (40.0, 5.0, 60.0, 15.0),
            100,
            visual_row_id=100,
        )
    ]
    for row_index in range(row_count):
        top = 30.0 + 12.0 * row_index
        lines.extend(
            [
                _text_line(
                    f"entry {row_index} . . . .",
                    (10.0, top, 80.0, top + 10.0),
                    row_index * 2,
                    visual_row_id=row_index,
                    run_index=0,
                    split_from_row=True,
                ),
                _text_line(
                    str(row_index + 1) if row_index in numeric_rows else "continuation",
                    (92.0, top, 96.0, top + 10.0),
                    row_index * 2 + 1,
                    visual_row_id=row_index,
                    run_index=1,
                    split_from_row=True,
                ),
            ]
        )
    return lines


@pytest.mark.parametrize(
    "content",
    [
        "entry 12",
        "entry １２",
        "entry xiv",
        "entry IV。",
        "entry 9 )",
    ],
)
def test_index_page_number_suffix_supports_expected_forms(content: str) -> None:
    """验证目录行尾支持半角、全角、罗马页码及有限尾随标点。"""

    assert index_blocks._index_row_ends_in_page_number(content)


@pytest.mark.parametrize("content", ["entry", "version A12x", "word LIVE"])
def test_index_page_number_suffix_rejects_non_page_number_tail(content: str) -> None:
    """验证普通词尾和字母数字混合标识不会被当作目录页码。"""

    assert not index_blocks._index_row_ends_in_page_number(content)


def test_index_block_merges_split_rows_and_preserves_heading() -> None:
    """验证目录正文合成一个换行块，居中目录标题保持段落标题。"""

    lines = _directory_lines(set(range(6)), row_count=6)

    blocks, remaining = index_blocks._extract_index_blocks(
        lines,
        (100.0, 130.0),
        [],
    )

    assert len(blocks) == 1
    assert blocks[0]["type"] == "index"
    assert blocks[0]["content"] == "\n".join(f"entry {row_index} . . . . {row_index + 1}" for row_index in range(6))
    assert not str(blocks[0]["content"]).endswith("\n")
    assert [line.text for line in remaining] == ["centered heading"]
    assert remaining[0].semantic_type == "paragraph_title"
    assert all(line.semantic_type == "index" for line in lines if line is not remaining[0])


def test_index_prepass_requires_geometric_heading_but_fallback_keeps_legacy_detection() -> None:
    """验证公式前目录预判拒绝无标题编号行，公式后的兼容识别仍可处理无标题目录。"""

    lines = _directory_lines(set(range(6)), row_count=6)[1:]

    prepass_blocks, prepass_remaining = index_blocks._extract_index_blocks(
        lines,
        (100.0, 130.0),
        [],
        require_heading=True,
    )
    fallback_blocks, fallback_remaining = index_blocks._extract_index_blocks(
        lines,
        (100.0, 130.0),
        [],
    )

    assert prepass_blocks == []
    assert prepass_remaining == lines
    assert len(fallback_blocks) == 1
    assert fallback_remaining == []


@pytest.mark.parametrize(
    ("numeric_rows", "expected_block_count"),
    [
        ({0, 1, 2, 4, 6, 8, 9}, 1),
        ({0, 1, 2, 4, 6, 9}, 0),
    ],
)
def test_index_block_requires_seventy_percent_numeric_rows(
    numeric_rows: set[int],
    expected_block_count: int,
) -> None:
    """验证目录候选在 70% 页码行尾阈值处命中，低于阈值则拒绝。"""

    lines = _directory_lines(numeric_rows)

    blocks, _remaining = index_blocks._extract_index_blocks(
        lines,
        (100.0, 160.0),
        [],
    )

    assert len(blocks) == expected_block_count


def test_numeric_lines_without_directory_width_or_sidecars_are_rejected() -> None:
    """验证仅有大量数字行尾而缺少宽行及页码侧栏时不会误判目录。"""

    lines = [
        _text_line(
            f"value {row_index}",
            (50.0, 10.0 + 12.0 * row_index, 90.0, 20.0 + 12.0 * row_index),
            row_index,
            visual_row_id=row_index,
        )
        for row_index in range(8)
    ]

    blocks, remaining = index_blocks._extract_index_blocks(
        lines,
        (100.0, 120.0),
        [],
    )

    assert blocks == []
    assert remaining == lines
