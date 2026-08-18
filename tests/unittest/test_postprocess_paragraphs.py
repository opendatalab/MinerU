"""跨页段落延续规则的聚焦回归测试。"""

from copy import deepcopy

import pytest

from mineru.backend.postprocess.paragraphs import can_auto_merge_text_blocks, merge_para_text_blocks
from mineru.types import BlockType


def _text_block(
    index: int,
    content: str,
    bbox: list[float],
    line_bboxes: list[list[float]],
) -> dict:
    """构造使用归一化 block/line bbox 的 dict text block。"""
    return {
        "index": index,
        "type": BlockType.TEXT,
        "bbox": list(bbox),
        "content": content,
        "lines": [{"bbox": list(line_bbox)} for line_bbox in line_bboxes],
    }


def _horizontal_pair() -> tuple[dict, dict]:
    """构造一组满足横排段落延续规则的前后 text block。"""
    previous_block = _text_block(
        0,
        "previous continuation",
        [0.1, 0.1, 0.9, 0.3],
        [[0.1, 0.1, 0.9, 0.15], [0.1, 0.2, 0.9, 0.25]],
    )
    current_block = _text_block(
        1,
        "current continuation",
        [0.1, 0.25, 0.9, 0.45],
        [[0.1, 0.25, 0.9, 0.3], [0.1, 0.35, 0.9, 0.4]],
    )
    return previous_block, current_block


def _vertical_pair() -> tuple[dict, dict]:
    """构造一组满足竖排段落延续规则的前后 text block。"""
    previous_block = _text_block(
        0,
        "previous vertical",
        [0.65, 0.1, 0.9, 0.9],
        [[0.82, 0.1, 0.87, 0.9], [0.72, 0.1, 0.77, 0.9]],
    )
    current_block = _text_block(
        1,
        "current vertical",
        [0.5, 0.1, 0.75, 0.9],
        [[0.62, 0.1, 0.67, 0.9], [0.52, 0.1, 0.57, 0.9]],
    )
    return previous_block, current_block


def test_merge_para_text_blocks_marks_reverse_chain_without_moving_content() -> None:
    """验证倒序处理会给连续链的后续块加标记，但不搬运正文或改写 bbox。"""
    first_block, second_block = _horizontal_pair()
    third_block = _text_block(
        2,
        "third continuation",
        [0.1, 0.4, 0.9, 0.6],
        [[0.1, 0.4, 0.9, 0.45], [0.1, 0.5, 0.9, 0.55]],
    )
    pages = [{"page_idx": 0, "blocks": [first_block, second_block, third_block]}]
    original_contents = [block["content"] for block in pages[0]["blocks"]]
    original_bboxes = deepcopy([block["bbox"] for block in pages[0]["blocks"]])

    merge_para_text_blocks(pages)

    assert "continues_prev" not in first_block
    assert second_block["continues_prev"] is True
    assert third_block["continues_prev"] is True
    assert [block["content"] for block in pages[0]["blocks"]] == original_contents
    assert [block["bbox"] for block in pages[0]["blocks"]] == original_bboxes
    assert all("lines" not in block for block in pages[0]["blocks"])

    first_result = deepcopy(pages)
    merge_para_text_blocks(pages)
    assert pages == first_result


@pytest.mark.parametrize(
    ("page_indices", "expected_continuation"),
    [
        ((0, 1), True),
        ((0, 2), False),
        ((2, 1), False),
    ],
)
def test_merge_para_text_blocks_requires_consecutive_cross_page_indices(
    page_indices: tuple[int, int],
    expected_continuation: bool,
) -> None:
    """验证跨页 text 仅允许从前一连续页延续。"""
    previous_block, current_block = _horizontal_pair()
    pages = [
        {"page_idx": page_indices[0], "blocks": [previous_block]},
        {"page_idx": page_indices[1], "blocks": [current_block]},
    ]

    merge_para_text_blocks(pages)

    assert current_block.get("continues_prev", False) is expected_continuation


@pytest.mark.parametrize(
    ("middle_type", "expected_continuation"),
    [
        (BlockType.IMAGE, True),
        (BlockType.CODE, True),
        (BlockType.HEADER, True),
        (BlockType.FOOTER, True),
        (BlockType.PAGE_NUMBER, True),
        (BlockType.PAGE_FOOTNOTE, True),
        (BlockType.ASIDE_TEXT, True),
        (BlockType.DOC_TITLE, False),
        (BlockType.LIST, False),
        (BlockType.REF_TEXT, False),
    ],
)
def test_merge_para_text_blocks_respects_transparent_and_barrier_types(
    middle_type: str,
    expected_continuation: bool,
) -> None:
    """验证视觉根块和页面装饰块可跨过，其他语义块会阻断 text 查找。"""
    previous_block, current_block = _horizontal_pair()
    middle_block = {
        "index": 1,
        "type": middle_type,
        "bbox": [0.1, 0.22, 0.9, 0.24],
        "content": "middle",
    }
    current_block["index"] = 2
    pages = [{"page_idx": 0, "blocks": [previous_block, middle_block, current_block]}]

    merge_para_text_blocks(pages)

    assert current_block.get("continues_prev", False) is expected_continuation


def test_merge_para_text_blocks_ignores_cross_page_decorations() -> None:
    """验证跨页正文会跳过前页尾部和后页头部的全部页面装饰块。"""
    previous_block, current_block = _horizontal_pair()
    current_block["index"] = 3
    pages = [
        {
            "page_idx": 0,
            "blocks": [
                previous_block,
                {"index": 1, "type": BlockType.FOOTER, "content": "footer"},
                {"index": 2, "type": BlockType.PAGE_FOOTNOTE, "content": "page footnote"},
            ],
        },
        {
            "page_idx": 1,
            "blocks": [
                {"index": 0, "type": BlockType.HEADER, "content": "header"},
                {"index": 1, "type": BlockType.PAGE_NUMBER, "content": "2"},
                {"index": 2, "type": BlockType.ASIDE_TEXT, "content": "aside"},
                current_block,
            ],
        },
    ]

    merge_para_text_blocks(pages)

    assert current_block["continues_prev"] is True
    assert previous_block["content"] == "previous continuation"
    assert current_block["content"] == "current continuation"
    assert "lines" not in previous_block
    assert "lines" not in current_block


@pytest.mark.parametrize(
    "barrier_type",
    [BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE, BlockType.EQUATION, BlockType.LIST],
)
def test_merge_para_text_blocks_keeps_cross_page_semantic_barriers(barrier_type: str) -> None:
    """验证页面装饰块透明后，标题、公式和列表仍会阻断跨页正文连接。"""
    previous_block, current_block = _horizontal_pair()
    current_block["index"] = 2
    pages = [
        {
            "page_idx": 0,
            "blocks": [
                previous_block,
                {"index": 1, "type": BlockType.PAGE_FOOTNOTE, "content": "page footnote"},
            ],
        },
        {
            "page_idx": 1,
            "blocks": [
                {"index": 0, "type": BlockType.HEADER, "content": "header"},
                {"index": 1, "type": barrier_type, "content": "barrier"},
                current_block,
            ],
        },
    ]

    merge_para_text_blocks(pages)

    assert "continues_prev" not in current_block


def test_can_auto_merge_horizontal_text_blocks_rejects_paragraph_boundaries() -> None:
    """验证横排规则会拒绝终止标点、特殊段首以及不满足几何条件的候选。"""
    previous_block, current_block = _horizontal_pair()
    assert can_auto_merge_text_blocks(current_block, previous_block)

    previous_block, current_block = _horizontal_pair()
    previous_block["content"] = "finished."
    assert not can_auto_merge_text_blocks(current_block, previous_block)

    for current_content in ("1 numbered", "Uppercase"):
        previous_block, current_block = _horizontal_pair()
        current_block["content"] = current_content
        assert not can_auto_merge_text_blocks(current_block, previous_block)

    previous_block, current_block = _horizontal_pair()
    current_block["lines"][0]["bbox"][0] = 0.2
    assert not can_auto_merge_text_blocks(current_block, previous_block)

    previous_block, current_block = _horizontal_pair()
    previous_block["lines"][-1]["bbox"][2] = 0.7
    assert not can_auto_merge_text_blocks(current_block, previous_block)

    previous_block, current_block = _horizontal_pair()
    for line in current_block["lines"]:
        line["bbox"][2] = 0.4
    assert not can_auto_merge_text_blocks(current_block, previous_block)

    previous_block, current_block = _horizontal_pair()
    previous_block["lines"] = previous_block["lines"][:1]
    current_block["lines"] = current_block["lines"][:1]
    assert not can_auto_merge_text_blocks(current_block, previous_block)

    previous_block, current_block = _horizontal_pair()
    current_block["bbox"][1] = 0.31
    assert not can_auto_merge_text_blocks(current_block, previous_block)


def test_merge_para_text_blocks_supports_vertical_but_rejects_mixed_orientation() -> None:
    """验证竖排块可以连续，而横竖方向不一致时不会误标记。"""
    previous_vertical, current_vertical = _vertical_pair()
    vertical_pages = [{"page_idx": 0, "blocks": [previous_vertical, current_vertical]}]

    merge_para_text_blocks(vertical_pages)

    assert current_vertical["continues_prev"] is True

    previous_vertical, _ = _vertical_pair()
    _, current_horizontal = _horizontal_pair()
    mixed_pages = [{"page_idx": 0, "blocks": [previous_vertical, current_horizontal]}]

    merge_para_text_blocks(mixed_pages)

    assert "continues_prev" not in current_horizontal


def test_merge_para_text_blocks_marks_adjacent_ref_lists_without_moving_items() -> None:
    """验证连续页的相邻参考文献 list 只增加标记，不搬运列表子项。"""
    previous_list = {
        "index": 0,
        "type": BlockType.LIST,
        "sub_type": BlockType.REF_TEXT,
        "bbox": [0.1, 0.1, 0.9, 0.3],
        "content": [{"type": BlockType.REF_TEXT, "content": "ref one", "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}]}],
    }
    current_list = {
        "index": 0,
        "type": BlockType.LIST,
        "sub_type": BlockType.REF_TEXT,
        "bbox": [0.1, 0.1, 0.9, 0.3],
        "content": [{"type": BlockType.REF_TEXT, "content": "ref two", "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}]}],
    }
    pages = [
        {"page_idx": 4, "blocks": [previous_list]},
        {"page_idx": 5, "blocks": [current_list]},
    ]

    merge_para_text_blocks(pages)

    assert "continues_prev" not in previous_list
    assert current_list["continues_prev"] is True
    assert [child["content"] for child in previous_list["content"]] == ["ref one"]
    assert [child["content"] for child in current_list["content"]] == ["ref two"]
    assert "lines" not in previous_list["content"][0]
    assert "lines" not in current_list["content"][0]

    previous_list, current_list = deepcopy(previous_list), deepcopy(current_list)
    current_list.pop("continues_prev", None)
    gap_pages = [
        {"page_idx": 4, "blocks": [previous_list]},
        {"page_idx": 6, "blocks": [current_list]},
    ]
    merge_para_text_blocks(gap_pages)
    assert "continues_prev" not in current_list


def test_merge_para_text_blocks_cleans_nested_lines_and_invalid_stale_markers() -> None:
    """验证非法 text 不合并，同时递归清理临时行框和过期标记。"""
    nested_child = {
        "type": BlockType.IMAGE_CAPTION,
        "content": "caption",
        "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}],
        "continues_prev": True,
    }
    visual_block = {
        "index": 0,
        "type": BlockType.IMAGE,
        "bbox": [0.1, 0.1, 0.9, 0.5],
        "content": [nested_child],
        "lines": [{"bbox": [0.1, 0.1, 0.9, 0.5]}],
    }
    invalid_text = {
        "index": 1,
        "type": BlockType.TEXT,
        "bbox": [0.1, 0.5, 0.9, 0.7],
        "content": "invalid lines",
        "lines": [{"bbox": [0.1, 0.5, 0.9]}],
        "continues_prev": True,
    }
    pages = [{"page_idx": 0, "blocks": [visual_block, invalid_text]}]

    merge_para_text_blocks(pages)

    assert "continues_prev" not in nested_child
    assert "continues_prev" not in invalid_text
    assert "lines" not in visual_block
    assert "lines" not in nested_child
    assert "lines" not in invalid_text
