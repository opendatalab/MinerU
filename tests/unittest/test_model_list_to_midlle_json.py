from copy import deepcopy

import pytest

from mineru.backend.model_list_to_midlle_json import model_list_to_pages
from mineru.types import (
    EquationBlock,
    ImageBlock,
    ListBlock,
    PageInfo,
    ParagraphTitleBlock,
    TableBlock,
    TextBlock,
)


def test_model_list_to_pages_keeps_equation_type_without_mutating_input() -> None:
    """验证 equation 经 MagicModel 和严格对象化后保持类型、内容与图片载荷。"""
    model_list = [[
        {
            "type": "equation",
            "bbox": [0.1, 0.2, 0.9, 0.4],
            "content": r"\[x+1\]",
            "image_base64": "data:image/jpeg;base64,/9j/2Q==",
        }
    ]]
    original = deepcopy(model_list)

    page = model_list_to_pages(model_list)[0]

    assert model_list == original
    assert isinstance(page.blocks[0], EquationBlock)
    assert page.blocks[0].type == "equation"
    assert page.blocks[0].content == "x+1"
    assert page.blocks[0].image_base64 == "data:image/jpeg;base64,/9j/2Q=="


def test_model_list_to_pages_returns_typed_pdf_tree_without_mutating_input() -> None:
    """验证 PDF raw dict 只在副本上分组，并返回具体 PageInfo/Block 对象。"""
    model_list = [[
        {"type": "image_caption", "bbox": [0.1, 0.05, 0.9, 0.1], "content": "Figure 1"},
        {
            "type": "image",
            "bbox": [0.1, 0.12, 0.9, 0.45],
            "content": None,
            "image_base64": "data:image/jpeg;base64,/9j/2Q==",
        },
        {"type": "list", "bbox": [0.1, 0.5, 0.9, 0.8], "content": ""},
        {"type": "text", "bbox": [0.15, 0.55, 0.85, 0.65], "content": "item"},
    ]]
    original = deepcopy(model_list)

    pages = model_list_to_pages(model_list, page_index_map=[7])

    assert model_list == original
    assert isinstance(pages[0], PageInfo)
    assert pages[0].page_idx == 7
    assert isinstance(pages[0].blocks[0], ImageBlock)
    assert [child.type for child in pages[0].blocks[0].content] == [
        "image_caption",
        "image_body",
    ]
    assert isinstance(pages[0].blocks[1], ListBlock)
    assert isinstance(pages[0].blocks[1].content[0], TextBlock)


def test_model_list_to_pages_preserves_recursive_office_list_and_index() -> None:
    """验证 Office list/index 任意深度递归对象化，并删除目录 text 的 anchor。"""
    model_list = [[
        {
            "type": "list",
            "attribute": "ordered",
            "ilevel": 0,
            "start": 1,
            "content": [
                {"type": "text", "content": "first"},
                {
                    "type": "list",
                    "attribute": "unordered",
                    "ilevel": 1,
                    "content": [
                        {"type": "text", "content": "child"},
                        {"type": "list", "content": [{"type": "text", "content": "deep"}]},
                    ],
                },
            ],
        },
        {
            "type": "index",
            "content": [
                {
                    "type": "index",
                    "content": [{"type": "text", "content": "section", "anchor": "a1"}],
                }
            ],
        },
    ]]

    page = model_list_to_pages(model_list)[0]
    list_block = page.blocks[0]
    index_block = page.blocks[1]

    assert isinstance(list_block, ListBlock)
    assert list_block.content[0].content == "1. first"
    assert isinstance(list_block.content[1], ListBlock)
    assert list_block.content[1].content[0].content == "- child"
    assert isinstance(list_block.content[1].content[1], ListBlock)
    assert index_block.content[0].content[0].content == "section"
    assert "anchor" not in index_block.content[0].content[0].model_fields_set


def test_office_paragraph_numbering_is_document_wide_and_copy_only() -> None:
    """验证 Office 标题跨页编号、显式编号同步和 raw 元数据保留。"""
    model_list = [
        [
            {"type": "paragraph_title", "content": "<b>A</b>", "level": 1, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "B", "level": 2, "is_numbered_style": True},
        ],
        [
            {"type": "paragraph_title", "content": "1.4 Explicit", "level": 2, "is_numbered_style": False},
            {"type": "paragraph_title", "content": "C", "level": 2, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "No level", "is_numbered_style": True},
        ],
    ]
    original = deepcopy(model_list)

    pages = model_list_to_pages(model_list)
    titles = [block for page in pages for block in page.blocks]

    assert all(isinstance(block, ParagraphTitleBlock) for block in titles)
    assert [block.content for block in titles] == [
        "1 <b>A</b>",
        "1.1 B",
        "1.4 Explicit",
        "1.5 C",
        "2 No level",
    ]
    assert titles[-1].level is None
    assert model_list == original


def test_office_paragraph_numbering_clears_deeper_levels() -> None:
    """验证标题返回浅层时会清理旧深层计数，后续重新从一开始编号。"""
    model_list = [[
        {"type": "paragraph_title", "content": "A", "level": 1, "is_numbered_style": True},
        {"type": "paragraph_title", "content": "B", "level": 2, "is_numbered_style": True},
        {"type": "paragraph_title", "content": "C", "level": 3, "is_numbered_style": True},
        {"type": "paragraph_title", "content": "D", "level": 2, "is_numbered_style": True},
        {"type": "paragraph_title", "content": "E", "level": 3, "is_numbered_style": True},
        {"type": "paragraph_title", "content": "F", "level": 1, "is_numbered_style": True},
        {"type": "paragraph_title", "content": "G", "level": 2, "is_numbered_style": True},
    ]]

    titles = model_list_to_pages(model_list)[0].blocks

    assert [title.content for title in titles] == [
        "1 A",
        "1.1 B",
        "1.1.1 C",
        "1.2 D",
        "1.2.1 E",
        "2 F",
        "2.1 G",
    ]


def test_visual_body_drops_empty_parent_subtype() -> None:
    """验证 raw visual 的空 subtype 只用于父块判断，不会泄漏到严格 body 模型。"""
    page = model_list_to_pages(
        [[{"type": "image", "content": None, "sub_type": None}]]
    )[0]

    assert isinstance(page.blocks[0], ImageBlock)
    assert page.blocks[0].sub_type is None
    assert page.blocks[0].content[0].type == "image_body"


def test_pdf_continuation_is_typed_and_line_metadata_is_removed() -> None:
    """验证 raw 段落延续在对象化前完成，公开 TextBlock 不保留临时 lines。"""
    model_list = [[
        {
            "type": "text",
            "bbox": [0.1, 0.1, 0.9, 0.3],
            "content": "previous continuation",
            "lines": [{"bbox": [0.1, 0.1, 0.9, 0.15]}, {"bbox": [0.1, 0.2, 0.9, 0.25]}],
        },
        {
            "type": "text",
            "bbox": [0.1, 0.25, 0.9, 0.45],
            "content": "current continuation",
            "lines": [{"bbox": [0.1, 0.25, 0.9, 0.3]}, {"bbox": [0.1, 0.35, 0.9, 0.4]}],
        },
    ]]

    page = model_list_to_pages(model_list)[0]

    assert all(isinstance(block, TextBlock) for block in page.blocks)
    assert page.blocks[0].continues_prev is None
    assert page.blocks[1].continues_prev is True
    assert all("lines" not in type(block).model_fields for block in page.blocks)


def test_pdf_detection_scans_past_empty_first_page() -> None:
    """验证整份文档 bbox 判定不会把 PDF 空白首页误认为 Office。"""
    pages = model_list_to_pages([
        [],
        [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.3], "content": "later", "lines": []}],
    ])

    assert pages[0].blocks == []
    assert isinstance(pages[1].blocks[0], TextBlock)


def test_cross_page_table_continuation_remains_raw_postprocess() -> None:
    """验证连续页表格仍在对象化前写入 continues_prev。"""
    html_a = "<table><tr><td>A</td><td>B</td></tr></table>"
    html_b = "<table><tr><td>C</td><td>D</td></tr></table>"
    pages = model_list_to_pages([
        [{"type": "table", "bbox": [0.1, 0.1, 0.9, 0.85], "content": html_a}],
        [{"type": "table", "bbox": [0.1, 0.1, 0.9, 0.85], "content": html_b}],
    ])

    assert isinstance(pages[0].blocks[0], TableBlock)
    assert pages[0].blocks[0].continues_prev is None
    assert pages[1].blocks[0].continues_prev is True


@pytest.mark.parametrize(
    ("page_index_map", "message"),
    [([0], "length mismatch"), ([0, 0], "unique"), ([1, 0], "increasing"), ([0, -1], "non-negative")],
)
def test_page_index_map_is_strict(page_index_map: list[int], message: str) -> None:
    """验证页号映射不允许截断、重复、逆序或负数。"""
    with pytest.raises(ValueError, match=message):
        model_list_to_pages([[], []], page_index_map=page_index_map)
