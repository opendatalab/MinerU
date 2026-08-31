from copy import deepcopy
from typing import Any

from mineru.backend.postprocess import pages as pages_module
from mineru.backend.postprocess.lists import fix_office_list_blocks
from mineru.backend.postprocess.pages import model_json_to_pages
from mineru.types import (
    ChartBlock,
    DocTitleBlock,
    EquationBlock,
    ImageBlock,
    ListBlock,
    ModelJson,
    PageInfo,
    ParagraphTitleBlock,
    RefTextBlock,
    TableBlock,
    TextBlock,
)
from _span_test_utils import equation, hyperlink, inline as _inline, inline_text


_RAW_INLINE_TYPES = {
    "text",
    "ref_text",
    "doc_title",
    "paragraph_title",
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "list",
    "index",
    "caption",
    "footnote",
    "image_caption",
    "image_footnote",
    "table_caption",
    "table_footnote",
    "chart_caption",
    "chart_footnote",
    "code_caption",
    "code_footnote",
}


def _spanize_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把本文件关注后处理逻辑的简写文字夹具转换为 Schema 2.0 Span。"""
    for block in blocks:
        block_type = str(block.get("type"))
        content = block.get("content")
        if block_type in _RAW_INLINE_TYPES and isinstance(content, str):
            block["content"] = _inline(content) if content else []
        elif block_type in {"list", "index"} and isinstance(content, list):
            _spanize_blocks([child for child in content if isinstance(child, dict)])
    return blocks


def _model_json(
    pages: list[list[dict[str, Any]]],
    *,
    page_index_map: list[int] | None = None,
) -> ModelJson:
    """为 PageInfo 后处理测试构造最小严格 ModelJson。"""
    return ModelJson(
        pages=[_spanize_blocks(deepcopy(page)) for page in pages],
        page_index_map=page_index_map or [],
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def test_model_json_to_pages_replaces_raw_model_list_entrypoint() -> None:
    """验证严格 ModelJson 是唯一文档级 PageInfo 转换入口。"""
    assert callable(pages_module.model_json_to_pages)
    assert not hasattr(pages_module, "model_list_to_pages")


def test_model_json_to_pages_keeps_equation_type_without_mutating_input() -> None:
    """验证 equation 经单页后处理和严格对象化后保持类型、内容与图片载荷。"""
    model_list = [
        [
            {
                "type": "equation",
                "bbox": [0.1, 0.2, 0.9, 0.4],
                "content": r"\[x+1\]",
                "image_base64": "data:image/jpeg;base64,/9j/2Q==",
            }
        ]
    ]
    original = deepcopy(model_list)

    page = model_json_to_pages(_model_json(model_list))[0]

    assert model_list == original
    assert isinstance(page.blocks[0], EquationBlock)
    assert page.blocks[0].type == "equation"
    assert page.blocks[0].content == "x+1"
    assert page.blocks[0].image_base64 == "data:image/jpeg;base64,/9j/2Q=="


def test_model_json_to_pages_returns_typed_pdf_tree_without_mutating_input() -> None:
    """验证 PDF raw dict 只在副本上分组，并返回具体 PageInfo/Block 对象。"""
    model_list = [
        [
            {"type": "image_caption", "bbox": [0.1, 0.05, 0.9, 0.1], "content": "Figure 1"},
            {
                "type": "image",
                "bbox": [0.1, 0.12, 0.9, 0.45],
                "content": None,
                "image_base64": "data:image/jpeg;base64,/9j/2Q==",
            },
            {"type": "list", "bbox": [0.1, 0.5, 0.9, 0.8], "content": ""},
            {"type": "text", "bbox": [0.15, 0.55, 0.85, 0.65], "content": "item"},
        ]
    ]
    original = deepcopy(model_list)

    pages = model_json_to_pages(_model_json(model_list, page_index_map=[7]))

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


def test_model_json_to_pages_preserves_recursive_office_list_and_index() -> None:
    """验证 Office list/index 任意深度递归对象化，并清理未匹配目录 anchor。"""
    model_list = [
        [
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
        ]
    ]

    page = model_json_to_pages(_model_json(model_list))[0]
    list_block = page.blocks[0]
    index_block = page.blocks[1]

    assert isinstance(list_block, ListBlock)
    assert inline_text(list_block.content[0].content) == "1. first"
    assert isinstance(list_block.content[1], ListBlock)
    assert inline_text(list_block.content[1].content[0].content) == "- child"
    assert isinstance(list_block.content[1].content[1], ListBlock)
    assert inline_text(index_block.content[0].content[0].content) == "section"
    assert "anchor" not in index_block.content[0].content[0].model_fields_set


def test_fix_office_list_blocks_uses_local_ordered_markers_at_each_depth() -> None:
    """验证多级 Office 有序列表按当前层独立编号，并保留富文本与各层起始值。"""
    rich_content = [*_inline("root", styles=["bold"]), equation("x"), hyperlink("https://example.com", "link")]
    list_block = {
        "type": "list",
        "attribute": "ordered",
        "ilevel": 0,
        "start": 3,
        "content": [
            {"type": "text", "content": rich_content},
            {
                "type": "list",
                "attribute": "unordered",
                "ilevel": 1,
                "content": [
                    {"type": "text", "content": "bullet before"},
                    {
                        "type": "list",
                        "attribute": "ordered",
                        "ilevel": 2,
                        "start": 1,
                        "content": [
                            {"type": "text", "content": "level two first"},
                            {"type": "text", "content": "level two second"},
                            {
                                "type": "list",
                                "attribute": "ordered",
                                "ilevel": 3,
                                "start": 0,
                                "content": [
                                    {"type": "text", "content": "zero"},
                                    {"type": "text", "content": "one"},
                                ],
                            },
                            {"type": "text", "content": "level two third"},
                            {
                                "type": "list",
                                "attribute": "ordered",
                                "ilevel": 3,
                                "start": 5,
                                "content": [
                                    {"type": "text", "content": "five"},
                                    {"type": "text", "content": "six"},
                                ],
                            },
                            {"type": "text", "content": "level two fourth"},
                        ],
                    },
                    {"type": "text", "content": "bullet after"},
                ],
            },
            {"type": "text", "content": "root after nested"},
        ],
    }
    invalid_start_list = {
        "type": "list",
        "attribute": "ordered",
        "ilevel": 0,
        "start": "invalid",
        "content": [{"type": "text", "content": "fallback"}],
    }

    _spanize_blocks([list_block, invalid_start_list])
    result = fix_office_list_blocks([list_block, invalid_start_list])

    assert result == [list_block, invalid_start_list]
    assert inline_text(list_block["content"][0]["content"]) == "3. rootxlink"
    assert inline_text(list_block["content"][2]["content"]) == "4. root after nested"
    unordered = list_block["content"][1]
    assert inline_text(unordered["content"][0]["content"]) == "- bullet before"
    assert inline_text(unordered["content"][2]["content"]) == "- bullet after"
    level_two = unordered["content"][1]
    assert [inline_text(level_two["content"][index]["content"]) for index in (0, 1, 3, 5)] == [
        "1. level two first",
        "2. level two second",
        "3. level two third",
        "4. level two fourth",
    ]
    zero_start = level_two["content"][2]
    assert [inline_text(child["content"]) for child in zero_start["content"]] == ["0. zero", "1. one"]
    five_start = level_two["content"][4]
    assert [inline_text(child["content"]) for child in five_start["content"]] == ["5. five", "6. six"]
    assert inline_text(invalid_start_list["content"][0]["content"]) == "1. fallback"

    pending_lists = list(result)
    while pending_lists:
        current = pending_lists.pop()
        assert {"attribute", "ilevel", "start"}.isdisjoint(current)
        pending_lists.extend(
            child
            for child in current.get("content", [])
            if isinstance(child, dict) and child.get("type") == "list"
        )


def test_model_json_to_pages_maps_index_anchor_to_document_title_type_and_level() -> None:
    """验证 Office 目录叶子按跨页目标 anchor 继承真实标题类型与层级。"""
    model_list = [
        [
            {
                "type": "index",
                "content": [
                    {"type": "text", "content": "Document", "anchor": "doc-anchor"},
                    {
                        "type": "index",
                        "content": [
                            {"type": "text", "content": "Section", "anchor": "section-anchor"},
                            {"type": "text", "content": "Missing", "anchor": "missing-anchor"},
                        ],
                    },
                ],
            },
        ],
        [
            {"type": "doc_title", "content": "Document title", "level": 1, "anchor": "doc-anchor"},
            {
                "type": "paragraph_title",
                "content": "Section title",
                "level": 3,
                "anchor": "section-anchor",
            },
        ],
    ]

    pages = model_json_to_pages(_model_json(model_list))
    index = pages[0].blocks[0]
    doc_leaf = index.content[0]
    section_leaf = index.content[1].content[0]
    missing_leaf = index.content[1].content[1]

    assert isinstance(doc_leaf, DocTitleBlock)
    assert doc_leaf.level == 1
    assert doc_leaf.anchor == "doc-anchor"
    assert inline_text(doc_leaf.content) == "Document"
    assert isinstance(section_leaf, ParagraphTitleBlock)
    assert section_leaf.level == 3
    assert section_leaf.anchor == "section-anchor"
    assert inline_text(section_leaf.content) == "Section"
    assert isinstance(missing_leaf, TextBlock)
    assert "anchor" not in missing_leaf.model_fields_set


def test_model_json_to_pages_uses_first_title_for_duplicate_anchor() -> None:
    """验证重复 anchor 按文档顺序使用首个标题目标。"""
    model_list = [
        [
            {"type": "paragraph_title", "content": "First", "level": 2, "anchor": "same"},
            {"type": "paragraph_title", "content": "Second", "level": 4, "anchor": "same"},
            {
                "type": "index",
                "content": [{"type": "text", "content": "Entry", "anchor": "same"}],
            },
        ]
    ]

    page = model_json_to_pages(_model_json(model_list))[0]
    index = page.blocks[2]

    assert isinstance(index.content[0], ParagraphTitleBlock)
    assert index.content[0].level == 2


def test_office_paragraph_numbering_is_document_wide_and_copy_only() -> None:
    """验证 Office 标题跨页编号、显式编号同步和 raw 元数据保留。"""
    model_list = [
        [
            {"type": "paragraph_title", "content": "<b>A</b>", "level": 1, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "B", "level": 3, "is_numbered_style": True},
        ],
        [
            {"type": "paragraph_title", "content": "1.4 Explicit", "level": 3, "is_numbered_style": False},
            {"type": "paragraph_title", "content": "C", "level": 3, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "No level", "is_numbered_style": True},
        ],
    ]
    original = deepcopy(model_list)

    pages = model_json_to_pages(_model_json(model_list))
    titles = [block for page in pages for block in page.blocks]

    assert all(isinstance(block, ParagraphTitleBlock) for block in titles)
    assert [inline_text(block.content) for block in titles] == [
        "1 <b>A</b>",
        "1.1 B",
        "1.4 Explicit",
        "1.5 C",
        "2 No level",
    ]
    assert [title.level for title in titles] == [2, 3, 3, 3, 2]
    assert model_list == original


def test_office_paragraph_numbering_clears_deeper_levels() -> None:
    """验证标题返回浅层时会清理旧深层计数，后续重新从一开始编号。"""
    model_list = [
        [
            {"type": "paragraph_title", "content": "A", "level": 2, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "B", "level": 3, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "C", "level": 4, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "D", "level": 3, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "E", "level": 4, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "F", "level": 2, "is_numbered_style": True},
            {"type": "paragraph_title", "content": "G", "level": 3, "is_numbered_style": True},
        ]
    ]

    titles = model_json_to_pages(_model_json(model_list))[0].blocks

    assert [inline_text(title.content) for title in titles] == [
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
    page = model_json_to_pages(_model_json([[{"type": "image", "content": None, "sub_type": None}]]))[0]

    assert isinstance(page.blocks[0], ImageBlock)
    assert page.blocks[0].sub_type is None
    assert page.blocks[0].content[0].type == "image_body"
    assert page.blocks[0].content[0].content == ""


def test_chart_none_content_is_normalized_to_empty_string() -> None:
    """验证 raw chart 的 null content 在严格对象化前规范为空字符串。"""
    page = model_json_to_pages(_model_json([[{"type": "chart", "content": None}]]))[0]

    assert isinstance(page.blocks[0], ChartBlock)
    assert page.blocks[0].content[0].type == "chart_body"
    assert page.blocks[0].content[0].content == ""


def test_raw_title_levels_are_normalized_to_global_hierarchy() -> None:
    """验证 raw 标题在严格对象化前归一为全局一至六级层级。"""
    page = model_json_to_pages(
        _model_json(
            [
                [
                    {"type": "doc_title", "content": "Document"},
                    {"type": "paragraph_title", "content": "Section", "level": 1},
                    {"type": "paragraph_title", "content": "Deep", "level": 9},
                ]
            ]
        )
    )[0]

    assert [block.level for block in page.blocks] == [1, 2, 6]


def test_pdf_raw_paragraph_title_level_is_clamped_to_six() -> None:
    """验证带 bbox 的 PDF raw 深层标题在严格对象化前归一为六级。"""
    page = model_json_to_pages(
        _model_json([[{"type": "paragraph_title", "content": "Deep", "level": 9, "bbox": [0.1, 0.1, 0.9, 0.2]}]])
    )[0]

    assert isinstance(page.blocks[0], ParagraphTitleBlock)
    assert page.blocks[0].level == 6


def test_office_title_level_is_clamped_before_numbering_and_index_mapping() -> None:
    """验证 Office 深层标题在编号和目录映射前归一为六级。"""
    model_list = [
        [
            {
                "type": "paragraph_title",
                "content": "Deep",
                "level": 9,
                "anchor": "deep-anchor",
                "is_numbered_style": True,
            },
            {
                "type": "index",
                "content": [{"type": "text", "content": "Deep entry", "anchor": "deep-anchor"}],
            },
        ]
    ]

    page = model_json_to_pages(_model_json(model_list))[0]
    title = page.blocks[0]
    index = page.blocks[1]

    assert isinstance(title, ParagraphTitleBlock)
    assert title.level == 6
    assert inline_text(title.content) == "1.1.1.1.1 Deep"
    assert isinstance(index.content[0], ParagraphTitleBlock)
    assert index.content[0].level == 6


def test_pdf_continuation_is_typed_and_line_metadata_is_removed() -> None:
    """验证 raw 段落延续在对象化前完成，公开 TextBlock 不保留临时 lines。"""
    model_list = [
        [
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
        ]
    ]

    page = model_json_to_pages(_model_json(model_list))[0]

    assert all(isinstance(block, TextBlock) for block in page.blocks)
    assert page.blocks[0].continues_prev is None
    assert page.blocks[1].continues_prev is True
    assert all("lines" not in type(block).model_fields for block in page.blocks)


def test_pdf_ref_text_continuation_is_typed_and_line_metadata_is_removed() -> None:
    """验证 ref_text 续段标记越过 raw 边界，严格对象不保留临时 lines。"""
    model_list = [
        [
            {
                "type": "ref_text",
                "bbox": [0.1, 0.1, 0.9, 0.3],
                "content": "previous continuation",
                "lines": [{"bbox": [0.1, 0.1, 0.9, 0.15]}, {"bbox": [0.1, 0.2, 0.9, 0.25]}],
            },
            {
                "type": "ref_text",
                "bbox": [0.1, 0.25, 0.9, 0.45],
                "content": "current continuation",
                "lines": [{"bbox": [0.2, 0.25, 0.9, 0.3]}, {"bbox": [0.1, 0.35, 0.9, 0.4]}],
            },
        ]
    ]

    page = model_json_to_pages(_model_json(model_list))[0]

    assert all(isinstance(block, RefTextBlock) for block in page.blocks)
    assert page.blocks[0].continues_prev is None
    assert page.blocks[1].continues_prev is True
    assert all("lines" not in type(block).model_fields for block in page.blocks)


def test_pdf_detection_scans_past_empty_first_page() -> None:
    """验证整份文档 bbox 判定不会把 PDF 空白首页误认为 Office。"""
    pages = model_json_to_pages(
        _model_json(
            [
                [],
                [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.3], "content": "later", "lines": []}],
            ]
        )
    )

    assert pages[0].blocks == []
    assert isinstance(pages[1].blocks[0], TextBlock)


def test_cross_page_table_continuation_remains_raw_postprocess() -> None:
    """验证连续页表格仍在对象化前写入 continues_prev。"""
    html_a = "<table><tr><td>A</td><td>B</td></tr></table>"
    html_b = "<table><tr><td>C</td><td>D</td></tr></table>"
    pages = model_json_to_pages(
        _model_json(
            [
                [{"type": "table", "bbox": [0.1, 0.1, 0.9, 0.85], "content": html_a}],
                [{"type": "table", "bbox": [0.1, 0.1, 0.9, 0.85], "content": html_b}],
            ]
        )
    )

    assert isinstance(pages[0].blocks[0], TableBlock)
    assert pages[0].blocks[0].continues_prev is None
    assert pages[1].blocks[0].continues_prev is True
