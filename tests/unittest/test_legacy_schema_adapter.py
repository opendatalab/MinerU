from copy import deepcopy

import pytest

from mineru.backend.postprocess.legacy_schema_adapter import legacy_page_to_model_list
from mineru.backend.postprocess.pages import model_json_to_pages
from mineru.doclib.background.compaction import _normalize_batch_pages
from mineru.types import BlockType, ModelJson


def _legacy_line(content: str, *, bbox: list[int] | None = None) -> dict:
    """构造 3.4.5 普通文字行。"""
    return {
        "bbox": bbox or [10, 10, 90, 20],
        "spans": [{"type": "text", "content": content}],
    }


def _current_pages(raw_pages: list[list[dict]]) -> list:
    """把适配后的 raw pages 送入当前严格后处理。"""
    model_json = ModelJson(
        pages=raw_pages,
        page_index_map=[],
        file_suffix="pdf",
        effort="medium",
        parse_mode="txt",
        mineru_version="3.4.5",
    )
    return model_json_to_pages(model_json)


def test_legacy_adapter_converts_inline_spans_without_mutating_input() -> None:
    """验证文字标签、公式、链接及危险链接降级，并保证输入不变。"""
    page = {
        "page_idx": 0,
        "page_size": [100, 100],
        "preproc_blocks": [
            {
                "type": "text",
                "bbox": [10, 10, 90, 30],
                "lines": [
                    {
                        "bbox": [10, 10, 90, 30],
                        "spans": [
                            {
                                "type": "text",
                                "content": (
                                    'plain <text style="bold,italic">styled</text> '
                                    "<eq>x^2</eq> "
                                    "<hyperlink><text>safe</text><url>https://example.com</url></hyperlink> "
                                    "<hyperlink><text>unsafe</text><url>javascript:alert(1)</url></hyperlink>"
                                ),
                            }
                        ],
                    }
                ],
            }
        ],
        "discarded_blocks": [],
    }
    original = deepcopy(page)

    raw_page = legacy_page_to_model_list(page)
    content = raw_page[0]["content"]

    assert page == original
    assert content == [
        {"type": "text", "content": "plain ", "styles": []},
        {"type": "text", "content": "styled", "styles": ["bold", "italic"]},
        {"type": "text", "content": " ", "styles": []},
        {"type": "equation_inline", "content": "x^2"},
        {"type": "text", "content": " ", "styles": []},
        {
            "type": "hyperlink",
            "url": "https://example.com",
            "content": [{"type": "text", "content": "safe", "styles": []}],
        },
        {"type": "text", "content": " unsafe", "styles": []},
    ]
    pages = _current_pages([raw_page])
    assert pages[0].blocks[0].type == BlockType.TEXT


def test_legacy_adapter_preserves_visual_metadata_and_algorithm_spans() -> None:
    """验证视觉父块展平后能由当前后处理恢复表格与算法块。"""
    page = {
        "page_idx": 0,
        "page_size": [100, 100],
        "para_blocks": [
            {
                "type": "table",
                "bbox": [10, 10, 90, 60],
                "cell_merge": [0, 1],
                "blocks": [
                    {
                        "type": "table_caption",
                        "bbox": [10, 10, 90, 20],
                        "lines": [_legacy_line("表 1")],
                    },
                    {
                        "type": "table_body",
                        "bbox": [10, 20, 90, 60],
                        "lines": [
                            {
                                "bbox": [10, 20, 90, 60],
                                "spans": [
                                    {
                                        "type": "table",
                                        "html": "<table><tr><td>x</td></tr></table>",
                                        "image_path": "images/table.png",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "type": "code",
                "sub_type": "algorithm",
                "bbox": [10, 70, 90, 90],
                "blocks": [
                    {
                        "type": "code_body",
                        "bbox": [10, 70, 90, 90],
                        "lines": [
                            {
                                "bbox": [10, 70, 90, 90],
                                "spans": [
                                    {"type": "text", "content": "a="},
                                    {"type": "inline_equation", "content": "x"},
                                ],
                            }
                        ],
                    }
                ],
            },
        ],
        "discarded_blocks": [{"type": "header", "bbox": [10, 1, 90, 5], "lines": [_legacy_line("Header")]}],
    }

    pages = _current_pages([legacy_page_to_model_list(page)])
    table, code, header = pages[0].blocks

    assert table.type == BlockType.TABLE
    assert table.cell_merge == [0, 1]
    assert table.content[1].image_path == "images/table.png"
    assert code.type == BlockType.CODE
    assert code.sub_type == "algorithm"
    assert code.content[0].type == BlockType.ALGORITHM_BODY
    assert [span.type for span in code.content[0].content] == ["text", "equation_inline"]
    assert header.type == BlockType.HEADER


def test_legacy_adapter_normalizes_bbox_and_block_aliases() -> None:
    """验证像素坐标、越界坐标及旧 discriminator 均转换为当前合法值。"""
    page = {
        "page_size": [200, 100],
        "preproc_blocks": [
            {
                "type": "title",
                "level": 1,
                "bbox": [-10, 0, 220, 20],
                "lines": [_legacy_line("Document")],
            },
            {
                "type": "interline_equation",
                "bbox": [0, 30, 100, 50],
                "lines": [
                    {
                        "bbox": [0, 30, 100, 50],
                        "spans": [{"type": "interline_equation", "content": "y=1"}],
                    }
                ],
            },
        ],
        "discarded_blocks": [],
    }

    raw_page = legacy_page_to_model_list(page)

    assert raw_page[0]["type"] == "doc_title"
    assert raw_page[0]["bbox"] == (0.0, 0.0, 1.0, 0.2)
    assert raw_page[1]["type"] == "equation"
    assert raw_page[1]["content"] == "y=1"
    _current_pages([raw_page])


def test_legacy_adapter_recovers_pdf_list_items_from_line_flags() -> None:
    """验证旧 PDF list 的物理行重新生成当前列表文本子块。"""
    page = {
        "page_size": [100, 100],
        "preproc_blocks": [
            {
                "type": "list",
                "bbox": [10, 10, 90, 40],
                "lines": [
                    {
                        "bbox": [10, 10, 90, 20],
                        "is_list_start_line": True,
                        "spans": [{"type": "text", "content": "one"}],
                    },
                    {
                        "bbox": [10, 25, 90, 35],
                        "is_list_start_line": True,
                        "spans": [{"type": "text", "content": "two"}],
                    },
                ],
            }
        ],
        "discarded_blocks": [],
    }

    pages = _current_pages([legacy_page_to_model_list(page)])
    list_block = pages[0].blocks[0]

    assert list_block.type == BlockType.LIST
    assert [child.content[0].content for child in list_block.content] == ["one", "two"]


def test_legacy_adapter_separates_office_blocks_with_synthetic_bbox_groups() -> None:
    """验证无 page_size 时列表不会吸收相邻 Office 正文块。"""
    page = {
        "para_blocks": [
            {"type": "text", "lines": [{"spans": [{"type": "text", "content": "before"}]}]},
            {
                "type": "list",
                "blocks": [
                    {"type": "text", "lines": [{"spans": [{"type": "text", "content": "item"}]}]},
                ],
            },
            {"type": "text", "lines": [{"spans": [{"type": "text", "content": "after"}]}]},
        ],
        "discarded_blocks": [],
    }

    pages = _current_pages([legacy_page_to_model_list(page)])

    assert [block.type for block in pages[0].blocks] == [BlockType.TEXT, BlockType.LIST, BlockType.TEXT]
    assert pages[0].blocks[0].content[0].content == "before"
    assert pages[0].blocks[1].content[0].content[0].content == "item"
    assert pages[0].blocks[2].content[0].content == "after"


def test_doclib_compaction_converts_schema_v1_pages_with_current_adapter() -> None:
    """验证 Doclib 历史调用点重新通过适配器生成 3.0 page 字典。"""
    payload = {
        "schema_version": "1.0",
        "pages": [
            {
                "page_idx": 3,
                "page_size": [100, 100],
                "preproc_blocks": [
                    {
                        "type": "text",
                        "bbox": [10, 10, 90, 20],
                        "lines": [_legacy_line("cached")],
                    }
                ],
                "discarded_blocks": [],
            }
        ],
    }

    pages = _normalize_batch_pages(payload)

    assert pages[0]["page_idx"] == 3
    assert pages[0]["blocks"][0]["content"] == [{"type": "text", "content": "cached"}]


def test_doclib_compaction_rejects_unknown_legacy_schema() -> None:
    """验证 2.0 等未支持旧 schema 不会被误当成 3.4.5 页面吞掉。"""
    with pytest.raises(ValueError, match="source reparse"):
        _normalize_batch_pages({"schema_version": "2.0", "pages": [{"page_idx": 0, "blocks": []}]})
