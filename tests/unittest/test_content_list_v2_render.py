from __future__ import annotations

from copy import deepcopy

import pytest

from _span_test_utils import inline as _inline
from mineru.render import render_content_list_v2
from mineru.types import (
    AlgorithmBodyBlock,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
    CodeBodyBlock,
    EquationBlock,
    ImageBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageFootnoteBlock,
    PageInfo,
    ParagraphTitleBlock,
    RefTextBlock,
    TableBlock,
    TextBlock,
)


def _middle(*pages: PageInfo) -> MiddleJson:
    """构造允许可选 bbox 的最小严格 MiddleJson。"""
    return MiddleJson(
        pages=list(pages),
        is_full_document=True,
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _page(page_idx: int, *blocks: object) -> PageInfo:
    """按调用方顺序构造测试页面。"""
    return PageInfo(page_idx=page_idx, blocks=list(blocks))  # type: ignore[arg-type]


def test_content_list_v2_preserves_pages_bbox_titles_aux_and_rich_spans() -> None:
    """验证 V2 按页结构、空页、标题、辅助块和规范 span。"""
    title = ParagraphTitleBlock(
        type="paragraph_title",
        index=0,
        bbox=(0.1, 0.1, 0.9, 0.2),
        level=6,
        anchor="section-a",
        content=[
            {"type": "text", "content": "Styled", "styles": ["bold"]},
            {"type": "text", "content": " "},
            {
                "type": "hyperlink",
                "url": "https://example.com/a",
                "content": _inline("single", styles=["underline"]),
            },
            {"type": "text", "content": " "},
            {
                "type": "hyperlink",
                "url": "#target",
                "content": [
                    {"type": "text", "content": "mix", "styles": ["italic"]},
                    {"type": "equation_inline", "content": "x"},
                    {"type": "code_inline", "content": "code"},
                ],
            },
        ],
    )
    middle = _middle(
        _page(
            3,
            title,
            PageAuxTextBlock(type="header", index=1, content=_inline("HEADER")),
            PageAuxTextBlock(type="footer", index=2, content=_inline("FOOTER")),
            PageAuxTextBlock(type="page_number", index=3, content=_inline("3")),
            PageAuxTextBlock(type="aside_text", index=4, content=_inline("ASIDE")),
            PageFootnoteBlock(type="page_footnote", index=5, anchor="note-one", content=_inline("note")),
        ),
        _page(8),
    )
    original = deepcopy(middle)

    output = render_content_list_v2(middle)

    assert len(output) == 2 and output[1] == []
    assert output[0][0]["type"] == "title"
    assert output[0][0]["content"]["level"] == 6
    assert output[0][0]["anchor"] == "section-a"
    assert output[0][0]["bbox"] == [100, 100, 900, 200]
    spans = output[0][0]["content"]["title_content"]
    assert spans[0] == {"type": "text", "content": "Styled", "style": ["bold"]}
    assert spans[2] == {
        "type": "hyperlink",
        "content": "single",
        "url": "https://example.com/a",
        "style": ["underline"],
    }
    assert spans[4] == {
        "type": "hyperlink",
        "content": "mixxcode",
        "url": "#target",
        "children": [
            {"type": "text", "content": "mix", "style": ["italic"]},
            {"type": "equation_inline", "content": "x"},
            {"type": "code_inline", "content": "code"},
        ],
    }
    assert [item["type"] for item in output[0][1:]] == [
        "page_header",
        "page_footer",
        "page_number",
        "page_aside_text",
        "page_footnote",
    ]
    assert output[0][-1]["anchor"] == "note-one"
    assert middle == original


def test_content_list_v2_normalizes_references_lists_and_indices() -> None:
    """验证连续参考文献、列表属性推断和目录 anchor。"""
    ordered_list = ListBlock(
        type="list",
        index=2,
        sub_type="text",
        content=[
            TextBlock(type="text", content=_inline("1. first")),
            TextBlock(type="text", content=_inline("2. second")),
            ListBlock(type="list", content=[TextBlock(type="text", content=_inline("    - nested"))]),
        ],
    )
    index = IndexBlock(
        type="index",
        index=3,
        content=[
            ParagraphTitleBlock(
                type="paragraph_title",
                level=2,
                anchor="section-b",
                content=_inline("Section B\tIV", styles=["bold"]),
            )
        ],
    )
    middle = _middle(
        _page(
            0,
            RefTextBlock(type="ref_text", index=0, bbox=(0.1, 0.2, 0.4, 0.3), content=_inline("[1] first")),
            RefTextBlock(type="ref_text", index=1, bbox=(0.05, 0.4, 0.5, 0.6), content=_inline("[2] second")),
            ordered_list,
            index,
        )
    )

    page = render_content_list_v2(middle)[0]

    assert page[0] == {
        "type": "list",
        "content": {
            "list_type": "reference_list",
            "list_items": [
                {"item_type": "text", "item_content": [{"type": "text", "content": "[1] first"}]},
                {"item_type": "text", "item_content": [{"type": "text", "content": "[2] second"}]},
            ],
        },
        "bbox": [50, 200, 500, 600],
    }
    assert page[1]["content"]["list_type"] == "text_list"
    assert page[1]["content"]["attribute"] == "ordered"
    assert [item["item_content"][0]["content"] for item in page[1]["content"]["list_items"]] == [
        "1. first",
        "2. second",
        "    - nested",
    ]
    assert page[2] == {
        "type": "index",
        "content": {
            "list_type": "text_list",
            "list_items": [
                {
                    "item_type": "text",
                    "item_content": [{"type": "text", "content": "Section B", "style": ["bold"]}],
                    "anchor": "section-b",
                }
            ],
        },
    }


def test_content_list_v2_renders_equation_visuals_tables_code_and_algorithm() -> None:
    """验证 V2 视觉资源、表格分类、说明 span 以及代码算法结构。"""
    image = ImageBlock.model_validate(
        {
            "type": "image",
            "index": 1,
            "sub_type": "diagram",
            "content": [
                {"type": "image_caption", "index": 8, "content": _inline("late")},
                {
                    "type": "image_body",
                    "index": 1,
                    "content": "<p>description</p>",
                    "image_path": "images/a b.png",
                    "image_base64": "data:image/png;base64,AAAA",
                },
                {"type": "image_caption", "index": 6, "content": _inline("early", styles=["bold"])},
                {"type": "image_footnote", "index": 7, "content": _inline("source")},
            ],
        }
    )
    table = TableBlock.model_validate(
        {
            "type": "table",
            "index": 2,
            "content": [
                {
                    "type": "table_body",
                    "index": 2,
                    "content": (
                        '<table><tr><td rowspan="2"><img src="cell.png"><eq>x</eq>'
                        "<table><tr><td>nested</td></tr></table></td></tr></table>"
                    ),
                    "image_path": "images/table.png",
                },
                {"type": "table_caption", "content": _inline("Table 1")},
                {"type": "table_footnote", "content": _inline("note")},
            ],
        }
    )
    chart = ChartBlock(
        type="chart",
        index=3,
        sub_type="bar",
        content=[
            ChartBodyBlock(
                type="chart_body",
                index=3,
                content="<div>chart</div>",
                image_base64="data:image/png;base64,BBBB",
            ),
            ChartAnnotationBlock(type="chart_caption", content=_inline("Chart 1")),
            ChartAnnotationBlock(type="chart_footnote", content=_inline("chart note")),
        ],
    )
    code = CodeBlock(
        type="code",
        index=4,
        sub_type="code",
        guess_lang="python",
        content=[
            CodeBodyBlock(type="code_body", index=4, content="print('x')"),
            {"type": "code_caption", "content": _inline("Example")},
            {"type": "code_footnote", "content": _inline("code note")},
        ],
    )
    algorithm = CodeBlock(
        type="code",
        index=5,
        sub_type="algorithm",
        content=[
            AlgorithmBodyBlock(
                type="algorithm_body",
                index=5,
                content=[{"type": "text", "content": "for "}, {"type": "equation_inline", "content": "n"}],
            )
        ],
    )
    middle = _middle(
        _page(
            0,
            EquationBlock(type="equation", index=0, content=" x=1 "),
            image,
            table,
            chart,
            code,
            algorithm,
        )
    )

    page = render_content_list_v2(middle, asset_base_url="https://cdn.example/doc")[0]

    assert page[0]["content"] == {
        "math_content": "x=1",
        "math_type": "latex",
        "image_source": {"path": ""},
    }
    assert page[1]["sub_type"] == "diagram"
    assert page[1]["content"]["image_source"] == {"path": "https://cdn.example/doc/images/a%20b.png"}
    assert page[1]["content"]["image_caption"] == [
        {"type": "text", "content": "early", "style": ["bold"]},
        {"type": "text", "content": "late"},
    ]
    assert page[1]["content"]["image_footnote"] == [{"type": "text", "content": "source"}]
    assert page[2]["content"]["table_type"] == "complex_table"
    assert page[2]["content"]["table_nest_level"] == 2
    assert 'src="https://cdn.example/doc/cell.png"' in page[2]["content"]["html"]
    assert "$x$" in page[2]["content"]["html"]
    assert page[3]["content"]["image_source"] == {"path": "data:image/png;base64,BBBB"}
    assert page[4]["content"]["code_content"] == [{"type": "text", "content": "print('x')"}]
    assert page[4]["content"]["code_language"] == "python"
    assert page[5] == {
        "type": "algorithm",
        "content": {
            "algorithm_caption": [],
            "algorithm_content": [
                {"type": "text", "content": "for "},
                {"type": "equation_inline", "content": "n"},
            ],
            "algorithm_footnote": [],
        },
    }


def test_content_list_v2_handles_empty_input_and_rejects_legacy_arguments() -> None:
    """验证空文档、严格 MiddleJson 输入和资源参数类型。"""
    middle = _middle()

    assert render_content_list_v2(middle) == []
    with pytest.raises(TypeError, match="MiddleJson"):
        render_content_list_v2(middle.to_dict())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="asset_base_url"):
        render_content_list_v2(middle, asset_base_url=1)  # type: ignore[arg-type]
