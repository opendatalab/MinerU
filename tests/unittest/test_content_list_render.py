from __future__ import annotations

from copy import deepcopy

import pytest

from _span_test_utils import inline as _inline
from mineru.render import render_content_list
from mineru.types import (
    AlgorithmBodyBlock,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    ImageBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
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


def _assert_no_internal_fields(value: object) -> None:
    """递归确认兼容输出不泄漏 MiddleJson 内部消费提示。"""
    if isinstance(value, list):
        for item in value:
            _assert_no_internal_fields(item)
        return
    if not isinstance(value, dict):
        return
    assert not ({"index", "continues_prev", "cell_merge", "guess_lang", "image_base64"} & value.keys())
    for item in value.values():
        _assert_no_internal_fields(item)


def test_content_list_v1_flattens_pages_titles_lists_references_and_indices() -> None:
    """验证 V1 页序扁平化、标题层级、参考文献分组和目录链接。"""
    nested_list = ListBlock(
        type="list",
        content=[TextBlock(type="text", content=_inline("    - nested"))],
    )
    list_block = ListBlock(
        type="list",
        index=5,
        bbox=(0.1, 0.6, 0.9, 0.8),
        sub_type="text",
        content=[TextBlock(type="text", content=_inline("1. first")), nested_list],
    )
    index_block = IndexBlock(
        type="index",
        index=6,
        content=[
            ParagraphTitleBlock(
                type="paragraph_title",
                level=2,
                anchor="section-a",
                content=_inline("Section A\t12"),
            )
        ],
    )
    middle = _middle(
        _page(
            2,
            DocTitleBlock(
                type="doc_title",
                index=0,
                bbox=(0.1, 0.1, 0.9, 0.2),
                level=1,
                anchor="document-title",
                content=_inline("Document"),
            ),
            TextBlock(
                type="text",
                index=1,
                content=[
                    {"type": "text", "content": "A "},
                    {"type": "text", "content": "B", "styles": ["bold"]},
                    {"type": "text", "content": " "},
                    {"type": "hyperlink", "url": "https://example.com/a b", "content": _inline("link")},
                    {"type": "text", "content": " "},
                    {"type": "equation_inline", "content": "x"},
                ],
            ),
            RefTextBlock(type="ref_text", index=2, bbox=(0.1, 0.2, 0.4, 0.3), content=_inline("[1] first")),
            RefTextBlock(type="ref_text", index=3, bbox=(0.05, 0.4, 0.5, 0.6), content=_inline("[2] second")),
            PageAuxTextBlock(type="header", index=4, content=_inline("HEADER")),
            list_block,
            index_block,
        ),
        _page(9),
    )
    original = deepcopy(middle)

    output = render_content_list(middle)

    assert [item["type"] for item in output] == ["text", "text", "list", "header", "list", "index"]
    assert output[0] == {
        "type": "text",
        "text": "Document",
        "text_level": 1,
        "anchor": "document-title",
        "bbox": [100, 100, 900, 200],
        "page_idx": 2,
    }
    assert output[1]["text"] == "A **B** [link](https://example.com/a%20b) $x$"
    assert output[2] == {
        "type": "list",
        "sub_type": "ref_text",
        "list_items": ["[1] first", "[2] second"],
        "bbox": [50, 200, 500, 600],
        "page_idx": 2,
    }
    assert output[4]["list_items"] == ["1. first", "    - nested"]
    assert output[5]["list_items"] == ["- [Section A](#section-a)"]
    assert all(item["page_idx"] == 2 for item in output)
    assert middle == original
    _assert_no_internal_fields(output)


def test_content_list_v1_renders_equation_visuals_code_and_resource_priority() -> None:
    """验证 V1 视觉字段、说明排序、代码算法和图片来源优先级。"""
    image = ImageBlock.model_validate(
        {
            "type": "image",
            "index": 1,
            "bbox": [0.1, 0.2, 0.8, 0.6],
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
                    "content": '<table><tr><td><img src="cell.png"><eq>x</eq></td></tr></table>',
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
                image_url="https://example.com/chart.png",
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
            EquationBlock(
                type="equation",
                index=0,
                content="  x=1  ",
                image_path="images/e q.png",
                image_base64="data:image/png;base64,BBBB",
            ),
            image,
            table,
            chart,
            code,
            algorithm,
        )
    )

    output = render_content_list(middle, asset_base_url="https://cdn.example/doc")

    assert output[0] == {
        "type": "equation",
        "img_path": "https://cdn.example/doc/images/e%20q.png",
        "text": "x=1",
        "text_format": "latex",
        "page_idx": 0,
    }
    assert output[1]["img_path"] == "https://cdn.example/doc/images/a%20b.png"
    assert output[1]["content"] == "<p>description</p>"
    assert output[1]["image_caption"] == ["**early**", "late"]
    assert output[1]["image_footnote"] == ["source"]
    assert output[1]["sub_type"] == "diagram"
    assert 'src="https://cdn.example/doc/cell.png"' in output[2]["table_body"]
    assert "$x$" in output[2]["table_body"]
    assert output[2]["img_path"] == "https://cdn.example/doc/images/table.png"
    assert output[3]["img_path"] == "https://example.com/chart.png"
    assert output[3]["sub_type"] == "bar"
    assert output[4]["code_body"] == "print('x')"
    assert output[4]["code_caption"] == ["Example"]
    assert output[5]["sub_type"] == "algorithm"
    assert output[5]["code_body"] == "for $n$"
    _assert_no_internal_fields(output)


def test_content_list_v1_handles_empty_input_and_rejects_legacy_arguments() -> None:
    """验证空文档、严格 MiddleJson 输入和资源参数类型。"""
    middle = _middle()

    assert render_content_list(middle) == []
    with pytest.raises(TypeError, match="MiddleJson"):
        render_content_list(middle.to_dict())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="asset_base_url"):
        render_content_list(middle, asset_base_url=1)  # type: ignore[arg-type]
