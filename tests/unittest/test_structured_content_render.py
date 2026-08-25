from __future__ import annotations

import json
from copy import deepcopy

import pytest

from mineru.config import Config
from mineru.render import render_markdown, render_structured_content
from mineru.types import (
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
    EquationBlock,
    ImageBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageBlock,
    PageInfo,
    ParagraphTitleBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
)


def _middle(*pages: PageInfo, file_suffix: str = "docx") -> MiddleJson:
    """构造最小严格 MiddleJson 测试对象。"""
    return MiddleJson(
        pages=list(pages),
        is_full_document=True,
        file_suffix=file_suffix,
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _page(page_idx: int, *blocks: PageBlock) -> PageInfo:
    """构造保持调用方 block 顺序的严格页面。"""
    return PageInfo(page_idx=page_idx, blocks=list(blocks))


def _assert_output_field_contract(value: object) -> None:
    """递归确认输出只在标题保留 level，且不暴露内部或重复图片字段。"""
    if isinstance(value, list):
        for item in value:
            _assert_output_field_contract(item)
        return
    if not isinstance(value, dict):
        return
    assert not ({"index", "guess_lang", "image_path", "image_base64"} & value.keys())
    if "level" in value:
        assert value.get("type") in {"doc_title", "paragraph_title"}
    for item in value.values():
        _assert_output_field_contract(item)


def test_structured_content_preserves_document_tree_without_merging_or_mutation() -> None:
    """验证文档树、辅助块和续段字段原样保留，且输入对象不被修改。"""
    middle = _middle(
        _page(
            0,
            ParagraphTitleBlock(
                type="paragraph_title",
                index=0,
                level=6,
                anchor="section-a",
                content='# <text style="bold">Section</text> <eq>x</eq>',
            ),
            TextBlock(type="text", index=1, content="first-"),
            PageAuxTextBlock(type="header", index=2, content="HEADER"),
        ),
        _page(
            1,
            TextBlock(type="text", index=0, content="continued", continues_prev=True),
        ),
    )
    original = deepcopy(middle)

    result = render_structured_content(middle)

    assert json.loads(json.dumps(result, ensure_ascii=False)) == result
    assert result["file_suffix"] == "docx"
    assert result["effort"] == "flash"
    assert [page["page_idx"] for page in result["pages"]] == [0, 1]
    assert [block["type"] for block in result["pages"][0]["blocks"]] == [
        "paragraph_title",
        "text",
        "header",
    ]
    assert result["pages"][0]["blocks"][0] == {
        "type": "paragraph_title",
        "anchor": "section-a",
        "level": 6,
        "content": r"\# **Section** $x$",
    }
    assert render_markdown(middle).startswith('<a id="section-a"></a>\n###### # **Section** $x$')
    assert result["pages"][0]["blocks"][1]["content"] == "first-"
    assert result["pages"][0]["blocks"][2]["content"] == "HEADER"
    assert result["pages"][1]["blocks"][0]["content"] == "continued"
    assert result["pages"][1]["blocks"][0]["continues_prev"] is True
    _assert_output_field_contract(result)
    assert middle == original


def test_structured_content_flattens_recursive_list_index_and_code_metadata() -> None:
    """验证递归容器和代码 body 上浮为 Markdown，渲染元数据不进入输出。"""
    nested = ListBlock(
        type="list",
        content=[TextBlock(type="text", content="- inner")],
    )
    list_block = ListBlock(
        type="list",
        index=0,
        sub_type="text",
        content=[TextBlock(type="text", content="- outer"), nested],
    )
    index_block = IndexBlock(
        type="index",
        index=1,
        content=[
            ParagraphTitleBlock(
                type="paragraph_title",
                level=2,
                anchor="section-b",
                content="Section\t12",
            )
        ],
    )
    code = CodeBlock.model_validate(
        {
            "type": "code",
            "index": 2,
            "sub_type": "code",
            "guess_lang": "python",
            "content": [
                {
                    "type": "code_caption",
                    "index": 4,
                    "bbox": [0.1, 0.1, 0.5, 0.2],
                    "content": '<text style="bold">Example</text>',
                },
                {"type": "code_body", "index": 2, "content": "print('x')\n```"},
                {
                    "type": "code_footnote",
                    "index": 5,
                    "bbox": [0.1, 0.8, 0.5, 0.9],
                    "content": "note <eq>x</eq>",
                },
            ],
        }
    )

    blocks = render_structured_content(_middle(_page(0, list_block, index_block, code)))["pages"][0]["blocks"]

    assert blocks[0]["content"] == "- outer\n    - inner"
    assert blocks[1]["content"] == "- [Section](#section-b)"
    assert blocks[2]["content"] == "````python\nprint('x')\n```\n````"
    assert blocks[2]["captions"] == [{"bbox": [0.1, 0.1, 0.5, 0.2], "content": "**Example**"}]
    assert blocks[2]["footnotes"] == [{"bbox": [0.1, 0.8, 0.5, 0.9], "content": "note $x$"}]
    assert "guess_lang" not in blocks[2]
    _assert_output_field_contract(blocks)


def test_structured_content_sorts_visual_annotations_and_selects_image_source() -> None:
    """验证视觉说明稳定排序、空项保留及图片 path 优先规则。"""
    image = ImageBlock.model_validate(
        {
            "type": "image",
            "index": 0,
            "sub_type": "diagram",
            "content": [
                {"type": "image_caption", "content": "missing index"},
                {
                    "type": "image_body",
                    "index": 0,
                    "content": "description",
                    "image_path": "images/a b.png",
                    "image_base64": "data:image/png;base64,AAAA",
                },
                {
                    "type": "image_caption",
                    "index": 5,
                    "bbox": [0.1, 0.5, 0.4, 0.6],
                    "content": "",
                },
                {
                    "type": "image_caption",
                    "index": 3,
                    "bbox": [0.1, 0.2, 0.4, 0.3],
                    "content": '<text style="bold">early</text>',
                },
                {
                    "type": "image_footnote",
                    "index": 4,
                    "bbox": [0.1, 0.7, 0.4, 0.8],
                    "content": "after",
                },
            ],
        }
    )

    result = render_structured_content(
        _middle(_page(0, image)),
        asset_base_url="https://cdn.example/doc",
    )["pages"][0]["blocks"][0]

    assert result["content"] == "description"
    assert result["image_source"] == "https://cdn.example/doc/images/a%20b.png"
    assert result["captions"] == [
        {"bbox": [0.1, 0.2, 0.4, 0.3], "content": "**early**"},
        {"bbox": [0.1, 0.5, 0.4, 0.6], "content": ""},
        {"content": "missing index"},
    ]
    assert result["footnotes"] == [{"bbox": [0.1, 0.7, 0.4, 0.8], "content": "after"}]
    assert "![](" not in result["content"]
    assert "<details>" not in result["content"]
    assert "data:image" not in json.dumps(result)
    _assert_output_field_contract(result)


def test_structured_content_keeps_table_image_source_and_cross_page_metadata() -> None:
    """验证表格 body 上浮后仍保留图片来源、续表字段且不执行跨页合并。"""
    first = TableBlock(
        type="table",
        index=0,
        content=[
            TableBodyBlock(
                type="table_body",
                index=0,
                content="<table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
                image_path="images/table.png",
            )
        ],
    )
    second = TableBlock.model_validate(
        {
            "type": "table",
            "index": 0,
            "continues_prev": True,
            "cell_merge": [1],
            "content": [
                {
                    "type": "table_body",
                    "index": 0,
                    "content": "<table><tr><th>A</th></tr><tr><td>2</td></tr></table>",
                },
                {
                    "type": "table_caption",
                    "index": 2,
                    "bbox": [0.1, 0.1, 0.9, 0.2],
                    "content": "Table 2",
                },
                {
                    "type": "table_footnote",
                    "index": 3,
                    "bbox": [0.1, 0.8, 0.9, 0.9],
                    "content": "note",
                },
            ],
        }
    )
    image_only = TableBlock(
        type="table",
        index=0,
        content=[
            TableBodyBlock(
                type="table_body",
                index=0,
                content="",
                image_path="images/image-only-table.png",
            )
        ],
    )

    result = render_structured_content(_middle(_page(0, first), _page(1, second), _page(2, image_only)))
    first_output = result["pages"][0]["blocks"][0]
    second_output = result["pages"][1]["blocks"][0]
    image_only_output = result["pages"][2]["blocks"][0]

    assert first_output["content"] == "| A |\n| --- |\n| 1 |"
    assert first_output["image_source"] == "images/table.png"
    assert first_output["captions"] == []
    assert first_output["footnotes"] == []
    assert second_output["content"] == "| A |\n| --- |\n| 2 |"
    assert second_output["continues_prev"] is True
    assert second_output["cell_merge"] == [1]
    assert second_output["captions"] == [{"bbox": [0.1, 0.1, 0.9, 0.2], "content": "Table 2"}]
    assert second_output["footnotes"] == [{"bbox": [0.1, 0.8, 0.9, 0.9], "content": "note"}]
    assert image_only_output["content"] == ""
    assert image_only_output["image_source"] == "images/image-only-table.png"


def test_structured_content_keeps_chart_content_separate_from_base64_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 chart 只在 image_source 保存 base64，结构内容继续复用行内公式配置。"""
    configured = Config(
        render={
            "latex_delimiters": {
                "display": {"left": "\\[", "right": "\\]"},
                "inline": {"left": "\\(", "right": "\\)"},
            }
        }
    )
    monkeypatch.setattr("mineru.render._internal.structured_content.renderer.config", configured)
    chart = ChartBlock(
        type="chart",
        index=0,
        content=[
            ChartAnnotationBlock(
                type="chart_caption",
                bbox=(0.1, 0.1, 0.9, 0.2),
                content='<text style="bold">Chart</text>',
            ),
            ChartBodyBlock(
                type="chart_body",
                index=0,
                content="<table><tr><th>A</th></tr><tr><td><eq>x</eq></td></tr></table>",
                image_base64="data:image/png;base64,AAAA",
            ),
            ChartAnnotationBlock(
                type="chart_footnote",
                content="source",
            ),
        ],
    )

    output = render_structured_content(_middle(_page(0, chart)))["pages"][0]["blocks"][0]

    assert output["image_source"] == "data:image/png;base64,AAAA"
    assert output["content"] == "| A |\n| --- |\n| \\(x\\) |"
    assert output["captions"] == [{"bbox": [0.1, 0.1, 0.9, 0.2], "content": "**Chart**"}]
    assert output["footnotes"] == [{"content": "source"}]
    assert json.dumps(output).count("data:image/png;base64,AAAA") == 1
    assert "![](" not in output["content"]
    assert "<details>" not in output["content"]


def test_structured_content_renders_equation_as_raw_latex_with_single_image_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 equation 不加行间定界符，并把选中的图片载荷唯一提升为 image_source。"""
    configured = Config(
        render={
            "latex_delimiters": {
                "display": {"left": "\\[", "right": "\\]"},
                "inline": {"left": "\\(", "right": "\\)"},
            }
        }
    )
    monkeypatch.setattr("mineru.render._internal.structured_content.renderer.config", configured)
    middle = _middle(
        _page(
            0,
            EquationBlock(type="equation", index=0, content="  x=1  "),
            EquationBlock(
                type="equation",
                index=1,
                content="",
                image_base64="data:image/png;base64,BBBB",
            ),
            EquationBlock(
                type="equation",
                index=2,
                content="y=2",
                image_base64="data:image/png;base64,CCCC",
            ),
            EquationBlock(
                type="equation",
                index=3,
                content="z=3",
                image_path="images/e q.png",
                image_base64="data:image/png;base64,DDDD",
            ),
        )
    )

    blocks = render_structured_content(middle, asset_base_url="https://cdn.example/doc")["pages"][0]["blocks"]

    assert blocks[0] == {"type": "equation", "content": "x=1"}
    assert blocks[1] == {
        "type": "equation",
        "content": "",
        "image_source": "data:image/png;base64,BBBB",
    }
    assert blocks[2] == {
        "type": "equation",
        "content": "y=2",
        "image_source": "data:image/png;base64,CCCC",
    }
    assert blocks[3] == {
        "type": "equation",
        "content": "z=3",
        "image_source": "https://cdn.example/doc/images/e%20q.png",
    }
    serialized = json.dumps(blocks)
    assert serialized.count("data:image/png;base64,BBBB") == 1
    assert serialized.count("data:image/png;base64,CCCC") == 1
    assert "data:image/png;base64,DDDD" not in serialized
    assert "\\[" not in serialized and "\\]" not in serialized
    _assert_output_field_contract(blocks)


def test_structured_content_rejects_legacy_dict_input() -> None:
    """验证公共入口只接受严格 MiddleJson。"""
    middle = _middle(_page(0))

    with pytest.raises(TypeError, match="MiddleJson"):
        render_structured_content(middle.to_dict())  # type: ignore[arg-type]
