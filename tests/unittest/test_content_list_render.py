from __future__ import annotations

import json
from copy import deepcopy

import pytest

from mineru.config import Config
from mineru.render import render_content_list
from mineru.types import (
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
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
        file_suffix=file_suffix,
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _page(page_idx: int, *blocks: PageBlock) -> PageInfo:
    """构造保持调用方 block 顺序的严格页面。"""
    return PageInfo(page_idx=page_idx, blocks=list(blocks))


def _assert_removed_fields_absent(value: object) -> None:
    """递归确认 content_list 没有暴露被移除的 block 字段。"""
    if isinstance(value, list):
        for item in value:
            _assert_removed_fields_absent(item)
        return
    if not isinstance(value, dict):
        return
    assert not ({"index", "level", "guess_lang"} & value.keys())
    for item in value.values():
        _assert_removed_fields_absent(item)


def test_content_list_preserves_document_tree_without_merging_or_mutation() -> None:
    """验证文档树、辅助块和续段字段原样保留，且输入对象不被修改。"""
    middle = _middle(
        _page(
            0,
            ParagraphTitleBlock(
                type="paragraph_title",
                index=0,
                level=2,
                anchor="section-a",
                content="Section",
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

    result = render_content_list(middle)

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
        "content": '<a id="section-a"></a>\n## Section',
    }
    assert result["pages"][0]["blocks"][1]["content"] == "first-"
    assert result["pages"][0]["blocks"][2]["content"] == "HEADER"
    assert result["pages"][1]["blocks"][0]["content"] == "continued"
    assert result["pages"][1]["blocks"][0]["continues_prev"] is True
    _assert_removed_fields_absent(result)
    assert middle == original


def test_content_list_flattens_recursive_list_index_and_code_metadata() -> None:
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
                {"type": "code_caption", "index": 4, "content": '<text style="bold">Example</text>'},
                {"type": "code_body", "index": 2, "content": "print('x')\n```"},
                {"type": "code_footnote", "index": 5, "content": "note <eq>x</eq>"},
            ],
        }
    )

    blocks = render_content_list(_middle(_page(0, list_block, index_block, code)))["pages"][0]["blocks"]

    assert blocks[0]["content"] == "- outer\n    - inner"
    assert blocks[1]["content"] == "- [Section](#section-b)"
    assert blocks[2]["content"] == "````python\nprint('x')\n```\n````"
    assert blocks[2]["captions"] == ["**Example**"]
    assert blocks[2]["footnotes"] == ["note $x$"]
    assert "guess_lang" not in blocks[2]
    _assert_removed_fields_absent(blocks)


def test_content_list_sorts_visual_annotations_and_selects_image_source() -> None:
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
                {"type": "image_caption", "index": 5, "content": ""},
                {
                    "type": "image_caption",
                    "index": 3,
                    "content": '<text style="bold">early</text>',
                },
                {"type": "image_footnote", "index": 4, "content": "after"},
            ],
        }
    )

    result = render_content_list(
        _middle(_page(0, image)),
        asset_base_url="https://cdn.example/doc",
    )["pages"][0]["blocks"][0]

    assert result["content"].startswith("![](https://cdn.example/doc/images/a%20b.png)\n\n<details>")
    assert "description" in result["content"]
    assert result["image_source"] == "https://cdn.example/doc/images/a%20b.png"
    assert result["captions"] == ["**early**", "", "missing index"]
    assert result["footnotes"] == ["after"]
    assert "data:image" not in result["content"]
    _assert_removed_fields_absent(result)


def test_content_list_keeps_table_image_source_and_cross_page_metadata() -> None:
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
                {"type": "table_caption", "index": 2, "content": "Table 2"},
                {"type": "table_footnote", "index": 3, "content": "note"},
            ],
        }
    )

    result = render_content_list(_middle(_page(0, first), _page(1, second)))
    first_output = result["pages"][0]["blocks"][0]
    second_output = result["pages"][1]["blocks"][0]

    assert first_output["content"] == "| A |\n| --- |\n| 1 |"
    assert first_output["image_source"] == "images/table.png"
    assert first_output["captions"] == []
    assert first_output["footnotes"] == []
    assert second_output["content"] == "| A |\n| --- |\n| 2 |"
    assert second_output["continues_prev"] is True
    assert second_output["cell_merge"] == [1]
    assert second_output["captions"] == ["Table 2"]
    assert second_output["footnotes"] == ["note"]


def test_content_list_uses_render_config_and_base64_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 content_list 复用公式配置，并在无路径时选择 base64 图片来源。"""
    configured = Config(
        render={
            "latex_delimiters": {
                "display": {"left": "\\[", "right": "\\]"},
                "inline": {"left": "\\(", "right": "\\)"},
            }
        }
    )
    monkeypatch.setattr("mineru.render.content_list.config", configured)
    chart = ChartBlock(
        type="chart",
        index=0,
        content=[
            ChartBodyBlock(
                type="chart_body",
                index=0,
                content="<table><tr><th>A</th></tr><tr><td><eq>x</eq></td></tr></table>",
                image_base64="data:image/png;base64,AAAA",
            )
        ],
    )

    output = render_content_list(_middle(_page(0, chart)))["pages"][0]["blocks"][0]

    assert output["image_source"] == "data:image/png;base64,AAAA"
    assert output["content"].startswith("![](data:image/png;base64,AAAA)\n\n<details>")
    assert "| \\(x\\) |" in output["content"]


def test_content_list_rejects_legacy_dict_input() -> None:
    """验证公共入口只接受严格 MiddleJson。"""
    middle = _middle(_page(0))

    with pytest.raises(TypeError, match="MiddleJson"):
        render_content_list(middle.to_dict())  # type: ignore[arg-type]
