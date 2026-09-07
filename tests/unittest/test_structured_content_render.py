from __future__ import annotations

import json
from copy import deepcopy

import pytest
from _span_test_utils import inline as _inline
from docvortex.schema import Producer

from mineru.config import Config
from mineru.integrations.docvortex import build_metadata
from mineru.render import render_markdown, render_structured_content
from mineru.types import (
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    EquationBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageBlock,
    PageFootnoteBlock,
    PageInfo,
    ParagraphTitleBlock,
    TextBlock,
)


def _middle(*pages: PageInfo, file_suffix: str = "docx") -> MiddleJson:
    """构造最小严格 MiddleJson 测试对象。"""
    return MiddleJson(
        pages=list(pages),
        is_full_document=True,
        file_suffix=file_suffix,
        producer=Producer(name="mineru", version="test"),
        extensions=build_metadata(effort="flash", parse_mode="txt", mineru_version="test"),
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
                content=[
                    {"type": "text", "content": "# "},
                    {"type": "text", "content": "Section", "styles": ["bold"]},
                    {"type": "text", "content": " "},
                    {"type": "equation_inline", "content": "x"},
                ],
            ),
            TextBlock(type="text", index=1, content=_inline("first-")),
            PageAuxTextBlock(type="header", index=2, content=_inline("HEADER")),
            PageFootnoteBlock(
                type="page_footnote",
                index=3,
                bbox=(0.1, 0.8, 0.9, 0.9),
                anchor="note-one",
                content=[{"type": "text", "content": "Foot "}, {"type": "equation_inline", "content": "x"}],
            ),
        ),
        _page(
            1,
            TextBlock(type="text", index=0, content=_inline("continued"), continues_prev=True),
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
        "page_footnote",
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
    assert result["pages"][0]["blocks"][3] == {
        "type": "page_footnote",
        "bbox": [0.1, 0.8, 0.9, 0.9],
        "anchor": "note-one",
        "content": "Foot $x$",
    }
    assert result["pages"][1]["blocks"][0]["content"] == "continued"
    assert result["pages"][1]["blocks"][0]["continues_prev"] is True
    _assert_output_field_contract(result)
    assert middle == original


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
    monkeypatch.setattr("mineru.config.config", configured)
    chart = ChartBlock(
        type="chart",
        index=0,
        content=[
            ChartAnnotationBlock(
                type="chart_caption",
                bbox=(0.1, 0.1, 0.9, 0.2),
                content=_inline("Chart", styles=["bold"]),
            ),
            ChartBodyBlock(
                type="chart_body",
                index=0,
                content="<table><tr><th>A</th></tr><tr><td><eq>x</eq></td></tr></table>",
                image_base64="data:image/png;base64,AAAA",
            ),
            ChartAnnotationBlock(
                type="chart_footnote",
                content=_inline("source"),
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
    monkeypatch.setattr("mineru.config.config", configured)
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
