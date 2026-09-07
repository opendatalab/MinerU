"""验证产品专用 Content List 契约与共享渲染片段的边界。"""

from __future__ import annotations

from typing import Any

import pytest

from docvortex.options import LatexDelimiterConfig, LatexDelimitersConfig
from docvortex.render import RenderFormat as EngineFormat
from docvortex.render.contracts import MarkdownRenderOptions as EngineMarkdownOptions
from docvortex.schema import MiddleJson, PageInfo, TextBlock, TextSpan, EquationInlineSpan, ChartBlock, ChartBodyBlock

from mineru.config import config
from mineru.render import RenderFormat, MarkdownRenderOptions, render_content_list, render_content_list_v2
from mineru.types import ContentType, ContentTypeV2


def test_product_owns_extended_contracts() -> None:
    """宿主九种格式独立于引擎七种格式，通用选项与语义类型继续共享。"""
    assert len(RenderFormat) == 9 and len(EngineFormat) == 7
    assert RenderFormat is not EngineFormat
    assert MarkdownRenderOptions is EngineMarkdownOptions
    assert ContentType.__module__ == ContentTypeV2.__module__ == "mineru.types"
    assert ContentType.TEXT == "text" and ContentTypeV2.SPAN_MD == "md"


@pytest.mark.parametrize("version", [1, 2])
def test_migrated_inline_delimiter_cases(version: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """迁入两种列表的公式分隔符断言，通过宿主配置显式传入实现。"""
    middle = MiddleJson(
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[
                    TextBlock(
                        type="text",
                        index=0,
                        content=[
                            TextSpan(type="text", content="Text"),
                            EquationInlineSpan(type="equation_inline", content="x"),
                        ],
                    ),
                    ChartBlock(
                        type="chart",
                        index=1,
                        content=[
                            ChartBodyBlock(type="chart_body", index=1, content="<table><tr><td><eq>x</eq></td></tr></table>")
                        ],
                    ),
                ],
            )
        ],
        file_suffix="html",
        is_full_document=True,
    )
    before = middle.to_dict(skip_defaults=False)
    delimiters = LatexDelimitersConfig(inline=LatexDelimiterConfig(left="\\(", right="\\)"))
    monkeypatch.setattr(config.render, "latex_delimiters", delimiters)
    value = render_content_list(middle) if version == 1 else render_content_list_v2(middle)

    def strings(item: Any) -> list[str]:
        """读取所有文本叶子，避免依赖两个列表各自的包装层级。"""
        if isinstance(item, str):
            return [item]
        if isinstance(item, dict):
            return [text for child in item.values() for text in strings(child)]
        if isinstance(item, list):
            return [text for child in item for text in strings(child)]
        return []

    assert any("\\(x\\)" in text for text in strings(value))
    assert middle.to_dict(skip_defaults=False) == before
