from __future__ import annotations

import pytest
from docvortex.schema import Producer

from mineru.config import Config
from mineru.integrations.docvortex import build_metadata
from mineru.render import render_markdown
from mineru.types import (
    EquationBlock,
    MiddleJson,
    PageBlock,
    PageInfo,
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
    """构造一页并保留调用方给定的 block 顺序。"""
    return PageInfo(page_idx=page_idx, blocks=list(blocks))


def test_equation_uses_content_then_image_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证行间公式定界符配置及空公式图片回退。"""
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
            EquationBlock(type="equation", index=0, content="x=1"),
            EquationBlock(type="equation", index=1, content="", image_path="images/e.png"),
            TextBlock(
                type="text",
                index=2,
                content=[{"type": "text", "content": "inline "}, {"type": "equation_inline", "content": "y"}],
            ),
        )
    )

    assert render_markdown(middle) == "\\[\nx=1\n\\]\n\n![](images/e.png)\n\ninline \\(y\\)"
