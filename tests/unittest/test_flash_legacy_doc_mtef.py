from __future__ import annotations

import asyncio

from _mtef_test_utils import (
    build_equation_doc,
    formula_corpus,
)

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.render.contracts import RenderMode
from mineru.render.html import render_html
from mineru.render.markdown import render_markdown
from mineru.types import BlockType, MiddleJson, ModelJson


def test_equation_editor_doc_sync_async_middle_json_and_renderers() -> None:
    """验证原生公式贯穿同步/异步 Analyze、严格 MiddleJson、Markdown 和 HTML。"""

    corpus = formula_corpus()
    file_bytes = build_equation_doc([(2000 + index, mtef) for index, (_name, mtef, _expected) in enumerate(corpus)])
    middle, model = doc_analyze(file_bytes, file_suffix="doc")
    async_middle, async_model = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="doc"))

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert async_model == model
    assert async_middle == middle
    assert [block.type for block in middle.pages[0].blocks] == [BlockType.EQUATION] * len(corpus)
    markdown = render_markdown(middle, mode=RenderMode.FULL)
    html = render_html(middle, mode=RenderMode.FULL)
    for _name, _mtef, expected in corpus:
        assert expected in markdown
        assert expected.replace("&", "&amp;") in html or expected in html
