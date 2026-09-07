from __future__ import annotations

import asyncio
from io import BytesIO

from _docx_equationxml_test_utils import (
    build_equationxml_docx,
    build_word_2003_fraction_equation_xml,
)
from docvortex.analyzers.native import DocxModel

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.types import BlockType, MiddleJson, ModelJson


def _equation_contents(pages: list[list[dict]]) -> list[str]:
    """按分页顺序提取独立公式 block 的 LaTeX 内容。"""

    return [block["content"] for page in pages for block in page if block["type"] == BlockType.EQUATION]


def test_docx_equationxml_analyze_lifecycle_and_strict_contracts() -> None:
    """验证 Equation XML 贯穿同步异步 Analyze 且调用方流保持打开。"""

    file_bytes = build_equationxml_docx([build_word_2003_fraction_equation_xml()])
    stream = BytesIO(file_bytes)
    pages = DocxModel().predict(stream)
    middle, model = doc_analyze(file_bytes, file_suffix="docx")
    async_middle, async_model = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="docx"))

    assert not stream.closed
    assert _equation_contents(pages) == [r"\frac{a}{b}"]
    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.file_suffix == middle.file_suffix == "docx"
    assert middle.pages[0].blocks[0].type == BlockType.EQUATION
    assert middle.pages[0].blocks[0].content == r"\frac{a}{b}"
    assert async_model == model
    assert async_middle == middle
