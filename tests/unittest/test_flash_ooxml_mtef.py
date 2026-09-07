from __future__ import annotations

import asyncio

import pytest
from _mtef_test_utils import formula_corpus
from _ooxml_mtef_test_utils import (
    build_equation_docx,
    build_equation_pptx,
    build_equation_xlsx,
)

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.types import BlockType, MiddleJson, ModelJson


@pytest.mark.parametrize("file_suffix", ["docx", "pptx", "xlsx"])
def test_ooxml_mtef_backend_analyze_sync_async_contract(file_suffix: str) -> None:
    """验证 MTEF 公式贯穿三种现代 Office 同步与异步严格契约。"""

    _name, mtef, expected = formula_corpus()[2]
    builders = {
        "docx": build_equation_docx,
        "pptx": build_equation_pptx,
        "xlsx": build_equation_xlsx,
    }
    file_bytes = builders[file_suffix]([mtef])
    middle, model = doc_analyze(file_bytes, file_suffix=file_suffix)  # type: ignore[arg-type]
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(file_bytes, file_suffix=file_suffix)  # type: ignore[arg-type]
    )

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.file_suffix == middle.file_suffix == file_suffix
    assert middle.pages[0].blocks[0].type == BlockType.EQUATION
    assert middle.pages[0].blocks[0].content == expected
    assert async_model == model
    assert async_middle == middle
