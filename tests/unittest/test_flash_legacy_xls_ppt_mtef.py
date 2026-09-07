from __future__ import annotations

import asyncio

import pytest
from _legacy_ppt_test_utils import build_equation_ppt
from _legacy_xls_test_utils import build_equation_xls
from _mtef_test_utils import formula_corpus

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.types import BlockType, MiddleJson, ModelJson


@pytest.mark.parametrize("file_suffix", ["xls", "ppt"])
def test_backend_analyze_preserves_native_equations_sync_and_async(
    file_suffix: str,
) -> None:
    """验证原生公式贯穿 XLS/PPT 同步与异步严格 Analyze 契约。"""

    _name, mtef, expected = formula_corpus()[3]
    file_bytes = (
        build_equation_xls([(42, mtef)], preview=False) if file_suffix == "xls" else build_equation_ppt([mtef], preview=False)
    )
    middle, model = doc_analyze(file_bytes, file_suffix=file_suffix)  # type: ignore[arg-type]
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(file_bytes, file_suffix=file_suffix)  # type: ignore[arg-type]
    )

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.metadata.file_suffix == middle.metadata.file_suffix == file_suffix
    assert middle.pages[0].blocks[0].type == BlockType.EQUATION
    assert middle.pages[0].blocks[0].content == expected
    assert async_model == model
    assert async_middle == middle
