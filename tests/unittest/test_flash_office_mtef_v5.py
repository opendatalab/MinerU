from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from _legacy_ppt_test_utils import build_equation_ppt
from _legacy_xls_test_utils import build_equation_xls
from _mtef_test_utils import build_equation_doc
from _mtef_v5_test_utils import v5_formula_corpus
from _ooxml_mtef_test_utils import (
    build_equation_docx,
    build_equation_pptx,
    build_equation_xlsx,
)

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.types import BlockType, MiddleJson, ModelJson


@pytest.mark.parametrize(
    "file_suffix",
    ["doc", "docx", "ppt", "pptx", "xls", "xlsx"],
)
def test_mtef_v5_runs_through_sync_async_analyze(
    file_suffix: str,
) -> None:
    """验证 v5 贯穿六格式同步/异步严格 Analyze 契约。"""

    _name, mtef, expected = v5_formula_corpus()[3]
    prog_id = "Equation.DSMT4"
    builders: dict[str, Callable[[], bytes]] = {
        "doc": lambda: build_equation_doc([(10, mtef)], prog_id=prog_id),
        "docx": lambda: build_equation_docx([mtef], prog_id=prog_id),
        "ppt": lambda: build_equation_ppt([mtef], preview=False, prog_id=prog_id),
        "pptx": lambda: build_equation_pptx([mtef], prog_id=prog_id),
        "xls": lambda: build_equation_xls([(10, mtef)], preview=False, prog_id=prog_id),
        "xlsx": lambda: build_equation_xlsx([mtef], prog_id=prog_id),
    }
    file_bytes = builders[file_suffix]()
    middle, model = doc_analyze(
        file_bytes,
        file_suffix=file_suffix,  # type: ignore[arg-type]
    )
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(
            file_bytes,
            file_suffix=file_suffix,  # type: ignore[arg-type]
        )
    )

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.metadata.file_suffix == middle.metadata.file_suffix == file_suffix
    assert middle.pages[0].blocks[0].type == BlockType.EQUATION
    assert middle.pages[0].blocks[0].content == expected
    assert async_model == model
    assert async_middle == middle
