from __future__ import annotations

import asyncio
from pathlib import Path

from _legacy_xls_test_utils import (
    SheetFixture,
    build_xls,
    label_cell,
)
from _span_test_utils import inline_text

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.parser import parse
from mineru.types import MiddleJson, ModelJson


def test_backend_analyze_accepts_xls_and_async_contract() -> None:
    """验证同步与异步 Analyze 都返回严格 XLS 文档契约。"""

    file_bytes = build_xls([SheetFixture("Data", label_cell(0, 0, "value"))])
    middle_json, model_json = doc_analyze(file_bytes, file_suffix="xls")
    async_middle, async_model = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="xls"))

    assert isinstance(model_json, ModelJson)
    assert isinstance(middle_json, MiddleJson)
    assert model_json.metadata.file_suffix == middle_json.metadata.file_suffix == "xls"
    assert model_json.extensions["mineru"]["tier"] == middle_json.extensions["mineru"]["tier"] == "flash"
    assert model_json.extensions["mineru"]["parse_mode"] == middle_json.extensions["mineru"]["parse_mode"] == "txt"
    assert model_json.is_full_document is middle_json.is_full_document is True
    assert async_model == model_json
    assert async_middle == middle_json


def test_xls_is_supported_by_public_parser(tmp_path: Path) -> None:
    """验证公共 parser 通过统一 MinerUParser 路由 XLS。"""

    path = tmp_path / "sample.xls"
    path.write_bytes(build_xls([SheetFixture("Data", label_cell(0, 0, "value"))]))

    result = parse(path, tier="flash")

    assert result.middle_json.metadata.file_suffix == "xls"
    assert inline_text(result.pages[0].blocks[0].content) == "value"
