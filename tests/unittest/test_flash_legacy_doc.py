from __future__ import annotations

import asyncio
from pathlib import Path

from _legacy_doc_test_utils import build_doc
from _span_test_utils import inline_text

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.parser import parse
from mineru.types import MiddleJson, ModelJson


def test_doc_analyze_sync_and_async_return_strict_doc_contract() -> None:
    """验证同步和异步 Analyze 均保留 DOC 严格后缀和 Flash/TXT 元数据。"""

    file_bytes = build_doc("Hello\r")
    middle, model = doc_analyze(file_bytes, file_suffix="doc")
    async_middle, async_model = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="doc"))

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.metadata.file_suffix == middle.metadata.file_suffix == "doc"
    assert model.extensions["mineru"]["tier"] == middle.extensions["mineru"]["tier"] == "flash"
    assert model.extensions["mineru"]["parse_mode"] == middle.extensions["mineru"]["parse_mode"] == "txt"
    assert async_model == model
    assert async_middle == middle


def test_doc_is_supported_by_public_parser(tmp_path: Path) -> None:
    """验证公共 parser 通过统一 MinerUParser 路由 DOC。"""

    path = tmp_path / "sample.doc"
    path.write_bytes(build_doc("Hello\r"))

    result = parse(path, tier="flash")

    assert result.middle_json.metadata.file_suffix == "doc"
    assert inline_text(result.pages[0].blocks[0].content) == "Hello"
