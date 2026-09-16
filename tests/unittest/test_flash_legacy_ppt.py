from __future__ import annotations

import asyncio
from pathlib import Path

from _legacy_ppt_test_utils import build_sparse_notes_ppt

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.parser import parse
from mineru.types import MiddleJson, ModelJson


def test_backend_analyze_accepts_ppt_and_async_contract() -> None:
    """验证同步与异步 Backend Analyze 均返回严格 PPT 文档契约。"""

    file_bytes = build_sparse_notes_ppt()
    middle_json, model_json = doc_analyze(file_bytes, file_suffix="ppt")
    async_middle_json, async_model_json = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="ppt"))

    assert isinstance(model_json, ModelJson)
    assert isinstance(middle_json, MiddleJson)
    assert model_json.metadata.file_suffix == "ppt"
    assert middle_json.metadata.file_suffix == "ppt"
    assert model_json.extensions["mineru"]["tier"] == middle_json.extensions["mineru"]["tier"] == "flash"
    assert model_json.extensions["mineru"]["parse_mode"] == middle_json.extensions["mineru"]["parse_mode"] == "txt"
    assert model_json.is_full_document is middle_json.is_full_document is True
    assert [page.page_idx for page in middle_json.pages] == [0, 1]
    assert async_middle_json == middle_json
    assert async_model_json == model_json


def test_ppt_is_supported_by_public_parser(tmp_path: Path) -> None:
    """验证公共 parser 通过统一 MinerUParser 路由 PPT。"""

    path = tmp_path / "sample.ppt"
    path.write_bytes(build_sparse_notes_ppt())

    result = parse(path, tier="flash")

    assert result.middle_json.metadata.file_suffix == "ppt"
    assert len(result.pages) == 2
