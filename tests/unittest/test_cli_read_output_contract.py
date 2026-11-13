"""阅读只返回片段，完整文本导出与图片保存保持各自的输出契约。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from test_doclib_page_selection import _DOC_REF, _SHA256, _cached_library, _page
from typer.testing import CliRunner

from mineru.cli.commands import parse, read
from mineru.cli.main import app
from mineru.doclib.types import (
    ContentNextRequest,
    ContentRequestScope,
    DocContentExportRequest,
    DocContentExportResponse,
    DocContentResponse,
    ParseResponse,
)


@pytest.mark.parametrize("json_mode", [False, True])
def test_read_stdout_alias_preserves_content_and_continuation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, json_mode: bool
) -> None:
    """显式 stdout 与省略 output 完全等价，不出现落盘字段或名为减号的文件。"""
    monkeypatch.chdir(tmp_path)

    class Client:
        def __init__(self, *, timeout: int) -> None:
            """模拟仅用于读取的客户端。"""

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            """返回带续读位置的文本片段。"""
            return DocContentResponse(
                sha256=_SHA256,
                short_id="aaaaaaa",
                tier="flash",
                content="正文",
                request_scope=ContentRequestScope(locator=locator),
                next_request=ContentNextRequest(locator=f"{_DOC_REF}/page:2"),
            )

    monkeypatch.setattr(read, "DoclibClient", Client)
    args = ["read", _DOC_REF, *(["--json"] if json_mode else [])]
    runner = CliRunner()
    normal = runner.invoke(app, args)
    alias = runner.invoke(app, [*args, "-o", "-"])
    assert normal.exit_code == alias.exit_code == 0
    assert normal.stdout == alias.stdout
    if json_mode:
        payload = json.loads(alias.stdout)
        assert payload["content"] == "正文"
        assert payload["next_request"]["locator"].endswith("page:2")
        assert "output" not in payload
    else:
        assert "<!-- Next: mineru read" in alias.stdout
    assert not (tmp_path / "-").exists()


def test_parse_output_exports_real_cache_beyond_read_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """真实缓存中的长文本完整导出；CLI 的 limit 不影响指定范围的产物。"""
    cache = tmp_path / "cache"
    cache.mkdir()
    body = "A" * 16000 + "B" * 16000 + "C" * 16000
    db, server = asyncio.run(_cached_library(cache, page_count=1, pages=[_page(0, body)]))
    output = tmp_path / "out" / "complete.md"

    class Client:
        def __init__(self, *, timeout: int) -> None:
            """仅替换传输，导出仍执行真实 Doclib 服务。"""

        def ensure_parse(self, request: Any) -> ParseResponse:
            """复用已经准备好的解析缓存。"""
            return ParseResponse(sha256=_SHA256, tier="flash", page_range="1", status="done", cache_hit=True)

        def export_doc_content(self, doc_ref: str, request: DocContentExportRequest) -> DocContentExportResponse:
            """将 CLI 导出请求转给真实服务。"""
            return asyncio.run(server.export_doc_content(doc_ref, request))

    monkeypatch.setattr(parse, "DoclibClient", Client)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    try:
        result = CliRunner().invoke(
            app, ["parse", str(source), "--tier", "flash", "--limit", "10", "-o", str(output), "--json"]
        )
        assert result.exit_code == 0, result.output
        assert body in output.read_text(encoding="utf-8")
        assert json.loads(result.stdout)["output"]["status"] == "written"
    finally:
        asyncio.run(db.close())
