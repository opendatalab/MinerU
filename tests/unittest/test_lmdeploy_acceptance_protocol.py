"""验收脚本使用正式 Tasks/健康协议；这里只替换推理，不代替 Windows 真机验收。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_doclib_page_selection import _page

from mineru.parser import api_server
from mineru.parser.base import ParseResult
from mineru.types import MiddleJson
from scripts.acceptance.lmdeploy_lifecycle import _convert, _wait_health


def test_acceptance_upload_poll_download_and_health_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """脚本能上传、轮询、校验和下载真实服务产物，并读取暂时不可用状态。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)

    async def parse(path: str, **kwargs: Any) -> ParseResult:
        """仅隔离 GPU 推理，后续序列化、文件存储和 HTTP 协议均使用正式实现。"""
        return ParseResult(
            MiddleJson(
                is_full_document=True,
                pages=[_page(0, "body")],
                metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
                extensions={"mineru": {"tier": kwargs["tier"], "parse_mode": "txt"}},
            )
        )

    monkeypatch.setattr(api_server, "parse_async", parse)

    async def scenario() -> None:
        """在单一循环内启动正式服务生命周期和真实 ASGI 传输。"""
        app = api_server.create_app(upload_dir=str(tmp_path / "uploads"), tiers=("standard",))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                task = await _convert(client, "http://worker", b"%PDF-1.7\n", "standard", tmp_path, "accepted")
                assert task["status"] == "completed"
                assert (tmp_path / "accepted.zip").is_file()
                owner = app.state.execution_store.runtime_owner
                owner.set_engine_unavailable("test", "timeout")
                result = await _wait_health(client, "http://worker", "engine_unavailable", 1)
                assert result["error"]["code"] == "engine_unavailable"
                owner.set_engine_unavailable("test", None)
                assert (await _wait_health(client, "http://worker", None, 1))["status"] == "ok"

    asyncio.run(scenario())
