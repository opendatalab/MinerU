"""仅公开 Standard 的服务经真实 TCP 支持 SDK 原生文档解析。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from test_http_api_example_script import _Server
from test_task_response_contract import _mixed_inputs

from mineru.kit.router import RouterSettings, create_app as create_router
from mineru.parser import MinerUApiParser, api_server


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native_index", [1, 2])
@pytest.mark.parametrize("deployment", ["api", "router"])
def test_native_sdk_uses_standard_only_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, native_index: int, deployment: str
) -> None:
    """同步和异步 SDK 均能上传、轮询、下载真实 HTML/DOCX，PDF Flash 仍被拒绝。"""
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", Mock())
    worker = _Server(api_server.create_app(upload_dir=str(tmp_path / "worker"), tiers=("standard",)))
    worker.start()
    router = None
    try:
        url = worker.url
        if deployment == "router":
            router = _Server(
                create_router(RouterSettings(upstream_urls=(url,), local_gpus="none", worker_refresh_interval_seconds=0))
            )
            router.start()
            url = router.url
        name, content = _mixed_inputs()[native_index]
        source = tmp_path / name
        source.write_bytes(content)
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert [item["id"] for item in client.get("/v1/tiers").json()["data"]] == ["standard"]
            rejected = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], "tier": "flash"})
            assert rejected.status_code == 400
            assert rejected.json()["error"]["param"] == "tier"
        parser = MinerUApiParser(api_url=url)
        result = asyncio.run(parser.parse_async(source)) if asynchronous else parser.parse(source)
        assert result.middle_json.extensions["mineru"]["tier"] == "flash"
        assert ("Mixed task HTML" if native_index == 1 else "Mixed task DOCX") in result.markdown()
    finally:
        if router is not None:
            router.stop()
        worker.stop()
