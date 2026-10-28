"""真实 Gradio 事件经 SDK 联动 HTTP Tasks 取消的回归。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mineru.kit.gradio.app import build_gradio_app
from mineru.kit.gradio.client import V1ArtifactClient, V1ServerCapabilities
from tests.unittest.test_tasks_sdk import bind_transport
from tests.unittest.test_task_api import service as _service_fixture

# 复用真实 API/双 worker Router，客户端不能绕过服务端取消与资源清理。
service = _service_fixture


@pytest.mark.parametrize("trigger", ["clear", "file_change", "resubmit", "unload"])
def test_ui_invalidation_cancels_only_its_upstream_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trigger: str) -> None:
    """通过真实组件回调和 HTTPX 传输验证四类事件，旧票据不能取消替换后的任务。"""
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n")
    client = V1ArtifactClient(api_url="https://api.test")
    capabilities = V1ServerCapabilities(client.base_url, ("flash",), ("default",), ("upload",))
    client._capabilities = capabilities
    demo = build_gradio_app(client, capabilities, output_root=tmp_path / "output", enable_example=False)
    callbacks = {function.name: function.fn for function in demo.fns.values()}
    request = SimpleNamespace(session_hash="browser-session")

    async def scenario() -> None:
        """让任务停在真实 SDK 查询中，界面事件必须发送一次对应服务端 ID 的 cancel。"""
        waiting: asyncio.Queue[str] = asyncio.Queue()
        canceled: list[str] = []
        created = 0

        async def handle(message: httpx.Request) -> httpx.Response:
            """创建独立的服务端身份，任何错误路由或旧 DELETE 都会失败。"""
            nonlocal created
            assert message.url.host == "api.test"
            if message.method == "POST" and message.url.path == "/v1/tasks":
                created += 1
                return httpx.Response(
                    202,
                    json={
                        "task_id": f"remote-{created}",
                        "status": "pending",
                        "files": [{"file_id": f"file-{created}", "file_name": source.name, "status": "pending"}],
                    },
                )
            remote = message.url.path.split("/")[3]
            if message.method == "POST" and message.url.path.endswith("/cancel"):
                canceled.append(remote)
                return httpx.Response(200, json={"task_id": remote, "status": "canceled"})
            assert message.method == "GET"
            waiting.put_nowait(remote)
            await asyncio.Future()
            raise AssertionError("unreachable")

        bind_transport(monkeypatch, handle)
        tickets = [json.dumps({"run_id": f"{index:032x}", "revision": index}) for index in (1, 2)]
        first = asyncio.create_task(callbacks["convert_ui"](str(source), 0, "", False, tickets[0], request))
        second = None
        try:
            assert await asyncio.wait_for(waiting.get(), 5) == "remote-1"
            if trigger == "resubmit":
                second = asyncio.create_task(callbacks["convert_ui"](str(source), 0, "", False, tickets[1], request))
                assert await asyncio.wait_for(waiting.get(), 5) == "remote-2"
            elif trigger == "unload":
                await callbacks["unload_session"](request)
            else:
                await callbacks["cancel_ui_conversion"](tickets[0], request)
            assert await asyncio.wait_for(first, 5) == ""
            assert canceled == ["remote-1"]
            # 重复的清除通知不能再次取消原任务，也不能误伤替换任务。
            await callbacks["cancel_ui_conversion"](tickets[0], request)
            assert canceled == ["remote-1"]
            if second is not None:
                assert not second.done()
                await callbacks["cancel_ui_conversion"](tickets[1], request)
                assert await asyncio.wait_for(second, 5) == ""
                assert canceled == ["remote-1", "remote-2"]
        finally:
            first.cancel()
            if second is not None:
                second.cancel()
            await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        demo.close()


def test_gradio_cancel_reaches_real_api_and_router(
    service: tuple[TestClient, FastAPI], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """只替换推理工作，贯通真正的提交、上传、取消路由及 worker 协程回收。"""
    import threading

    from mineru.kit.gradio import client as gradio_client
    from mineru.parser import api_client, api_server
    from tests.unittest.test_kit_gradio import _httpx_proxy_for_test_client

    http_client, app = service
    started, stopped = threading.Event(), threading.Event()
    source = tmp_path / "real.html"
    source.write_text("<h1>cancel actual task</h1>")

    async def parse(*_args: Any, **_kwargs: Any) -> None:
        """保持实际解析作业运行，取消必须最终到达该生产工作协程。"""
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    monkeypatch.setattr(api_server, "parse_async", parse)
    calls = []
    proxy = _httpx_proxy_for_test_client(http_client, calls)
    monkeypatch.setattr(gradio_client, "httpx", proxy)
    monkeypatch.setattr(api_client, "httpx", proxy)
    monkeypatch.setattr(api_client, "_POLL_INTERVAL_SECONDS", 0.01)
    client = V1ArtifactClient(api_url="http://testserver")

    async def scenario() -> None:
        """在真实解析开始后取消等待，服务端必须保留 canceled 终态且清理作业。"""
        operation = asyncio.create_task(client.parse_file(source, tier="flash", page_range=""))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(operation, 5)
            assert await asyncio.to_thread(stopped.wait, 5)
        finally:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)

    asyncio.run(scenario())
    cancellations = [path for method, path, _ in calls if method == "POST" and path.endswith("/cancel")]
    assert len(cancellations) == 1
    response = http_client.get(cancellations[0].removesuffix("/cancel"))
    assert response.status_code == 200
    assert response.json()["status"] == "canceled"
    assert response.json()["files"][0]["status"] == "canceled"
