"""统一文件状态的准备、排队、取消和耗时边界，使用真实 API 生命周期验证。"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from task_test_utils import task_url
from test_task_api import HTML, _archive, _submit, _terminal, _upload
from test_tasks_sdk import middle

from mineru.parser import api_server
from mineru.parser.base import ParseResult


def _install_downloads(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """只隔离源站下载，保留客户端到 API 的真实协议和任务调度。"""

    def client(**kwargs: Any) -> httpx.AsyncClient:
        """为每次源下载创建独立 HTTPX 客户端，不修改全局 HTTPX 对象。"""
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(api_server, "httpx", SimpleNamespace(AsyncClient=client))


def _url_task(client: TestClient, name: str = "remote") -> dict[str, Any]:
    """创建无需客户端上传的 URL 文件，初始必须处于文件准备阶段。"""
    response = client.post(
        "/v1/tasks", json={"tier": "flash", "files": [{"file_name": name + ".html", "url": "https://source.test/" + name}]}
    )
    assert response.status_code == 202, response.text
    task = response.json()
    assert task["status"] == "waiting-file"
    assert task["files"][0].get("upload") is None and "upload_status" not in task["files"][0]
    return task


@pytest.mark.parametrize("cancel", [False, True])
def test_url_preparation_does_not_block_ready_files_or_use_upload_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    """单解析名额下慢下载不阻塞已上传文件，URL 准备不受客户端上传截止时间影响。"""
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Source(httpx.AsyncByteStream):
        """提供可取消的源下载流，确认取消真正关闭了网络输入。"""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """等待测试放行后输出有效 HTML，取消必须中断等待。"""
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            yield HTML

        async def aclose(self) -> None:
            """记录源流关闭，区别取消和正常放行。"""
            closed.set()

    _install_downloads(monkeypatch, lambda _: httpx.Response(200, stream=Source()))
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",), concurrency=1)
    try:
        with TestClient(app) as client:
            remote = _url_task(client)
            assert started.wait(3)
            app.state.task_store._records[remote["task_id"]].expires_at = 0
            snapshot = client.get(task_url(client, remote)).json()
            assert snapshot["status"] == "waiting-file" and snapshot["files"][0].get("error") is None
            assert snapshot["files"][0].get("parse") is None
            local = _submit(client, [("local.html", HTML)])
            _upload(client, local, [("local.html", HTML)])
            assert _terminal(client, local)["status"] == "completed"
            assert not release.is_set()
            if cancel:
                assert client.post(task_url(client, remote) + "/cancel").json()["status"] == "canceled"
                assert closed.wait(3)
            else:
                release.set()
                completed = _terminal(client, remote)
                assert completed["status"] == "completed"
                assert completed["files"][0].get("bytes") == len(HTML)
                assert b"Preserved content" in _archive(client, completed)["markdown.md"]
            assert all(not job.prepared_input for job in app.state.execution_store._executions.values())
    finally:
        release.set()


@pytest.mark.parametrize("cancel_ready", [False, True])
def test_ready_url_inputs_wait_in_pending_with_bounded_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_ready: bool
) -> None:
    """文件下载完成后排队，准备缓冲有界，排队取消会释放缓冲及准备名额。"""
    parsing, release = threading.Event(), threading.Event()
    downloads: list[str] = []
    original = api_server.parse_async

    async def infer(path: str, **kwargs: Any) -> ParseResult:
        """仅阻塞占位文件的解析，让后续 URL 输入稳定停在排队阶段。"""
        if b"blocker" in Path(path).read_bytes():
            parsing.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        return await original(path, **kwargs)

    def download(request: httpx.Request) -> httpx.Response:
        """记录实际开始的下载，尚未取得准备名额时不应访问源站。"""
        downloads.append(request.url.path)
        return httpx.Response(200, content=HTML)

    monkeypatch.setattr(api_server, "parse_async", infer)
    _install_downloads(monkeypatch, download)
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",), concurrency=1)
    try:
        with TestClient(app) as client:
            blocking = [("blocker.html", b"<html><body>blocker</body></html>")]
            first = _submit(client, blocking)
            _upload(client, first, blocking)
            assert parsing.wait(3)
            second = _url_task(client, "second")
            deadline = time.monotonic() + 3
            while client.get(task_url(client, second)).json()["status"] != "pending":
                assert time.monotonic() < deadline
                time.sleep(0.01)
            record = app.state.task_store._records[second["task_id"]]
            job = app.state.execution_store._executions[record.file_states[0].execution_id]
            assert job.prepared_input and job.parse is None
            third = _url_task(client, "third")
            assert client.get(task_url(client, third)).json()["status"] == "waiting-file"
            assert downloads == ["/second"]
            if cancel_ready:
                assert client.post(task_url(client, second) + "/cancel").json()["status"] == "canceled"
                deadline = time.monotonic() + 3
                while "/third" not in downloads:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                assert not job.prepared_input
            release.set()
            assert _terminal(client, first)["status"] == "completed"
            assert _terminal(client, second)["status"] == ("canceled" if cancel_ready else "completed")
            assert _terminal(client, third)["status"] == "completed"
            assert all(not item.prepared_input for item in app.state.execution_store._executions.values())
    finally:
        release.set()


def test_file_duration_includes_preparation_but_excludes_queue_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """确定性时钟区分两秒准备、千秒排队和三秒解析，最终耗时应为五秒。"""
    clock = [0.0]
    monkeypatch.setattr(api_server, "time", SimpleNamespace(time=time.time, monotonic=lambda: clock[0]))

    def download(request: httpx.Request) -> httpx.Response:
        """源下载推进两秒执行时钟，真实准备逻辑仍负责完整性检查。"""
        clock[0] += 2
        return httpx.Response(200, content=HTML)

    async def infer(path: str, **kwargs: Any) -> ParseResult:
        """解析推进三秒时钟，保留真实导出和统计发布。"""
        clock[0] += 3
        return ParseResult(middle())

    _install_downloads(monkeypatch, download)
    monkeypatch.setattr(api_server, "parse_async", infer)
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",), concurrency=1)
    with TestClient(app) as client:
        client.portal.call(app.state.execution_store._semaphore.acquire)
        task = _url_task(client)
        deadline = time.monotonic() + 3
        while client.get(task_url(client, task)).json()["status"] != "pending":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        clock[0] += 1000
        client.portal.call(app.state.execution_store._semaphore.release)
        result = _terminal(client, task)
        assert result["files"][0].get("parse")["duration_ms"] == 5000
