"""级联 Router 的源文件交接、取消及逐跳结果租约回归。"""

from __future__ import annotations

from task_test_utils import pending_files, task_url

import asyncio
import hashlib
import io
import time
import zipfile
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from mineru.kit.router import RouterSettings, create_app
from mineru.kit.router.tasks import maintain_task_result_leases
from mineru.parser import api_server

HTML = b"<h1>Cascade document</h1><p>Preserved content.</p>"


class _WorkerTransport(httpx.AsyncBaseTransport):
    """只控制内层 Router 提交前的等待，所有任务和产物仍经过真实 API。"""

    def __init__(self, worker: FastAPI, *, block_submission: bool) -> None:
        """用事件固定上传就绪但源 File 尚未生成的真实调度窗口。"""
        self.transport = httpx.ASGITransport(app=worker)
        self.submission_started = asyncio.Event()
        self.allow_submission = asyncio.Event()
        if not block_submission:
            self.allow_submission.set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """延迟 Tasks 创建请求，不修改任何服务端响应。"""
        if request.method == "POST" and request.url.path == "/v1/tasks":
            self.submission_started.set()
            await self.allow_submission.wait()
        return await self.transport.handle_async_request(request)

    async def aclose(self) -> None:
        """只关闭传输，应用由共享生命周期栈按依赖顺序关闭。"""
        await self.transport.aclose()


@asynccontextmanager
async def _cascade(
    tmp_path: Path, *, block_submission: bool = False
) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI, FastAPI, FastAPI, _WorkerTransport]]:
    """在同一事件循环启动外层、内层 Router 与实际解析 API。"""
    worker = api_server.create_app(upload_dir=str(tmp_path / "worker"), tiers=("flash",), retention_seconds=1)
    transport = _WorkerTransport(worker, block_submission=block_submission)
    inner = create_app(
        RouterSettings(
            upstream_urls=("http://worker",), local_gpus="none", worker_refresh_interval_seconds=0, retention_seconds=3600
        ),
        transport=transport,
    )
    outer = create_app(
        RouterSettings(
            upstream_urls=("http://inner",), local_gpus="none", worker_refresh_interval_seconds=0, retention_seconds=3600
        ),
        transport=httpx.ASGITransport(app=inner),
    )
    async with AsyncExitStack() as stack:
        for app in (worker, inner, outer):
            await stack.enter_async_context(app.router.lifespan_context(app))
        client = await stack.enter_async_context(
            httpx.AsyncClient(transport=httpx.ASGITransport(app=outer), base_url="http://outer")
        )
        try:
            yield client, outer, inner, worker, transport
        finally:
            transport.allow_submission.set()


async def _submit(client: httpx.AsyncClient, *, filename: str = "input.html", url: str | None = None) -> dict[str, Any]:
    """提交单文件清单，上传与 URL 输入采用同一份源内容声明。"""
    response = await client.post(
        "/v1/tasks",
        json={
            "tier": "flash",
            "files": [
                {
                    "file_name": filename,
                    "bytes": len(HTML),
                    "sha256sum": hashlib.sha256(HTML).hexdigest(),
                    **({"url": url} if url is not None else {}),
                }
            ],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


async def _terminal(client: httpx.AsyncClient, task: dict[str, Any]) -> dict[str, Any]:
    """按真实公共链接等待终态，超时不允许测试永久挂起。"""
    async with asyncio.timeout(10):
        while True:
            response = await client.get(task_url(client, task))
            assert response.status_code == 200, response.text
            result = response.json()
            if result["status"] in {"completed", "partial", "failed", "canceled"}:
                return result
            await asyncio.sleep(0.01)


async def _check_download(client: httpx.AsyncClient, result: dict[str, Any]) -> None:
    """验证逐跳 Files 代理返回真实且内容完整的解析 ZIP。"""
    response = await client.get(result["files"][0].get("result_url"))
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert b"Cascade document" in archive.read("markdown.md")
        assert b"Preserved content" in archive.read("markdown.md")


def test_cascade_accepts_ready_upload_without_source_file_and_reuses_cache(tmp_path: Path) -> None:
    """下游尚未提交解析也可确认上传，级联只依赖任务身份和私有缓存。"""

    async def scenario() -> None:
        """通过提交等待固定竞争窗口，验证内外层均不提前判错。"""
        async with _cascade(tmp_path, block_submission=True) as (client, outer, inner, worker, transport):
            task = await _submit(client)
            response = await client.put(pending_files(task)[0].get("upload").get("url"), content=HTML)
            assert response.status_code == 200, response.text
            await asyncio.wait_for(transport.submission_started.wait(), timeout=5)
            inner_record = next(iter(inner.state.task_store._records.values()))
            inner_route = outer.state.registry.list("job")[0]
            ready = await outer.state.worker_pool.client.get(f"http://inner/v1/tasks/{inner_route.upstream_id}")
            assert ready.json()["files"][0]["status"] == "pending"
            assert ready.json()["files"][0].get("upload") is None
            assert ready.json()["files"][0]["file_id"] == inner_record.inputs[0].file_id
            assert ready.json()["files"][0]["file_id"] != task["files"][0]["file_id"]
            async with asyncio.timeout(5):
                while not outer.state.task_store._records[task["task_id"]].file_states[0].submitted:
                    await asyncio.sleep(0.01)
            assert not outer.state.registry.list("file")
            assert outer.state.source_store._active_references
            pending = (await client.get(task_url(client, task))).json()
            assert pending["files"][0]["status"] == "pending", pending
            assert inner_record.status == "pending"
            transport.allow_submission.set()
            result = await _terminal(client, task)
            assert result["status"] == "completed", result
            assert result["files"][0]["file_id"] == task["files"][0]["file_id"]
            assert result["files"][0]["result_url"].endswith(f"/{task['files'][0]['file_id']}/content")
            await _check_download(client, result)
            cached = await _submit(client, filename="renamed.html")
            assert pending_files(cached) == []
            result = await _terminal(client, cached)
            assert result["status"] == "completed", result
            await _check_download(client, result)
            assert len(worker.state.execution_store._executions) == 2
            assert all(item.active_jobs == 0 for app in (outer, inner) for item in app.state.worker_pool.workers)

    asyncio.run(scenario())


def test_cascade_can_cancel_while_downstream_submission_is_blocked(tmp_path: Path) -> None:
    """下游提交阻塞时仍可取消，内外层预占和准备引用都正常释放。"""

    async def scenario() -> None:
        """阻塞实际任务提交后取消外层父任务，确认没有遗留解析或缓存租约。"""
        async with _cascade(tmp_path, block_submission=True) as (client, outer, inner, worker, transport):
            task = await _submit(client)
            response = await client.put(pending_files(task)[0].get("upload").get("url"), content=HTML)
            assert response.status_code == 200, response.text
            await asyncio.wait_for(transport.submission_started.wait(), timeout=5)
            response = await client.post(task_url(client, task) + "/cancel")
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "canceled"
            async with asyncio.timeout(5):
                await outer.state.task_store._records[task["task_id"]].operation
                for record in inner.state.task_store._records.values():
                    # 下游仍有结果租约，取消清理只需等待实际子文件退出。
                    await asyncio.gather(*(file.operation for file in record.file_states if file.operation is not None))
            assert not worker.state.execution_store._executions
            for app in (outer, inner):
                assert all(item.active_jobs == 0 for item in app.state.worker_pool.workers)
                assert not app.state.source_store._pins

    asyncio.run(scenario())


def test_cascade_renews_worker_results_while_outer_lease_is_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """内层等待外部租约结束期间，仍须续租底层结果并保持下载可用。"""

    def download_client(**kwargs: Any) -> httpx.AsyncClient:
        """仅隔离 URL 源站，Tasks、Files、租约和打包都执行真实实现。"""
        return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=HTML)), **kwargs)

    monkeypatch.setattr(api_server, "httpx", SimpleNamespace(AsyncClient=download_client))

    async def scenario() -> None:
        """用短 worker 保留期验证三级链路，并检查外部租约释放后的清理交接。"""
        async with _cascade(tmp_path) as (client, outer, inner, worker, _transport):
            task = await _submit(client, url="https://source.test/input.html")
            result = await _terminal(client, task)
            assert result["status"] == "completed", result
            await outer.state.task_store._records[task["task_id"]].operation
            inner_record = next(iter(inner.state.task_store._records.values()))
            inner_route = inner.state.registry.list("job")[0]
            worker_record = next(iter(worker.state.task_store._records.values()))
            assert inner_record.lease_expires_at > time.time() + 100
            assert inner_route.lease.retain_until is None
            await _check_download(client, result)
            worker_record.lease_expires_at = time.time() + 0.1
            await maintain_task_result_leases(inner.state.registry, inner.state.worker_pool)
            assert worker_record.lease_expires_at > time.time() + 100
            await asyncio.sleep(1.1)
            worker.state.file_store.collect_expired()
            assert worker_record.retention_epoch is None
            await _check_download(client, result)
            response = await outer.state.worker_pool.client.get(
                f"http://inner/v1/tasks/{inner_record.id}", headers={"x-mineru-task-lease-seconds": "0"}
            )
            assert response.status_code == 200, response.text
            await asyncio.wait_for(inner_record.operation, timeout=5)
            assert inner_route.lease.retain_until is not None and inner_route.lease.retain_until > time.time()
            inner_route.lease.retain_until = time.time() - 1
            await maintain_task_result_leases(inner.state.registry, inner.state.worker_pool)
            assert inner_route.lease.held is False
            assert worker_record.lease_expires_at == 0

    asyncio.run(scenario())
