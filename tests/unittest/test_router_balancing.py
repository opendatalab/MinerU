"""验证真实 Router 调度时序、并发预占与 generation 安全。"""

from __future__ import annotations

from task_test_utils import pending_files

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from test_v1_router import _FakeV1Upstream, _make_router
from test_task_api import HTML, _WorkerTransport, _submit, _terminal, _upload
from mineru.parser import api_server

from mineru.kit.router import RouterSettings, create_app
from mineru.kit.router.workers import JobReservation

BODY = {"tier": "standard", "files": [{"file_name": "source.pdf", "bytes": 5}]}


@pytest.mark.parametrize("formats", [["unknown"], "markdown", None, 1, {}, [{}], [1]])
def test_output_format_validation_precedes_worker_selection(formats: Any, tmp_path: Path) -> None:
    """未知格式和错误 JSON 类型应与直连 API 一样报 400，且不能预占 worker。"""
    from mineru.parser.api_server import create_app as create_api_app

    upstream = _FakeV1Upstream("worker-a", ("flash",))
    app, _ = _make_router(upstream)
    body = {**BODY, "tier": "flash", "output_formats": formats}
    with TestClient(create_api_app(upload_dir=str(tmp_path / "files"), tiers=("flash",))) as direct:
        expected = direct.post("/v1/tasks", json=body)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/tasks", json=body)
        assert response.status_code == expected.status_code == 400, response.text
        assert response.json().get("error")["code"] == expected.json().get("error")["code"]
        assert not app.state.registry.list("job")
        assert app.state.worker_pool.workers[0].active_jobs == 0


def test_task_auth_preflight_uses_a_capable_worker_without_reservation() -> None:
    """待上传任务只在匹配 tier 的 worker 校验凭证，不能被无关 worker 的权限误判。"""
    first = _FakeV1Upstream("worker-a", ("flash",))
    second = _FakeV1Upstream("worker-b", ("standard",))
    checked: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """只有 standard worker 的资源接口要求认证，健康检查保持公开。"""
        if request.url.path == "/v1/health":
            response = await (first if request.url.host == "worker-a" else second).handle(request)
            payload = response.json()
            payload["features"]["task_output_formats"] = ["default", "docx", "html", "latex"]
            return httpx.Response(200, json=payload, request=request)
        if request.url.path == "/v1/usage":
            checked.append(request.url.host)
            if request.url.host == "worker-b" and request.headers.get("authorization") != "Bearer owner":
                return httpx.Response(
                    401,
                    json={
                        "error": {
                            "code": "invalid_api_key",
                            "message": "Invalid key",
                        }
                    },
                    request=request,
                )
            return httpx.Response(200, json={"object": "usage", "current": {}}, request=request)
        return await (first if request.url.host == "worker-a" else second).handle(request)

    app = create_app(
        RouterSettings(
            upstream_urls=("http://worker-a", "http://worker-b"), local_gpus="none", worker_refresh_interval_seconds=0
        ),
        transport=httpx.MockTransport(handler),
    )
    form = {
        "tier": "standard",
        "files": [
            {
                "file_name": "source.pdf",
                "bytes": 3,
            }
        ],
    }
    with TestClient(app) as client:
        cursors = dict(app.state.worker_pool._selection_cursors)
        assert client.post("/v1/tasks", json=form).status_code == 401
        assert client.post("/v1/tasks", json=form, headers={"authorization": "Bearer wrong"}).status_code == 401
        accepted = client.post("/v1/tasks", json=form, headers={"authorization": "Bearer owner"})
        assert accepted.status_code == 202, accepted.text
        assert checked == ["worker-b", "worker-b", "worker-b"]
        assert app.state.worker_pool._selection_cursors == cursors
        assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [0, 0]


def test_cached_tasks_still_balance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 tasks 命中缓存后仍按负载选择 worker，等待上传不预占模型名额。"""
    release = threading.Event()
    original = api_server.parse_async

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        """保持已选任务处于活动状态，明确观察十个任务的并发负载分配。"""
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await original(*args, **kwargs)

    monkeypatch.setattr(api_server, "parse_async", delayed)
    workers = {
        host: api_server.create_app(upload_dir=str(tmp_path / host), tiers=("flash",)) for host in ("worker-a", "worker-b")
    }
    transport = _WorkerTransport(workers)
    app = create_app(
        RouterSettings(
            upstream_urls=tuple(f"http://{host}" for host in workers), local_gpus="none", worker_refresh_interval_seconds=0
        ),
        transport=transport,
    )
    tasks = []
    with TestClient(app) as client:
        try:
            for index in range(10):
                task = _submit(client, [("source.html", HTML)])
                if index == 0:
                    assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [0, 0]
                    _upload(client, task, [("source.html", HTML)])
                else:
                    assert pending_files(task) == []
                tasks.append(task)
                # submitted 只表示 Router 开始转发；等待上游实际建好 job，包含仍在排队的任务。
                deadline = time.monotonic() + 5
                while sum(len(worker.state.execution_store._executions) for worker in workers.values()) < len(tasks):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
            assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [5, 5]
            assert [len(worker.state.execution_store._executions) for worker in workers.values()] == [5, 5]
            assert sum(method == "PUT" for _, method, _ in transport.calls) == 2
        finally:
            release.set()
        for task in tasks:
            assert _terminal(client, task)["status"] == "completed"
        assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [0, 0]


def test_generation_change_and_duplicate_release_do_not_touch_new_jobs() -> None:
    """旧预占的迟到释放不能扣掉 replacement 新任务的名额。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))
    app, _ = _make_router(upstream)
    with TestClient(app):
        pool = app.state.worker_pool
        old = pool.reserve_job(tier="standard", required_sources={"upload"})
        assert isinstance(old, JobReservation)
        worker = old.worker
        worker.generation += 1
        worker.active_jobs = 0
        new = pool.reserve_job(tier="standard", required_sources={"upload"})
        assert isinstance(new, JobReservation)
        old.release()
        assert worker.active_jobs == 1
        new.release()
        new.release()
        assert worker.active_jobs == 0


def test_task_submission_failure_and_replacement_release_only_owned_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新 tasks 的网络失败和迟到响应释放自身预占，不能扣除新 generation 的任务。"""
    for replacement in (False, True):
        worker_app = api_server.create_app(upload_dir=str(tmp_path / str(replacement)), tiers=("flash",))
        transport = _WorkerTransport({"worker-a": worker_app})
        app = create_app(
            RouterSettings(upstream_urls=("http://worker-a",), local_gpus="none", worker_refresh_interval_seconds=0),
            transport=transport,
        )
        original = transport.handle_async_request
        new_reservations = []

        async def interrupted(request: httpx.Request) -> httpx.Response:
            """在上游创建请求的网络边界模拟失败或 worker 替换。"""
            if request.method == "POST" and request.url.path == "/v1/tasks":
                if not replacement:
                    raise httpx.ConnectError("simulated disconnect", request=request)
                response = await original(request)
                pool = app.state.worker_pool
                worker = pool.workers[0]
                worker.generation += 1
                worker.active_jobs = 0
                reservation = pool.reserve_job(tier="flash", required_sources={"upload"})
                assert reservation is not None
                new_reservations.append(reservation)
                return response
            return await original(request)

        monkeypatch.setattr(transport, "handle_async_request", interrupted)
        with TestClient(app) as client:
            task = _submit(client, [("source.html", HTML)])
            _upload(client, task, [("source.html", HTML)])
            failed = _terminal(client, task)
            assert failed["status"] == "failed" and failed["files"][0].get("result_url") is None
            assert not app.state.registry.list("job")
            assert not app.state.source_store._files and not app.state.source_store._uploads
            assert app.state.worker_pool.workers[0].active_jobs == (1 if replacement else 0)
            for reservation in new_reservations:
                reservation.release()
