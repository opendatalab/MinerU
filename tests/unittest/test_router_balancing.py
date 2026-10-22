"""验证真实 Router 调度时序、并发预占与 generation 安全。"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from test_v1_router import _FakeV1Upstream, _make_router, _upload_file
from test_task_api import HTML, _WorkerTransport, _submit, _terminal, _upload
from mineru.parser import api_server

from mineru.kit.router import RouterSettings, create_app
from mineru.kit.router.workers import JobReservation

BODY = {"tier": "standard", "files": [{"source": {"type": "inline", "name": "source.pdf", "data": "aGVsbG8="}}]}


@pytest.mark.parametrize("formats", [["unknown"], "markdown", None, 1, {}, [{}], [1]])
def test_output_format_validation_precedes_worker_selection(formats: Any, tmp_path: Path) -> None:
    """未知格式和错误 JSON 类型应与直连 API 一样报 400，且不能预占 worker。"""
    from mineru.parser.api_server import create_app as create_api_app

    upstream = _FakeV1Upstream("worker-a", ("flash",))
    app, _ = _make_router(upstream)
    body = {**BODY, "tier": "flash", "output_formats": formats}
    with TestClient(create_api_app(upload_dir=str(tmp_path / "files"), tier="flash")) as direct:
        expected = direct.post("/v1/parse/jobs", json=body)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/parse/jobs", json=body)
        assert response.status_code == expected.status_code == 400, response.text
        assert response.json()["error"]["code"] == expected.json()["error"]["code"]
        assert upstream.job_counter == 0
        assert app.state.worker_pool.workers[0].active_jobs == 0


def test_valid_unavailable_output_format_remains_a_capability_error() -> None:
    """合法格式确实未被 worker 支持时，继续返回 503 能力不可用。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))
    app, _ = _make_router(upstream)
    with TestClient(app) as client:
        response = client.post("/v1/parse/jobs", json={**BODY, "output_formats": ["structured_content"]})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "quality_tier_unavailable"
        assert upstream.job_counter == 0 and app.state.worker_pool.workers[0].active_jobs == 0


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
        if request.url.path == "/v1/files":
            checked.append(request.url.host)
            if request.url.host == "worker-b" and request.headers.get("authorization") != "Bearer owner":
                return httpx.Response(
                    401,
                    json={
                        "error": {
                            "type": "authentication_error",
                            "code": "invalid_api_key",
                            "message": "Invalid key",
                        }
                    },
                    request=request,
                )
            return httpx.Response(200, json={"object": "list", "data": []}, request=request)
        return await (first if request.url.host == "worker-a" else second).handle(request)

    app = create_app(
        RouterSettings(
            upstream_urls=("http://worker-a", "http://worker-b"), local_gpus="none", worker_refresh_interval_seconds=0
        ),
        transport=httpx.MockTransport(handler),
    )
    form = {
        "tier": "standard",
        "manifest": [
            {
                "client_file_id": "f1",
                "filename": "source.pdf",
                "bytes": 3,
            }
        ],
    }
    with TestClient(app) as client:
        cursors = dict(app.state.worker_pool._selection_cursors)
        assert client.post("/v1/tasks", json=form).status_code == 401
        accepted = client.post("/v1/tasks", json=form, headers={"authorization": "Bearer owner"})
        assert accepted.status_code == 202, accepted.text
        assert checked == ["worker-b", "worker-b"]
        assert app.state.worker_pool._selection_cursors == cursors
        assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [0, 0]


def test_same_caller_uploads_round_robin_without_affinity() -> None:
    """同一 IP 与凭证的独立文件应均衡上传，不再固定在一个 worker。"""
    first, second = _FakeV1Upstream("worker-a", ("standard",)), _FakeV1Upstream("worker-b", ("standard",))
    app, _ = _make_router(first, second)
    with TestClient(app) as client:
        for index in range(10):
            _upload_file(client, token="same-caller", content=f"unique-pdf-{index}".encode())
    assert (first.upload_counter, second.upload_counter) == (5, 5)


def test_completed_inline_jobs_rotate_and_reads_do_not_advance_cursors() -> None:
    """逐个完成的 inline 任务也轮询，状态和能力查询不能改变分配顺序。"""
    first, second = _FakeV1Upstream("worker-a", ("standard",)), _FakeV1Upstream("worker-b", ("standard",))
    app, _ = _make_router(first, second)
    with TestClient(app) as client:
        for index in range(10):
            response = client.post("/v1/parse/jobs", json=BODY)
            assert response.status_code == 202, response.text
            task_id = response.json().get("task_id") or response.json()["job_id"]
            assert client.get(f"/v1/parse/jobs/{task_id}").json()["status"] == "completed"
            client.get("/v1/health")
            client.get("/v1/tiers")
            client.get("/v1/models")
        assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [0, 0]
    assert (first.job_counter, second.job_counter) == (5, 5)


def test_delayed_concurrent_submission_pre_reserves_worker_load() -> None:
    """上游创建响应延迟时，十个并发请求仍应在网络等待前分摊负载。"""
    first, second = _FakeV1Upstream("worker-a", ("standard",)), _FakeV1Upstream("worker-b", ("standard",))
    app, _ = _make_router(first, second)
    for upstream in (first, second):
        original = upstream._create_job

        async def delayed(request: httpx.Request, create: Any = original) -> httpx.Response:
            """让所有选择发生在首个上游成功响应之前。"""
            await asyncio.sleep(0.05)
            return await create(request)

        upstream._create_job = delayed  # type: ignore[method-assign]

    async def run() -> list[httpx.Response]:
        """在 Router 所属事件循环内同时提交新旧入口。"""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
            return await asyncio.gather(*(client.post("/v1/parse/jobs", json=BODY) for index in range(10)))

    with TestClient(app) as client:
        responses = client.portal.call(run)
        assert all(response.status_code == 202 for response in responses)
        assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [5, 5]
    assert (first.job_counter, second.job_counter) == (5, 5)


def test_file_ownership_cannot_override_lower_load() -> None:
    """同一文件的十个排队任务不能都固定在其所属 worker。"""
    first, second = _FakeV1Upstream("worker-a", ("standard",)), _FakeV1Upstream("worker-b", ("standard",))
    app, _ = _make_router(first, second)
    headers = {"authorization": "Bearer owner"}
    with TestClient(app) as client:
        file_id = _upload_file(client, token="owner", content=b"source")
        for _ in range(10):
            response = client.post(
                "/v1/parse/jobs",
                headers=headers,
                json={"tier": "standard", "files": [{"source": {"type": "file_id", "file_id": file_id}}]},
            )
            assert response.status_code == 202, response.text
    assert (first.job_counter, second.job_counter) == (5, 5)
    assert first.source_download_attempts == second.source_download_attempts == 0


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
    workers = {host: api_server.create_app(upload_dir=str(tmp_path / host), tier="flash") for host in ("worker-a", "worker-b")}
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
                    assert task["uploads"] == []
                tasks.append(task)
                deadline = time.monotonic() + 5
                while not app.state.task_store._records[task["task_id"]].files[0].submitted:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
            assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [5, 5]
            assert [len(worker.state.job_store._jobs) for worker in workers.values()] == [5, 5]
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
        old = pool.reserve_job(tier="standard", required_sources={"inline"})
        assert isinstance(old, JobReservation)
        worker = old.worker
        worker.generation += 1
        worker.active_jobs = 0
        new = pool.reserve_job(tier="standard", required_sources={"inline"})
        assert isinstance(new, JobReservation)
        old.release()
        assert worker.active_jobs == 1
        new.release()
        new.release()
        assert worker.active_jobs == 0


def test_late_job_response_after_replacement_is_not_published() -> None:
    """旧进程的创建响应不能在 replacement 后注册为新 worker 的任务。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))
    app, _ = _make_router(upstream)
    original = upstream._create_job
    new_reservations: list[JobReservation] = []

    async def replaced(request: httpx.Request) -> httpx.Response:
        """在响应到达前模拟重启及新 generation 接单。"""
        response = await original(request)
        pool = app.state.worker_pool
        worker = pool.workers[0]
        worker.generation += 1
        worker.active_jobs = 0
        reservation = pool.reserve_job(tier="standard", required_sources={"inline"})
        assert reservation is not None
        new_reservations.append(reservation)
        return response

    upstream._create_job = replaced  # type: ignore[method-assign]
    with TestClient(app) as client:
        response = client.post("/v1/parse/jobs", json=BODY)
        assert response.status_code == 503, response.text
        assert not app.state.registry.list("job")
        assert app.state.worker_pool.workers[0].active_jobs == 1
        new_reservations[0].release()


def test_output_redirect_does_not_fetch_remote_body() -> None:
    """Files 下载只转发重定向，不在 Router 内回源下载外部结果正文。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))

    async def handler(request: httpx.Request) -> httpx.Response:
        """worker 返回外部下载地址时，Router 不能主动请求 CDN 字节。"""
        assert request.url.host != "cdn.example"
        if request.url.path.endswith("/content"):
            return httpx.Response(302, headers={"location": "https://cdn.example/output"}, request=request)
        return await upstream.handle(request)

    app = create_app(
        RouterSettings(upstream_urls=("http://worker-a",), local_gpus="none", worker_refresh_interval_seconds=0),
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        headers = {"authorization": "Bearer private-key"}
        created = client.post("/v1/parse/jobs", headers=headers, json=BODY).json()
        result = client.get(f"/v1/parse/jobs/{created['job_id']}", headers=headers).json()
        file_id = result["files"][0]["output_files"]["markdown"]["file_id"]
        response = client.get(f"/v1/files/{file_id}/content", headers=headers, follow_redirects=False)
        assert response.status_code == 302 and response.headers["location"] == "https://cdn.example/output"


def test_creation_failure_and_cancelled_copy_release_reservations() -> None:
    """既有 Jobs 创建失败与跨 worker 复制取消仍归还预占，不损坏原源文件。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",), fail_jobs=True)
    app, _ = _make_router(upstream)
    with TestClient(app) as client:
        failed = client.post("/v1/parse/jobs", json=BODY)
        assert failed.status_code == 502
        assert app.state.worker_pool.workers[0].active_jobs == 0

    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("standard",))
    original = second.handle

    async def canceled_put(request: httpx.Request) -> httpx.Response:
        """仅取消目标 worker 的复制上传，原文件及其私有缓存仍可用。"""
        if request.method == "PUT":
            raise asyncio.CancelledError()
        return await original(request)

    second.handle = canceled_put  # type: ignore[method-assign]
    app, _ = _make_router(first, second)
    with TestClient(app) as client:
        file_id = _upload_file(client, token="owner", content=b"source")
        app.state.worker_pool.workers[0].active_jobs = 1

        async def run_cancel() -> None:
            """在 Router 所属事件循环捕获模拟取消，随后核对清理结果。"""
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as asynchronous:
                try:
                    await asynchronous.post(
                        "/v1/parse/jobs",
                        headers={"authorization": "Bearer owner"},
                        json={"tier": "standard", "files": [{"source": {"type": "file_id", "file_id": file_id}}]},
                    )
                except asyncio.CancelledError:
                    pass

        client.portal.call(run_cancel)
        assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [1, 0]
        assert app.state.source_store.find_file(file_id) is not None
        assert not app.state.source_store._uploads
        assert all(upload["status"] != "pending" for upload in second.uploads.values())
        app.state.worker_pool.workers[0].active_jobs = 0


def test_task_submission_failure_and_replacement_release_only_owned_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新 tasks 的网络失败和迟到响应释放自身预占，不能扣除新 generation 的任务。"""
    for replacement in (False, True):
        worker_app = api_server.create_app(upload_dir=str(tmp_path / str(replacement)), tier="flash")
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
                reservation = pool.reserve_job(tier="flash", required_sources={"file_id"})
                assert reservation is not None
                new_reservations.append(reservation)
                return response
            return await original(request)

        monkeypatch.setattr(transport, "handle_async_request", interrupted)
        with TestClient(app) as client:
            task = _submit(client, [("source.html", HTML)])
            _upload(client, task, [("source.html", HTML)])
            failed = _terminal(client, task)
            assert failed["status"] == "failed" and failed["files"][0]["result_url"] is None
            assert not app.state.registry.list("job")
            assert not app.state.source_store._files and not app.state.source_store._uploads
            assert app.state.worker_pool.workers[0].active_jobs == (1 if replacement else 0)
            for reservation in new_reservations:
                reservation.release()
