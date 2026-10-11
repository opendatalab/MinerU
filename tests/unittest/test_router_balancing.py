"""验证真实 Router 调度时序、并发预占与 generation 安全。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from test_v1_router import _FakeV1Upstream, _make_router, _upload_file

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
            response = client.post("/v1/tasks" if index % 2 else "/v1/parse/jobs", json=BODY)
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
            return await asyncio.gather(
                *(client.post("/v1/tasks" if index % 2 else "/v1/parse/jobs", json=BODY) for index in range(10))
            )

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
                "/v1/tasks",
                headers=headers,
                json={"tier": "standard", "files": [{"source": {"type": "file_id", "file_id": file_id}}]},
            )
            assert response.status_code == 202, response.text
    assert (first.job_counter, second.job_counter) == (5, 5)
    assert first.source_download_attempts == second.source_download_attempts == 0


def test_cached_raw_tasks_still_balance() -> None:
    """新入口哈希命中仅复用字节，不能让后续任务跳过负载选择。"""
    first, second = _FakeV1Upstream("worker-a", ("standard",)), _FakeV1Upstream("worker-b", ("standard",))
    app, _ = _make_router(first, second)
    with TestClient(app) as client:
        for _ in range(10):
            response = client.post(
                "/v1/tasks",
                params={"tier": "standard", "filename": "source.pdf", "sha256sum": hashlib.sha256(b"pdf").hexdigest()},
                headers={"content-type": "application/octet-stream"},
                content=b"pdf",
            )
            assert response.status_code == 202, response.text
        assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [5, 5]
    assert (first.job_counter, second.job_counter) == (5, 5)


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
        response = client.post("/v1/tasks", json=BODY)
        assert response.status_code == 503, response.text
        assert not app.state.registry.list("job")
        assert app.state.worker_pool.workers[0].active_jobs == 1
        new_reservations[0].release()


def test_inline_output_follows_redirect_without_forwarding_auth() -> None:
    """产物重定向保持可用，调用方鉴权不能被转发到不同源的下载地址。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))

    async def handler(request: httpx.Request) -> httpx.Response:
        """模拟 worker 返回签名下载地址，并检查 CDN 请求头。"""
        if request.url.host == "cdn.example":
            assert "authorization" not in request.headers
            return httpx.Response(200, content=b"Redirected markdown", request=request)
        if request.url.path.endswith("/content"):
            return httpx.Response(302, headers={"location": "https://cdn.example/output"}, request=request)
        return await upstream.handle(request)

    app = create_app(
        RouterSettings(upstream_urls=("http://worker-a",), local_gpus="none", worker_refresh_interval_seconds=0),
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post("/v1/file_parse", headers={"authorization": "Bearer private-key"}, json=BODY)
        assert response.status_code == 200, response.text
        assert response.json()["files"][0]["content"]["markdown"] == "Redirected markdown"


def test_creation_failure_and_cancelled_copy_release_reservations() -> None:
    """创建失败和传输取消均归还名额，临时文件不会遗留。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",), fail_jobs=True)
    app, _ = _make_router(upstream)
    with TestClient(app) as client:
        failed = client.post("/v1/tasks", json=BODY)
        assert failed.status_code == 502
        assert app.state.worker_pool.workers[0].active_jobs == 0
        uploaded = client.post(
            "/v1/tasks?filename=source.pdf&tier=standard", headers={"content-type": "application/octet-stream"}, content=b"pdf"
        )
        assert uploaded.status_code == 502
        assert not upstream.files
        assert not app.state.source_store._files
        assert not app.state.source_store._uploads
        assert app.state.worker_pool.workers[0].active_jobs == 0

    upstream = _FakeV1Upstream("worker-a", ("standard",))
    original = upstream.handle

    async def canceled_put(request: httpx.Request) -> httpx.Response:
        """在目标 PUT 中模拟提交方的协程取消。"""
        if request.method == "PUT":
            raise asyncio.CancelledError()
        return await original(request)

    upstream.handle = canceled_put  # type: ignore[method-assign]
    app, _ = _make_router(upstream)

    async def run_cancel() -> None:
        """捕获模拟取消，随后核对同一事件循环中的清理结果。"""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
            try:
                await client.post(
                    "/v1/tasks?filename=source.pdf&tier=standard",
                    content=b"pdf",
                    headers={"content-type": "application/octet-stream"},
                )
            except asyncio.CancelledError:
                pass

    with TestClient(app) as client:
        client.portal.call(run_cancel)
        assert app.state.worker_pool.workers[0].active_jobs == 0
        assert not app.state.source_store._uploads
        assert all(upload["status"] != "pending" for upload in upstream.uploads.values())
