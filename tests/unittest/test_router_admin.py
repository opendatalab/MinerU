"""验证远程 upstream 动态注册、独立认证和不破坏既有任务的安全注销。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.types import Message

from mineru.kit.router import RouterSettings, create_app
from mineru.kit.router.proxy import RouterProxyError, request_upstream, stream_upstream
from mineru.kit.router.tasks import _worker_path, maintain_task_result_leases
from mineru.kit.router.workers import WorkerState, normalize_base_url
from mineru.parser import api_server
from mineru.tasks.errors import ApiServerError
from test_router_cascade import HTML, _WorkerTransport, _check_download, _submit, _terminal
from test_v1_router import _FakeV1Upstream

ADMIN = {"authorization": "Bearer management-only-secret"}
WORKERS = "/v1/router/workers"


def _settings(*urls: str, retention: int = 86400, enabled: bool = True) -> RouterSettings:
    """构造禁用本地模型与周期探测的管理测试配置，固定保留期避免环境变量干扰。"""
    return RouterSettings(
        upstream_urls=urls,
        local_gpus="none",
        worker_refresh_interval_seconds=0,
        retention_seconds=retention,
        admin_api_key="management-only-secret" if enabled else None,
    )


def _fake_app(settings: RouterSettings) -> tuple[FastAPI, list[httpx.Request]]:
    """只模拟无认证的能力端点，记录管理凭证是否意外传给下层。"""
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        """所有测试地址均返回相同的合法能力，不替换 Router 自身的行为。"""
        requests.append(request)
        return await _FakeV1Upstream("worker", ("flash",)).handle(request)

    return create_app(settings, transport=httpx.MockTransport(handle)), requests


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """在一个事件循环中运行完整 Router 生命周期，便于控制请求之间的竞争窗口。"""
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
            yield client


@pytest.mark.parametrize("enabled", [False, True])
def test_management_authentication_and_openapi(enabled: bool) -> None:
    """未配置 Key 时接口不存在；启用后所有管理操作均使用独立认证且不向下传递。"""
    app, requests = _fake_app(_settings(enabled=enabled))
    with TestClient(app) as client:
        for method, path in (("GET", WORKERS), ("POST", WORKERS), ("DELETE", f"{WORKERS}/remote-1")):
            for headers in ({}, {"authorization": "Bearer parse-key"}):
                response = client.request(method, path, headers=headers, json={"url": "http://worker"})
                assert response.status_code == (401 if enabled else 404), response.text
                if enabled:
                    assert response.json()["error"]["code"] == "invalid_api_key"
        schema = client.get("/openapi.json").json()
        assert (WORKERS in schema["paths"]) is enabled
        if enabled:
            response = client.post(WORKERS, json={"url": "http://worker"}, headers=ADMIN)
            assert response.status_code == 201, response.text
            assert response.json()["healthy"] is True
            assert requests and all("authorization" not in request.headers for request in requests)
            assert "management-only-secret" not in response.text + str(schema) + repr(app.state.settings)
        else:
            assert client.get(WORKERS, headers=ADMIN).status_code == 404
            assert not requests


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"url": 123},
        {"url": ""},
        {"url": "ftp://worker"},
        {"url": "//worker"},
        {"url": "http://user:secret@worker"},
        {"url": "http://@worker"},
        {"url": "http://worker?"},
        {"url": "http://worker#"},
        {"url": "http://worker?x=1"},
        {"url": "http://worker/#part"},
        {"url": "http://work er"},
        {"url": "http://worker:bad"},
        {"url": "http://worker/\\path"},
        {"url": "http://worker", "healthy": True},
    ],
)
def test_registration_rejects_invalid_input_before_probe(body: dict[str, Any]) -> None:
    """非法地址和额外字段必须在节点入池及网络探测前拒绝。"""
    app, requests = _fake_app(_settings())
    with TestClient(app) as client:
        response = client.post(WORKERS, json=body, headers=ADMIN)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_request"
        assert not app.state.worker_pool.workers
        assert not requests


def test_registration_drain_resume_and_non_reused_ids() -> None:
    """启动节点与动态节点共用身份；draining 停止新调度，重复注册恢复，最终删除不复用 ID。"""
    app, _ = _fake_app(_settings("HTTP://WORKER:80/"))
    with TestClient(app) as client:
        first = client.post(WORKERS, json={"url": "http://worker/"}, headers=ADMIN)
        assert first.status_code == 200
        worker_id = first.json()["worker_id"]
        pool = app.state.worker_pool
        reservation = client.portal.call(partial(pool.reserve_job, tier="flash", required_sources={"upload"}))
        assert reservation is not None
        drained = client.delete(f"{WORKERS}/{worker_id}", headers=ADMIN)
        assert drained.status_code == 202
        assert drained.json()["active_jobs"] == 1
        client.portal.call(pool.refresh, reservation.worker)
        assert reservation.worker.healthy
        assert reservation.worker.state == "draining"
        assert client.get("/v1/health").status_code == 503
        assert client.get("/v1/tiers").status_code == 503
        assert client.get("/v1/models").status_code == 503
        assert pool.select(tier="flash") is None
        restored = client.post(WORKERS, json={"url": "http://worker"}, headers=ADMIN)
        assert restored.status_code == 200
        assert restored.json()["worker_id"] == worker_id
        assert restored.json()["active_jobs"] == 1
        client.portal.call(reservation.release)
        assert client.delete(f"{WORKERS}/{worker_id}", headers=ADMIN).status_code == 204
        assert client.delete(f"{WORKERS}/{worker_id}", headers=ADMIN).status_code == 404
        assert client.get(WORKERS, headers=ADMIN).json()["data"] == []
        replacement = client.post(WORKERS, json={"url": "http://worker"}, headers=ADMIN)
        assert replacement.status_code == 201
        assert replacement.json()["worker_id"] != worker_id


def test_restarts_restore_only_startup_upstreams_and_exclude_managed_workers() -> None:
    """运行期注销不改启动配置；本地托管节点不能经管理接口列出或删除。"""
    settings = _settings("http://configured")
    app, _ = _fake_app(settings)
    with TestClient(app) as client:
        assert client.delete(f"{WORKERS}/remote-1", headers=ADMIN).status_code == 204
        assert client.post(WORKERS, json={"url": "http://dynamic"}, headers=ADMIN).status_code == 201
        local = WorkerState("local-test", "http://local", "local")
        app.state.worker_pool._workers[local.worker_id] = local
        assert client.delete(f"{WORKERS}/{local.worker_id}", headers=ADMIN).status_code == 404
        assert [item["url"] for item in client.get(WORKERS, headers=ADMIN).json()["data"]] == ["http://dynamic"]
    restarted, _ = _fake_app(settings)
    with TestClient(restarted) as client:
        assert [item["url"] for item in client.get(WORKERS, headers=ADMIN).json()["data"]] == ["http://configured"]


def test_concurrent_registration_and_drain_during_initial_probe() -> None:
    """探测等待期间重复注册只创建一个节点，并发注销不会被迟到的健康结果撤销。"""

    async def scenario() -> None:
        """固定两次注册都持有请求引用的窗口，再发起注销并放行探测。"""
        release = asyncio.Event()

        async def handle(request: httpx.Request) -> httpx.Response:
            """延迟所有探测请求以观察节点登记与 HTTP 完成之间的状态。"""
            await release.wait()
            return await _FakeV1Upstream("worker", ("flash",)).handle(request)

        app = create_app(_settings(), transport=httpx.MockTransport(handle))
        async with _client(app) as client:
            operations = [
                asyncio.create_task(client.post(WORKERS, json={"url": url}, headers=ADMIN))
                for url in ("http://worker", "http://worker/")
            ]
            try:
                async with asyncio.timeout(5):
                    while not app.state.worker_pool.workers or app.state.worker_pool.workers[0].active_requests < 2:
                        await asyncio.sleep(0.01)
                assert len(app.state.worker_pool.workers) == 1
                drained = await client.delete(f"{WORKERS}/remote-1", headers=ADMIN)
                assert drained.status_code == 202
            finally:
                release.set()
            responses = await asyncio.gather(*operations)
            assert sorted(response.status_code for response in responses) == [200, 201]
            assert all(response.json()["state"] == "draining" for response in responses)
            assert app.state.worker_pool.workers[0].active_requests == 0
            app.state.collect_resources_once()
            assert not app.state.worker_pool.workers

    asyncio.run(scenario())


def test_failed_probe_recovers_without_reregistration() -> None:
    """注册暂时离线的节点仍保留记录，后续现有健康检查可恢复其可调度状态。"""

    async def scenario() -> None:
        """切换模拟 upstream 可达性，验证探测失败与管理状态互相独立。"""
        healthy = False

        async def handle(request: httpx.Request) -> httpx.Response:
            """离线时使用真实 httpx 连接异常，恢复后返回标准能力。"""
            if not healthy:
                raise httpx.ConnectError("offline", request=request)
            return await _FakeV1Upstream("worker", ("flash",)).handle(request)

        app = create_app(_settings(), transport=httpx.MockTransport(handle))
        async with _client(app) as client:
            response = await client.post(WORKERS, json={"url": "http://worker"}, headers=ADMIN)
            assert response.status_code == 201
            assert response.json()["healthy"] is False
            assert response.json()["state"] == "active"
            healthy = True
            await app.state.worker_pool.refresh_all()
            assert (await client.get("/v1/health")).status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "invalid_tiers",
    [
        {"data": [{"id": "unknown-tier"}]},
        {"data": [{"id": "flash"}, {"id": "unknown-tier"}]},
        {"data": [{"id": ["flash"]}]},
        {"data": [{}]},
        {"data": ["flash"]},
        {"data": None},
        {"data": {}},
        {},
    ],
)
def test_invalid_tiers_do_not_poison_management_or_replace_valid_capabilities(invalid_tiers: dict[str, Any]) -> None:
    """异常节点与错误刷新不能使整个管理列表返回 500，也不能发布未校验的部分能力。"""
    invalid = True

    async def handle(request: httpx.Request) -> httpx.Response:
        """仅破坏一个节点的 tier，并修改其 health 内容以检查失败刷新是否原子化。"""
        response = await _FakeV1Upstream("worker", ("flash",)).handle(request)
        if request.url.host == "bad" and invalid:
            if request.url.path == "/v1/tiers":
                return httpx.Response(200, json=invalid_tiers)
            if request.url.path == "/v1/health":
                return httpx.Response(200, json={"status": "ok", "features": {"sources": ["url"]}})
        return response

    app = create_app(_settings("http://good"), transport=httpx.MockTransport(handle))
    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(WORKERS, json={"url": "http://bad"}, headers=ADMIN)
        assert created.status_code == 201, created.text
        assert created.json()["healthy"] is False
        assert created.json()["tiers"] == []
        worker = app.state.worker_pool.get(created.json()["worker_id"])
        assert worker.features == {} and worker.models == []
        listed = client.get(WORKERS, headers=ADMIN)
        assert listed.status_code == 200
        assert len(listed.json()["data"]) == 2
        assert [item.worker_id for item in app.state.worker_pool.capable_workers(tier="flash")] == ["remote-1"]
        invalid = False
        client.portal.call(app.state.worker_pool.refresh, worker)
        assert worker.healthy and worker.tiers == {"flash"}
        features = dict(worker.features)
        invalid = True
        client.portal.call(app.state.worker_pool.refresh, worker)
        assert worker.healthy is False
        assert worker.tiers == {"flash"} and worker.features == features
        assert client.get(WORKERS, headers=ADMIN).status_code == 200
        assert client.delete(f"{WORKERS}/{worker.worker_id}", headers=ADMIN).status_code == 204


@pytest.mark.parametrize("cancel", [False, True])
def test_pending_request_blocks_removal_and_releases_on_failure(cancel: bool) -> None:
    """即使没有执行预占和路由，进行中的请求也阻止删除，异常或取消后可正常回收。"""

    async def scenario() -> None:
        """在请求已发出但尚未返回时注销节点，分别模拟断网与调用方取消。"""
        started, release = asyncio.Event(), asyncio.Event()

        async def handle(request: httpx.Request) -> httpx.Response:
            """能力探测正常，仅业务请求在指定窗口失败。"""
            if request.url.path == "/blocked":
                started.set()
                await release.wait()
                raise httpx.ConnectError("disconnected", request=request)
            return await _FakeV1Upstream("worker", ("flash",)).handle(request)

        app = create_app(_settings("http://worker"), transport=httpx.MockTransport(handle))
        async with _client(app) as client:
            pool = app.state.worker_pool
            worker = pool.workers[0]
            operation = asyncio.create_task(request_upstream(pool, worker, "GET", "/blocked"))
            await asyncio.wait_for(started.wait(), 5)
            assert (await client.delete(f"{WORKERS}/{worker.worker_id}", headers=ADMIN)).status_code == 202
            if cancel:
                operation.cancel()
            else:
                release.set()
            with pytest.raises(asyncio.CancelledError if cancel else RouterProxyError):
                await operation
            assert worker.active_requests == 0
            app.state.collect_resources_once()
            assert not pool.workers

    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["complete", "cancel", "headers_fail"])
def test_stream_holds_worker_until_asgi_response_finishes(ending: str) -> None:
    """流式下载持有节点直到整个 ASGI 响应结束，含 body 未启动与中途取消两种失败。"""

    async def scenario() -> None:
        """使用真实 httpx 流配合 ASGI send 控制下载窗口，验证关闭与引用释放。"""
        body_started, release = asyncio.Event(), asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            """记录流是否被关闭，并把正文传输停在第一块之后。"""

            closed = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                """先返回一块正文，再等待测试控制结束。"""
                yield b"first"
                body_started.set()
                await release.wait()
                yield b"last"

            async def aclose(self) -> None:
                """记录上游流清理，即使正文未消费也应调用。"""
                self.closed = True

        stream = Stream()

        async def handle(request: httpx.Request) -> httpx.Response:
            """下载端点使用阻塞字节流，其他请求使用标准能力响应。"""
            if request.url.path == "/download":
                return httpx.Response(200, stream=stream, request=request)
            return await _FakeV1Upstream("worker", ("flash",)).handle(request)

        async def receive() -> Message:
            """ASGI 2.4 的响应不读取请求体，保留合法 receive 协议。"""
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: Message) -> None:
            """模拟响应头发送失败，其他模式允许正文正常发送。"""
            if ending == "headers_fail" and message["type"] == "http.response.start":
                raise OSError("client disconnected before body")

        app = create_app(_settings("http://worker"), transport=httpx.MockTransport(handle))
        async with _client(app) as client:
            pool = app.state.worker_pool
            worker = pool.workers[0]
            scope = {"type": "http", "method": "GET", "headers": [], "asgi": {"spec_version": "2.4"}}
            response = await stream_upstream(pool, worker, "GET", "/download", request=Request(scope))
            assert (await client.delete(f"{WORKERS}/{worker.worker_id}", headers=ADMIN)).status_code == 202
            operation = asyncio.create_task(response(scope, receive, send))
            if ending != "headers_fail":
                await asyncio.wait_for(body_started.wait(), 5)
                app.state.collect_resources_once()
                assert pool.workers == [worker]
                if ending == "cancel":
                    operation.cancel()
                else:
                    release.set()
            if ending == "complete":
                await operation
            elif ending == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await operation
            else:
                # Starlette 将 ASGI send 的 OSError 包装为 ClientDisconnect。
                from starlette.requests import ClientDisconnect

                with pytest.raises(ClientDisconnect):
                    await operation
            assert stream.closed
            assert worker.active_requests == 0
            app.state.collect_resources_once()
            assert not pool.workers

    asyncio.run(scenario())


def test_reverse_alias_and_route_lease_block_removal() -> None:
    """删除条件覆盖反向别名及过期但仍持有租约的路由，不能只统计正向归属。"""
    app, _ = _fake_app(_settings("http://worker"))
    with TestClient(app) as client:
        registry = app.state.registry
        route = registry.register_file(owner_scope="anonymous", worker_id="other", upstream_id="file-other")
        registry.alias_upstream(route, worker_id="remote-1", upstream_id="alias")
        route.payload = {"expires_at": time.time() - 10}
        route.lease.held = True
        route.lease.retain_until = time.time() - 1
        response = client.delete(f"{WORKERS}/remote-1", headers=ADMIN)
        assert response.status_code == 202
        assert response.json()["retained_routes"] == 1
        route.lease.held = False
        client.portal.call(app.state.collect_resources_once)
        assert client.get(WORKERS, headers=ADMIN).json()["data"] == []


@pytest.mark.parametrize("retention", [0, 60])
@pytest.mark.parametrize("prefix", ["", "/service", "/模型", "/worker%20one", "/%e6%a8%a1%e5%9e%8b", "/50%25done"])
def test_dynamic_cascade_preserves_results_and_routes_new_tasks_elsewhere(tmp_path: Path, retention: int, prefix: str) -> None:
    """动态接入下层 Router，注销后旧结果仍可下载，新任务进入另一节点且注销不会向下传播。"""

    async def scenario() -> None:
        """完整启动两级 Router 和解析 API，检查真实 ZIP、逐跳租约与结果保留后移除。"""
        worker = api_server.create_app(upload_dir=str(tmp_path / "worker"), tiers=("flash",), retention_seconds=1)
        inner = create_app(_settings("http://worker", retention=retention), transport=httpx.ASGITransport(app=worker))
        entry = FastAPI()
        entry.mount(unquote(prefix) or "/", inner)
        apps = {"inner": entry, "worker": worker}

        async def dispatch(request: httpx.Request) -> httpx.Response:
            """顶层可连接下层 Router 或 API，转发仍由实际应用执行。"""
            return await httpx.ASGITransport(app=apps[request.url.host]).handle_async_request(request)

        outer = create_app(_settings(retention=retention), transport=httpx.MockTransport(dispatch))
        async with AsyncExitStack() as stack:
            for app in (worker, inner):
                await stack.enter_async_context(app.router.lifespan_context(app))
            client = await stack.enter_async_context(_client(outer))
            assert (await client.get("/v1/health")).status_code == 503
            registered = await client.post(WORKERS, json={"url": f"http://inner{prefix}"}, headers=ADMIN)
            assert registered.status_code == 201 and registered.json()["healthy"]
            task = await _submit(client)
            detail = (await client.get(f"/v1/tasks/{task['task_id']}")).json()
            uploaded = await client.put(detail["files"][0]["upload"]["url"], content=HTML)
            assert uploaded.status_code == 200
            result = await _terminal(client, task)
            await outer.state.task_store._records[task["task_id"]].operation
            assert result["status"] == "completed", str(result)
            await _check_download(client, result)
            worker_id = registered.json()["worker_id"]
            drained = await client.delete(f"{WORKERS}/{worker_id}", headers=ADMIN)
            assert drained.status_code == 202 and drained.json()["retained_routes"] >= 2
            outer.state.collect_resources_once()
            await maintain_task_result_leases(outer.state.registry, outer.state.worker_pool)
            await maintain_task_result_leases(inner.state.registry, inner.state.worker_pool)
            await _check_download(client, result)
            assert inner.state.worker_pool.workers[0].state == "active"
            assert (await client.post(WORKERS, json={"url": "http://worker"}, headers=ADMIN)).status_code == 201
            second = await _submit(client, filename="another.html")
            second_result = await _terminal(client, second)
            assert second_result["status"] == "completed"
            await _check_download(client, second_result)
            assert len(inner.state.task_store._records) == 1
            routes = [route for route in outer.state.registry.list("job") if route.worker_id == worker_id]
            assert len(routes) == 1
            if retention == 0:
                assert routes[0].lease.retain_until is None
                outer.state.collect_resources_once()
                assert outer.state.worker_pool.get(worker_id).state == "draining"
            else:
                # 模拟结果保留期到期，先正常释放下游租约再执行既有回收循环。
                routes[0].lease.retain_until = time.time() - 1
                routes[0].finalized_at = "2000-01-01T00:00:00Z"
                for route in outer.state.registry.list("file", include_expired=True):
                    if route.worker_id == worker_id:
                        route.payload["expires_at"] = time.time() - 1
                await maintain_task_result_leases(outer.state.registry, outer.state.worker_pool)
                outer.state.collect_resources_once()
                assert [item.worker_id for item in outer.state.worker_pool.workers] == ["remote-2"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("base", "target", "expected"),
    [
        ("/模型", "/模型/v1/tasks/task/uploads/file", "/v1/tasks/task/uploads/file"),
        (
            "/模型",
            "/%e6%a8%a1%e5%9e%8b/v1/tasks/task%2fpart/uploads/file%20name",
            "/v1/tasks/task%2Fpart/uploads/file%20name",
        ),
        ("/worker%20one", "/worker one/v1/files/file/content", "/v1/files/file/content"),
        ("/50%25done", "/50%25done/v1/files/file/content", "/v1/files/file/content"),
    ],
)
def test_worker_resource_paths_normalize_equivalent_encodings(base: str, target: str, expected: str) -> None:
    """原始 Unicode、百分号转义和大小写差异可匹配，资源 ID 内部的编码斜杠保持不变。"""
    worker = WorkerState("worker", normalize_base_url("http://worker" + base), "remote")
    assert _worker_path(worker, "http://worker" + target) == expected


@pytest.mark.parametrize(
    "target",
    [
        "http://other/模型/v1/files/file/content",
        "http://worker:8080/模型/v1/files/file/content",
        "https://worker/模型/v1/files/file/content",
        "http://user:pass@worker/模型/v1/files/file/content",
        "http://worker/模型-other/v1/files/file/content",
        "http://worker/模型%2Foutside/v1/files/file/content",
        "http://worker/模型/v1/files/file/content?token=secret",
        "http://worker/模型/v1/files/file/content#fragment",
        "http://worker/模型/%FF/v1/files/file/content",
    ],
)
def test_worker_resource_path_normalization_preserves_origin_and_prefix_checks(target: str) -> None:
    """编码兼容不放宽同源、端口或路径前缀限制，非法 UTF-8 仍返回结构化代理错误。"""
    worker = WorkerState("worker", normalize_base_url("http://worker/模型"), "remote")
    with pytest.raises(ApiServerError) as error:
        _worker_path(worker, target)
    assert error.value.status_code == 502
    assert error.value.error.code == "invalid_upstream_response"


@pytest.mark.parametrize("cancel", [False, True])
def test_draining_preserves_reserved_submission_and_cancel(tmp_path: Path, cancel: bool) -> None:
    """注销前已预占但尚未返回下游任务 ID 的请求可继续提交，也可沿原链路取消。"""

    async def scenario() -> None:
        """阻塞真实 API 创建之前的窗口，注销后再选择完成或取消原任务。"""
        worker = api_server.create_app(upload_dir=str(tmp_path / "worker"), tiers=("flash",))
        transport = _WorkerTransport(worker, block_submission=True)
        router = create_app(_settings("http://worker"), transport=transport)
        async with worker.router.lifespan_context(worker), _client(router) as client:
            try:
                task = await _submit(client)
                path = f"/v1/tasks/{task['task_id']}"
                detail = (await client.get(path)).json()
                assert (await client.put(detail["files"][0]["upload"]["url"], content=HTML)).status_code == 200
                await asyncio.wait_for(transport.submission_started.wait(), 5)
                drained = await client.delete(f"{WORKERS}/remote-1", headers=ADMIN)
                assert drained.status_code == 202
                assert drained.json()["active_jobs"] == 1
                assert drained.json()["retained_routes"] == 0
                assert (await client.get(path)).status_code == 200
                if cancel:
                    canceled = await client.post(path + "/cancel")
                    assert canceled.status_code == 200 and canceled.json()["status"] == "canceled"
                transport.allow_submission.set()
                terminal = await _terminal(client, task)
                assert terminal["status"] == ("canceled" if cancel else "completed")
                await router.state.task_store._records[task["task_id"]].operation
                assert router.state.worker_pool.get("remote-1").active_jobs == 0
                if cancel:
                    assert not worker.state.execution_store._executions
                    router.state.collect_resources_once()
                    assert not router.state.worker_pool.workers
                else:
                    await _check_download(client, terminal)
            finally:
                transport.allow_submission.set()

    asyncio.run(scenario())


def test_waiting_upload_fails_explicitly_when_last_worker_removed(tmp_path: Path) -> None:
    """允许移除最后一个空闲节点；未分派文件随后上传时明确失败，不改档或隐式排队。"""

    async def scenario() -> None:
        """先受理待上传任务再移除唯一节点，确认既有失败契约与管理入口仍有效。"""
        worker = api_server.create_app(upload_dir=str(tmp_path / "worker"), tiers=("flash",))
        router = create_app(_settings("http://worker"), transport=httpx.ASGITransport(app=worker))
        async with worker.router.lifespan_context(worker), _client(router) as client:
            task = await _submit(client)
            detail = (await client.get(f"/v1/tasks/{task['task_id']}")).json()
            assert (await client.delete(f"{WORKERS}/remote-1", headers=ADMIN)).status_code == 204
            assert (await client.put(detail["files"][0]["upload"]["url"], content=HTML)).status_code == 200
            result = await _terminal(client, task)
            assert result["status"] == "failed"
            assert result["files"][0]["error"]["code"] == "quality_tier_unavailable"
            assert (await client.get(WORKERS, headers=ADMIN)).status_code == 200

    asyncio.run(scenario())


def test_background_collection_removes_drained_node_without_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    """永久保留模式仍运行节点回收循环，无资源节点的请求引用释放后可以自动移除。"""
    from mineru.kit.router import app as router_app

    monkeypatch.setattr(router_app, "RETENTION_SCAN_INTERVAL_SECONDS", 0.01)

    async def scenario() -> None:
        """仅持有请求引用触发 draining，释放后等待实际后台循环删除节点。"""
        app, _ = _fake_app(_settings("http://worker", retention=0))
        async with _client(app) as client:
            worker = app.state.worker_pool.workers[0]
            reference = worker.acquire_request()
            assert (await client.delete(f"{WORKERS}/{worker.worker_id}", headers=ADMIN)).status_code == 202
            reference.release()
            async with asyncio.timeout(5):
                while app.state.worker_pool.workers:
                    await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_cli_admin_key_precedence_and_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式管理 Key 优先于环境变量，reload 工厂恢复相同设置且不改变运行期清单语义。"""
    from typer.testing import CliRunner

    from mineru.kit.main import app as cli_app
    from mineru.kit.router import cli

    monkeypatch.setenv("MINERU_ROUTER_ADMIN_API_KEY", "environment-key")
    seen: list[RouterSettings] = []

    def run(target: FastAPI | str, **kwargs: Any) -> None:
        """截获启动设置，reload 模式从真实环境恢复工厂配置。"""
        settings = RouterSettings.from_env() if isinstance(target, str) else target.state.settings
        seen.append(settings)
        assert kwargs["reload"] is True

    monkeypatch.setattr(cli.uvicorn, "run", run)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["router", "--reload", "--local-gpus", "none"])
    assert result.exit_code == 0, result.output
    assert seen[-1].admin_api_key == "environment-key"
    result = runner.invoke(cli_app, ["router", "--reload", "--local-gpus", "none", "--admin-api-key", "explicit-key"])
    assert result.exit_code == 0, result.output
    assert seen[-1].admin_api_key == "explicit-key"
    assert "explicit-key" not in result.output + repr(seen[-1])
