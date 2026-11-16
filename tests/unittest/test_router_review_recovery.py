"""复现代理环境、首次启动重试及 Router 提前退出的恢复边界。"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from mineru.kit.gradio.client import ManagedLocalRouter, V1ArtifactClient, V1ArtifactError
from mineru.kit.router.app import create_app
from mineru.kit.router.workers import ManagedLocalWorker, RouterSettings, WorkerPool, WorkerState
from mineru.parser.managed_temp import create_managed_temp_dir
from mineru.parser.process_tree import _group_running
from mineru.tasks.contracts import ErrorDetail
from mineru.tasks.errors import ApiServerError


@contextmanager
def http_endpoint(label: str) -> Iterator[str]:
    """提供可区分直连与代理路径的真实 HTTP 服务，不用 transport 替身绕过代理选择。"""

    class Handler(BaseHTTPRequestHandler):
        """返回当前服务标签，记录请求是否真正绕经代理。"""

        def do_GET(self) -> None:
            """给直连请求和代理的绝对路径请求返回同一简单响应。"""
            self.send_response(200)
            self.end_headers()
            self.wfile.write(label.encode())

        def log_message(self, format: str, *args: object) -> None:
            """关闭无关的访问日志，失败信息由测试断言提供。"""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("proxy_variable", ["HTTP_PROXY", "ALL_PROXY"])
def test_managed_router_bypasses_proxy_without_changing_download_environment(
    monkeypatch: pytest.MonkeyPatch,
    managed: bool,
    proxy_variable: str,
) -> None:
    """托管回环请求绕过代理；普通 Router 和模型下载的环境代理保持原语义。"""
    for key in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.upper(), raising=False)
    with http_endpoint("worker") as worker_url, http_endpoint("proxy") as proxy_url:
        monkeypatch.setenv(proxy_variable, proxy_url)

        async def scenario() -> None:
            """使用正式池的共享 client 验证探测和转发共用的网络路径。"""
            pool = WorkerPool(RouterSettings(local_gpus="none", managed_webui=managed))
            try:
                response = await pool.client.get(f"{worker_url}/v1/health")
                assert response.text == ("worker" if managed else "proxy")
                assert os.environ[proxy_variable] == proxy_url
            finally:
                await pool.close()

        asyncio.run(scenario())


def test_initial_start_retry_does_not_restart_the_healthy_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """首次启动失败后的成功重试必须结束恢复状态，不能再次失效并加载模型。"""

    async def scenario() -> None:
        """注入一次启动失败，随后让正式监督循环检查已恢复实例。"""
        starts, stops, invalidated = [], [], []

        async def start(local: ManagedLocalWorker, client: httpx.AsyncClient) -> None:
            """第一次没有成功创建进程，重试时发布可用实例。"""
            starts.append(True)
            if len(starts) == 1:
                raise RuntimeError("first startup failed")
            local.process = SimpleNamespace(poll=lambda: None)
            local.base_url = "http://worker"

        async def stop(local: ManagedLocalWorker) -> None:
            """记录意外的重复回收，不创建真实模型进程。"""
            stops.append(True)
            local.process = None

        async def invalidate(worker: WorkerState) -> None:
            """记录任务失效所属实例代次。"""
            invalidated.append(worker.generation)

        monkeypatch.setattr(ManagedLocalWorker, "start", start)
        monkeypatch.setattr(ManagedLocalWorker, "stop", stop)
        pool = WorkerPool(
            RouterSettings(local_gpus="inherit", worker_refresh_interval_seconds=0),
            on_local_worker_replaced=invalidate,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok", "data": []})),
        )
        try:
            await pool.start()
            pool._supervisor_task.cancel()
            await asyncio.gather(pool._supervisor_task, return_exceptions=True)
            worker = pool.workers[0]
            assert worker.restart_requested and worker.generation == 0
            worker.retry_at = 0
            await pool.refresh(worker)
            assert worker.healthy and not worker.restart_requested
            for _ in range(2):
                await pool.supervise_once()
                if worker.recovery_task is not None:
                    await worker.recovery_task
            assert len(starts) == 2 and not stops
            assert worker.generation == 1 and 1 not in invalidated
        finally:
            await pool.close()

    asyncio.run(scenario())


@pytest.mark.skipif(os.name == "nt", reason="POSIX 会话和进程组验收")
@pytest.mark.parametrize("action", ["exit", "stop_worker"])
def test_owned_worker_cleanup_survives_router_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    """Router 先退出仍回收 worker/模型；单独回收 worker 时 Router 必须保持存活。"""
    marker, gate = tmp_path / "pids.json", tmp_path / "action"
    fixture = Path("tests/fixtures/owned_worker_tree.py").resolve()
    # 独立脚本进程需要导入本次工作区，而不是环境中其它可编辑安装的快照。
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))

    def command(server: ManagedLocalRouter, port: int, upload_dir: Path) -> list[str]:
        """替换 Router 入口，嵌套 worker 仍使用正式生命周期实现。"""
        return [sys.executable, str(fixture), str(marker), str(gate)]

    def wait_ready(server: ManagedLocalRouter) -> None:
        """以模型子进程实际创建作为故障注入起点。"""
        until = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < until:
            assert server.process.poll() is None
            time.sleep(0.01)
        assert marker.exists()

    monkeypatch.setattr(ManagedLocalRouter, "_command", command)
    monkeypatch.setattr(ManagedLocalRouter, "_wait_until_ready", wait_ready)
    server = ManagedLocalRouter(tier="flash")
    worker_pid = None
    try:
        server.start()
        worker_pid, model_pid = json.loads(marker.read_text())
        assert os.getpgid(model_pid) == worker_pid
        root = Path(server._temp_dir.name)
        gate.write_text(action)
        if action == "exit":
            server.process.wait(timeout=5)
            assert _group_running(worker_pid)
        else:
            until = time.monotonic() + 5
            while gate.read_text() != "stopped" and time.monotonic() < until:
                time.sleep(0.01)
            assert gate.read_text() == "stopped"
            assert server.process.poll() is None
            assert not _group_running(worker_pid)
        server.stop()
        assert not _group_running(worker_pid)
        assert not root.exists()
    finally:
        # 断言失败时也显式清理注入的进程组，不能把回归测试变成真实资源泄漏。
        if worker_pid is not None and _group_running(worker_pid):
            os.killpg(worker_pid, signal.SIGKILL)
        server.stop()


@pytest.mark.skipif(os.name == "nt", reason="Unix socket 路径和真实 POSIX 进程验收")
@pytest.mark.parametrize("mode", ["ipc", "preload_error"])
@pytest.mark.parametrize("long_temp_root", [False, True])
def test_managed_startup_ipc_and_preload_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    long_temp_root: bool,
) -> None:
    """真实两级启动检查 IPC 可用、长环境路径回退、错误及时透传和所有临时目录回收。"""
    marker = tmp_path / "startup.json"
    ambient = tmp_path / ("长临时目录" * 12) if long_temp_root else Path("/tmp")
    ambient.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(ambient))
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    monkeypatch.setenv("MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS", "8")
    fixture = Path("tests/fixtures/managed_startup_probe.py").resolve()
    roots = []

    def command(server: ManagedLocalRouter, port: int, upload_dir: Path) -> list[str]:
        """保存最外层目录，验证嵌套目录没有因缩短路径而逃出父进程所有权。"""
        roots.append(Path(server._temp_dir.name))
        return [sys.executable, str(fixture), "router", str(port), str(marker), mode]

    monkeypatch.setattr(ManagedLocalRouter, "_command", command)
    server = ManagedLocalRouter(tier="flash", preload_models=True)
    started = time.monotonic()
    try:
        if mode == "preload_error":
            with pytest.raises(V1ArtifactError) as caught:
                server.start()
            assert caught.value.code == "model_preload_dependency_missing"
            assert str(caught.value) == "startup probe: dependency unavailable"
            assert time.monotonic() - started < 5
            assert server.process is None
        else:
            server.start()
        evidence = json.loads(marker.read_text())
        assert Path(evidence["scratch"]).is_relative_to(roots[0])
        if mode == "ipc":
            assert evidence["error"] is None
            assert evidence["socket_bytes"] < 104 and evidence["listener_bytes"] < 104
    finally:
        server.stop()
    assert roots and all(not root.exists() for root in roots)
    assert not _group_running(evidence["pid"])
    assert not _group_running(evidence["router_pid"])


@pytest.mark.parametrize("code", ["model_preload_dependency_missing", "model_preload_files_missing", "model_preload_failed"])
def test_worker_preload_error_keeps_original_detail(code: str) -> None:
    """明确的预加载错误只探测一次，并使用公共结构化异常保留错误身份。"""

    async def scenario() -> None:
        """直接覆盖正式 worker 等待入口，不把预加载失败归类成连接超时。"""
        calls = []

        def health(request: httpx.Request) -> httpx.Response:
            """返回 API 已完成初始化后的稳定错误响应。"""
            calls.append(request)
            return httpx.Response(503, json={"error": {"code": code, "message": "preload failed"}})

        worker = ManagedLocalWorker("probe", "127.0.0.1", None, RouterSettings())
        worker.base_url = "http://worker"
        async with httpx.AsyncClient(transport=httpx.MockTransport(health)) as client:
            with pytest.raises(ApiServerError) as caught:
                await worker.wait_until_ready(client, timeout_seconds=0.3)
        assert caught.value.error.code == code and caught.value.error.message == "preload failed"
        assert len(calls) == 1

    asyncio.run(scenario())


def test_worker_startup_keeps_waiting_for_transient_failures() -> None:
    """网络未就绪、普通 503 以及非 JSON 响应不能误触发永久预加载失败。"""

    async def scenario() -> None:
        """让同一实例经过短暂不可用后成功就绪。"""
        replies = [
            httpx.ConnectError("still starting"),
            httpx.Response(503, json={"error": {"code": "starting", "message": "loading"}}),
            httpx.Response(503, text="not ready"),
            httpx.Response(200),
        ]

        def health(request: httpx.Request) -> httpx.Response:
            """按实际启动时序给出连接失败及后续 HTTP 状态。"""
            response = replies.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        worker = ManagedLocalWorker("probe", "127.0.0.1", None, RouterSettings())
        worker.base_url = "http://worker"
        async with httpx.AsyncClient(transport=httpx.MockTransport(health)) as client:
            await worker.wait_until_ready(client, timeout_seconds=2)
        assert not replies

    asyncio.run(scenario())


def test_waiting_webui_request_surfaces_recovery_preload_error() -> None:
    """已排队的 WebUI 请求也应及时看到恢复阶段的预加载失败。"""

    async def scenario() -> None:
        """用与 Router health 相同的响应验证正在等待的客户端收尾。"""
        client = V1ArtifactClient(
            api_url="http://router",
            managed=True,
            recovery_timeout_seconds=0.01,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    503,
                    json={"error": {"code": "model_preload_device_unavailable", "message": "device unavailable"}},
                )
            ),
        )
        with pytest.raises(V1ArtifactError) as caught:
            await client.wait_for_ready(lambda message: None)
        assert caught.value.code == "model_preload_device_unavailable"
        assert str(caught.value) == "device unavailable"

    asyncio.run(scenario())


@pytest.mark.parametrize("managed", [False, True])
def test_preload_failure_is_cleared_after_successful_retry(monkeypatch: pytest.MonkeyPatch, managed: bool) -> None:
    """托管 UI 保留原始错误，普通 Router 保留聚合语义；成功重试后不再发布过时失败。"""
    starts = []

    async def start(worker: ManagedLocalWorker, client: httpx.AsyncClient) -> None:
        """第一次返回真实类型的预加载错误，第二次模拟重新配置后成功。"""
        starts.append(True)
        if len(starts) == 1:
            raise ApiServerError(503, ErrorDetail(code="model_preload_files_missing", message="missing model"))
        worker.base_url = "http://worker"
        worker.process = SimpleNamespace(poll=lambda: None)

    async def stop(worker: ManagedLocalWorker) -> None:
        """仅回收测试的进程状态，不执行系统进程操作。"""
        worker.process = None

    monkeypatch.setattr(ManagedLocalWorker, "start", start)
    monkeypatch.setattr(ManagedLocalWorker, "stop", stop)
    app = create_app(
        RouterSettings(local_gpus="inherit", managed_webui=managed, worker_refresh_interval_seconds=0),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok", "data": []})),
    )
    with TestClient(app) as client:
        pool = app.state.worker_pool
        worker = pool.workers[0]
        response = client.get("/v1/health")
        assert response.status_code == 503
        expected = "model_preload_files_missing" if managed else "upstream_unavailable"
        assert response.json()["error"]["code"] == expected
        worker.retry_at = 0
        client.portal.call(pool.refresh, worker)
        assert worker.startup_error is None and worker.restart_failures == 0
        assert client.get("/v1/health").status_code == 200
        assert len(starts) == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX IPC 路径预算")
def test_temp_root_reserves_space_for_owned_nested_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """父目录单独够短但两层会超长时，必须提前选根，不能让 worker 跳出父目录。"""
    with tempfile.TemporaryDirectory(prefix="ambient-", dir="/tmp") as base:
        ambient = Path(base) / "12345678"
        ambient.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(ambient))
        with create_managed_temp_dir(reserve_worker=True) as outer:
            scratch = Path(outer) / "tmp"
            scratch.mkdir()
            monkeypatch.setattr(tempfile, "tempdir", str(scratch))
            with create_managed_temp_dir() as worker:
                assert Path(worker).is_relative_to(outer)
                assert len(os.fsencode(Path(worker) / "tmp")) <= 60
