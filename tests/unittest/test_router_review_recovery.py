"""复现代理环境、首次启动重试及 Router 提前退出的恢复边界。"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import httpx
import pytest

from mineru.kit.gradio.client import ManagedLocalRouter
from mineru.kit.router.workers import ManagedLocalWorker, RouterSettings, WorkerPool, WorkerState
from mineru.parser.process_tree import _group_running


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
