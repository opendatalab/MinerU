"""共享真实 TCP 测试服务及统一 HTTP 示例入口回归。旧协议场景已迁入 test_task_http。"""

from __future__ import annotations

import socket
import threading
import time
import subprocess
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI


class _Server:
    """持有已绑定的本地 socket，在独立线程运行真实 FastAPI 服务。"""

    def __init__(self, app: Any, host: str = "127.0.0.1") -> None:
        """从分配端口到 Uvicorn 接管始终保留 socket，消除重复绑定的竞争窗口。"""
        import uvicorn

        self._socket = socket.socket()
        try:
            self._socket.bind((host, 0))
            self.port = self._socket.getsockname()[1]
            self.host = host
            self.url = f"http://{host}:{self.port}"
            self._server = uvicorn.Server(uvicorn.Config(app, host=host, port=self.port, log_level="error"))
            self._thread = threading.Thread(target=self._server.run, kwargs={"sockets": [self._socket]}, daemon=True)
        except BaseException:
            self._socket.close()
            raise

    def start(self, timeout: float = 20.0) -> None:
        """等待监听就绪；启动退出或超时时释放线程和预留 socket。"""
        try:
            self._thread.start()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._server.started:
                    return
                if not self._thread.is_alive():
                    raise RuntimeError("test server exited before startup")
                time.sleep(0.05)
            raise RuntimeError("test server did not start within timeout")
        except BaseException:
            self.stop()
            raise

    def stop(self, timeout: float = 10.0) -> None:
        """回收已启动的服务，也允许清理尚未启动或已经关闭的实例。"""
        self._server.should_exit = True
        try:
            if self._thread.ident is not None:
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    raise RuntimeError("test server did not stop within timeout")
        finally:
            self._socket.close()


def test_server_reserves_port_before_start() -> None:
    """服务创建后持续占用端口，并把同一 socket 交给真实 HTTP 服务。"""
    server = _Server(FastAPI())
    try:
        with socket.socket() as contender:
            with pytest.raises(OSError):
                contender.bind((server.host, server.port))
        server.start()
        with httpx.Client(trust_env=False) as client:
            assert client.get(f"{server.url}/openapi.json").status_code == 200
    finally:
        server.stop()


def test_http_example_delegates_to_tasks(tmp_path: Path) -> None:
    """原脚本入口执行同一 Tasks 客户端并保留退出码，不残留旧 HTTP 流程。"""
    from mineru.parser.api_server import create_app

    server = _Server(create_app(upload_dir=str(tmp_path / "api"), tiers=("flash",)))
    source = tmp_path / "source.html"
    source.write_text("<h1>Unified HTTP example</h1>")
    server.start()
    try:
        script = Path(__file__).resolve().parents[2] / "scripts/http_api_example.sh"
        result = subprocess.run(
            ["bash", str(script), str(source)],
            text=True,
            capture_output=True,
            timeout=30,
            env={
                **os.environ,
                "MINERU_API_URL": server.url,
                "MINERU_TIER": "flash",
                "MINERU_API_KEY": "",
                "RESULT_DIR": str(tmp_path / "result"),
                "POLL_INTERVAL": "0.01",
            },
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert list((tmp_path / "result").rglob("*.zip"))
    finally:
        server.stop()
