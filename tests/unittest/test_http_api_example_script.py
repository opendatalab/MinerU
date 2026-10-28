"""共享真实 TCP 测试服务及统一 HTTP 示例入口回归。旧协议场景已迁入 test_task_http。"""

from __future__ import annotations

import socket
import threading
import time
import subprocess
import os
from pathlib import Path
from typing import Any


class _Server:
    """Run a FastAPI app on a pre-picked localhost port in a daemon thread."""

    def __init__(self, app: Any, host: str = "127.0.0.1") -> None:
        import uvicorn

        with socket.socket() as sock:
            sock.bind((host, 0))
            self.port = sock.getsockname()[1]
        self.host = host
        self.url = f"http://{host}:{self.port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host=host, port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout: float = 20.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("test server did not start within timeout")

    def stop(self, timeout: float = 10.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout)


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
