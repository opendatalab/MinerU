"""通过真实托管 Router/worker 进程验证 IPC 路径与预加载错误传递，不加载 GPU 模型。"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener
from pathlib import Path
from uuid import uuid4

from mineru.parser.process_control import ManagedProcessControlWatcher


def run_worker(port: int, marker: Path, mode: str) -> None:
    """用标准库真实创建 IPC，或按正式 API 错误协议报告预加载失败。"""
    scratch = tempfile.gettempdir()
    error = None
    evidence = {"pid": os.getpid(), "scratch": scratch, "router_pid": os.getppid()}
    if mode == "preload_error":
        error = {"code": "model_preload_dependency_missing", "message": "startup probe: dependency unavailable"}
    else:
        path = str(Path(scratch) / str(uuid4()))
        evidence["socket_bytes"] = len(os.fsencode(path))
        try:
            with socket.socket(socket.AF_UNIX) as endpoint:
                endpoint.bind(path)
            Path(path).unlink()
            with Listener(family="AF_UNIX") as listener:
                evidence["listener_bytes"] = len(os.fsencode(listener.address))
        except OSError as exc:
            error = {"code": "model_preload_failed", "message": str(exc)}
    evidence["error"] = error
    marker.write_text(json.dumps(evidence))

    class Handler(BaseHTTPRequestHandler):
        """实现 Router 就绪探测使用的最小真实 HTTP 协议。"""

        def do_GET(self) -> None:
            """保留 API 的 503 错误结构和成功时所需的能力发现端点。"""
            if error:
                status, payload = 503, {"error": error}
            elif self.path == "/v1/health":
                status, payload = 200, {"status": "ok", "features": {"sources": ["upload"]}}
            else:
                status, payload = 200, {"data": [{"id": "flash"}] if self.path == "/v1/tiers" else []}
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """由测试保存结构化证据，不输出每次轮询的重复访问日志。"""

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        control = ManagedProcessControlWatcher.from_environment(server.shutdown)
        control.start()
        try:
            server.serve_forever(poll_interval=0.05)
        finally:
            control.close()


def main() -> None:
    """Router 使用正式私有入口，只将其 worker 命令替换成受控探针。"""
    role, port, marker, mode = sys.argv[1:]
    if role == "worker":
        run_worker(int(port), Path(marker), mode)
        return
    from mineru.kit.gradio.managed_router import main as router_main
    from mineru.kit.router.workers import ManagedLocalWorker

    def command(worker: ManagedLocalWorker, worker_port: int, upload_dir: Path) -> list[str]:
        """保持真实启动环境、进程所有权和 HTTP 就绪检查，仅替换模型初始化。"""
        return [sys.executable, __file__, "worker", str(worker_port), marker, mode]

    ManagedLocalWorker.command = command
    sys.argv = ["managed-router-probe", "--port", port]
    router_main()


if __name__ == "__main__":
    main()
