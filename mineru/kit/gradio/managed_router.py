"""WebUI 的私有 Router 进程入口，复用正式 Router 应用与 worker 恢复。"""

from __future__ import annotations

import argparse
import threading

import uvicorn

from ...parser.process_control import ManagedProcessControlWatcher
from ...utils.server import ServerWithDocs
from ...utils.stdio import configure_standard_streams
from ...utils.logger import configure_global_log_level
from ..router.app import create_app
from ..router.workers import RouterSettings


def main() -> None:
    """启动仅监听回环地址的 Router；父进程退出时沿正式 lifespan 关闭 worker。"""
    configure_standard_streams()
    configure_global_log_level()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    settings = RouterSettings.from_env()
    if not settings.managed_webui or settings.local_gpus != "inherit" or settings.upstream_urls:
        raise ValueError("Managed WebUI requires exactly one inherited local worker")
    stopped = threading.Event()
    server_ref: list[uvicorn.Server] = []

    def shutdown() -> None:
        """控制线程只请求退出，进程强制回收由拥有实例的父进程负责。"""
        stopped.set()
        if server_ref:
            server_ref[0].should_exit = True

    control = ManagedProcessControlWatcher.from_environment(shutdown)
    if control is not None:
        control.start()
    try:
        server = ServerWithDocs(uvicorn.Config(create_app(settings), host="127.0.0.1", port=args.port, log_level="warning"))
        server_ref.append(server)
        server.should_exit = stopped.is_set()
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        if control is not None:
            control.close()


if __name__ == "__main__":
    main()

__all__ = ["main"]
