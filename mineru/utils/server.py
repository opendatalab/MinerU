"""在 Uvicorn 成功监听后展示可直接访问的 API 文档地址。"""

from __future__ import annotations

import logging
import socket

import uvicorn
from starlette.types import ASGIApp
from uvicorn.main import STARTUP_FAILURE
from uvicorn.supervisors import ChangeReload, Multiprocess

logger = logging.getLogger("uvicorn.error")


class ServerWithDocs(uvicorn.Server):
    """沿用 Uvicorn 生命周期，从实际监听端口生成浏览器文档链接。"""

    def __init__(self, config: uvicorn.Config, *, docs_path: str | None = "/docs", docs_require_auth: bool = False) -> None:
        """接收实际文档路径与认证要求，关闭文档时传入 None。"""
        super().__init__(config)
        self.docs_path = docs_path
        self.docs_require_auth = docs_require_auth

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        """成功绑定后输出文档链接，通配监听地址转换为本机可访问地址。"""
        await super().startup(sockets=sockets)
        if not self.started or self.should_exit or self.docs_path is None:
            return
        scheme = "https" if self.config.is_ssl else "http"
        auth_note = " (requires Bearer authentication)" if self.docs_require_auth else ""
        for server in self.servers:
            for listener in server.sockets or ():
                if listener.family not in {socket.AF_INET, socket.AF_INET6}:
                    continue
                host, port = listener.getsockname()[:2]
                host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
                authority = f"[{host}]" if ":" in host else host
                logger.info("API documentation available at %s://%s:%d%s%s", scheme, authority, port, self.docs_path, auth_note)


def run_server(
    app: ASGIApp | str,
    *,
    host: str,
    port: int,
    reload: bool = False,
    factory: bool = False,
) -> None:
    """统一启动带文档提示的 API，保留 Uvicorn 的重载、多进程和正常中断处理。"""
    config = uvicorn.Config(app, host=host, port=port, reload=reload, factory=factory)
    if (config.reload or config.workers > 1) and not isinstance(app, str):
        logger.warning("You must pass the application as an import string to enable 'reload' or 'workers'.")
        raise SystemExit(1)
    server = ServerWithDocs(config)
    bound_socket: socket.socket | None = None
    try:
        if config.should_reload:
            bound_socket = config.bind_socket()
            ChangeReload(config, target=server.run, sockets=[bound_socket]).run()
        elif config.workers > 1:
            bound_socket = config.bind_socket()
            Multiprocess(config, target=server.run, sockets=[bound_socket]).run()
        else:
            server.run()
    except KeyboardInterrupt:
        pass
    finally:
        if bound_socket is not None:
            bound_socket.close()
    if not server.started and not config.should_reload and config.workers == 1:
        raise SystemExit(STARTUP_FAILURE)


__all__ = ["ServerWithDocs", "run_server"]
