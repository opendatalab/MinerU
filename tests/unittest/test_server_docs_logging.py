"""验证文档提示使用真实监听地址，并遵守启动结果和文档开关。"""

from __future__ import annotations

import asyncio
import logging
import socket

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from mineru.utils import server as server_utils


@pytest.mark.parametrize(
    ("host", "expected_host"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "127.0.0.1"),
        pytest.param("::1", "[::1]", marks=pytest.mark.skipif(not socket.has_ipv6, reason="系统未启用 IPv6")),
        pytest.param("::", "[::1]", marks=pytest.mark.skipif(not socket.has_ipv6, reason="系统未启用 IPv6")),
    ],
)
@pytest.mark.parametrize(("docs_path", "requires_auth"), [("/docs", False), ("/api/docs", True)])
def test_docs_url_uses_bound_port_and_connectable_host(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    expected_host: str,
    docs_path: str,
    requires_auth: bool,
) -> None:
    """真实绑定随机端口，验证通配地址、IPv6、自定义文档路径和 HTTP 可达性。"""
    monkeypatch.setattr(server_utils, "logger", logging.getLogger("test_server_docs"))

    async def scenario() -> None:
        """启动实际 HTTP 监听并访问日志中的文档链接，最后执行完整关闭。"""
        config = uvicorn.Config(FastAPI(docs_url=docs_path), host=host, port=0, log_config=None)
        config.load()
        server = server_utils.ServerWithDocs(config, docs_path=docs_path, docs_require_auth=requires_auth)
        server.lifespan = config.lifespan_class(config)
        await server.startup()
        try:
            port = server.servers[0].sockets[0].getsockname()[1]
            docs_url = f"http://{expected_host}:{port}{docs_path}"
            assert f"API documentation available at {docs_url}" in caplog.text
            assert ("requires Bearer authentication" in caplog.text) is requires_auth
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(docs_url)
                assert response.status_code == 200
        finally:
            await server.shutdown()

    with caplog.at_level(logging.INFO, logger="test_server_docs"):
        asyncio.run(scenario())


def test_disabled_docs_have_no_startup_link(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """文档关闭后，成功监听也不打印无法访问的链接。"""
    monkeypatch.setattr(server_utils, "logger", logging.getLogger("test_server_docs"))

    async def scenario() -> None:
        """为关闭文档的应用建立真实监听，核对日志并回收资源。"""
        config = uvicorn.Config(FastAPI(docs_url=None, openapi_url=None), port=0, log_config=None)
        config.load()
        server = server_utils.ServerWithDocs(config, docs_path=None)
        server.lifespan = config.lifespan_class(config)
        await server.startup()
        await server.shutdown()

    with caplog.at_level(logging.INFO, logger="test_server_docs"):
        asyncio.run(scenario())
    assert "API documentation available at" not in caplog.text


def test_failed_bind_has_no_docs_link(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """端口被占用导致启动失败时，禁止提前打印成功访问提示。"""
    monkeypatch.setattr(server_utils, "logger", logging.getLogger("test_server_docs"))
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()

        async def scenario() -> None:
            """让服务绑定已占用的端口，保留 Uvicorn 原有失败退出行为。"""
            config = uvicorn.Config(FastAPI(), port=occupied.getsockname()[1], log_config=None)
            config.load()
            server = server_utils.ServerWithDocs(config)
            server.lifespan = config.lifespan_class(config)
            with pytest.raises(SystemExit):
                await server.startup()

        with caplog.at_level(logging.INFO, logger="test_server_docs"):
            asyncio.run(scenario())
    assert "API documentation available at" not in caplog.text
