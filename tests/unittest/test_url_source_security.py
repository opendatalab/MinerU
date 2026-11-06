"""URL 来源的公网边界、DNS 固定、代理隔离及 API/Router 回显回归。"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
import test_task_api
from test_http_api_example_script import _Server
from test_task_api import _archive, _execution_apps, _terminal

from mineru.errors import MineruError
from mineru.parser import api_server, url_source

HTML = b"<h1>Public source</h1><p>Validated download content.</p>"
service = test_task_api.service


@pytest.fixture
def dns(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
    """模拟确定的 DNS 应答并记录解析次数，实际地址策略仍由生产代码执行。"""
    addresses = ["93.184.216.34"]
    calls: list[str] = []

    async def resolve(
        self: asyncio.BaseEventLoop, host: str, port: int, **kwargs: Any
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        """允许测试混合 A/AAAA、重复地址和 DNS 重绑定后的第二次响应。"""
        calls.append(host)
        return [
            (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))
            for address in addresses
        ]

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolve)
    return addresses, calls


@pytest.fixture
def connections(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """仅替换最底层 HTTP 发送，观察安全传输最终交给网络的目标和 TLS 参数。"""
    calls: list[httpx.Request] = []

    async def respond(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        """记录已固定的网络请求，并返回可由真实解析器处理的 HTML。"""
        calls.append(request)
        return httpx.Response(200, content=HTML, headers={"content-type": "text/html; charset=utf-8"})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    return calls


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "169.254.169.254",
        "169.254.170.2",
        "100.100.100.200",
        "0.0.0.0",
        "192.0.0.8",
        "192.88.99.1",
        "192.0.2.1",
        "198.18.0.1",
        "224.0.0.1",
        "240.0.0.1",
        "255.255.255.255",
        "168.63.129.16",
        "::",
        "::1",
        "fe80::1",
        "fd00:ec2::254",
        "ff02::1",
        "2001:db8::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "64:ff9b::7f00:1",
        "64:ff9b:1::7f00:1",
        "2002:7f00:1::",
        "2001::7f00:1",
        "2001:4860:4860::8888%eth0",
    ],
)
def test_reject_non_public_addresses(address: str) -> None:
    """覆盖内网、元数据、IPv4 映射、转换机制和作用域地址的常见绕过。"""
    with pytest.raises(MineruError) as caught:
        url_source._public_address(address)
    assert caught.value.code == "unsupported_source"


@pytest.mark.parametrize("address", ["93.184.216.34", "2606:4700:4700::1111", "::ffff:93.184.216.34"])
def test_accept_public_addresses(address: str) -> None:
    """允许公网 IPv4/IPv6，并把映射公网 IPv4 归一化为同一连接目标。"""
    result = url_source._public_address(address)
    assert result == ("93.184.216.34" if address.startswith("::ffff:") else address)


@pytest.mark.parametrize("host", ["127.1", "2130706433", "0x7f000001", "[::ffff:127.0.0.1]", "[::1]"])
def test_system_resolver_cannot_bypass_loopback_policy(host: str, tmp_path: Path) -> None:
    """真实系统解析器把缩写、整数和映射形式归一化后，仍必须拒绝回环目标。"""
    with pytest.raises(MineruError, match="public unicast"):
        asyncio.run(
            api_server._extract_bytes(api_server._UrlSource(url=f"https://{host}/internal"), api_server.FileStore(tmp_path))
        )


def test_scoped_ipv6_is_rejected_before_dns(dns: tuple[list[str], list[str]], tmp_path: Path) -> None:
    """拒绝带接口编号的 IPv6，避免解析器剥离作用域后丢失原始限制信息。"""
    with pytest.raises(MineruError, match="zone identifier"):
        asyncio.run(
            api_server._extract_bytes(
                api_server._UrlSource(url="https://[2001:4860:4860::8888%251]/"), api_server.FileStore(tmp_path)
            )
        )
    assert dns[1] == []


@pytest.mark.parametrize("private", ["127.0.0.1", "10.0.0.5", "::1", "::ffff:169.254.169.254"])
def test_reject_entire_mixed_dns_answer(
    dns: tuple[list[str], list[str]], connections: list[httpx.Request], tmp_path: Path, private: str
) -> None:
    """任一解析结果受限时，在建立首个公网连接前拒绝整个 URL。"""
    addresses, calls = dns
    addresses.append(private)
    with pytest.raises(MineruError, match="public unicast"):
        asyncio.run(
            api_server._extract_bytes(
                api_server._UrlSource(url="https://source.example/opaque"), api_server.FileStore(tmp_path)
            )
        )
    assert calls == ["source.example"]
    assert connections == []


def test_pin_ip_preserve_host_sni_and_signed_url(
    dns: tuple[list[str], list[str]], connections: list[httpx.Request], tmp_path: Path
) -> None:
    """连接使用校验后的 IP，原始域名、签名路径、查询串和来源上下文完整传递。"""
    original = "https://source.example:8443/a%2Fb?signature=a%2Fb&part=1"
    extracted = asyncio.run(api_server._extract_bytes(api_server._UrlSource(url=original), api_server.FileStore(tmp_path)))
    assert dns[1] == ["source.example"]
    assert len(connections) == 1
    request = connections[0]
    assert str(request.url) == "https://93.184.216.34:8443/a%2Fb?signature=a%2Fb&part=1"
    assert request.headers["host"] == "source.example:8443"
    assert request.extensions["sni_hostname"] == "source.example"
    assert request.extensions["timeout"]["connect"] <= 60
    assert request.extensions["timeout"]["read"] == 60
    assert extracted.source_uri == original
    assert extracted.data == HTML


def test_dns_rebinding_and_connection_fallback(
    dns: tuple[list[str], list[str]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """首个地址失败后仅使用首次校验过的候选地址，DNS 变更不会重新进入连接。"""
    addresses, dns_calls = dns
    addresses[:] = ["2606:4700:4700::1111", "93.184.216.34", "93.184.216.34"]
    connections: list[httpx.Request] = []

    async def respond(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        """首个 IPv6 连接失败时把后续 DNS 应答改成回环地址。"""
        connections.append(request)
        if len(connections) == 1:
            addresses[:] = ["127.0.0.1"]
            raise httpx.ConnectError("simulated unreachable IPv6")
        return httpx.Response(200, content=HTML)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    extracted = asyncio.run(
        api_server._extract_bytes(api_server._UrlSource(url="https://source.example/a.html"), api_server.FileStore(tmp_path))
    )
    assert extracted.data == HTML
    assert dns_calls == ["source.example"]
    assert [request.url.host for request in connections] == ["2606:4700:4700::1111", "93.184.216.34"]
    assert connections[1].extensions["timeout"]["connect"] <= connections[0].extensions["timeout"]["connect"]


def test_redirect_target_is_never_requested(
    dns: tuple[list[str], list[str]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """公网响应重定向至内网时停止下载，重定向目标不会收到请求。"""
    calls: list[httpx.Request] = []

    async def redirect(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        """模拟公网源站把下载重定向至回环管理端口。"""
        calls.append(request)
        return httpx.Response(302, headers={"location": "https://127.0.0.1/admin"})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", redirect)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            api_server._extract_bytes(
                api_server._UrlSource(url="https://source.example/redirect"), api_server.FileStore(tmp_path)
            )
        )
    assert len(calls) == 1
    assert dns[1] == ["source.example"]


def test_dns_timeout_and_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS 等待受连接预算限制，任务取消也能直接中断解析等待。"""

    async def resolve(self: asyncio.BaseEventLoop, host: str, port: int, **kwargs: Any) -> list[Any]:
        """模拟 DNS 服务一直没有应答。"""
        await asyncio.Future()
        return []

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolve)

    async def check() -> None:
        """分别验证超时异常与取消异常，确保没有转成无限等待。"""
        request = httpx.Request("GET", "https://source.example/", extensions={"timeout": {"connect": 0.01}})
        with pytest.raises(httpx.ConnectTimeout):
            await url_source._resolve_public_addresses(request)
        task = asyncio.create_task(url_source._resolve_public_addresses(request))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(check())


@pytest.mark.parametrize("api_key", [None, "test-key"])
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_api_and_router_block_loopback_before_contact(
    service: tuple[TestClient, FastAPI], api_key: str | None, scheme: str
) -> None:
    """直接 API 与 Router 在匿名及已认证调用下均拦截本机源站，阻止内容回显。"""
    client, app = service
    calls: list[str] = []
    source = FastAPI()

    @source.get("/internal")
    async def internal(request: Request) -> Response:
        """仅提供合成敏感标记，任何实际访问都视为防护失败。"""
        calls.append(request.url.path)
        return Response(b'{"secret":"SSRF_LOCAL_CANARY"}', media_type="application/json")

    server = _Server(source)
    server.start()
    try:
        for worker in _execution_apps(app):
            worker.state.allow_http_source = True
            worker.state.api_key = api_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = client.post(
            "/v1/tasks",
            json={"files": [{"file_name": "internal.html", "url": f"{scheme}://127.0.0.1:{server.port}/internal"}]},
            headers=headers,
        )
        assert response.status_code == 202, response.text
        result = _terminal(client, response.json(), headers=headers)
        assert result["status"] == "failed"
        assert result["files"][0]["error"]["code"] == "unsupported_source"
        assert result["files"][0].get("result_url") is None
        assert calls == []
    finally:
        server.stop()


def test_api_and_router_allow_public_download_with_proxy_environment(
    service: tuple[TestClient, FastAPI],
    dns: tuple[list[str], list[str]],
    connections: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境代理不会接管下载，API/Router 仍通过校验后的公网连接产出真实结果。"""
    client, _ = service
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    response = client.post("/v1/tasks", json={"files": [{"file_name": "public.html", "url": "https://source.example/opaque"}]})
    assert response.status_code == 202
    result = _terminal(client, response.json())
    assert result["status"] == "completed", result
    assert b"Validated download content." in _archive(client, result)["markdown.md"]
    assert dns[1] == ["source.example"]
    assert len(connections) == 1 and connections[0].url.host == "93.184.216.34"
