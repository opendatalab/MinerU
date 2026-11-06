# Copyright (c) Opendatalab. All rights reserved.
"""用户提供的 URL 仅直连已校验的公网地址，避免 DNS 重绑定和代理绕过。"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import time

import httpx

from ..errors import MineruError

# 补齐不同 Python 版本的特殊地址分类，并拒绝可嵌入内网 IPv4 的转换地址。
_BLOCKED_NETWORKS = (
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("168.63.129.16/32"),  # Azure 平台虚拟地址。
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2001::/23"),
    ipaddress.ip_network("2002::/16"),
)


def _public_address(value: str) -> str:
    """归一化映射地址，只接受公网单播地址并拒绝 IPv6 zone identifier。"""
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if (
        "%" in value
        or not address.is_global
        or address.is_multicast
        or address.is_reserved
        or any(address in network for network in _BLOCKED_NETWORKS)
    ):
        raise MineruError("unsupported_source", "URL sources must resolve only to public unicast IP addresses")
    return str(address)


async def _resolve_public_addresses(request: httpx.Request) -> list[str]:
    """在连接超时内解析全部地址，整组校验成功后才允许建立任何连接。"""
    host = request.url.raw_host.decode("ascii")
    if "%" in host:
        raise MineruError("unsupported_source", "URL sources must not contain an IPv6 zone identifier")
    port = request.url.port if request.url.port is not None else (443 if request.url.scheme == "https" else 80)
    timeout = request.extensions.get("timeout", {}).get("connect")
    try:
        async with asyncio.timeout(timeout):
            resolved = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
            )
    except TimeoutError as exc:
        raise httpx.ConnectTimeout("URL source DNS resolution timed out", request=request) from exc
    except OSError as exc:
        raise httpx.ConnectError("URL source DNS resolution failed", request=request) from exc
    if not resolved:
        raise httpx.ConnectError("URL source DNS resolution returned no addresses", request=request)
    return list(dict.fromkeys(_public_address(item[4][0]) for item in resolved))


class UrlSourceTransport(httpx.AsyncHTTPTransport):
    """每次下载固定已校验的目标 IP，Host、SNI 和证书校验仍使用原域名。"""

    def __init__(self) -> None:
        """直接连接目标并保留环境 CA 配置，关闭跨域名共享 IP 的连接复用。"""
        super().__init__(trust_env=True, limits=httpx.Limits(max_keepalive_connections=0))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """验证全部候选地址，再依次连接；仅连接阶段失败时尝试下一地址。"""
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("connect")
        deadline = time.monotonic() + timeout if timeout is not None else None
        addresses = await _resolve_public_addresses(request)
        for index, address in enumerate(addresses):
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise httpx.ConnectTimeout("URL source connection timed out", request=request)
            pinned = httpx.Request(
                method=request.method,
                url=request.url.copy_with(host=address),
                headers=request.headers,
                stream=request.stream,
                extensions={
                    **request.extensions,
                    "sni_hostname": request.url.raw_host.decode("ascii"),
                    "timeout": {**timeouts, "connect": remaining},
                },
            )
            try:
                return await super().handle_async_request(pinned)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if index == len(addresses) - 1:
                    raise
        raise httpx.ConnectError("URL source has no reachable public address", request=request)


__all__ = ["UrlSourceTransport"]
