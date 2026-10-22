# Copyright (c) Opendatalab. All rights reserved.
"""Router 到 V1 upstream 的 HTTP 转发、资源重写与流式响应。"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .resources import ResourceRegistry, ResourceRoute
from .workers import JobReservation, WorkerPool, WorkerState

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
_TERMINAL_JOB_STATUSES = frozenset({"completed", "partial", "failed", "canceled"})


class RouterProxyError(RuntimeError):
    """表示 Router 自身产生且应使用 V1 error shape 返回的错误。"""

    def __init__(self, status_code: int, code: str, message: str, param: str | None = None) -> None:
        """保存 HTTP 状态、稳定错误码、消息和可选参数路径。"""
        self.status_code = status_code
        self.code = code
        self.message = message
        self.param = param
        super().__init__(message)


def router_error_response(exc: RouterProxyError) -> JSONResponse:
    """把 RouterProxyError 转成统一 V1 error response。"""
    error: dict[str, Any] = {
        "type": "api_error" if exc.status_code >= 500 else "invalid_request_error",
        "code": exc.code,
        "message": exc.message,
    }
    if exc.param is not None:
        error["param"] = exc.param
    return JSONResponse(status_code=exc.status_code, content={"error": error})


def forwarded_headers(request: Request) -> dict[str, str]:
    """选择可安全传给 upstream 的入站请求头，包括 Authorization。"""
    return {name: value for name, value in request.headers.items() if name.lower() not in _HOP_BY_HOP_HEADERS}


def response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """过滤 upstream response 中不应由 Router 原样返回的 hop-by-hop 头。"""
    return {name: value for name, value in headers.items() if name.lower() not in _HOP_BY_HOP_HEADERS}


async def request_upstream(
    pool: WorkerPool,
    worker: WorkerState,
    method: str,
    path: str,
    *,
    request: Request | None = None,
    json_body: Any = None,
    content: bytes | AsyncIterator[bytes] | None = None,
    headers: Mapping[str, str] | None = None,
    follow_redirects: bool = False,
) -> httpx.Response:
    """执行普通 upstream 请求，并把连接失败与超时转成稳定 Router 错误。"""
    generation = worker.generation
    outgoing_headers = forwarded_headers(request) if request is not None else {}
    outgoing_headers.pop("expect", None)
    if json_body is not None:
        outgoing_headers["content-type"] = "application/json"
    outgoing_headers.update(headers or {})
    try:
        response = await pool.client.request(
            method,
            f"{worker.base_url}{path}",
            json=json_body,
            content=content,
            headers=outgoing_headers,
            follow_redirects=follow_redirects,
        )
        if response.status_code == 503 and worker.generation == generation:
            try:
                if (response.json().get("error") or {}).get("code") == "engine_dead":
                    pool.notify_engine_dead(worker)
            except (ValueError, AttributeError):
                pass
        return response
    except httpx.TimeoutException as exc:
        raise RouterProxyError(504, "upstream_timeout", f"Upstream {worker.worker_id} timed out") from exc
    except httpx.HTTPError as exc:
        raise RouterProxyError(502, "upstream_unavailable", f"Upstream {worker.worker_id} is unavailable") from exc


async def stream_upstream(
    pool: WorkerPool,
    worker: WorkerState,
    method: str,
    path: str,
    *,
    request: Request,
) -> StreamingResponse:
    """以流式方式代理 upstream 文件内容并在完成后关闭响应。"""
    upstream_request = pool.client.build_request(
        method,
        f"{worker.base_url}{path}",
        headers=forwarded_headers(request),
    )
    try:
        upstream = await pool.client.send(upstream_request, stream=True)
    except httpx.TimeoutException as exc:
        raise RouterProxyError(504, "upstream_timeout", f"Upstream {worker.worker_id} timed out") from exc
    except httpx.HTTPError as exc:
        raise RouterProxyError(502, "upstream_unavailable", f"Upstream {worker.worker_id} is unavailable") from exc

    async def _body() -> AsyncIterator[bytes]:
        """逐块读取 upstream body，并确保响应最终关闭。"""
        try:
            if upstream.is_stream_consumed:
                yield upstream.content
                return
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        _body(),
        status_code=upstream.status_code,
        headers=response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


def passthrough_response(upstream: httpx.Response) -> Response:
    """保留 upstream 状态、body 与安全响应头生成普通响应。"""
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


def json_or_error(upstream: httpx.Response) -> dict[str, Any]:
    """读取成功 JSON；非成功响应或非法 JSON 直接转成代理错误。"""
    if upstream.status_code >= 400:
        message = upstream.text or f"Upstream returned HTTP {upstream.status_code}"
        try:
            payload = upstream.json()
            message = str((payload.get("error") or {}).get("message") or message)
        except ValueError:
            pass
        raise RouterProxyError(upstream.status_code, "upstream_error", message)
    try:
        payload = upstream.json()
    except ValueError as exc:
        raise RouterProxyError(502, "invalid_upstream_response", "Upstream returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RouterProxyError(502, "invalid_upstream_response", "Upstream JSON response must be an object")
    return payload


def rewrite_file_payload(
    payload: dict[str, Any],
    worker: WorkerState,
    registry: ResourceRegistry,
    owner_scope: str,
) -> dict[str, Any]:
    """注册 upstream file 并把响应中的 file ID 改成 Router 公共 ID。"""
    rewritten = copy.deepcopy(payload)
    upstream_id = str(rewritten.get("id") or rewritten.get("file_id") or "")
    if not upstream_id:
        return rewritten
    route = registry.register(
        "file",
        owner_scope=owner_scope,
        worker_id=worker.worker_id,
        upstream_id=upstream_id,
    )
    if "id" in rewritten:
        rewritten["id"] = route.public_id
    if "file_id" in rewritten:
        rewritten["file_id"] = route.public_id
    route.metadata["payload"] = copy.deepcopy(rewritten)
    return rewritten


def release_job_load(route: ResourceRoute, pool: WorkerPool) -> None:
    """所有终态入口共用一次性释放，预占对象校验 worker generation。"""
    if route.metadata.pop("active_counted", False):
        reservation = route.metadata.pop("reservation", None)
        if isinstance(reservation, JobReservation):
            reservation.release()
        else:
            pool.mark_job_finished(route.worker_id)


def require_worker_generation(worker: WorkerState, generation: int) -> None:
    """阻止 replacement 期间的迟到响应发布旧进程的资源标识。"""
    if worker.generation != generation or not worker.base_url:
        raise RouterProxyError(503, "upstream_unavailable", "Worker changed while the request was being submitted")


__all__ = [
    "RouterProxyError",
    "forwarded_headers",
    "json_or_error",
    "passthrough_response",
    "request_upstream",
    "require_worker_generation",
    "release_job_load",
    "rewrite_file_payload",
    "router_error_response",
    "stream_upstream",
]
