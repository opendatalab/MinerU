# Copyright (c) Opendatalab. All rights reserved.
"""Router 到 V1 upstream 的 HTTP 转发、资源重写与流式响应。"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .resources import ResourceRegistry, ResourceRoute
from .workers import WorkerPool, WorkerState

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
    content: bytes | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """执行普通 upstream 请求，并把连接失败与超时转成稳定 Router 错误。"""
    outgoing_headers = forwarded_headers(request) if request is not None else {}
    outgoing_headers.update(headers or {})
    try:
        return await pool.client.request(
            method,
            f"{worker.base_url}{path}",
            json=json_body,
            content=content,
            headers=outgoing_headers,
        )
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
) -> dict[str, Any]:
    """注册 upstream file 并把响应中的 file ID 改成 Router 公共 ID。"""
    rewritten = copy.deepcopy(payload)
    upstream_id = str(rewritten.get("id") or rewritten.get("file_id") or "")
    if not upstream_id:
        return rewritten
    route = registry.register("file", worker_id=worker.worker_id, upstream_id=upstream_id)
    if "id" in rewritten:
        rewritten["id"] = route.public_id
    if "file_id" in rewritten:
        rewritten["file_id"] = route.public_id
    route.metadata["payload"] = copy.deepcopy(rewritten)
    return rewritten


def rewrite_upload_payload(
    payload: dict[str, Any],
    worker: WorkerState,
    registry: ResourceRegistry,
) -> dict[str, Any]:
    """注册 upload 与完成后的 file，并重写 upload URL 和资源标识。"""
    rewritten = copy.deepcopy(payload)
    upstream_id = str(rewritten.get("id") or "")
    if not upstream_id:
        return rewritten
    route = registry.register("upload", worker_id=worker.worker_id, upstream_id=upstream_id)
    rewritten["id"] = route.public_id
    if rewritten.get("upload_url") is not None:
        rewritten["upload_url"] = f"/v1/uploads/{route.public_id}/content"
    if isinstance(rewritten.get("file"), dict):
        rewritten["file"] = rewrite_file_payload(rewritten["file"], worker, registry)
    route.metadata["payload"] = copy.deepcopy(rewritten)
    return rewritten


def rewrite_job_payload(
    payload: dict[str, Any],
    worker: WorkerState,
    registry: ResourceRegistry,
    pool: WorkerPool,
) -> dict[str, Any]:
    """注册 job、输入/输出 files，重写 links，并在终态归还 worker 负载。"""
    rewritten = copy.deepcopy(payload)
    upstream_id = str(rewritten.get("job_id") or "")
    if not upstream_id:
        return rewritten
    route = registry.register("job", worker_id=worker.worker_id, upstream_id=upstream_id)
    rewritten["job_id"] = route.public_id
    input_aliases = cast(dict[str, str], route.metadata.get("input_aliases") or {})
    for file_result in rewritten.get("files") or []:
        if not isinstance(file_result, dict):
            continue
        upstream_file_id = file_result.get("file_id")
        if isinstance(upstream_file_id, str):
            alias = input_aliases.get(upstream_file_id)
            if alias is not None:
                file_result["file_id"] = alias
            else:
                file_route = registry.register("file", worker_id=worker.worker_id, upstream_id=upstream_file_id)
                file_result["file_id"] = file_route.public_id
        output_files = file_result.get("output_files")
        if isinstance(output_files, dict):
            for output in output_files.values():
                if not isinstance(output, dict) or not isinstance(output.get("file_id"), str):
                    continue
                output_route = registry.register(
                    "file",
                    worker_id=worker.worker_id,
                    upstream_id=output["file_id"],
                )
                output["file_id"] = output_route.public_id
    links = rewritten.get("links")
    if isinstance(links, dict):
        links["self"] = f"/v1/parse/jobs/{route.public_id}"
        if links.get("cancel") is not None:
            links["cancel"] = f"/v1/parse/jobs/{route.public_id}"
    status = str(rewritten.get("status") or "")
    if status in _TERMINAL_JOB_STATUSES and route.metadata.pop("active_counted", False):
        pool.mark_job_finished(worker.worker_id)
    route.metadata["payload"] = copy.deepcopy(rewritten)
    return rewritten


async def copy_file_to_worker(
    route: ResourceRoute,
    target: WorkerState,
    *,
    request: Request,
    pool: WorkerPool,
    registry: ResourceRegistry,
) -> str:
    """通过 V1 Files/Uploads API 把已有输入文件复制到目标 worker。"""
    source = pool.get(route.worker_id)
    metadata_response = await request_upstream(
        pool,
        source,
        "GET",
        f"/v1/files/{route.upstream_id}",
        request=request,
    )
    metadata = json_or_error(metadata_response)
    content_response = await request_upstream(
        pool,
        source,
        "GET",
        f"/v1/files/{route.upstream_id}/content",
        request=request,
    )
    if content_response.status_code >= 400:
        json_or_error(content_response)
    create_payload = {
        "filename": metadata.get("filename") or "input.bin",
        "bytes": len(content_response.content),
        "mime_type": content_response.headers.get("content-type") or "application/octet-stream",
        "purpose": metadata.get("purpose") or "parse",
        **({"sha256sum": metadata["sha256sum"]} if metadata.get("sha256sum") else {}),
    }
    upload_response = await request_upstream(
        pool,
        target,
        "POST",
        "/v1/uploads",
        request=request,
        json_body=create_payload,
    )
    upload_payload = json_or_error(upload_response)
    if isinstance(upload_payload.get("file"), dict):
        target_file_id = str(upload_payload["file"]["id"])
        registry.alias_upstream(route, worker_id=target.worker_id, upstream_id=target_file_id)
        return target_file_id
    upload_id = str(upload_payload["id"])
    put_response = await request_upstream(
        pool,
        target,
        "PUT",
        f"/v1/uploads/{upload_id}/content",
        request=request,
        content=content_response.content,
        headers={"content-type": "application/octet-stream"},
    )
    if put_response.status_code >= 400:
        json_or_error(put_response)
    complete_response = await request_upstream(
        pool,
        target,
        "POST",
        f"/v1/uploads/{upload_id}/complete",
        request=request,
        json_body={"sha256sum": metadata.get("sha256sum")} if metadata.get("sha256sum") else {},
    )
    complete_payload = json_or_error(complete_response)
    file_payload = complete_payload.get("file")
    if not isinstance(file_payload, dict) or not isinstance(file_payload.get("id"), str):
        raise RouterProxyError(502, "invalid_upstream_response", "Completed upload did not return a file")
    target_file_id = file_payload["id"]
    registry.alias_upstream(route, worker_id=target.worker_id, upstream_id=target_file_id)
    return target_file_id


__all__ = [
    "RouterProxyError",
    "copy_file_to_worker",
    "forwarded_headers",
    "json_or_error",
    "passthrough_response",
    "request_upstream",
    "rewrite_file_payload",
    "rewrite_job_payload",
    "rewrite_upload_payload",
    "router_error_response",
    "stream_upstream",
]
