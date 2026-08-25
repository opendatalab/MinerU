# Copyright (c) Opendatalab. All rights reserved.
"""独立于 cli_old 的 MinerU V1 Router FastAPI 实现。"""

from __future__ import annotations

import copy
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response

from ...types import TIERS, select_default_quality_tier, validate_tier
from ...version import __version__
from .proxy import (
    RouterProxyError,
    copy_file_to_worker,
    json_or_error,
    passthrough_response,
    request_upstream,
    rewrite_file_payload,
    rewrite_job_payload,
    rewrite_upload_payload,
    router_error_response,
    stream_upstream,
)
from .resources import ResourceKind, ResourceRegistry, ResourceRoute, SourceFileStore, stored_file_chunks
from .workers import RouterSettings, WorkerPool, WorkerState


def _route_or_404(registry: ResourceRegistry, kind: ResourceKind, public_id: str) -> ResourceRoute:
    """读取资源路由，不存在时返回稳定 V1 404 错误。"""
    route = registry.find(kind, public_id)
    if route is None:
        raise RouterProxyError(404, f"{kind}_not_found", f"{kind.title()} {public_id} not found")
    return route


def _affinity_key(request: Request) -> str:
    """用 Authorization 与客户端地址构造同一调用方的 upload affinity。"""
    authorization = request.headers.get("authorization", "")
    client_host = request.client.host if request.client is not None else "unknown"
    return f"{authorization}\0{client_host}"


def _successful_json(upstream: httpx.Response) -> dict[str, Any] | Response:
    """成功时返回 JSON 对象，业务错误时保持 upstream response 不变。"""
    if upstream.status_code >= 400:
        return passthrough_response(upstream)
    return json_or_error(upstream)


def _healthy_worker_or_503(pool: WorkerPool) -> WorkerState:
    """选择任意健康 worker，无可用 upstream 时抛出 503。"""
    worker = pool.select(tier=None)
    if worker is None:
        raise RouterProxyError(503, "upstream_unavailable", "No healthy MinerU V1 upstream is available")
    return worker


def _list_registered_files(
    registry: ResourceRegistry,
    *,
    after: str | None,
    limit: int,
    order: Literal["asc", "desc"],
    purpose: str | None,
) -> dict[str, Any]:
    """从 Router 注册表构造 V1 Files 列表与游标。"""
    data = [
        copy.deepcopy(route.metadata["payload"])
        for route in registry.list("file")
        if isinstance(route.metadata.get("payload"), dict)
        and (purpose is None or route.metadata["payload"].get("purpose") == purpose)
    ]
    data.sort(key=lambda item: int(item.get("created_at") or 0), reverse=order == "desc")
    start = next((index + 1 for index, item in enumerate(data) if item.get("id") == after), 0) if after else 0
    page = data[start : start + limit]
    return {
        "object": "list",
        "data": page,
        "first_id": page[0]["id"] if page else None,
        "last_id": page[-1]["id"] if page else None,
        "has_more": start + limit < len(data),
    }


def _list_registered_jobs(
    registry: ResourceRegistry,
    *,
    status_filter: str | None,
    after: str | None,
    limit: int,
    order: Literal["asc", "desc"],
    created_after: str | None,
) -> dict[str, Any]:
    """从 Router 注册表构造 V1 Jobs 列表并应用过滤与分页。"""
    allowed_statuses = set(status_filter.split(",")) if status_filter else None
    payloads = [
        copy.deepcopy(route.metadata["payload"])
        for route in registry.list("job")
        if isinstance(route.metadata.get("payload"), dict)
    ]
    payloads = [
        payload
        for payload in payloads
        if (allowed_statuses is None or payload.get("status") in allowed_statuses)
        and (created_after is None or str(payload.get("created_at") or "") >= created_after)
    ]
    payloads.sort(key=lambda item: str(item.get("created_at") or ""), reverse=order == "desc")
    start = next((index + 1 for index, item in enumerate(payloads) if item.get("job_id") == after), 0) if after else 0
    page = payloads[start : start + limit]
    return {
        "object": "list",
        "data": [
            {
                "job_id": payload["job_id"],
                "status": payload.get("status", "queued"),
                "created_at": payload.get("created_at", ""),
                "file_count": len(payload.get("files") or []),
            }
            for payload in page
        ],
        "first_id": page[0]["job_id"] if page else None,
        "last_id": page[-1]["job_id"] if page else None,
        "has_more": start + limit < len(payloads),
    }


def _usage_payload(request: Request, registry: ResourceRegistry, pool: WorkerPool) -> dict[str, Any]:
    """聚合当前 Router 进程观察到的 jobs/files 与 worker 并发上限。"""
    jobs = [route.metadata.get("payload") or {} for route in registry.list("job")]
    completed_jobs = [payload for payload in jobs if payload.get("status") in {"completed", "partial"}]
    files_processed = sum(
        1
        for payload in completed_jobs
        for file_result in payload.get("files") or []
        if isinstance(file_result, dict) and file_result.get("status") == "completed"
    )
    return {
        "object": "usage",
        "access_level": "registered" if request.headers.get("authorization") else "anonymous",
        "billing_period": {"start": request.app.state.started_at, "end": None},
        "current": {
            "pages_processed": 0,
            "files_processed": files_processed,
            "jobs_created": len(jobs),
        },
        "limits": {
            "max_pages_per_file": 1000,
            "max_file_size_bytes": 209715200,
            "max_files_per_job": 100,
            "max_concurrent_jobs": sum(worker.max_concurrent_jobs for worker in pool.healthy_workers()),
            "max_file_retention_days": None,
        },
    }


async def _cleanup_copied_files(
    worker: WorkerState,
    upstream_file_ids: list[str],
    *,
    request: Request,
    pool: WorkerPool,
) -> None:
    """在 job 创建失败时尽力删除已经复制到目标 worker 的临时输入。"""
    for upstream_file_id in upstream_file_ids:
        try:
            await request_upstream(
                pool,
                worker,
                "DELETE",
                f"/v1/files/{upstream_file_id}",
                request=request,
            )
        except RouterProxyError:
            continue


def create_app(
    settings: RouterSettings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """创建完整代理 MinerU V1 资源面的 Router FastAPI 应用。"""
    resolved_settings = settings or RouterSettings.from_env()
    registry = ResourceRegistry()
    source_store = SourceFileStore()
    pool = WorkerPool(resolved_settings, transport=transport)

    @asynccontextmanager
    async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
        """启动 worker pool，并在应用关闭时释放全部网络和子进程资源。"""
        application.state.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            await pool.start()
            yield
        finally:
            await pool.close()
            source_store.close()

    application = FastAPI(
        title="MinerU V1 Router",
        version="1.0.0",
        lifespan=_lifespan,
    )
    application.state.settings = resolved_settings
    application.state.registry = registry
    application.state.source_store = source_store
    application.state.worker_pool = pool
    application.state.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @application.exception_handler(RouterProxyError)
    async def _router_error_handler(_request: Request, exc: RouterProxyError) -> JSONResponse:
        """把 Router 内部错误转换成稳定 V1 error shape。"""
        return router_error_response(exc)

    @application.get("/v1/health")
    async def health() -> Response:
        """聚合健康 worker 的 source/output 能力并返回 Router V1 health。"""
        workers = pool.healthy_workers()
        if not workers:
            raise RouterProxyError(503, "upstream_unavailable", "No healthy MinerU V1 upstream is available")
        sources = sorted({source for worker in workers for source in worker.features.get("sources") or []})
        output_formats = sorted(
            {output_format for worker in workers for output_format in worker.features.get("output_formats") or []}
        )
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "features": {"webhook": False, "output_formats": output_formats, "sources": sources},
            }
        )

    @application.get("/v1/models")
    async def models() -> dict[str, Any]:
        """按 model id 保序去重聚合健康 worker 模型列表。"""
        _healthy_worker_or_503(pool)
        by_id: dict[str, dict[str, Any]] = {}
        for worker in pool.healthy_workers():
            for model in worker.models:
                if model.get("id"):
                    by_id.setdefault(str(model["id"]), copy.deepcopy(model))
        return {"object": "list", "data": list(by_id.values())}

    @application.get("/v1/models/{model_id}")
    async def model(model_id: str) -> dict[str, Any]:
        """返回任一健康 worker 暴露的指定模型信息。"""
        for worker in pool.healthy_workers():
            found = next((item for item in worker.models if item.get("id") == model_id), None)
            if found is not None:
                return copy.deepcopy(found)
        raise RouterProxyError(404, "model_not_found", f"Model '{model_id}' not found")

    @application.get("/v1/tiers")
    async def tiers() -> dict[str, Any]:
        """按公共 tier 顺序聚合健康 worker 的能力。"""
        available = pool.available_tiers()
        if not available:
            raise RouterProxyError(503, "upstream_unavailable", "No healthy MinerU V1 upstream is available")
        data: list[dict[str, Any]] = []
        for tier in TIERS:
            if tier not in available:
                continue
            model_id = next(
                (
                    item.get("id")
                    for worker in pool.healthy_workers()
                    if tier in worker.tiers
                    for item in worker.models
                    if item.get("id")
                ),
                None,
            )
            data.append({"id": tier, "description": f"Router-discovered {tier} parsing tier.", "current_model": model_id})
        return {"object": "list", "data": data}

    @application.post("/v1/uploads")
    async def create_upload(request: Request) -> Response:
        """按调用方 affinity 选择 worker 并创建 Router upload。"""
        worker = pool.select(tier=None, required_sources={"file_id"}, affinity_key=_affinity_key(request))
        if worker is None:
            raise RouterProxyError(503, "upstream_unavailable", "No upstream accepts file uploads")
        body = await request.json()
        if not isinstance(body, dict):
            raise RouterProxyError(400, "invalid_request", "Upload request body must be an object")
        upstream_body = dict(body)
        # Router 必须收到源字节才能跨 worker 转移，因此禁止 upstream 在 PUT 前按 sha256 提前完成。
        upstream_body.pop("sha256sum", None)
        upstream = await request_upstream(pool, worker, "POST", "/v1/uploads", request=request, json_body=upstream_body)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        rewritten = rewrite_upload_payload(result, worker, registry)
        if isinstance(body.get("sha256sum"), str):
            rewritten["sha256sum"] = body["sha256sum"]
        route = _route_or_404(registry, "upload", str(rewritten["id"]))
        route.metadata["declared"] = copy.deepcopy(body)
        return JSONResponse(rewritten, status_code=upstream.status_code)

    @application.get("/v1/uploads/{upload_id}")
    async def get_upload(upload_id: str, request: Request) -> Response:
        """把 Router upload 查询路由到其所属 worker。"""
        route = _route_or_404(registry, "upload", upload_id)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "GET", f"/v1/uploads/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        return JSONResponse(rewrite_upload_payload(result, worker, registry), status_code=upstream.status_code)

    @application.put("/v1/uploads/{upload_id}/content")
    async def upload_content(upload_id: str, request: Request) -> Response:
        """把上传内容流式暂存到 Router，再写入 upload 所属 worker。"""
        route = _route_or_404(registry, "upload", upload_id)
        worker = pool.get(route.worker_id)
        declared = route.metadata.get("declared") if isinstance(route.metadata.get("declared"), dict) else {}
        mime_type = str(declared.get("mime_type") or request.headers.get("content-type") or "application/octet-stream")
        stored = await source_store.stage_upload(upload_id, request.stream(), mime_type=mime_type)
        declared_bytes = declared.get("bytes")
        if isinstance(declared_bytes, int) and stored.bytes != declared_bytes:
            source_store.discard_upload(upload_id)
            raise RouterProxyError(
                400,
                "upload_size_mismatch",
                f"Expected {declared_bytes} upload bytes, received {stored.bytes}",
            )
        declared_sha256 = declared.get("sha256sum")
        if isinstance(declared_sha256, str) and stored.sha256sum != declared_sha256:
            source_store.discard_upload(upload_id)
            raise RouterProxyError(400, "upload_sha256_mismatch", "Uploaded content SHA256 does not match request")
        upstream = await request_upstream(
            pool,
            worker,
            "PUT",
            f"/v1/uploads/{route.upstream_id}/content",
            request=request,
            content=stored_file_chunks(stored.path),
            headers={"content-type": "application/octet-stream"},
        )
        return passthrough_response(upstream)

    @application.post("/v1/uploads/{upload_id}/complete")
    async def complete_upload(upload_id: str, request: Request) -> Response:
        """完成所属 worker 的 upload，并注册返回的 V1 file。"""
        route = _route_or_404(registry, "upload", upload_id)
        worker = pool.get(route.worker_id)
        stored = source_store.find_upload(upload_id)
        if stored is None:
            raise RouterProxyError(409, "upload_not_ready", "Upload bytes have not been received by Router")
        raw_body = await request.body()
        body = await request.json() if raw_body else {}
        if not isinstance(body, dict):
            raise RouterProxyError(400, "invalid_request", "Upload complete body must be an object")
        body.setdefault("sha256sum", stored.sha256sum)
        upstream = await request_upstream(
            pool,
            worker,
            "POST",
            f"/v1/uploads/{route.upstream_id}/complete",
            request=request,
            json_body=body,
        )
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        rewritten = rewrite_upload_payload(result, worker, registry)
        file_payload = rewritten.get("file")
        if not isinstance(file_payload, dict) or not isinstance(file_payload.get("id"), str):
            raise RouterProxyError(502, "invalid_upstream_response", "Completed upload did not return a file")
        source_store.bind_file(upload_id, file_payload["id"])
        return JSONResponse(rewritten, status_code=upstream.status_code)

    @application.post("/v1/uploads/{upload_id}/cancel")
    async def cancel_upload(upload_id: str, request: Request) -> Response:
        """取消所属 worker 中尚未完成的 upload。"""
        route = _route_or_404(registry, "upload", upload_id)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(
            pool,
            worker,
            "POST",
            f"/v1/uploads/{route.upstream_id}/cancel",
            request=request,
        )
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        source_store.discard_upload(upload_id)
        return JSONResponse(rewrite_upload_payload(result, worker, registry), status_code=upstream.status_code)

    @application.get("/v1/files")
    async def list_files(
        after: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        order: Literal["asc", "desc"] = "desc",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """列出当前 Router 进程注册的 V1 files。"""
        return _list_registered_files(registry, after=after, limit=limit, order=order, purpose=purpose)

    @application.get("/v1/files/{file_id}")
    async def get_file(file_id: str, request: Request) -> Response:
        """查询 Router file 所属 worker 的最新元数据。"""
        route = _route_or_404(registry, "file", file_id)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "GET", f"/v1/files/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        return JSONResponse(rewrite_file_payload(result, worker, registry), status_code=upstream.status_code)

    @application.get("/v1/files/{file_id}/content")
    async def get_file_content(file_id: str, request: Request) -> Response:
        """流式代理 Router file 所属 worker 的内容。"""
        route = _route_or_404(registry, "file", file_id)
        return await stream_upstream(
            pool,
            pool.get(route.worker_id),
            "GET",
            f"/v1/files/{route.upstream_id}/content",
            request=request,
        )

    @application.delete("/v1/files/{file_id}")
    async def delete_file(file_id: str, request: Request) -> Response:
        """删除所属 worker 的 file，并在成功后清除 Router 映射。"""
        route = _route_or_404(registry, "file", file_id)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "DELETE", f"/v1/files/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        registry.remove("file", file_id)
        source_store.delete_file(file_id)
        result["id"] = file_id
        return JSONResponse(result, status_code=upstream.status_code)

    @application.post("/v1/parse/jobs")
    async def create_job(request: Request) -> Response:
        """选择匹配 tier/source 的 worker，迁移跨 worker files 后创建 V1 job。"""
        body = await request.json()
        if not isinstance(body, dict):
            raise RouterProxyError(400, "invalid_request", "Job request body must be an object")
        files = body.get("files")
        if not isinstance(files, list) or not files:
            raise RouterProxyError(400, "invalid_request", "Job request must contain at least one file", "files")
        raw_tier = body.get("tier")
        try:
            tier = validate_tier(raw_tier) if raw_tier is not None else select_default_quality_tier(pool.available_tiers())
        except ValueError as exc:
            raise RouterProxyError(400, "invalid_request", str(exc), "tier") from exc
        if tier is None:
            raise RouterProxyError(503, "quality_tier_unavailable", "No default quality tier is available")
        body["tier"] = tier
        required_sources: set[str] = set()
        file_routes: list[tuple[dict[str, Any], ResourceRoute]] = []
        for index, entry in enumerate(files):
            source = entry.get("source") if isinstance(entry, dict) else None
            if not isinstance(source, dict) or not isinstance(source.get("type"), str):
                raise RouterProxyError(400, "invalid_request", "Each file must contain a typed source", f"files.{index}.source")
            source_type = source["type"]
            required_sources.add(source_type)
            if source_type == "file_id":
                public_file_id = source.get("file_id")
                if not isinstance(public_file_id, str):
                    raise RouterProxyError(400, "invalid_request", "file_id source requires a string file_id")
                file_routes.append((source, _route_or_404(registry, "file", public_file_id)))
        owner_ids = {route.worker_id for _, route in file_routes}
        preferred_worker_id = next(iter(owner_ids)) if len(owner_ids) == 1 else None
        worker = pool.select(
            tier=tier,
            required_sources=required_sources,
            preferred_worker_id=preferred_worker_id,
        )
        if worker is None:
            raise RouterProxyError(503, "quality_tier_unavailable", f"No healthy upstream supports tier '{tier}'")
        input_aliases: dict[str, str] = {}
        copied_file_ids: list[str] = []
        try:
            for source, route in file_routes:
                target_file_id = route.upstream_id
                if route.worker_id != worker.worker_id:
                    target_file_id = await copy_file_to_worker(
                        route,
                        worker,
                        request=request,
                        pool=pool,
                        registry=registry,
                        source_store=source_store,
                    )
                    copied_file_ids.append(target_file_id)
                input_aliases[target_file_id] = route.public_id
                source["file_id"] = target_file_id
            upstream = await request_upstream(pool, worker, "POST", "/v1/parse/jobs", request=request, json_body=body)
        except RouterProxyError:
            await _cleanup_copied_files(worker, copied_file_ids, request=request, pool=pool)
            raise
        result = _successful_json(upstream)
        if isinstance(result, Response):
            await _cleanup_copied_files(worker, copied_file_ids, request=request, pool=pool)
            return result
        upstream_job_id = result.get("job_id")
        if not isinstance(upstream_job_id, str):
            raise RouterProxyError(502, "invalid_upstream_response", "Created job did not return a job_id")
        job_route = registry.register("job", worker_id=worker.worker_id, upstream_id=upstream_job_id)
        job_route.metadata["input_aliases"] = input_aliases
        job_route.metadata["active_counted"] = True
        pool.mark_job_started(worker.worker_id)
        rewritten = rewrite_job_payload(result, worker, registry, pool)
        job_route.metadata["payload"] = copy.deepcopy(rewritten)
        return JSONResponse(rewritten, status_code=upstream.status_code)

    @application.get("/v1/parse/jobs")
    async def list_jobs(
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = None,
        order: Literal["asc", "desc"] = "desc",
        created_after: str | None = None,
    ) -> dict[str, Any]:
        """列出当前 Router 进程创建的 jobs。"""
        return _list_registered_jobs(
            registry,
            status_filter=status_filter,
            after=after,
            limit=limit,
            order=order,
            created_after=created_after,
        )

    @application.get("/v1/parse/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> Response:
        """查询 Router job 所属 worker 的最新状态并重写所有资源标识。"""
        route = _route_or_404(registry, "job", job_id)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "GET", f"/v1/parse/jobs/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        return JSONResponse(rewrite_job_payload(result, worker, registry, pool), status_code=upstream.status_code)

    @application.delete("/v1/parse/jobs/{job_id}")
    async def cancel_job(job_id: str, request: Request) -> Response:
        """取消 Router job，并保持公共 job ID。"""
        route = _route_or_404(registry, "job", job_id)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "DELETE", f"/v1/parse/jobs/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        result["job_id"] = job_id
        if route.metadata.pop("active_counted", False):
            pool.mark_job_finished(worker.worker_id)
        payload = route.metadata.get("payload")
        if isinstance(payload, dict):
            payload["status"] = "canceled"
        return JSONResponse(result, status_code=upstream.status_code)

    @application.get("/v1/usage")
    async def usage(request: Request) -> dict[str, Any]:
        """返回当前 Router 进程聚合的 V1 usage。"""
        return _usage_payload(request, registry, pool)

    return application


def create_app_from_env() -> FastAPI:
    """为 Uvicorn reload worker 按环境变量创建新的 Router 应用。"""
    return create_app(RouterSettings.from_env())


__all__ = ["create_app", "create_app_from_env"]
