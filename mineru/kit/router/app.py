# Copyright (c) Opendatalab. All rights reserved.
"""MinerU V1 Router 的独立 FastAPI 实现。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import secrets
import time
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal, get_args

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response

from ...filetypes import is_flash_only_parse_extension
from ...parser.api_server import CreateUploadRequest, OutputFormat
from ...parser.task_api import TaskUpload, build_task_router
from ...types import TIERS, select_default_quality_tier, validate_tier
from ...utils.async_utils import drain_future
from ...version import __version__
from .proxy import (
    RouterProxyError,
    copy_file_to_worker,
    json_or_error,
    passthrough_response,
    release_job_load,
    request_upstream,
    require_worker_generation,
    rewrite_file_payload,
    rewrite_job_payload,
    rewrite_upload_payload,
    router_error_response,
    stream_upstream,
    upload_stored_source,
)
from .resources import (
    CopiedInputFile,
    ResourceKind,
    ResourceRegistry,
    ResourceRoute,
    SourceFileStore,
    stored_file_chunks,
    utc_now_iso,
)
from ...utils.retention import RETENTION_SCAN_INTERVAL_SECONDS
from .workers import RouterSettings, WorkerPool, WorkerState


def _caller_scope(request: Request) -> str:
    """从 Authorization 派生不可逆的 Router 调用方资源 scope。"""
    authorization = request.headers.get("authorization", "")
    if not authorization:
        return "anonymous"
    return "auth:" + hashlib.sha256(authorization.encode("utf-8")).hexdigest()


def _authorization_headers(request: Request) -> dict[str, str]:
    """提取后台 Job reconciliation 需要的私有 upstream Authorization。"""
    authorization = request.headers.get("authorization")
    return {"authorization": authorization} if authorization else {}


def _route_or_404(
    registry: ResourceRegistry,
    kind: ResourceKind,
    public_id: str,
    owner_scope: str,
) -> ResourceRoute:
    """读取当前调用方资源路由，不存在或 scope 不匹配时返回稳定 404。"""
    route = registry.find(kind, public_id)
    if route is None or route.owner_scope != owner_scope:
        raise RouterProxyError(404, f"{kind}_not_found", f"{kind.title()} {public_id} not found")
    return route


def _successful_json(upstream: httpx.Response) -> dict[str, Any] | Response:
    """成功时返回 JSON 对象，业务错误时保持 upstream response 不变。"""
    if upstream.status_code >= 400:
        return passthrough_response(upstream)
    return json_or_error(upstream)


async def _read_json_object(request: Request, *, allow_empty: bool = False) -> dict[str, Any]:
    """读取 JSON object body，并把空体、解码错误和非 object 映射为稳定 400。"""
    raw_body = await request.body()
    if not raw_body:
        if allow_empty:
            return {}
        raise RouterProxyError(400, "invalid_json", "Request body must be a non-empty JSON object")
    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RouterProxyError(400, "invalid_json", "Request body contains malformed JSON") from exc
    if not isinstance(payload, dict):
        raise RouterProxyError(400, "invalid_request", "Request JSON body must be an object")
    return payload


def _healthy_worker_or_503(pool: WorkerPool) -> WorkerState:
    """只检查健康 worker，不推进用于实际分配的轮询游标。"""
    workers = pool.healthy_workers()
    if not workers:
        raise RouterProxyError(503, "upstream_unavailable", "No healthy MinerU V1 upstream is available")
    return workers[0]


def _list_registered_files(
    registry: ResourceRegistry,
    *,
    owner_scope: str,
    after: str | None,
    limit: int,
    order: Literal["asc", "desc"],
    purpose: str | None,
) -> dict[str, Any]:
    """从 Router 注册表构造 V1 Files 列表与游标。"""
    data = [
        copy.deepcopy(route.metadata["payload"])
        for route in registry.list("file", owner_scope=owner_scope)
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
    owner_scope: str,
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
        for route in registry.list("job", owner_scope=owner_scope)
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


def _usage_payload(request: Request, registry: ResourceRegistry, pool: WorkerPool, owner_scope: str) -> dict[str, Any]:
    """聚合当前 Router 进程的累计 jobs/files 与 worker 并发上限。"""
    return {
        "object": "usage",
        "access_level": "registered" if request.headers.get("authorization") else "anonymous",
        "billing_period": {"start": request.app.state.started_at, "end": None},
        "current": {
            "pages_processed": 0,
            **registry.usage(owner_scope),
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
    copied_files: list[CopiedInputFile],
    *,
    request: Request | None,
    pool: WorkerPool,
    registry: ResourceRegistry,
    headers: dict[str, str] | None = None,
) -> list[CopiedInputFile]:
    """幂等删除 cross-worker 输入副本，返回等待后续重试的失败项。"""
    pending: list[CopiedInputFile] = []
    for copied_file in copied_files:
        worker = pool.get(copied_file.worker_id)
        try:
            response = await request_upstream(
                pool,
                worker,
                "DELETE",
                f"/v1/files/{copied_file.upstream_file_id}",
                request=request,
                headers=headers,
            )
        except RouterProxyError:
            pending.append(copied_file)
            continue
        if response.status_code not in {200, 204, 404}:
            pending.append(copied_file)
            continue
        registry.remove_upstream_alias(
            "file",
            owner_scope=copied_file.owner_scope,
            worker_id=copied_file.worker_id,
            upstream_id=copied_file.upstream_file_id,
        )
    return pending


async def _hydrate_job_output_files(
    payload: dict[str, Any],
    *,
    request: Request | None,
    pool: WorkerPool,
    registry: ResourceRegistry,
    headers: dict[str, str] | None = None,
) -> None:
    """读取 Job output files 的完整元数据，使其立即进入 `/v1/files`。"""
    seen_public_ids: set[str] = set()
    for file_result in payload.get("files") or []:
        if not isinstance(file_result, dict) or not isinstance(file_result.get("output_files"), dict):
            continue
        for output in file_result["output_files"].values():
            if not isinstance(output, dict) or not isinstance(output.get("file_id"), str):
                continue
            public_file_id = output["file_id"]
            if public_file_id in seen_public_ids:
                continue
            seen_public_ids.add(public_file_id)
            route = registry.find("file", public_file_id)
            if route is None or isinstance(route.metadata.get("payload"), dict):
                continue
            worker = pool.get(route.worker_id)
            generation = worker.generation
            try:
                response = await request_upstream(
                    pool,
                    worker,
                    "GET",
                    f"/v1/files/{route.upstream_id}",
                    request=request,
                    headers=headers,
                )
                if response.status_code >= 400:
                    route.metadata["hydration_error"] = {
                        "status_code": response.status_code,
                        "message": response.text,
                    }
                    continue
                metadata = json_or_error(response)
            except RouterProxyError as exc:
                route.metadata["hydration_error"] = {
                    "status_code": exc.status_code,
                    "code": exc.code,
                    "message": exc.message,
                }
                continue
            if worker.generation != generation or registry.find("file", public_file_id) is not route:
                continue
            rewrite_file_payload(metadata, worker, registry, route.owner_scope)
            route.metadata.pop("hydration_error", None)


async def _finalize_terminal_job(
    route: ResourceRoute,
    payload: dict[str, Any],
    *,
    request: Request | None,
    pool: WorkerPool,
    registry: ResourceRegistry,
    headers: dict[str, str] | None = None,
) -> None:
    """在 Job 终态补齐 outputs，并回收或保留待重试的输入副本。"""
    if payload.get("status") not in {"completed", "partial", "failed", "canceled"}:
        return
    try:
        await _hydrate_job_output_files(
            payload,
            request=request,
            pool=pool,
            registry=registry,
            headers=headers,
        )
    finally:
        copied_files = list(route.metadata.get("copied_inputs") or [])
        pending = await _cleanup_copied_files(
            copied_files,
            request=request,
            pool=pool,
            registry=registry,
            headers=headers,
        )
        if pending:
            route.metadata["copied_inputs"] = pending
        else:
            route.metadata.pop("copied_inputs", None)
            route.metadata.pop("upstream_headers", None)
            route.metadata.setdefault("finalized_at", utc_now_iso())


def _mark_missing_job(route: ResourceRoute, pool: WorkerPool) -> dict[str, Any]:
    """上游已丢失任务时保存失败快照并归还负载，避免重启后永久占用名额。"""
    payload = route.metadata.get("payload") or {}
    if payload.get("status") not in {"completed", "partial", "failed", "canceled"}:
        payload["status"] = "failed"
        payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for file in payload.get("files") or []:
            if file.get("status") != "completed":
                file["status"] = "failed"
                file["error"] = {
                    "type": "engine_error",
                    "code": "upstream_job_lost",
                    "message": "Upstream task no longer exists",
                }
        progress = payload.get("progress")
        if isinstance(progress, dict):
            progress["failed"] = progress.get("total", 0) - progress.get("completed", 0)
    route.metadata["payload"] = payload
    route.metadata["upstream_lost"] = True
    release_job_load(route, pool)
    return payload


async def _reconcile_jobs_once(registry: ResourceRegistry, pool: WorkerPool) -> None:
    """后台查询活动或待清理 Job，并统一执行终态资源收敛。"""
    for route in registry.list("job", include_expired=True):
        if not route.metadata.get("active_counted") and not route.metadata.get("copied_inputs"):
            continue
        headers = dict(route.metadata.get("upstream_headers") or {})
        if route.metadata.get("upstream_lost"):
            await _finalize_terminal_job(
                route, route.metadata["payload"], request=None, pool=pool, registry=registry, headers=headers
            )
            continue
        worker = pool.get(route.worker_id)
        generation = worker.generation
        try:
            response = await request_upstream(
                pool,
                worker,
                "GET",
                f"/v1/parse/jobs/{route.upstream_id}",
                headers=headers,
            )
            if worker.generation != generation or route.metadata.get("upstream_lost"):
                continue
            if response.status_code >= 400:
                if response.status_code == 404:
                    cached = _mark_missing_job(route, pool)
                    await _finalize_terminal_job(route, cached, request=None, pool=pool, registry=registry, headers=headers)
                continue
            payload = json_or_error(response)
            rewritten = rewrite_job_payload(payload, worker, registry, pool, route.owner_scope)
            route.metadata["payload"] = copy.deepcopy(rewritten)
            await _finalize_terminal_job(
                route,
                rewritten,
                request=None,
                pool=pool,
                registry=registry,
                headers=headers,
            )
        except RouterProxyError:
            continue


async def _job_reconcile_loop(
    registry: ResourceRegistry,
    pool: WorkerPool,
    interval_seconds: float,
) -> None:
    """按固定间隔运行 Job reconciliation，单次失败不终止循环。"""
    while True:
        await asyncio.sleep(interval_seconds)
        await _reconcile_jobs_once(registry, pool)


def create_app(
    settings: RouterSettings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """创建完整代理 MinerU V1 资源面的 Router FastAPI 应用。"""
    resolved_settings = settings or RouterSettings.from_env()
    registry = ResourceRegistry(resolved_settings.retention_seconds)
    source_store = SourceFileStore(resolved_settings.retention_seconds)

    async def _invalidate_replaced_worker(worker: WorkerState) -> None:
        """在本地 worker replacement 发布前移除旧 generation 的资源状态。"""
        for route in registry.list("job"):
            if route.worker_id != worker.worker_id:
                continue
            payload = route.metadata.get("payload")
            if not isinstance(payload, dict) or payload.get("status") in {"completed", "partial", "failed", "canceled"}:
                continue
            release_job_load(route, pool)
            route.metadata["upstream_lost"] = True
            payload["status"] = "failed"
            payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for file in payload.get("files") or []:
                if file.get("status") != "completed":
                    file["status"] = "failed"
                    file["error"] = {"type": "engine_error", "code": "engine_dead", "message": "Worker engine was replaced"}
            progress = payload.get("progress")
            if isinstance(progress, dict):
                progress["failed"] = progress.get("total", 0) - progress.get("completed", 0)
            route.metadata.pop("copied_inputs", None)
        for route in registry.remove_worker(worker.worker_id):
            if route.kind == "upload":
                source_store.discard_upload(route.public_id)
            elif route.kind == "file":
                source_store.delete_file(route.public_id)

    pool = WorkerPool(
        resolved_settings,
        transport=transport,
        on_local_worker_replaced=_invalidate_replaced_worker,
    )

    @asynccontextmanager
    async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
        """启动 worker pool，并在应用关闭时释放全部网络和子进程资源。"""
        application.state.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        reconcile_task: asyncio.Task[None] | None = None
        collection_task: asyncio.Task[None] | None = None

        async def collection_loop() -> None:
            """每五分钟同步回收过期公共路由与无引用源文件缓存。"""
            while True:
                await asyncio.sleep(RETENTION_SCAN_INTERVAL_SECONDS)
                registry.collect_expired(source_store)

        try:
            if resolved_settings.retention_seconds:
                collection_task = asyncio.create_task(collection_loop(), name="mineru-router-retention")
            await pool.start()
            if resolved_settings.worker_refresh_interval_seconds > 0:
                reconcile_task = asyncio.create_task(
                    _job_reconcile_loop(
                        registry,
                        pool,
                        resolved_settings.worker_refresh_interval_seconds,
                    ),
                    name="mineru-v1-router-job-reconciler",
                )
            yield
        finally:
            if collection_task is not None:
                collection_task.cancel()
                await asyncio.gather(collection_task, return_exceptions=True)
            if reconcile_task is not None:
                reconcile_task.cancel()
                try:
                    await reconcile_task
                except asyncio.CancelledError:
                    pass
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

    async def _reconcile_once_for_state() -> None:
        """向测试和诊断入口暴露一次显式 Job reconciliation。"""
        await _reconcile_jobs_once(registry, pool)

    application.state.reconcile_jobs_once = _reconcile_once_for_state

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
            tier_info = next(
                (
                    copy.deepcopy(worker.tier_metadata[tier])
                    for worker in pool.healthy_workers()
                    if tier in worker.tier_metadata
                ),
                {"id": tier, "description": f"Router-discovered {tier} parsing tier.", "current_model": None},
            )
            data.append(tier_info)
        return {"object": "list", "data": data}

    @application.post("/v1/uploads")
    async def create_upload(request: Request) -> Response:
        """按负载和上传轮询选择 worker，并复用本调用方已有的源字节。"""
        owner_scope = _caller_scope(request)
        body = await _read_json_object(request)
        try:
            validated = CreateUploadRequest.model_validate(body)
        except ValueError as exc:
            raise RouterProxyError(400, "invalid_request", str(exc)) from exc
        body = validated.model_dump(exclude_none=True)
        worker = pool.select(tier=None, required_sources={"file_id"}, selection_kind="upload")
        if worker is None:
            raise RouterProxyError(503, "upstream_unavailable", "No upstream accepts file uploads")
        cached = source_store.find_hash(owner_scope, body["sha256sum"]) if isinstance(body.get("sha256sum"), str) else None
        if cached is not None:
            if body.get("bytes") != cached.bytes:
                raise RouterProxyError(400, "upload_size_mismatch", "Declared size does not match cached bytes")
            with source_store.pin(cached):
                result = await upload_stored_source(
                    cached, str(body.get("filename") or "input.bin"), worker, request=request, pool=pool, declaration=body
                )
                rewritten = rewrite_upload_payload(result, worker, registry, owner_scope)
                source_store.bind_source(
                    str(rewritten["file"]["id"]),
                    cached,
                    owner_scope=owner_scope,
                    expires_at=rewritten["file"].get("expires_at"),
                )
            return JSONResponse(rewritten)
        upstream_body = dict(body)
        # 本地未命中时必须收到源字节；仅凭 worker 的 blob 命中无法支持跨 worker 复制。
        upstream_body.pop("sha256sum", None)
        upstream = await request_upstream(pool, worker, "POST", "/v1/uploads", request=request, json_body=upstream_body)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        rewritten = rewrite_upload_payload(result, worker, registry, owner_scope)
        if isinstance(body.get("sha256sum"), str):
            rewritten["sha256sum"] = body["sha256sum"]
        route = _route_or_404(registry, "upload", str(rewritten["id"]), owner_scope)
        route.metadata["declared"] = copy.deepcopy(body)
        return JSONResponse(rewritten, status_code=upstream.status_code)

    @application.get("/v1/uploads/{upload_id}")
    async def get_upload(upload_id: str, request: Request) -> Response:
        """把 Router upload 查询路由到其所属 worker。"""
        owner_scope = _caller_scope(request)
        route = _route_or_404(registry, "upload", upload_id, owner_scope)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "GET", f"/v1/uploads/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        return JSONResponse(
            rewrite_upload_payload(result, worker, registry, owner_scope),
            status_code=upstream.status_code,
        )

    @application.put("/v1/uploads/{upload_id}/content")
    async def upload_content(upload_id: str, request: Request) -> Response:
        """把上传内容流式暂存到 Router，再写入 upload 所属 worker。"""
        route = _route_or_404(registry, "upload", upload_id, _caller_scope(request))
        with registry.pin_route(route):
            upload_payload = route.metadata.get("payload")
            upload_status = upload_payload.get("status") if isinstance(upload_payload, dict) else None
            if source_store.is_bound_upload(upload_id) or upload_status == "completed":
                raise RouterProxyError(409, "upload_already_completed", f"Upload {upload_id} is already completed")
            if upload_status == "canceled":
                raise RouterProxyError(409, "upload_already_canceled", f"Upload {upload_id} is already canceled")
            worker = pool.get(route.worker_id)
            declared = route.metadata.get("declared") if isinstance(route.metadata.get("declared"), dict) else {}
            mime_type = str(declared.get("mime_type") or request.headers.get("content-type") or "application/octet-stream")
            try:
                stored = await source_store.stage_upload(upload_id, request.stream(), mime_type=mime_type)
            except ValueError as exc:
                raise RouterProxyError(409, "upload_not_writable", str(exc)) from exc
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
        owner_scope = _caller_scope(request)
        route = _route_or_404(registry, "upload", upload_id, owner_scope)
        worker = pool.get(route.worker_id)
        stored = source_store.find_upload(upload_id)
        if stored is None:
            raise RouterProxyError(409, "upload_not_ready", "Upload bytes have not been received by Router")
        body = await _read_json_object(request, allow_empty=True)
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
        rewritten = rewrite_upload_payload(result, worker, registry, owner_scope)
        file_payload = rewritten.get("file")
        if not isinstance(file_payload, dict) or not isinstance(file_payload.get("id"), str):
            raise RouterProxyError(502, "invalid_upstream_response", "Completed upload did not return a file")
        source_store.bind_file(
            upload_id, file_payload["id"], owner_scope=owner_scope, expires_at=file_payload.get("expires_at")
        )
        return JSONResponse(rewritten, status_code=upstream.status_code)

    @application.post("/v1/uploads/{upload_id}/cancel")
    async def cancel_upload(upload_id: str, request: Request) -> Response:
        """取消所属 worker 中尚未完成的 upload。"""
        owner_scope = _caller_scope(request)
        route = _route_or_404(registry, "upload", upload_id, owner_scope)
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
        return JSONResponse(
            rewrite_upload_payload(result, worker, registry, owner_scope),
            status_code=upstream.status_code,
        )

    @application.get("/v1/files")
    async def list_files(
        request: Request,
        after: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        order: Literal["asc", "desc"] = "desc",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """列出当前 Router 进程注册的 V1 files。"""
        return _list_registered_files(
            registry,
            owner_scope=_caller_scope(request),
            after=after,
            limit=limit,
            order=order,
            purpose=purpose,
        )

    @application.get("/v1/files/{file_id}")
    async def get_file(file_id: str, request: Request) -> Response:
        """查询 Router file 所属 worker 的最新元数据。"""
        owner_scope = _caller_scope(request)
        route = _route_or_404(registry, "file", file_id, owner_scope)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "GET", f"/v1/files/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        return JSONResponse(
            rewrite_file_payload(result, worker, registry, owner_scope),
            status_code=upstream.status_code,
        )

    @application.get("/v1/files/{file_id}/content")
    async def get_file_content(file_id: str, request: Request) -> Response:
        """流式代理 Router file 所属 worker 的内容。"""
        route = _route_or_404(registry, "file", file_id, _caller_scope(request))
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
        owner_scope = _caller_scope(request)
        route = _route_or_404(registry, "file", file_id, owner_scope)
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "DELETE", f"/v1/files/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        for job_route in registry.list("job", owner_scope=owner_scope):
            copied_inputs = list(job_route.metadata.get("copied_inputs") or [])
            matching = [item for item in copied_inputs if item.source_public_id == file_id]
            if not matching:
                continue
            job_payload = job_route.metadata.get("payload")
            if not isinstance(job_payload, dict) or job_payload.get("status") not in {
                "completed",
                "partial",
                "failed",
                "canceled",
            }:
                continue
            pending = await _cleanup_copied_files(
                matching,
                request=request,
                pool=pool,
                registry=registry,
            )
            unrelated = [item for item in copied_inputs if item.source_public_id != file_id]
            if unrelated or pending:
                job_route.metadata["copied_inputs"] = [*unrelated, *pending]
            else:
                job_route.metadata.pop("copied_inputs", None)
                job_route.metadata.setdefault("finalized_at", utc_now_iso())
        registry.remove("file", file_id)
        source_store.delete_file(file_id)
        result["id"] = file_id
        return JSONResponse(result, status_code=upstream.status_code)

    async def submit_job(body: dict[str, Any], request: Request, uploads: dict[int, TaskUpload] | None = None) -> Response:
        """统一选择并预占 worker，V1 Job 和便捷任务共用上传与副本清理。"""
        owner_scope = _caller_scope(request)
        body = copy.deepcopy(body)
        uploads = uploads or {}
        output_formats = body.get("output_formats", ["markdown"])
        if not isinstance(output_formats, list) or any(not isinstance(fmt, str) for fmt in output_formats):
            raise RouterProxyError(400, "invalid_request", "output_formats must be a list of strings", "output_formats")
        for fmt in output_formats:
            if fmt not in get_args(OutputFormat):
                raise RouterProxyError(400, "unsupported_output_format", f"Unknown output format: {fmt}", "output_formats")
        files = body.get("files")
        if not isinstance(files, list) or not files:
            raise RouterProxyError(400, "invalid_request", "Job request must contain at least one file", "files")
        required_sources: set[str] = set()
        file_routes: list[tuple[dict[str, Any], ResourceRoute]] = []
        names: list[str] = []
        for index, entry in enumerate(files):
            source = entry.get("source") if isinstance(entry, dict) else None
            if not isinstance(source, dict) or not isinstance(source.get("type"), str):
                raise RouterProxyError(400, "invalid_request", "Each file must contain a typed source", f"files.{index}.source")
            required_sources.add(source["type"])
            if index in uploads:
                names.append(uploads[index].filename)
            elif source["type"] == "file_id":
                public_file_id = source.get("file_id")
                if not isinstance(public_file_id, str):
                    raise RouterProxyError(400, "invalid_request", "file_id source requires a string file_id")
                route = _route_or_404(registry, "file", public_file_id, owner_scope)
                file_routes.append((source, route))
                names.append(str((route.metadata.get("payload") or {}).get("filename") or ""))
            else:
                if source["type"] == "url":
                    names.append(str(source.get("url") or "").rsplit("/", 1)[-1].split("?")[0] or "unknown")
                elif source["type"] == "local":
                    names.append(Path(str(source.get("path") or "")).name)
                else:
                    names.append(str(source.get("name") or ""))
        can_default_to_flash = all(
            is_flash_only_parse_extension(name) or (entry["source"]["type"] == "url" and not Path(name).suffix)
            for name, entry in zip(names, files)
        )
        raw_tier = body.get("tier")
        try:
            tier = validate_tier(raw_tier) if raw_tier is not None else select_default_quality_tier(pool.available_tiers())
        except ValueError as exc:
            raise RouterProxyError(400, "invalid_request", str(exc), "tier") from exc
        if tier is None and can_default_to_flash and "flash" in pool.available_tiers():
            tier = "flash"
        if tier is None:
            raise RouterProxyError(503, "quality_tier_unavailable", "No default quality tier is available")
        body["tier"] = tier
        selection_tier = "flash" if can_default_to_flash and tier not in pool.available_tiers() else tier
        owner_ids = {route.worker_id for _, route in file_routes}
        preferred = next(iter(owner_ids)) if len(owner_ids) == 1 else None
        reservation = pool.reserve_job(
            tier=selection_tier,
            required_sources=required_sources,
            preferred_worker_id=preferred,
            required_outputs=set(output_formats),
            require_flash=any(is_flash_only_parse_extension(name) for name in names),
        )
        if reservation is None:
            raise RouterProxyError(
                503, "quality_tier_unavailable", f"No healthy upstream supports tier '{tier}' and requested formats"
            )
        worker = reservation.worker
        input_aliases: dict[str, str] = {}
        copied_files: list[CopiedInputFile] = []
        registered_inputs: list[ResourceRoute] = []
        staged: list[str] = []
        transferred = False

        async def rollback() -> None:
            """回收未被任务接管的跨 worker 副本和本次创建的源文件视图。"""
            if worker.generation == reservation.generation and worker.base_url:
                await _cleanup_copied_files(copied_files, request=request, pool=pool, registry=registry)
            for route in registered_inputs:
                try:
                    if worker.generation == reservation.generation and worker.base_url:
                        await request_upstream(pool, worker, "DELETE", f"/v1/files/{route.upstream_id}", request=request)
                except RouterProxyError:
                    pass
                registry.remove("file", route.public_id)
                source_store.delete_file(route.public_id)

        try:
            with ExitStack() as pins:
                # 哈希命中输入在第一次网络等待前固定缓存引用，避免并发删除原文件。
                cached_inputs = {}
                for index, upload in uploads.items():
                    cached = source_store.find_hash(owner_scope, upload.sha256sum)
                    if cached is not None:
                        pins.enter_context(source_store.pin(cached))
                        cached_inputs[index] = cached
                for index, upload in uploads.items():
                    stored = cached_inputs.get(index)
                    if stored is None:
                        staging_id = "task-" + secrets.token_hex(12)
                        staged.append(staging_id)
                        stored = await source_store.stage_upload(
                            staging_id, stored_file_chunks(upload.path), mime_type=upload.mime_type, max_bytes=200 * 1024 * 1024
                        )
                        pins.enter_context(source_store.pin(stored))
                    payload = await upload_stored_source(
                        stored, upload.filename, worker, request=request, pool=pool, generation=reservation.generation
                    )
                    rewritten = rewrite_file_payload(payload["file"], worker, registry, owner_scope)
                    public_id = str(rewritten["id"])
                    route = _route_or_404(registry, "file", public_id, owner_scope)
                    registered_inputs.append(route)
                    source_store.bind_source(public_id, stored, owner_scope=owner_scope, expires_at=rewritten.get("expires_at"))
                    input_aliases[route.upstream_id] = public_id
                    files[index]["source"] = {"type": "file_id", "file_id": route.upstream_id}
                for source, route in file_routes:
                    target_file_id = route.upstream_id
                    if route.worker_id != worker.worker_id:
                        target_file_id = await copy_file_to_worker(
                            route, worker, request=request, pool=pool, registry=registry, source_store=source_store
                        )
                        copied_files.append(
                            CopiedInputFile(
                                source_public_id=route.public_id,
                                owner_scope=owner_scope,
                                worker_id=worker.worker_id,
                                upstream_file_id=target_file_id,
                            )
                        )
                    input_aliases[target_file_id] = route.public_id
                    source["file_id"] = target_file_id
                upstream = await request_upstream(pool, worker, "POST", "/v1/parse/jobs", request=request, json_body=body)
                require_worker_generation(worker, reservation.generation)
                result = _successful_json(upstream)
                if isinstance(result, Response):
                    return result
                upstream_job_id = result.get("job_id")
                if not isinstance(upstream_job_id, str):
                    raise RouterProxyError(502, "invalid_upstream_response", "Created job did not return a job_id")
                job_route = registry.register(
                    "job", owner_scope=owner_scope, worker_id=worker.worker_id, upstream_id=upstream_job_id
                )
                job_route.metadata.update(
                    input_aliases=input_aliases,
                    copied_inputs=copied_files,
                    active_counted=True,
                    reservation=reservation,
                    upstream_headers=_authorization_headers(request),
                )
                transferred = True
                rewritten = rewrite_job_payload(result, worker, registry, pool, owner_scope)
                await _finalize_terminal_job(job_route, rewritten, request=request, pool=pool, registry=registry)
                return JSONResponse(rewritten, status_code=upstream.status_code)
        finally:
            for staging_id in staged:
                source_store.discard_upload(staging_id)
            if not transferred:
                reservation.release()
                await drain_future(asyncio.create_task(rollback()))

    @application.post("/v1/parse/jobs")
    async def create_job(request: Request) -> Response:
        """沿用 V1 请求响应，通过共享服务提交并预占解析负载。"""
        return await submit_job(await _read_json_object(request), request)

    @application.get("/v1/parse/jobs")
    async def list_jobs(
        request: Request,
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = None,
        order: Literal["asc", "desc"] = "desc",
        created_after: str | None = None,
    ) -> dict[str, Any]:
        """列出当前 Router 进程创建的 jobs。"""
        return _list_registered_jobs(
            registry,
            owner_scope=_caller_scope(request),
            status_filter=status_filter,
            after=after,
            limit=limit,
            order=order,
            created_after=created_after,
        )

    @application.get("/v1/parse/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> Response:
        """查询 Router job 所属 worker 的最新状态并重写所有资源标识。"""
        owner_scope = _caller_scope(request)
        route = _route_or_404(registry, "job", job_id, owner_scope)
        if route.metadata.get("upstream_lost"):
            return JSONResponse(route.metadata["payload"])
        worker = pool.get(route.worker_id)
        generation = worker.generation
        upstream = await request_upstream(pool, worker, "GET", f"/v1/parse/jobs/{route.upstream_id}", request=request)
        if worker.generation != generation or route.metadata.get("upstream_lost"):
            return JSONResponse(route.metadata["payload"])
        if upstream.status_code == 404:
            failed = _mark_missing_job(route, pool)
            await _finalize_terminal_job(route, failed, request=request, pool=pool, registry=registry)
            return JSONResponse(failed)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        rewritten = rewrite_job_payload(result, worker, registry, pool, owner_scope)
        await _finalize_terminal_job(
            route,
            rewritten,
            request=request,
            pool=pool,
            registry=registry,
        )
        if worker.generation != generation or route.metadata.get("upstream_lost"):
            return JSONResponse(route.metadata["payload"])
        return JSONResponse(rewritten, status_code=upstream.status_code)

    @application.delete("/v1/parse/jobs/{job_id}")
    async def cancel_job(job_id: str, request: Request) -> Response:
        """取消 Router job，并保持公共 job ID。"""
        route = _route_or_404(registry, "job", job_id, _caller_scope(request))
        if route.metadata.get("upstream_lost"):
            raise RouterProxyError(409, "job_already_terminal", "Job is failed")
        worker = pool.get(route.worker_id)
        upstream = await request_upstream(pool, worker, "DELETE", f"/v1/parse/jobs/{route.upstream_id}", request=request)
        result = _successful_json(upstream)
        if isinstance(result, Response):
            return result
        result["job_id"] = job_id
        release_job_load(route, pool)
        payload = route.metadata.get("payload")
        if isinstance(payload, dict):
            payload["status"] = "canceled"
            await _finalize_terminal_job(
                route,
                payload,
                request=request,
                pool=pool,
                registry=registry,
            )
        return JSONResponse(result, status_code=upstream.status_code)

    @application.get("/v1/usage")
    async def usage(request: Request) -> dict[str, Any]:
        """返回当前 Router 进程聚合的 V1 usage。"""
        return _usage_payload(request, registry, pool, _caller_scope(request))

    from .tasks import RouterTaskBackend, install_task_error_handlers

    def task_backend(request: Request) -> RouterTaskBackend:
        """显式组装 Router 适配器，便捷入口不通过 HTTP 请求自身。"""
        return RouterTaskBackend(request, submit_job, get_job, cancel_job, _caller_scope(request))

    install_task_error_handlers(application)
    application.include_router(build_task_router(task_backend))
    return application


def create_app_from_env() -> FastAPI:
    """为 Uvicorn reload worker 按环境变量创建新的 Router 应用。"""
    return create_app(RouterSettings.from_env())


__all__ = ["create_app", "create_app_from_env"]
