# Copyright (c) Opendatalab. All rights reserved.
"""MinerU V1 Router 的独立 FastAPI 实现。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from ...tasks.errors import ApiServerError
from ...tasks.execution import TaskExecution
from ...tasks.http import build_task_router
from ...tasks.service import TaskStore
from ...types import TIERS
from ...version import __version__
from .proxy import (
    RouterProxyError,
    json_or_error,
    passthrough_response,
    request_upstream,
    rewrite_file_payload,
    router_error_response,
    stream_upstream,
)
from .resources import (
    ExecutionRoute,
    ResourceKind,
    ResourceRegistry,
    ResourceRoute,
    SourceFileStore,
    utc_now_iso,
)
from ...utils.retention import RETENTION_SCAN_INTERVAL_SECONDS
from .workers import RouterSettings, WorkerPool, WorkerState
from .admin import build_admin_router


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
            "max_pages_per_file": None,
            "max_file_size_bytes": None,
            "max_files_per_job": 100,
            "max_concurrent_jobs": sum(worker.max_concurrent_jobs for worker in pool.healthy_workers()),
            "max_file_retention_days": None,
        },
    }


async def _finalize_terminal_job(
    route: ExecutionRoute,
    execution: TaskExecution,
) -> None:
    """终态释放一次执行预占，结果映射和租约由 Tasks 专用路径管理。"""
    if execution.status in {"completed", "failed", "canceled"}:
        route.state.release_reservation()
        if not route.lease.held:
            route.state.upstream_headers.clear()
        if route.finalized_at is None:
            route.finalized_at = utc_now_iso()


async def _reconcile_jobs_once(registry: ResourceRegistry, pool: WorkerPool) -> None:
    """后台仅查询 worker Tasks，HTTP 查询方断连也能归还执行负载。"""
    from .tasks import read_task_execution

    for route in registry.list("job", include_expired=True):
        if route.state.reservation is None:
            continue
        try:
            execution = await read_task_execution(
                route,
                pool,
                registry,
                headers=dict(route.state.upstream_headers),
            )
            await _finalize_terminal_job(route, execution)
        except (RouterProxyError, ApiServerError):
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
    task_store = TaskStore(resolved_settings.retention_seconds)

    async def _invalidate_replaced_worker(worker: WorkerState) -> None:
        """在本地 worker replacement 发布前移除旧 generation 的资源状态。"""
        for route in registry.list("job"):
            if route.worker_id != worker.worker_id:
                continue
            payload = route.payload
            if not isinstance(payload, dict) or payload.get("status") in {"completed", "partial", "failed", "canceled"}:
                continue
            route.state.release_reservation()
            route.state.upstream_lost = True
            payload["status"] = "failed"
            payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for file in payload.get("files") or []:
                if file.get("status") != "completed":
                    file["status"] = "failed"
                    file["error"] = {"code": "engine_dead", "message": "Worker engine was replaced"}
            progress = payload.get("progress")
            if isinstance(progress, dict):
                progress["failed"] = progress.get("total", 0) - progress.get("completed", 0)
        for route in registry.remove_worker(worker.worker_id):
            if route.kind == "file":
                source_store.delete_file(route.public_id)

    pool = WorkerPool(
        resolved_settings,
        transport=transport,
        on_local_worker_replaced=_invalidate_replaced_worker,
    )

    def collect_resources_once() -> None:
        """先按原保留契约回收路由，再移除无执行、传输或结果引用的 draining 节点。"""
        if resolved_settings.retention_seconds:
            registry.collect_expired(source_store)
            source_store.collect_expired()
        pool.remove_drained_workers(registry.worker_route_count)

    @asynccontextmanager
    async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
        """启动 worker pool，并在应用关闭时释放全部网络和子进程资源。"""
        application.state.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        reconcile_task: asyncio.Task[None] | None = None
        collection_task: asyncio.Task[None] | None = None
        lease_task: asyncio.Task[None] | None = None

        async def task_lease_loop() -> None:
            """独立维护 Tasks 的结果租约，禁用 worker 刷新时也保证下载保留期。"""
            from .tasks import maintain_task_result_leases

            while True:
                await asyncio.sleep(30)
                await maintain_task_result_leases(registry, pool)

        async def collection_loop() -> None:
            """复用五分钟扫描回收路由和节点；永久保留模式仍可回收无引用节点。"""
            while True:
                await asyncio.sleep(RETENTION_SCAN_INTERVAL_SECONDS)
                collect_resources_once()

        try:
            collection_task = asyncio.create_task(collection_loop(), name="mineru-router-retention")
            await pool.start()
            task_store.start()
            lease_task = asyncio.create_task(task_lease_loop(), name="mineru-router-task-result-leases")
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
            await task_store.shutdown()
            if lease_task is not None:
                lease_task.cancel()
                await asyncio.gather(lease_task, return_exceptions=True)
            from .tasks import maintain_task_result_leases

            await maintain_task_result_leases(registry, pool, release=True)
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
    application.state.task_store = task_store
    application.state.worker_pool = pool
    application.state.collect_resources_once = collect_resources_once
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
        task_output_formats = sorted(
            {output_format for worker in workers for output_format in worker.features.get("task_output_formats") or []}
        )
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "features": {
                    "webhook": False,
                    "task_output_formats": task_output_formats,
                    "sources": sources,
                },
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

    @application.get("/v1/usage")
    async def usage(request: Request) -> dict[str, Any]:
        """返回当前 Router 进程聚合的 V1 usage。"""
        return _usage_payload(request, registry, pool, _caller_scope(request))

    from .tasks import RouterTaskBackend, install_task_error_handlers

    async def finalize_task_job(route: ExecutionRoute, execution: TaskExecution, request: Request | None) -> None:
        """新任务沿用既有产物元数据补齐及负载释放边界。"""
        await _finalize_terminal_job(route, execution)

    def task_backend(request: Request) -> RouterTaskBackend:
        """显式组装 Router 适配器，便捷入口不通过 HTTP 请求自身。"""
        return RouterTaskBackend(request, _caller_scope(request), finalize_task_job)

    install_task_error_handlers(application)
    application.include_router(build_task_router(task_backend))
    if resolved_settings.admin_api_key:
        application.include_router(
            build_admin_router(
                pool,
                registry,
                api_key=resolved_settings.admin_api_key,
                collect_resources=collect_resources_once,
            )
        )
    return application


def create_app_from_env() -> FastAPI:
    """为 Uvicorn reload worker 按环境变量创建新的 Router 应用。"""
    return create_app(RouterSettings.from_env())


__all__ = ["create_app", "create_app_from_env"]
