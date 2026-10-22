"""Router 的逐文件任务适配器；worker 负责解析与单文件产物，Router 仅映射资源。"""

from __future__ import annotations

import asyncio
import copy
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import ExitStack
from dataclasses import replace
from typing import Any
from urllib.parse import SplitResult, quote, unquote, urljoin, urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ...filetypes import is_flash_only_parse_extension
from ...parser.api_server import ApiServerError, ErrorDetail, _JobInputMetadata, _raise_api_error
from ...parser.task_api import (
    TERMINAL_TASK_STATUSES,
    TaskExecution,
    TaskManifestItem,
    TaskStatusResponse,
    TaskUpload,
    UploadTaskOptions,
    file_chunks,
    resolve_task_tier,
)
from ...types import select_default_quality_tier
from ...utils.async_utils import drain_future
from .proxy import (
    RouterProxyError,
    release_job_load,
    request_upstream,
    require_worker_generation,
    rewrite_file_payload,
    rewrite_job_payload,
)
from .resources import ResourceRegistry, ResourceRoute, SourceFileStore, StoredSourceFile, utc_now_iso
from .workers import WorkerPool, WorkerState

FinalizeJob = Callable[[ResourceRoute, dict[str, Any], Request | None], Awaitable[None]]


def _checked_json(response: httpx.Response) -> dict[str, Any]:
    """保留 worker 的业务错误码，非法响应不能伪装成成功任务。"""
    try:
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
    except ValueError:
        _raise_api_error(
            response.status_code if response.status_code >= 400 else 502,
            error_type="api_error",
            code="invalid_upstream_response",
            message="Upstream did not return a JSON object",
        )
    if response.status_code >= 400:
        detail = payload.get("error") or {}
        raise ApiServerError(
            response.status_code,
            ErrorDetail.model_validate(
                {
                    "type": detail.get("type", "api_error"),
                    "code": detail.get("code", "upstream_error"),
                    "message": detail.get("message", "Upstream request failed"),
                    "param": detail.get("param"),
                }
            ),
        )
    return payload


async def _worker_request(
    pool: WorkerPool,
    worker: WorkerState,
    method: str,
    path: str,
    *,
    request: Request | None = None,
    content: bytes | AsyncIterator[bytes] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """把网络错误转换成共享 tasks 错误，监控阶段可识别临时失败并继续查询。"""
    try:
        return await request_upstream(
            pool, worker, method, path, request=request, content=content, json_body=json_body, headers=headers
        )
    except RouterProxyError as exc:
        _raise_api_error(exc.status_code, error_type="api_error", code=exc.code, message=exc.message, param=exc.param)


def _worker_path(worker: WorkerState, url: str) -> str:
    """只接受配置 worker 自身的资源链接，禁止把私有认证或源文件转发到其他地址。"""
    base = urlsplit(worker.base_url)
    target = urlsplit(urljoin(worker.base_url.rstrip("/") + "/", url))

    def origin(item: SplitResult) -> tuple[str, str | None, int]:
        """按协议、主机及有效端口比较来源，端口不同也属于跨源。"""
        return item.scheme, item.hostname, item.port or (443 if item.scheme == "https" else 80)

    prefix = base.path.rstrip("/")
    if origin(base) != origin(target) or target.username or target.password or target.query or target.fragment:
        _raise_api_error(502, error_type="api_error", code="invalid_upstream_response", message="Invalid worker resource URL")
    if prefix and not target.path.startswith(prefix + "/"):
        _raise_api_error(502, error_type="api_error", code="invalid_upstream_response", message="Invalid worker resource path")
    return target.path[len(prefix) :]


def _job_payload(payload: dict[str, Any], upstream_id: str, *, authenticated: bool) -> dict[str, Any]:
    """把 worker 的 tasks 元数据转换成既有 Router Job 视图，不泄露上传或 worker 链接。"""
    file_keys = ("file_id", "name", "page_range", "status", "parse", "output_files", "error")
    files = [{key: item.get(key) for key in file_keys} for item in payload["files"]]
    for file in files:
        if file["status"] in {"waiting_upload", "canceled"}:
            file["status"] = "queued"
    return {
        "job_id": upstream_id,
        "status": "queued" if payload["status"] == "waiting_upload" else payload["status"],
        "created_at": payload["created_at"],
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "tier": payload["tier"],
        "output_formats": [],
        "progress": payload["progress"],
        "access_level": "registered" if authenticated else "anonymous",
        "files": files,
        "links": {"self": f"/v1/parse/jobs/{upstream_id}", "cancel": f"/v1/parse/jobs/{upstream_id}"},
    }


def _lost_execution(route: ResourceRoute, pool: WorkerPool) -> TaskExecution:
    """worker 丢失时收敛失败和负载，保留已经成功文件的诊断信息。"""
    payload = copy.deepcopy(route.metadata.get("payload") or {})
    payload.update(job_id=route.public_id, status="failed", finished_at=utc_now_iso())
    error = ErrorDetail(type="engine_error", code="upstream_task_lost", message="Upstream task no longer exists")
    for file in payload.get("files") or []:
        if file.get("status") != "completed":
            file.update(status="failed", error=error.model_dump())
    route.metadata.update(payload=payload, upstream_lost=True)
    release_job_load(route, pool)
    return TaskExecution(payload, error=error, input_metadata=dict(route.metadata.get("task_input_metadata") or {}))


async def read_task_execution(
    route: ResourceRoute,
    pool: WorkerPool,
    registry: ResourceRegistry,
    *,
    request: Request | None = None,
    headers: Mapping[str, str] | None = None,
) -> TaskExecution:
    """查询 worker 的轻量任务状态，并把单文件产物登记为 Router 的公共 File。"""
    if route.metadata.get("upstream_lost"):
        return _lost_execution(route, pool)
    worker = pool.get(route.worker_id)
    generation = worker.generation
    response = await _worker_request(
        pool,
        worker,
        "GET",
        f"/v1/tasks/{quote(route.upstream_id, safe='')}",
        request=request,
        headers={
            **(headers or {}),
            **({"x-mineru-task-lease-seconds": "120"} if route.metadata.get("task_result_held") else {}),
        },
    )
    if response.status_code == 404 or worker.generation != generation or route.metadata.get("upstream_lost"):
        return _lost_execution(route, pool)
    try:
        payload = TaskStatusResponse.model_validate(_checked_json(response)).model_dump()
    except ValidationError as exc:
        _raise_api_error(502, error_type="api_error", code="invalid_upstream_response", message=str(exc))
    if payload["task_id"] != route.upstream_id:
        _raise_api_error(502, error_type="api_error", code="invalid_upstream_response", message="Upstream task ID changed")
    result_file_id: str | None = None
    if len(payload["files"]) != 1:
        _raise_api_error(502, error_type="api_error", code="invalid_upstream_response", message="Expected one task file")
    if payload["files"][0]["status"] == "completed":
        result_url = payload["files"][0].get("result_url")
        if not isinstance(result_url, str):
            _raise_api_error(502, error_type="api_error", code="invalid_upstream_response", message="Missing result URL")
        path = _worker_path(worker, result_url)
        parts = path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["v1", "files"] or parts[-1] != "content":
            _raise_api_error(502, error_type="api_error", code="invalid_upstream_response", message="Invalid result File URL")
        upstream_file_id = unquote(parts[2])
        existing = registry.find_upstream("file", route.owner_scope, worker.worker_id, upstream_file_id)
        if existing is None or not isinstance(existing.metadata.get("payload"), dict):
            metadata = _checked_json(
                await _worker_request(
                    pool,
                    worker,
                    "GET",
                    f"/v1/files/{quote(upstream_file_id, safe='')}",
                    request=request,
                    headers=headers,
                )
            )
            require_worker_generation(worker, generation)
            if metadata.get("id") != upstream_file_id or metadata.get("purpose") != "parse_output":
                _raise_api_error(
                    502, error_type="api_error", code="invalid_upstream_response", message="Invalid result File metadata"
                )
            result_file_id = rewrite_file_payload(metadata, worker, registry, route.owner_scope)["id"]
        else:
            result_file_id = existing.public_id
        result_route = registry.find("file", result_file_id)
        if result_route is not None:
            result_route.metadata["task_result_held"] = bool(route.metadata.get("task_result_held"))
    authenticated = bool(request.headers.get("authorization") if request is not None else (headers or {}).get("authorization"))
    rewritten = rewrite_job_payload(
        _job_payload(payload, route.upstream_id, authenticated=authenticated),
        worker,
        registry,
        pool,
        route.owner_scope,
    )
    route.metadata["task_result_file_id"] = result_file_id
    error = ErrorDetail.model_validate(payload["error"]) if payload.get("error") else None
    input_metadata = {
        index: _JobInputMetadata(item["bytes"], item["sha256sum"])
        for index, item in enumerate(payload["files"])
        if item.get("url") is not None and item["bytes"] is not None and item["sha256sum"] is not None
    }
    route.metadata["task_input_metadata"] = input_metadata
    return TaskExecution(rewritten, result_file_id, error, input_metadata)


class RouterTaskBackend:
    """客户端上传项就绪后才选择 worker；URL 原样交由 worker 下载。"""

    def __init__(self, request: Request, owner_scope: str, finalize: FinalizeJob) -> None:
        """保存轻量上下文与显式收尾回调，不跨请求保存上传主体。"""
        self.request = Request(dict(request.scope))
        self.owner_scope = owner_scope
        self.finalize = finalize
        self.sources: SourceFileStore = request.app.state.source_store
        self.registry: ResourceRegistry = request.app.state.registry
        self.pool: WorkerPool = request.app.state.worker_pool
        self._pins = ExitStack()
        self._owned_files: list[str] = []

    def fork(self) -> RouterTaskBackend:
        """每个输入文件拥有独立暂存引用和 worker 调度上下文。"""
        return RouterTaskBackend(self.request, self.owner_scope, self.finalize)

    async def release_result(self, task_id: str, finished: float) -> None:
        """父任务完成后记录公开保留期限，后台续租直到该期限再释放 worker。"""
        route = self.registry.find("job", task_id)
        if route is None:
            return
        route.metadata["task_result_retain_until"] = (
            finished + self.registry.retention_seconds if self.registry.retention_seconds else None
        )
        route.metadata["finalized_at"] = utc_now_iso()
        result_id = route.metadata.get("task_result_file_id")
        result_route = self.registry.find("file", result_id) if result_id else None
        if result_route is not None:
            result_route.metadata["payload"]["expires_at"] = route.metadata["task_result_retain_until"]
            result_route.metadata["task_result_retain_until"] = route.metadata["task_result_retain_until"]

    async def validate(self, options: UploadTaskOptions, manifest: list[TaskManifestItem]) -> None:
        """能力预检不推进轮询游标，也不为尚未上传的任务预占 worker。"""
        tiers = self.pool.available_tiers()
        options.tier = resolve_task_tier(options, manifest, tiers, select_default_quality_tier(tiers))
        checked_workers: set[str] = set()
        for item in manifest:
            lightweight = is_flash_only_parse_extension(item.filename)
            selection_tier = "flash" if lightweight else options.tier
            candidates = self.pool.capable_workers(
                tier=selection_tier,
                required_sources={"url" if item.url is not None else "file_id"},
                required_task_outputs=set(options.output_formats),
                require_flash=lightweight,
            )
            if not candidates:
                _raise_api_error(
                    503,
                    error_type="api_error",
                    code="quality_tier_unavailable",
                    message="No worker supports requested task formats",
                )
            # 能力预检不预占名额，不推进轮询游标；认证沿用所选部署的服务认证。
            worker = candidates[0]
            if worker.worker_id not in checked_workers:
                _checked_json(await _worker_request(self.pool, worker, "GET", "/v1/files?limit=1", request=self.request))
                checked_workers.add(worker.worker_id)

    async def find_upload(self, sha256sum: str, filename: str) -> TaskUpload | None:
        """仅复用当前调用方持有的 Router 源字节，worker 哈希命中不能替代本地缓存。"""
        stored = self.sources.find_hash(self.owner_scope, sha256sum)
        if stored is None:
            return None
        self._pins.enter_context(self.sources.pin(stored))
        return TaskUpload(stored.path, filename, stored.bytes, stored.sha256sum, stored.mime_type)

    async def retain_upload(self, upload: TaskUpload) -> TaskUpload:
        """为已校验文件建立准备引用，同一调用方相同内容复用现有暂存字节。"""
        async with self.request.app.state.task_store.storage_lock:
            private_id = "task-source-" + secrets.token_hex(12)
            self._owned_files.append(private_id)
            stored = self.sources.find_hash(self.owner_scope, upload.sha256sum)
            if stored is None:
                staging_id = "task-" + secrets.token_hex(12)
                try:
                    stored = await self.sources.stage_upload(staging_id, file_chunks(upload.path), mime_type=upload.mime_type)
                    self.sources.bind_file(staging_id, private_id, owner_scope=self.owner_scope)
                finally:
                    self.sources.discard_upload(staging_id)
            else:
                self.sources.bind_source(private_id, stored, owner_scope=self.owner_scope)
            self._pins.enter_context(self.sources.pin(stored))
            return replace(upload, path=stored.path, file_id=None)

    def release_uploads(self, *, rollback: bool) -> None:
        """关闭准备引用；成功提交后公开 File 绑定继续持有可复用源字节。"""
        for file_id in self._owned_files:
            self.sources.delete_file(file_id)
        self._owned_files.clear()
        self._pins.close()

    async def submit(
        self, task_id: str, options: UploadTaskOptions, manifest: list[TaskManifestItem], uploads: dict[int, TaskUpload]
    ) -> TaskExecution:
        """预占后提交 worker 任务，只向 worker 上传它实际缺失的文件。"""
        lightweight = all(is_flash_only_parse_extension(item.filename) for item in manifest)
        tier = "flash" if lightweight else options.tier
        reservation = self.pool.reserve_job(
            tier=tier,
            required_sources={"url" if item.url is not None else "file_id" for item in manifest},
            required_task_outputs=set(options.output_formats),
            require_flash=any(is_flash_only_parse_extension(item.filename) for item in manifest),
        )
        if reservation is None:
            _raise_api_error(
                503, error_type="api_error", code="quality_tier_unavailable", message="No capable worker is available"
            )
        worker = reservation.worker
        upstream_id: str | None = None
        route: ResourceRoute | None = None
        registered_inputs: list[str] = []
        transferred = False
        try:
            worker_manifest = [
                item.model_dump(exclude_none=True)
                if item.url is not None
                else {**item.model_dump(exclude_none=True), "sha256sum": uploads[index].sha256sum}
                for index, item in enumerate(manifest)
            ]
            response = await _worker_request(
                self.pool,
                worker,
                "POST",
                "/v1/tasks",
                request=self.request,
                json_body={
                    "manifest": worker_manifest,
                    "tier": options.tier,
                    "ocr_mode": options.ocr_mode,
                    "output_formats": options.output_formats,
                },
                headers={"x-mineru-task-lease-seconds": "120"},
            )
            require_worker_generation(worker, reservation.generation)
            payload = TaskStatusResponse.model_validate(_checked_json(response)).model_dump()
            upstream_id = payload["task_id"]
            route = self.registry.register(
                "job",
                owner_scope=self.owner_scope,
                worker_id=worker.worker_id,
                upstream_id=upstream_id,
                public_id=task_id,
            )
            authorization = self.request.headers.get("authorization")
            route.metadata.update(
                task_protocol=True,
                task_result_held=True,
                active_counted=True,
                reservation=reservation,
                upstream_headers={"authorization": authorization} if authorization else {},
                payload={"job_id": task_id, "status": "queued", "files": []},
            )
            by_client_id = {item.client_file_id: index for index, item in enumerate(manifest)}
            for link in payload["uploads"]:
                index = by_client_id.get(link["client_file_id"])
                if index not in uploads or link["upload_method"] != "PUT":
                    _raise_api_error(
                        502, error_type="api_error", code="invalid_upstream_response", message="Unexpected worker upload"
                    )
                path = _worker_path(worker, link["upload_url"])
                expected_path = f"/v1/tasks/{quote(upstream_id, safe='')}/uploads/{quote(link['client_file_id'], safe='')}"
                if path != expected_path:
                    _raise_api_error(
                        502, error_type="api_error", code="invalid_upstream_response", message="Invalid worker upload path"
                    )
                response = await _worker_request(
                    self.pool,
                    worker,
                    "PUT",
                    path,
                    request=self.request,
                    content=file_chunks(uploads[index].path),
                    headers={
                        **link["upload_headers"],
                        "content-length": str(uploads[index].bytes),
                        "x-mineru-task-lease-seconds": "120",
                    },
                )
                if response.status_code >= 400:
                    _checked_json(response)
                require_worker_generation(worker, reservation.generation)
            # 上传就绪即可获取源 File 引用，后台 Job 尚未启动也不影响缓存交接。
            response = await _worker_request(self.pool, worker, "GET", f"/v1/tasks/{upstream_id}", request=self.request)
            require_worker_generation(worker, reservation.generation)
            ready = TaskStatusResponse.model_validate(_checked_json(response)).model_dump()
            if len(ready["files"]) != len(manifest) or ready["status"] == "waiting_upload":
                _raise_api_error(
                    502, error_type="api_error", code="invalid_upstream_response", message="Worker inputs are not ready"
                )
            aliases: dict[str, str] = {}
            for index, item in enumerate(ready["files"]):
                if item["client_file_id"] != manifest[index].client_file_id:
                    _raise_api_error(
                        502, error_type="api_error", code="invalid_upstream_response", message="Worker input order changed"
                    )
                if manifest[index].url is not None:
                    if item["upload_status"] != "not_required":
                        _raise_api_error(
                            502, error_type="api_error", code="invalid_upstream_response", message="Unexpected URL upload"
                        )
                    continue
                upstream_file_id = item.get("file_id")
                if not isinstance(upstream_file_id, str):
                    _raise_api_error(
                        502, error_type="api_error", code="invalid_upstream_response", message="Missing worker source File"
                    )
                upload = uploads[index]
                rewritten = rewrite_file_payload(
                    {
                        "id": upstream_file_id,
                        "object": "file",
                        "filename": upload.filename,
                        "bytes": upload.bytes,
                        "sha256sum": upload.sha256sum,
                        "purpose": "parse",
                        "created_at": int(time.time()),
                        "expires_at": None,
                    },
                    worker,
                    self.registry,
                    self.owner_scope,
                )
                public_id = rewritten["id"]
                registered_inputs.append(public_id)
                self.sources.bind_source(
                    public_id,
                    StoredSourceFile(upload.path, upload.bytes, upload.sha256sum, upload.mime_type),
                    owner_scope=self.owner_scope,
                )
                aliases[upstream_file_id] = public_id
            route.metadata["input_aliases"] = aliases
            result = await self.get(task_id)
            transferred = True
            return result
        finally:
            if not transferred:
                if upstream_id is not None and worker.generation == reservation.generation and worker.base_url:
                    await drain_future(asyncio.create_task(self._cancel_remote(worker, upstream_id)))
                if route is not None:
                    release_job_load(route, self.pool)
                    route.metadata.update(
                        payload={"job_id": task_id, "status": "failed", "files": []}, finalized_at=utc_now_iso()
                    )
                    route.metadata["task_result_retain_until"] = time.time()
                else:
                    reservation.release()
                for public_id in registered_inputs:
                    self.registry.remove("file", public_id)
                    self.sources.delete_file(public_id)

    async def _cancel_remote(self, worker: WorkerState, upstream_id: str) -> None:
        """尽力取消已经创建的 worker 任务，清理失败不覆盖最初的提交错误。"""
        try:
            await _worker_request(self.pool, worker, "DELETE", f"/v1/tasks/{upstream_id}", request=self.request)
        except Exception:
            # worker 已不可达或被替换时，原始错误及任务保留策略仍负责后续诊断。
            return

    def _route(self, task_id: str) -> ResourceRoute:
        """检查本调用方的公共执行映射，不能用其他调用方的任务探测 worker。"""
        route = self.registry.find("job", task_id)
        if route is None or route.owner_scope != self.owner_scope:
            _raise_api_error(
                404, error_type="invalid_request_error", code="task_not_found", message=f"Task {task_id} not found"
            )
        return route

    async def get(self, task_id: str) -> TaskExecution:
        """只查询 worker 元数据并登记结果引用，产物字节走已有 Files 流式代理。"""
        route = self._route(task_id)
        execution = await read_task_execution(route, self.pool, self.registry, request=self.request)
        await self.finalize(route, execution.job, self.request)
        return execution

    async def cancel(self, task_id: str) -> None:
        """通过 worker tasks 取消准备或执行，成功后沿用一次性负载释放。"""
        route = self.registry.find("job", task_id)
        if route is None:
            return
        if route.metadata.get("upstream_lost"):
            release_job_load(route, self.pool)
            return
        worker = self.pool.get(route.worker_id)
        response = await _worker_request(self.pool, worker, "DELETE", f"/v1/tasks/{route.upstream_id}", request=self.request)
        if response.status_code not in {404, 409}:
            _checked_json(response)
        execution = await self.get(task_id)
        if execution.job["status"] in TERMINAL_TASK_STATUSES:
            release_job_load(route, self.pool)


def install_task_error_handlers(app: FastAPI) -> None:
    """让共享任务输入和后台代理错误使用与 API server 相同的 V1 envelope。"""

    @app.exception_handler(ApiServerError)
    async def task_error(_request: Request, exc: ApiServerError) -> JSONResponse:
        """保留共享错误的 HTTP 状态和明确错误码。"""
        return JSONResponse({"error": exc.error.model_dump()}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """把路径及 JSON 校验错误统一转换成 400。"""
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "code": "invalid_request", "message": str(exc)}},
            status_code=400,
        )


async def maintain_task_result_leases(registry: ResourceRegistry, pool: WorkerPool, *, release: bool = False) -> None:
    """续租已结束父任务的子产物，进程关闭或公开保留期结束时释放上游租约。"""
    for route in registry.list("job", include_expired=True):
        if not route.metadata.get("task_result_held"):
            continue
        if not release and "task_result_retain_until" not in route.metadata:
            continue
        until = route.metadata.get("task_result_retain_until")
        ended = release or (until is not None and until <= time.time())
        worker = pool.get(route.worker_id)
        headers = dict(route.metadata.get("upstream_headers") or {})
        headers["x-mineru-task-lease-seconds"] = "0" if ended else "120"
        try:
            response = await _worker_request(pool, worker, "GET", f"/v1/tasks/{route.upstream_id}", headers=headers)
            if response.status_code >= 400 and response.status_code != 404 and not ended:
                continue
        except ApiServerError:
            if not ended:
                continue
        if ended:
            route.metadata["task_result_held"] = False
            route.metadata.pop("upstream_headers", None)
            result_id = route.metadata.get("task_result_file_id")
            result_route = next(
                (item for item in registry.list("file", include_expired=True) if item.public_id == result_id), None
            )
            if result_route is not None:
                result_route.metadata["task_result_held"] = False


__all__ = ["RouterTaskBackend", "install_task_error_handlers", "read_task_execution", "maintain_task_result_leases"]
