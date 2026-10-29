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

from ...filetypes import batch_effective_parse_tier, is_flash_only_parse_extension
from ...tasks.contracts import (
    CancelTaskResponse,
    CreateTaskResponse,
    ErrorDetail,
    ErrorResponse,
    TaskDetailResponse,
    TaskFileInput,
    TaskFileItem,
    UploadFileResponse,
    UploadTaskOptions,
)
from ...tasks.errors import ApiServerError, raise_api_error
from ...tasks.execution import TaskExecution, TaskInputMetadata, TaskUpload
from ...tasks.service import TERMINAL_TASK_STATUSES, file_chunks, resolve_task_tier
from ...utils.async_utils import drain_future
from .proxy import (
    RouterProxyError,
    request_upstream,
    require_worker_generation,
    rewrite_file_payload,
)
from .resources import ExecutionRoute, ResourceRegistry, SourceFileStore, utc_now_iso
from .route_state import ExecutionRouteState
from .workers import WorkerPool, WorkerState

FinalizeExecution = Callable[[ExecutionRoute, TaskExecution, Request | None], Awaitable[None]]


def _checked_json(response: httpx.Response) -> dict[str, Any]:
    """保留 worker 的业务错误码，非法响应不能伪装成成功任务。"""
    try:
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
    except ValueError:
        raise_api_error(
            response.status_code if response.status_code >= 400 else 502,
            code="invalid_upstream_response",
            message="Upstream did not return a JSON object",
        )
    if response.status_code >= 400:
        try:
            error = ErrorResponse.model_validate(payload).error
        except ValidationError:
            raise_api_error(502, code="invalid_upstream_response", message="Upstream returned an invalid error response")
        raise ApiServerError(response.status_code, error)
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
        raise_api_error(exc.status_code, code=exc.code, message=exc.message, param=exc.param)


def _worker_path(worker: WorkerState, url: str) -> str:
    """只接受配置 worker 自身的资源链接，禁止把私有认证或源文件转发到其他地址。"""
    base = urlsplit(worker.base_url)
    target = urlsplit(urljoin(worker.base_url.rstrip("/") + "/", url))

    def origin(item: SplitResult) -> tuple[str, str | None, int]:
        """按协议、主机及有效端口比较来源，端口不同也属于跨源。"""
        return item.scheme, item.hostname, item.port or (443 if item.scheme == "https" else 80)

    prefix = base.path.rstrip("/")
    if origin(base) != origin(target) or target.username or target.password or target.query or target.fragment:
        raise_api_error(502, code="invalid_upstream_response", message="Invalid worker resource URL")
    if prefix and not target.path.startswith(prefix + "/"):
        raise_api_error(502, code="invalid_upstream_response", message="Invalid worker resource path")
    return target.path[len(prefix) :]


def _lost_execution(route: ExecutionRoute) -> TaskExecution:
    """worker 丢失时终结有类型快照，并保留既有资源计数与诊断缓存。"""
    previous = route.state.execution
    error = ErrorDetail(code="upstream_task_lost", message="Upstream task no longer exists")
    execution = replace(previous, status="failed", finished_at=utc_now_iso(), result_file_id=None, error=error)
    payload = copy.deepcopy(route.payload or {})
    payload.update(status="failed", finished_at=execution.finished_at)
    for file in payload.get("files") or []:
        if file.get("status") != "completed":
            file.update(status="failed", error=error.model_dump())
    route.payload = payload
    route.state.execution = execution
    route.state.upstream_lost = True
    route.state.release_reservation()
    return execution


async def read_task_execution(
    route: ExecutionRoute,
    pool: WorkerPool,
    registry: ResourceRegistry,
    *,
    request: Request | None = None,
    headers: Mapping[str, str] | None = None,
) -> TaskExecution:
    """只在 HTTP 边界拆出 worker 的唯一文件，内部传递有类型的执行快照。"""
    if route.state.upstream_lost:
        return _lost_execution(route)
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
            **({"x-mineru-task-lease-seconds": "120"} if route.lease.held else {}),
        },
    )
    if response.status_code == 404 or worker.generation != generation or route.state.upstream_lost:
        return _lost_execution(route)
    try:
        payload = TaskDetailResponse.model_validate(_checked_json(response))
    except ValidationError as exc:
        raise_api_error(502, code="invalid_upstream_response", message=str(exc))
    if payload.task_id != route.upstream_id:
        raise_api_error(502, code="invalid_upstream_response", message="Upstream task ID changed")
    if len(payload.files) != 1:
        raise_api_error(502, code="invalid_upstream_response", message="Expected one task file")
    if payload.status == "partial":
        raise_api_error(502, code="invalid_upstream_response", message="Invalid single-file task status")
    (remote_file,) = payload.files
    if remote_file.file_id != route.state.upstream_file_id or remote_file.file_name != route.state.execution.file_name:
        raise_api_error(502, code="invalid_upstream_response", message="Worker file identity changed")
    upstream_file_id = remote_file.file_id
    public_file_id = route.state.execution.file_id
    result_file_id: str | None = None
    if remote_file.status == "completed":
        result_url = remote_file.result_url
        if not isinstance(result_url, str):
            raise_api_error(502, code="invalid_upstream_response", message="Missing result URL")
        resource_path = _worker_path(worker, result_url)
        parts = resource_path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["v1", "files"] or parts[-1] != "content":
            raise_api_error(502, code="invalid_upstream_response", message="Invalid result File URL")
        if unquote(parts[2]) != upstream_file_id:
            raise_api_error(502, code="invalid_upstream_response", message="Result file ID changed")
        existing = registry.find_upstream("file", route.owner_scope, worker.worker_id, upstream_file_id)
        if existing is None and not registry.is_file_reserved(public_file_id):
            # 取消、删除或过期的结果不因迟到响应重新登记；父 Task 保持已经发布的终态。
            result_file_id = public_file_id
        elif existing is None or existing.payload is None:
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
                raise_api_error(502, code="invalid_upstream_response", message="Invalid result File metadata")
            result_file_id = rewrite_file_payload(metadata, worker, registry, route.owner_scope, public_id=public_file_id)["id"]
        else:
            if existing.public_id != public_file_id:
                raise_api_error(502, code="invalid_upstream_response", message="Public file ID changed")
            result_file_id = existing.public_id
        result_route = registry.find("file", result_file_id)
        if result_route is not None:
            result_route.lease.held = route.lease.held
    previous = route.state.execution
    execution = TaskExecution(
        execution_id=route.public_id,
        file_id=public_file_id,
        file_name=remote_file.file_name,
        status=payload.status,
        created_at=payload.created_at,
        started_at=payload.started_at,
        finished_at=payload.finished_at,
        page_range=previous.page_range,
        parse=remote_file.parse,
        result_file_id=result_file_id,
        error=payload.error or remote_file.error,
        input_metadata=TaskInputMetadata(remote_file.bytes)
        if remote_file.url is not None and remote_file.bytes is not None
        else None,
    )
    if execution.status in TERMINAL_TASK_STATUSES:
        route.state.release_reservation()
    # 既有 Registry 仍消费 HTTP 状态缓存做回收和统计；执行协议不再经过 Job 字典转换。
    route.payload = payload.model_dump()
    route.state.execution = execution
    route.state.result_file_id = result_file_id
    return execution


class RouterTaskBackend:
    """客户端上传项就绪后才选择 worker；URL 原样交由 worker 下载。"""

    def __init__(self, request: Request, owner_scope: str, finalize: FinalizeExecution) -> None:
        """保存轻量上下文与显式收尾回调，不跨请求保存上传主体。"""
        self.request = Request(dict(request.scope))
        self.owner_scope = owner_scope
        self.finalize = finalize
        self.sources: SourceFileStore = request.app.state.source_store
        self.registry: ResourceRegistry = request.app.state.registry
        self.pool: WorkerPool = request.app.state.worker_pool
        self._pins = ExitStack()
        self._prepared_sources: list[str] = []
        self._task_sources: list[str] = []

    def fork(self) -> RouterTaskBackend:
        """每个输入文件拥有独立暂存引用和 worker 调度上下文。"""
        return RouterTaskBackend(self.request, self.owner_scope, self.finalize)

    def reserve_file_id(self) -> str:
        """每层 Router 独立预留公共 ID，通过内部映射关联下游生成的标识。"""
        return self.registry.reserve_file_id()

    def discard_file_id(self, file_id: str) -> None:
        """取消本层预留或尚未发布的结果映射，不触碰共享源缓存。"""
        self.registry.discard_file_id(file_id)

    def forget_file_id(self, file_id: str) -> None:
        """父任务记录过期后回收本层 ID 索引，不依赖 worker 路由仍然存在。"""
        self.registry.forget_file_id(file_id)

    async def release_result(self, task_id: str, finished: float) -> None:
        """父任务完成后记录公开保留期限，后台续租直到该期限再释放 worker。"""
        # 私有源缓存独立于 worker 路由，即使 worker 已丢失也必须开始回收倒计时。
        for reference_id in self._task_sources:
            self.sources.release_reference(reference_id, finished)
        self._task_sources.clear()
        route = self.registry.find("job", task_id)
        if route is None:
            return
        route.lease.retain_until = finished + self.registry.retention_seconds if self.registry.retention_seconds else None
        route.finalized_at = utc_now_iso()
        result_id = route.state.result_file_id
        result_route = self.registry.find("file", result_id) if result_id else None
        if result_route is not None:
            assert result_route.payload is not None
            result_route.payload["expires_at"] = route.lease.retain_until
            result_route.lease.retain_until = route.lease.retain_until

    async def validate(self, options: UploadTaskOptions, inputs: list[TaskFileInput]) -> None:
        """能力预检不推进轮询游标，也不为尚未上传的任务预占 worker。"""
        tiers = self.pool.available_tiers()
        options.tier = resolve_task_tier(options, inputs, tiers)
        checked_workers: set[str] = set()
        for item in inputs:
            lightweight = is_flash_only_parse_extension(item.file_name)
            selection_tier = None if lightweight else options.tier
            candidates = self.pool.capable_workers(
                tier=selection_tier,
                required_sources={"url" if item.url is not None else "upload"},
                required_task_outputs=set(options.output_formats),
            )
            if not candidates:
                raise_api_error(
                    503,
                    code="quality_tier_unavailable",
                    message="No worker supports requested task formats",
                )
            # 能力预检不预占名额，不推进轮询游标；认证沿用所选部署的服务认证。
            worker = candidates[0]
            if worker.worker_id not in checked_workers:
                _checked_json(await _worker_request(self.pool, worker, "GET", "/v1/usage", request=self.request))
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
            self._prepared_sources.append(private_id)
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
            self.sources.hold_reference(private_id)
            self._pins.enter_context(self.sources.pin(stored))
            return replace(upload, path=stored.path, file_id=None)

    def release_uploads(self, *, rollback: bool) -> None:
        """回滚未交接的源引用并释放传输租约，成功提交的缓存由父任务持有。"""
        if rollback:
            for reference_id in self._prepared_sources:
                self.sources.delete_file(reference_id)
            self._prepared_sources.clear()
        self._pins.close()

    async def _wait_for_upload_ready(
        self, worker: WorkerState, upstream_id: str, generation: int, item: TaskFileItem, upstream_file_id: str
    ) -> TaskDetailResponse:
        """等待唯一上传项被接受，URL 输入仍由下游在独立准备阶段下载。"""
        while True:
            response = await _worker_request(
                self.pool,
                worker,
                "GET",
                f"/v1/tasks/{quote(upstream_id, safe='')}",
                request=self.request,
                headers={"x-mineru-task-lease-seconds": "120"},
            )
            require_worker_generation(worker, generation)
            ready = TaskDetailResponse.model_validate(_checked_json(response))
            if ready.task_id != upstream_id or len(ready.files) != 1:
                raise_api_error(502, code="invalid_upstream_response", message="Worker inputs are not ready")
            (remote_file,) = ready.files
            if remote_file.file_id != upstream_file_id or remote_file.file_name != item.file_name:
                raise_api_error(502, code="invalid_upstream_response", message="Worker input order changed")
            if item.url is not None:
                if remote_file.upload is not None:
                    raise_api_error(502, code="invalid_upstream_response", message="URL input requests upload")
                return ready
            if remote_file.status != "waiting-file":
                return ready
            await asyncio.sleep(1)

    async def submit(
        self, execution_id: str, options: UploadTaskOptions, item: TaskFileItem, upload: TaskUpload | None
    ) -> TaskExecution:
        """为单文件预占 worker，并把公开文件身份与下游身份一一关联。"""
        if options.tier is None:
            raise ValueError("Task tier must be resolved before execution")
        lightweight = is_flash_only_parse_extension(item.file_name)
        tier = batch_effective_parse_tier(options.tier, item.file_name)
        reservation = self.pool.reserve_job(
            tier=None if lightweight else tier,
            required_sources={"url" if item.url is not None else "upload"},
            required_task_outputs=set(options.output_formats),
        )
        if reservation is None:
            raise_api_error(503, code="quality_tier_unavailable", message="No capable worker is available")
        worker = reservation.worker
        upstream_id: str | None = None
        route: ExecutionRoute | None = None
        transferred = False
        try:
            worker_input = item.model_dump(exclude_none=True, exclude={"file_id"})
            if item.url is None:
                if upload is None:
                    raise ValueError("Prepared source file is missing")
                worker_input["sha256sum"] = upload.sha256sum
            response = await _worker_request(
                self.pool,
                worker,
                "POST",
                "/v1/tasks",
                request=self.request,
                json_body={
                    "files": [worker_input],
                    **({"tier": tier} if not lightweight else {}),
                    "ocr_mode": options.ocr_mode,
                    "output_formats": options.output_formats,
                },
                headers={"x-mineru-task-lease-seconds": "120"},
            )
            require_worker_generation(worker, reservation.generation)
            created = CreateTaskResponse.model_validate(_checked_json(response))
            upstream_id = created.task_id
            authorization = self.request.headers.get("authorization")
            initial = TaskExecution(
                execution_id=execution_id,
                file_id=item.file_id,
                file_name=item.file_name,
                status="pending",
                created_at=utc_now_iso(),
                page_range=item.page_range or "",
            )
            route = self.registry.register_execution(
                owner_scope=self.owner_scope,
                worker_id=worker.worker_id,
                upstream_id=upstream_id,
                state=ExecutionRouteState(
                    execution=initial,
                    reservation=reservation,
                    upstream_headers={"authorization": authorization} if authorization else {},
                ),
                payload={"task_id": upstream_id, "status": "pending", "files": []},
            )
            route.lease.held = True
            if len(created.files) != 1:
                raise_api_error(502, code="invalid_upstream_response", message="Worker input order changed")
            (remote_file,) = created.files
            if remote_file.file_name != item.file_name:
                raise_api_error(502, code="invalid_upstream_response", message="Worker input order changed")
            route.state.upstream_file_id = remote_file.file_id
            link = remote_file.upload
            if link is not None:
                if upload is None:
                    raise_api_error(502, code="invalid_upstream_response", message="Unexpected worker upload")
                resource_path = _worker_path(worker, link.url)
                expected_path = f"/v1/tasks/{quote(upstream_id, safe='')}/uploads/{quote(remote_file.file_id, safe='')}"
                if resource_path != expected_path:
                    raise_api_error(502, code="invalid_upstream_response", message="Invalid worker upload path")
                response = await _worker_request(
                    self.pool,
                    worker,
                    "PUT",
                    resource_path,
                    request=self.request,
                    content=file_chunks(upload.path),
                    headers={
                        **link.headers,
                        "content-length": str(upload.bytes),
                        "x-mineru-task-lease-seconds": "120",
                    },
                )
                try:
                    acknowledged = UploadFileResponse.model_validate(_checked_json(response))
                except ValidationError as exc:
                    raise_api_error(502, code="invalid_upstream_response", message=str(exc))
                if acknowledged.task_id != upstream_id or acknowledged.file_id != remote_file.file_id:
                    raise_api_error(
                        502,
                        code="invalid_upstream_response",
                        message="Upload acknowledgement ID changed",
                    )
                require_worker_generation(worker, reservation.generation)
            await self._wait_for_upload_ready(worker, upstream_id, reservation.generation, item, remote_file.file_id)
            result = await self.get(execution_id)
            self._task_sources.extend(self._prepared_sources)
            self._prepared_sources.clear()
            transferred = True
            return result
        finally:
            if not transferred:
                if upstream_id is not None and worker.generation == reservation.generation and worker.base_url:
                    await drain_future(asyncio.create_task(self._cancel_remote(worker, upstream_id)))
                if route is not None:
                    route.state.release_reservation()
                    route.payload = {"task_id": upstream_id, "status": "failed", "files": []}
                    route.state.execution = replace(route.state.execution, status="failed", finished_at=utc_now_iso())
                    route.finalized_at = utc_now_iso()
                    route.lease.retain_until = time.time()
                else:
                    reservation.release()

    async def _cancel_remote(self, worker: WorkerState, upstream_id: str) -> None:
        """尽力取消已经创建的 worker 任务，清理失败不覆盖最初的提交错误。"""
        try:
            await _worker_request(
                self.pool, worker, "POST", f"/v1/tasks/{quote(upstream_id, safe='')}/cancel", request=self.request
            )
        except Exception:
            # worker 已不可达或被替换时，原始错误及任务保留策略仍负责后续诊断。
            return

    def _route(self, task_id: str) -> ExecutionRoute:
        """检查本调用方的公共执行映射，不能用其他调用方的任务探测 worker。"""
        route = self.registry.find("job", task_id)
        if route is None or route.owner_scope != self.owner_scope:
            raise_api_error(404, code="task_not_found", message=f"Task {task_id} not found")
        return route

    async def get(self, task_id: str) -> TaskExecution:
        """查询有类型单文件快照，产物字节继续经过现有 Files 流式代理。"""
        route = self._route(task_id)
        execution = await read_task_execution(route, self.pool, self.registry, request=self.request)
        await self.finalize(route, execution, self.request)
        return execution

    async def cancel(self, task_id: str) -> None:
        """核对唯一文件身份后取消下游任务，并沿原有边界释放执行预占。"""
        route = self.registry.find("job", task_id)
        if route is None:
            return
        if route.state.upstream_lost:
            route.state.release_reservation()
            return
        worker = self.pool.get(route.worker_id)
        response = await _worker_request(
            self.pool, worker, "POST", f"/v1/tasks/{quote(route.upstream_id, safe='')}/cancel", request=self.request
        )
        if response.status_code not in {404, 409}:
            try:
                canceled = CancelTaskResponse.model_validate(_checked_json(response))
            except ValidationError as exc:
                raise_api_error(502, code="invalid_upstream_response", message=str(exc))
            if canceled.task_id != route.upstream_id or len(canceled.files) != 1:
                raise_api_error(502, code="invalid_upstream_response", message="Cancellation file identity changed")
            (remote_file,) = canceled.files
            if remote_file.file_id != route.state.upstream_file_id or remote_file.file_name != route.state.execution.file_name:
                raise_api_error(502, code="invalid_upstream_response", message="Cancellation file identity changed")
        execution = await self.get(task_id)
        if execution.status in TERMINAL_TASK_STATUSES:
            route.state.release_reservation()


def install_task_error_handlers(app: FastAPI) -> None:
    """让共享任务输入和后台代理错误使用与 API server 相同的 V1 envelope。"""

    @app.exception_handler(ApiServerError)
    async def task_error(_request: Request, exc: ApiServerError) -> JSONResponse:
        """保留共享错误的 HTTP 状态和明确错误码。"""
        return JSONResponse({"error": exc.error.model_dump(exclude_none=True)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """把路径及 JSON 校验错误统一转换成 400。"""
        return JSONResponse(
            {"error": {"code": "invalid_request", "message": str(exc)}},
            status_code=400,
        )


_TASK_LEASE_REFRESH_TIMEOUT_SECONDS = 10.0
# 单个慢任务只占用部分 worker 预算，为同组其他任务留出续租机会。
_TASK_LEASE_REQUEST_TIMEOUT_SECONDS = 2.0


async def maintain_task_result_leases(registry: ResourceRegistry, pool: WorkerPool, *, release: bool = False) -> None:
    """按 worker 隔离续租，并把整轮网络等待限制在远小于 120 秒租约的预算内。"""
    groups: dict[str, list[ExecutionRoute]] = {}
    for route in registry.list("job", include_expired=True):
        # 父任务运行或等待外部租约时尚无本地保留期限，但下游结果同样需要持续保护。
        if route.lease.held:
            groups.setdefault(route.worker_id, []).append(route)

    async def maintain_worker(worker_id: str, routes: list[ExecutionRoute]) -> None:
        """按最近尝试时间公平轮转；单请求和整轮分别限时，慢任务不会持续占据队首。"""
        worker = pool.get(worker_id)
        routes.sort(key=lambda route: route.lease.attempted_at)
        try:
            async with asyncio.timeout(_TASK_LEASE_REFRESH_TIMEOUT_SECONDS):
                for route in routes:
                    until = route.lease.retain_until
                    ended = release or (until is not None and until <= time.time())
                    headers = dict(route.state.upstream_headers)
                    headers["x-mineru-task-lease-seconds"] = "0" if ended else "120"
                    route.lease.attempted_at = time.monotonic()
                    try:
                        async with asyncio.timeout(_TASK_LEASE_REQUEST_TIMEOUT_SECONDS):
                            response = await _worker_request(
                                pool, worker, "GET", f"/v1/tasks/{route.upstream_id}", headers=headers
                            )
                        if response.status_code < 400:
                            route.lease.refreshed_at = time.monotonic()
                    except (ApiServerError, TimeoutError):
                        continue
        except TimeoutError:
            pass
        finally:
            # 上游释放超时也必须结束已经到期的本地保留，避免路由永久占用资源。
            for route in routes:
                until = route.lease.retain_until
                if release or (until is not None and until <= time.time()):
                    route.lease.held = False
                    route.state.upstream_headers.clear()
                    result_id = route.state.result_file_id
                    result_route = next(
                        (item for item in registry.list("file", include_expired=True) if item.public_id == result_id), None
                    )
                    if result_route is not None:
                        result_route.lease.held = False

    await asyncio.gather(*(maintain_worker(worker_id, routes) for worker_id, routes in groups.items()))


__all__ = ["RouterTaskBackend", "install_task_error_handlers", "read_task_execution", "maintain_task_result_leases"]
