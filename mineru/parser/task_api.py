"""本地服务与 Router 共用的 JSON 任务、逐文件上传、执行和结果链接协议。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol
from urllib.parse import quote

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.requests import ClientDisconnect

from ..errors import MineruError
from ..filetypes import is_flash_only_parse_extension
from ..types import Tier
from ..utils.async_utils import drain_future, run_sync
from .api_server import (
    AccessLevel,
    ApiServerError,
    CreateJobRequest,
    ErrorDetail,
    FileIdSource,
    FileParseInfo,
    FileStore,
    JobFileEntry,
    JobProgress,
    JobStore,
    UrlSource,
    _JobInputMetadata,
    _JobInputSpec,
    _raise_api_error,
    _require_model_preload_ready,
    _validate_source_policy,
    submit_parse_job,
)
from .page_range import normalize_page_range_input
from .task_outputs import TaskOutputFormat

MAX_TASK_FILES = 100
TASK_UPLOAD_TTL = 3600
TERMINAL_TASK_STATUSES = frozenset({"completed", "partial", "failed", "canceled"})
TaskStatus = Literal["waiting_upload", "queued", "running", "completed", "partial", "failed", "canceled"]
TaskFileState = Literal["waiting_upload", "queued", "running", "completed", "failed", "canceled"]
logger = logging.getLogger(__name__)


class TaskManifestItem(BaseModel):
    """按客户端标识关联上传或 URL 输入，不依赖文件名唯一性。"""

    model_config = ConfigDict(extra="forbid")
    client_file_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    filename: str = Field(min_length=1, max_length=1024)
    bytes: int | None = Field(default=None, gt=0, strict=True)
    url: str | None = None
    sha256sum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    page_range: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> TaskManifestItem:
        """上传项必须声明大小；URL 只校验合法性，保留签名地址的原始字符串。"""
        if self.url is None:
            if self.bytes is None:
                raise ValueError("bytes is required when url is not supplied")
        else:
            AnyHttpUrl(self.url)
        return self


class UploadTaskOptions(BaseModel):
    """解析选项仅来自 JSON 主体，文件元数据独立保存在 manifest 中。"""

    model_config = ConfigDict(extra="forbid")
    tier: Tier | None = None
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto"
    page_range: str = ""
    output_formats: list[TaskOutputFormat] = Field(default_factory=lambda: ["default"], min_length=1)


class CreateTaskRequest(UploadTaskOptions):
    """严格 JSON 创建请求，清单是数组而不是编码后的字符串。"""

    manifest: list[TaskManifestItem] = Field(min_length=1, max_length=MAX_TASK_FILES)


class TaskUploadLink(BaseModel):
    """描述一个文件的上传目标，客户端原样使用服务端给出的地址和请求头。"""

    client_file_id: str
    upload_url: str
    upload_method: Literal["PUT"] = "PUT"
    upload_headers: dict[str, str]
    expires_at: int


class TaskFileStatus(TaskManifestItem):
    """同时提供上传准备情况与逐文件解析状态，禁止内联解析正文。"""

    upload_status: Literal["pending", "ready", "not_required", "expired"]
    file_id: str | None = None
    name: str
    status: TaskFileState
    parse: FileParseInfo | None = None
    result_url: str | None = None
    error: ErrorDetail | None = None


class TaskStatusResponse(BaseModel):
    """任务状态只携带元数据；每个成功文件提供自己的下载链接。"""

    task_id: str
    status: TaskStatus
    status_url: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    tier: Tier | None = None
    output_formats: list[TaskOutputFormat]
    progress: JobProgress
    files: list[TaskFileStatus]
    uploads: list[TaskUploadLink] = Field(default_factory=list)
    error: ErrorDetail | None = None


@dataclass(frozen=True)
class TaskUpload:
    """保存已校验的源字节位置，文件名视图独立于内容去重路径。"""

    path: Path
    filename: str
    bytes: int
    sha256sum: str
    mime_type: str = "application/octet-stream"
    file_id: str | None = None


@dataclass(frozen=True)
class TaskExecution:
    """在适配器之间传递单文件执行快照、实际输入元数据及本部署的产物标识。"""

    job: dict[str, Any]
    result_file_id: str | None = None
    error: ErrorDetail | None = None
    input_metadata: dict[int, _JobInputMetadata] = field(default_factory=dict)


class TaskBackend(Protocol):
    """上传准备复用各部署的存储，执行仍交给已有 Job 服务。"""

    owner_scope: str

    def fork(self) -> TaskBackend:
        """为每个文件创建独立的引用和执行上下文。"""
        ...

    async def release_result(self, task_id: str, finished: float) -> None:
        """父任务收尾后解除子结果租约，并开始正常保留期。"""
        ...

    async def validate(self, options: UploadTaskOptions, manifest: list[TaskManifestItem]) -> None:
        """在接单之前校验部署能力，不能预占解析名额。"""
        ...

    async def find_upload(self, sha256sum: str, filename: str) -> TaskUpload | None:
        """原子查找并固定当前调用方可以复用的源文件。"""
        ...

    async def retain_upload(self, upload: TaskUpload) -> TaskUpload:
        """持久保存已校验的单个输入，并持有跨请求准备阶段的引用。"""
        ...

    def release_uploads(self, *, rollback: bool) -> None:
        """提交后交接输入引用；未提交的取消或失败同时清理文件视图。"""
        ...

    async def submit(
        self, task_id: str, options: UploadTaskOptions, manifest: list[TaskManifestItem], uploads: dict[int, TaskUpload]
    ) -> TaskExecution:
        """使用预先分配的公开 ID 提交一次后台解析。"""
        ...

    async def get(self, task_id: str) -> TaskExecution:
        """只查询执行和产物元数据，不读取结果文件内容。"""
        ...

    async def cancel(self, task_id: str) -> None:
        """取消对应后台工作，并收敛已有资源和预占。"""
        ...


def _invalid(message: str, *, code: str = "invalid_request", status_code: int = 400, param: str | None = None) -> NoReturn:
    """把协议校验错误转换成两类服务共用的 V1 error envelope。"""
    _raise_api_error(
        status_code,
        error_type="api_error" if status_code >= 500 else "invalid_request_error",
        code=code,
        message=message,
        param=param,
    )


def resolve_task_tier(
    options: UploadTaskOptions, manifest: list[TaskManifestItem], tiers: set[Tier], default: Tier | None
) -> Tier:
    """按现有轻量格式回退规则解析 tier，不触发 worker 选择或模型执行。"""
    lightweight = [is_flash_only_parse_extension(item.filename) for item in manifest]
    if any(lightweight) and "flash" not in tiers:
        _invalid("Flash parsing is disabled, but an input requires it", param="manifest")
    tier = options.tier or default
    if tier is None and all(lightweight) and "flash" in tiers:
        tier = "flash"
    if tier is None:
        _invalid("No default quality tier is available; pass tier explicitly", code="quality_tier_unavailable", status_code=503)
    if tier not in tiers and not (all(lightweight) and "flash" in tiers):
        _invalid(f"Tier '{tier}' is not available", param="tier")
    return tier


async def file_chunks(path: Path) -> AsyncIterator[bytes]:
    """以固定大小从磁盘读取，避免阻塞事件循环和完整文件缓冲。"""
    with path.open("rb") as source:
        while chunk := await run_sync(source.read, 1024 * 1024):
            yield chunk


async def _save_upload(
    chunks: AsyncIterator[bytes],
    path: Path,
    filename: str,
    mime_type: str,
    *,
    expected: TaskManifestItem | None = None,
    canceled: Callable[[], bool] | None = None,
) -> TaskUpload:
    """流式落盘并校验完整性；声明大小用于一致性检查，不是上传大小上限。"""
    hasher = hashlib.sha256()
    size = 0
    try:
        with path.open("wb") as target:
            async for chunk in chunks:
                if canceled is not None and canceled():
                    _invalid("Task was canceled during upload", status_code=409, code="task_already_terminal")
                size += len(chunk)
                if expected is not None and expected.bytes is not None and size > expected.bytes:
                    _invalid("Declared size does not match uploaded bytes", code="upload_size_mismatch")
                await run_sync(target.write, chunk)
                hasher.update(chunk)
        if size == 0:
            _invalid("Empty files are not supported", param="files")
        if expected is not None and expected.bytes is not None and size != expected.bytes:
            _invalid("Declared size does not match uploaded bytes", code="upload_size_mismatch")
        sha = hasher.hexdigest()
        if expected is not None and expected.sha256sum is not None and sha != expected.sha256sum:
            _invalid("SHA-256 mismatch", code="file_hash_mismatch")
        return TaskUpload(path, filename, size, sha, mime_type)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


async def _read_submission(request: Request) -> tuple[UploadTaskOptions, list[TaskManifestItem]]:
    """只读取 JSON 清单，严格拒绝表单、直传字节和 URL 查询参数。"""
    if request.query_params:
        _invalid("POST parameters must be supplied in the JSON body, not the URL")
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        _invalid("Use application/json", status_code=415)
    try:
        body = CreateTaskRequest.model_validate(await request.json())
        if len({item.client_file_id for item in body.manifest}) != len(body.manifest):
            _invalid("client_file_id must be unique within a task", param="manifest")
        options = UploadTaskOptions.model_validate(body.model_dump(exclude={"manifest"}))
        options.page_range = normalize_page_range_input(options.page_range)
        options.output_formats = list(dict.fromkeys(options.output_formats))
        for item in body.manifest:
            item.page_range = normalize_page_range_input(item.page_range if item.page_range is not None else options.page_range)
        return options, body.manifest
    except ClientDisconnect:
        _invalid("Request interrupted", code="request_interrupted")
    except MineruError as exc:
        _invalid(str(exc), code=exc.code, param=exc.param)
    except (ValidationError, ValueError) as exc:
        _invalid(str(exc))


def _retention_lease(request: Request) -> int | None:
    """读取 Router 的有界结果租约，避免父任务结束前 worker 提前清理子文件。"""
    value = request.headers.get("x-mineru-task-lease-seconds")
    if value is None:
        return None
    try:
        seconds = int(value)
        if not 0 <= seconds <= TASK_UPLOAD_TTL:
            raise ValueError
    except ValueError:
        _invalid("Task result lease must be between 0 and 3600 seconds")
    return seconds


@dataclass
class _TaskFileRecord:
    """每个文件独立拥有准备引用、执行 ID 和一次性提交句柄。"""

    backend: TaskBackend
    execution_id: str = field(default_factory=JobStore._new_job_id)
    status: TaskFileState = "waiting_upload"
    upload: TaskUpload | None = None
    upload_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    upload_active: bool = False
    expired: bool = False
    execution: TaskExecution | None = None
    operation: asyncio.Task[None] | None = None
    submitted: bool = False
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    lease_refresh_at: float = 0
    error: ErrorDetail | None = None


@dataclass
class _TaskRecord:
    """父 Task 只聚合文件状态，独立子 Job 继续使用现有解析并发。"""

    id: str
    backend: TaskBackend
    options: UploadTaskOptions
    manifest: list[TaskManifestItem]
    directory: Path
    expires_at: int
    files: list[_TaskFileRecord]
    created_at: str = field(default_factory=JobStore._now)
    status: TaskStatus = "waiting_upload"
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    operation: asyncio.Task[None] | None = None
    finished_at: str | None = None
    finished_epoch: float | None = None
    retention_epoch: float | None = None
    lease_expires_at: float = 0
    error: ErrorDetail | None = None


class TaskStore:
    """管理固定上传窗口和逐文件执行，不建立第二个解析队列。"""

    def __init__(self, retention_seconds: int) -> None:
        """只在创建应用时分配暂存目录，模块导入不产生副作用。"""
        self.retention_seconds = retention_seconds
        self._temporary = tempfile.TemporaryDirectory(prefix="mineru-tasks-")
        self._root = Path(self._temporary.name)
        self._records: dict[str, _TaskRecord] = {}
        self.storage_lock = asyncio.Lock()
        self._collector: asyncio.Task[None] | None = None
        self._closing = False

    def start(self) -> None:
        """启动过期扫描，关闭正常保留期也仍然执行上传截止时间。"""
        if self._collector is None and not self._closing:
            self._collector = asyncio.create_task(self._collect_loop(), name="mineru-task-retention")

    async def _collect_loop(self) -> None:
        """定期清理过期元数据；上传与状态请求也执行同一截止检查。"""
        while True:
            await asyncio.sleep(60)
            self.collect_expired()

    def _aggregate(self, record: _TaskRecord) -> None:
        """按运行、排队、等待的优先级聚合，所有文件终结后再发布成功或部分成功。"""
        if record.status == "canceled":
            return
        previous = record.status
        states = [file.status for file in record.files]
        if "running" in states:
            record.status = "running"
        elif "queued" in states:
            record.status = "queued"
        elif "waiting_upload" in states:
            record.status = "waiting_upload"
        else:
            completed = states.count("completed")
            record.status = "completed" if completed == len(states) else ("partial" if completed else "failed")
        if record.status in TERMINAL_TASK_STATUSES and record.finished_epoch is None:
            record.finished_at = JobStore._now()
            record.finished_epoch = time.time()
        if record.status != previous:
            record.changed.set()

    def _expire_uploads(self, record: _TaskRecord) -> None:
        """只终结尚未受理上传的文件；截止前已经受理的流允许继续完成。"""
        if record.expires_at > time.time() or record.status == "canceled":
            return
        for file in record.files:
            if file.status == "waiting_upload" and not file.upload_active:
                file.expired = True
                file.status = "failed"
                file.error = ErrorDetail(type="invalid_request_error", code="upload_expired", message="Upload expired")
        self._aggregate(record)

    def collect_expired(self) -> None:
        """上传到期按文件处理，记录仅在清理结束并超过父任务保留期后移除。"""
        now = time.time()
        for task_id, record in tuple(self._records.items()):
            self._expire_uploads(record)
            if (
                self.retention_seconds
                and record.retention_epoch is not None
                and record.retention_epoch + self.retention_seconds <= now
                and record.operation is not None
                and record.operation.done()
            ):
                self._records.pop(task_id, None)
                shutil.rmtree(record.directory, ignore_errors=True)

    def lookup(self, task_id: str, owner_scope: str) -> _TaskRecord:
        """未知与越权任务使用相同错误，避免跨调用方探测。"""
        self.collect_expired()
        record = self._records.get(task_id)
        if record is None or record.backend.owner_scope != owner_scope:
            _invalid(f"Task {task_id} not found", status_code=404, code="task_not_found")
        return record

    def update_lease(self, record: _TaskRecord, seconds: int | None) -> None:
        """续租只保护结果保留期，从不延长上传窗口，也不复活已释放的结果。"""
        if seconds is None:
            return
        if seconds and record.retention_epoch is not None:
            _invalid("Task result lease has already ended", status_code=409, code="task_lease_ended")
        record.lease_expires_at = time.time() + seconds if seconds else 0
        record.changed.set()

    async def create(
        self, backend: TaskBackend, options: UploadTaskOptions, manifest: list[TaskManifestItem], *, lease_seconds: int = 0
    ) -> _TaskRecord:
        """先完整校验清单并固定缓存，再为各个就绪文件独立提交执行。"""
        if self._closing:
            _invalid("Server is shutting down", status_code=503, code="server_shutting_down")
        await backend.validate(options, manifest)
        task_id = JobStore._new_job_id()
        record = _TaskRecord(
            task_id,
            backend,
            options,
            manifest,
            self._root / task_id,
            int(time.time()) + TASK_UPLOAD_TTL,
            [_TaskFileRecord(backend.fork()) for _ in manifest],
        )
        self.update_lease(record, lease_seconds)
        try:
            for index, item in enumerate(manifest):
                file = record.files[index]
                if item.url is None and item.sha256sum:
                    upload = await file.backend.find_upload(item.sha256sum, item.filename)
                    if upload is not None:
                        if upload.bytes != item.bytes:
                            _invalid("Declared size does not match cached bytes", code="upload_size_mismatch")
                        file.upload = await file.backend.retain_upload(upload)
            self._records[task_id] = record
            self.start()
            for index, item in enumerate(manifest):
                if item.url is not None or record.files[index].upload is not None:
                    self._start_file(record, index)
            record.operation = asyncio.create_task(self._monitor(record), name=f"mineru-task-{task_id}")
            return record
        except BaseException:
            for file in record.files:
                file.backend.release_uploads(rollback=True)
            raise

    def _start_file(self, record: _TaskRecord, index: int) -> None:
        """事件循环内同步登记提交句柄，重复 PUT 不能创建第二个子作业。"""
        file = record.files[index]
        if file.operation is not None or file.status != "waiting_upload" or record.status == "canceled":
            return
        file.status = "queued"
        record.changed.set()
        file.operation = asyncio.create_task(self._dispatch_file(record, index), name=f"mineru-task-file-{file.execution_id}")
        self._aggregate(record)

    def _apply_execution(self, record: _TaskRecord, file: _TaskFileRecord, execution: TaskExecution) -> None:
        """只接受单文件执行快照，取消和已发布终态不被迟到响应覆盖。"""
        if file.status in TERMINAL_TASK_STATUSES:
            return
        status = execution.job.get("status")
        if status not in {"queued", "running", "completed", "failed", "canceled"}:
            _invalid("Invalid file status from execution backend", code="invalid_response", status_code=502)
        if status == "completed" and execution.result_file_id is None:
            _invalid("Completed file has no result", code="invalid_response", status_code=502)
        if len(execution.job.get("files") or []) != 1:
            _invalid("Expected one file from execution backend", code="invalid_response", status_code=502)
        if status != file.status:
            record.changed.set()
        file.execution = execution
        file.status = status
        file.error = execution.error
        if status in TERMINAL_TASK_STATUSES:
            file.finished.set()
            file.lease_refresh_at = time.monotonic() + 30
        self._aggregate(record)

    async def refresh(self, record: _TaskRecord) -> None:
        """并发刷新独立文件；成功文件仍续租 worker 结果直到父任务收尾。"""
        async with record.refresh_lock:
            self._expire_uploads(record)
            if record.status in TERMINAL_TASK_STATUSES:
                return
            targets = [
                file
                for file in record.files
                if file.submitted
                and file.status not in {"failed", "canceled"}
                and (file.status != "completed" or file.lease_refresh_at <= time.monotonic())
            ]
            results = await asyncio.gather(*(file.backend.get(file.execution_id) for file in targets), return_exceptions=True)
            for file, result in zip(targets, results):
                if isinstance(result, BaseException):
                    if (
                        isinstance(result, ApiServerError)
                        and result.status_code in {408, 429, 500, 502, 503, 504}
                        and result.error.code not in {"invalid_response", "invalid_upstream_response"}
                    ):
                        logger.warning("Task file refresh failed: %s: %s", file.execution_id, result.error.code)
                        continue
                    if file.status not in TERMINAL_TASK_STATUSES:
                        file.status = "failed"
                        file.error = (
                            result.error
                            if isinstance(result, ApiServerError)
                            else ErrorDetail(type="api_error", code="task_refresh_failed", message=str(result))
                        )
                        if file.operation is not None:
                            file.operation.cancel()
                    continue
                try:
                    self._apply_execution(record, file, result)
                    if file.status == "completed":
                        file.lease_refresh_at = time.monotonic() + 30
                except ApiServerError as exc:
                    file.status = "failed"
                    file.error = exc.error
                    if file.operation is not None:
                        file.operation.cancel()
            self._aggregate(record)

    async def _dispatch_file(self, record: _TaskRecord, index: int) -> None:
        """每个文件独立提交，失败或取消只清理自己的作业与源文件租约。"""
        file = record.files[index]
        try:
            execution = await file.backend.submit(
                file.execution_id, record.options, [record.manifest[index]], {0: file.upload} if file.upload is not None else {}
            )
            file.submitted = True
            self._apply_execution(record, file, execution)
            file.backend.release_uploads(rollback=False)
            if file.status not in TERMINAL_TASK_STATUSES:
                await file.finished.wait()
        except asyncio.CancelledError:
            try:
                await drain_future(asyncio.create_task(file.backend.cancel(file.execution_id)))
            except Exception:
                logger.exception("Task file cancellation failed: %s", file.execution_id)
            if file.status not in TERMINAL_TASK_STATUSES:
                file.status = "canceled"
        except Exception as exc:
            if file.status not in TERMINAL_TASK_STATUSES:
                file.status = "failed"
                file.error = (
                    exc.error
                    if isinstance(exc, ApiServerError)
                    else ErrorDetail(type="api_error", code="task_submission_failed", message=str(exc))
                )
            try:
                await file.backend.cancel(file.execution_id)
            except Exception:
                logger.exception("Task file failure cleanup failed: %s", file.execution_id)
        finally:
            file.finished.set()
            record.changed.set()
            file.backend.release_uploads(rollback=not file.submitted)
            self._aggregate(record)

    async def _monitor(self, record: _TaskRecord) -> None:
        """应用持有父任务监控；HTTP 断连不影响执行，所有子文件清理后才开始保留期。"""
        try:
            while record.status not in TERMINAL_TASK_STATUSES:
                record.changed.clear()
                await self.refresh(record)
                if record.status not in TERMINAL_TASK_STATUSES:
                    await self._wait_for_change(record)
            operations = [file.operation for file in record.files if file.operation is not None]
            if operations:
                await asyncio.gather(*operations, return_exceptions=True)
            while any(file.upload_active for file in record.files) or record.lease_expires_at > time.time():
                record.changed.clear()
                await self._wait_for_change(record)
            finished = time.time()
            for file in record.files:
                await file.backend.release_result(file.execution_id, finished)
                file.backend.release_uploads(rollback=not file.submitted)
            record.retention_epoch = finished
        except Exception:
            logger.exception("Task result cleanup failed: %s", record.id)
        finally:
            if not any(file.upload_active for file in record.files):
                shutil.rmtree(record.directory, ignore_errors=True)

    async def _wait_for_change(self, record: _TaskRecord) -> None:
        """状态变更即时唤醒；没有事件时每秒检查执行与到期边界。"""
        try:
            await asyncio.wait_for(record.changed.wait(), timeout=1)
        except asyncio.TimeoutError:
            pass

    async def upload(self, record: _TaskRecord, client_file_id: str, request: Request) -> bool:
        """在接收主体前检查固定期限，已受理流不因父任务状态变化中止。"""
        index = next((i for i, item in enumerate(record.manifest) if item.client_file_id == client_file_id), None)
        if index is None:
            _invalid("Unknown client_file_id", status_code=404, code="upload_not_found")
        expected = record.manifest[index]
        if expected.url is not None:
            _invalid("URL inputs do not accept uploads")
        file = record.files[index]
        async with file.upload_lock:
            if record.expires_at <= time.time():
                self._expire_uploads(record)
                _invalid("Upload link expired", status_code=410, code="upload_expired")
            if record.status == "canceled":
                _invalid("Task was canceled", status_code=409, code="task_already_terminal")
            if file.upload is not None:
                return True
            if file.status != "waiting_upload":
                _invalid("File no longer accepts uploads", status_code=409, code="file_already_terminal")
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError:
                    _invalid("Invalid Content-Length", code="upload_size_mismatch")
                if declared != expected.bytes:
                    _invalid("Content-Length does not match manifest bytes", code="upload_size_mismatch")
            file.upload_active = True
            record.directory.mkdir(parents=True, exist_ok=True)
            path = record.directory / secrets.token_hex(16)

            def was_canceled() -> bool:
                """只在本文件被取消时中止已受理上传，不把上传截止时间当作传输超时。"""
                return record.status == "canceled" or file.status == "canceled"

            try:
                upload = await _save_upload(
                    request.stream(),
                    path,
                    expected.filename,
                    request.headers.get("content-type", "application/octet-stream"),
                    expected=expected,
                    canceled=was_canceled,
                )
                if was_canceled():
                    _invalid("Task was canceled during upload", status_code=409, code="task_already_terminal")
                retained = await file.backend.retain_upload(upload)
                if was_canceled():
                    file.backend.release_uploads(rollback=True)
                    _invalid("Task was canceled during upload", status_code=409, code="task_already_terminal")
                file.upload = retained
                self._start_file(record, index)
                return False
            except ClientDisconnect:
                _invalid("Upload interrupted; retry this file on the same task", code="upload_interrupted")
            finally:
                path.unlink(missing_ok=True)
                file.upload_active = False
                record.changed.set()
                self._expire_uploads(record)

    async def cancel(self, record: _TaskRecord, *, wait: bool = True) -> None:
        """保留已完成结果，只取消未完成文件并等待真实子作业清理。"""
        if record.status in TERMINAL_TASK_STATUSES:
            _invalid(f"Task is {record.status}", status_code=409, code="task_already_terminal")
        record.status = "canceled"
        record.finished_at = JobStore._now()
        record.finished_epoch = time.time()
        record.changed.set()
        for file in record.files:
            if file.status not in TERMINAL_TASK_STATUSES:
                file.status = "canceled"
                if file.operation is not None:
                    file.operation.cancel()
        if wait and record.operation is not None:
            record.lease_expires_at = 0
            record.changed.set()
            await drain_future(record.operation)

    async def shutdown(self) -> None:
        """停止接单并排空全部文件执行和结果租约，最后释放临时目录。"""
        self._closing = True
        if self._collector is not None:
            self._collector.cancel()
            await asyncio.gather(self._collector, return_exceptions=True)
        for record in tuple(self._records.values()):
            record.lease_expires_at = 0
            if record.status not in TERMINAL_TASK_STATUSES:
                await self.cancel(record, wait=False)
        await asyncio.gather(
            *(record.operation for record in self._records.values() if record.operation is not None), return_exceptions=True
        )
        self._temporary.cleanup()


def _url(request: Request, path: str) -> str:
    """生成包含反向代理 root_path 的本部署地址，客户端无需自行拼接路径。"""
    return str(request.base_url).rstrip("/") + path


def task_payload(record: _TaskRecord, request: Request) -> dict[str, Any]:
    """逐文件合并上传与执行元数据，成功文件立即给出独立下载链接。"""
    files: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    starts: list[str] = []
    for item, state in zip(record.manifest, record.files):
        file = item.model_dump()
        upload = state.upload
        file.update(
            name=item.filename,
            upload_status="not_required"
            if item.url is not None
            else ("ready" if upload is not None else ("expired" if state.expired else "pending")),
            status=state.status,
            file_id=upload.file_id if upload else None,
            result_url=None,
        )
        if state.execution is not None:
            job = state.execution.job
            results = job.get("files") or []
            if results:
                file.update({key: results[0].get(key) for key in ("name", "file_id", "page_range", "parse", "error")})
            if job.get("started_at"):
                starts.append(job["started_at"])
            if (metadata := state.execution.input_metadata.get(0)) is not None:
                file.update(bytes=metadata.bytes, sha256sum=metadata.sha256sum)
            if state.status == "completed" and state.execution.result_file_id:
                file["result_url"] = _url(request, f"/v1/files/{quote(state.execution.result_file_id, safe='')}/content")
        if state.error is not None:
            file["error"] = state.error.model_dump()
        files.append(file)
        if (
            item.url is None
            and state.status == "waiting_upload"
            and record.status != "canceled"
            and record.expires_at > time.time()
        ):
            links.append(
                {
                    "client_file_id": item.client_file_id,
                    "upload_url": _url(request, f"/v1/tasks/{record.id}/uploads/{quote(item.client_file_id, safe='')}"),
                    "upload_method": "PUT",
                    "upload_headers": {"Content-Type": "application/octet-stream"},
                    "expires_at": record.expires_at,
                }
            )
    payload = {
        "task_id": record.id,
        "status": record.status,
        "status_url": _url(request, f"/v1/tasks/{record.id}"),
        "created_at": record.created_at,
        "started_at": min(starts) if starts else None,
        "finished_at": record.finished_at,
        "tier": record.options.tier,
        "output_formats": record.options.output_formats,
        "progress": {
            "completed": sum(file.status == "completed" for file in record.files),
            "failed": sum(file.status == "failed" for file in record.files),
            "total": len(files),
        },
        "files": files,
        "uploads": links,
        "error": record.error.model_dump() if record.error else None,
    }
    return TaskStatusResponse.model_validate(payload).model_dump()


def _request_schema() -> dict[str, Any]:
    """OpenAPI 使用与运行时相同的 JSON 清单和格式约束。"""
    schema = CreateTaskRequest.model_json_schema()
    definitions = schema.pop("$defs", {})

    def expand(value: Any) -> Any:
        """内联局部模型引用，避免 OpenAPI 把 JSON Schema 的 $defs 当成全局路径。"""
        if isinstance(value, dict):
            if "$ref" in value:
                return expand(definitions[value["$ref"].rsplit("/", 1)[-1]])
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return {"requestBody": {"required": True, "content": {"application/json": {"schema": expand(schema)}}}}


def build_task_router(factory: Callable[[Request], TaskBackend]) -> APIRouter:
    """两类部署使用同一任务协议，下载交给已有 Files 入口。"""
    router = APIRouter(prefix="/v1", tags=["Tasks"])

    @router.post("/tasks", status_code=202, response_model=TaskStatusResponse, openapi_extra=_request_schema())
    async def create_task(request: Request) -> Response:
        """先验证 JSON 再建立稳定任务 ID，HTTP 结束后逐文件工作独立继续。"""
        backend = factory(request)
        store: TaskStore = request.app.state.task_store
        try:
            options, manifest = await _read_submission(request)
            record = await store.create(backend, options, manifest, lease_seconds=_retention_lease(request) or 0)
        except BaseException:
            backend.release_uploads(rollback=True)
            raise
        return JSONResponse(task_payload(record, request), status_code=202)

    @router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task(task_id: str, request: Request) -> Response:
        """读取逐文件状态与结果地址，上传链接截止时间保持不变。"""
        store: TaskStore = request.app.state.task_store
        record = store.lookup(task_id, factory(request).owner_scope)
        store.update_lease(record, _retention_lease(request))
        await store.refresh(record)
        return JSONResponse(task_payload(record, request))

    @router.put(
        "/tasks/{task_id}/uploads/{client_file_id}",
        response_model=TaskStatusResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    async def upload_task_file(task_id: str, client_file_id: str, request: Request) -> Response:
        """每条上传链接只接收一个文件，该文件就绪后立即提交执行。"""
        store: TaskStore = request.app.state.task_store
        record = store.lookup(task_id, factory(request).owner_scope)
        lease_seconds = _retention_lease(request)
        store.update_lease(record, lease_seconds)
        skipped = await store.upload(record, client_file_id, request)
        # 大文件转传可超过租约长度；受理完成时续租，保护紧接着发布的结果。
        store.update_lease(record, lease_seconds)
        return JSONResponse(
            task_payload(record, request),
            headers={"Connection": "close"} if skipped and request.scope.get("http_version") == "1.1" else None,
        )

    @router.delete("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def cancel_task(task_id: str, request: Request) -> Response:
        """统一取消等待上传与正在执行的任务，不重建后台作业。"""
        store: TaskStore = request.app.state.task_store
        record = store.lookup(task_id, factory(request).owner_scope)
        await store.refresh(record)
        await store.cancel(record, wait=False)
        return JSONResponse(task_payload(record, request))

    return router


class ApiTaskBackend:
    """把上传准备接入 API server 的唯一文件存储和 JobStore。"""

    def __init__(self, request: Request) -> None:
        """保留轻量请求上下文，不跨请求持有上传主体。"""
        self.request = Request(dict(request.scope))
        self.files: FileStore = request.app.state.file_store
        self.jobs: JobStore = request.app.state.job_store
        self.owner_scope = "local"
        self.access: AccessLevel = "registered" if request.app.state.api_key else "anonymous"
        self._pins: list[str] = []
        self._owned_files: list[str] = []

    def fork(self) -> ApiTaskBackend:
        """为单个文件隔离源字节引用，其他文件提交不释放本文件资源。"""
        return ApiTaskBackend(self.request)

    async def release_result(self, task_id: str, finished: float) -> None:
        """由父 Task 统一开始子作业与结果文件的保留期。"""
        self.jobs.release_task_result(task_id, finished)

    async def validate(self, options: UploadTaskOptions, manifest: list[TaskManifestItem]) -> None:
        """复用引擎就绪状态，提前校验显式 tier 与轻量格式能力。"""
        _require_model_preload_ready(self.request)
        if self.jobs._closing:
            _invalid("Server is shutting down", code="server_shutting_down", status_code=503)
        state = self.request.app.state
        for index, item in enumerate(manifest):
            if item.url is not None:
                try:
                    _validate_source_policy(
                        UrlSource(url=item.url),
                        allow_local_source=False,
                        max_inline_bytes=state.max_inline_bytes,
                        allow_http_source=state.allow_http_source,
                    )
                except ValueError as exc:
                    _invalid(str(exc), param=f"manifest.{index}.url")
        options.tier = resolve_task_tier(options, manifest, set(state.tier_runtime_options), state.default_tier)

    async def find_upload(self, sha256sum: str, filename: str) -> TaskUpload | None:
        """原子查找并取得缓存租约，准备阶段不会因原文件视图到期丢失字节。"""
        path = self.files.pin_cached_blob(sha256sum)
        if path is None:
            return None
        self._pins.append(sha256sum)
        return TaskUpload(path, filename, path.stat().st_size, sha256sum)

    async def retain_upload(self, upload: TaskUpload) -> TaskUpload:
        """立即按内容去重并登记独立文件视图，后续文件失败不丢弃已成功上传项。"""
        file_id = self.files._new_file_id()
        self._owned_files.append(file_id)
        await run_sync(
            self.files.register_source_file, upload.filename, upload.path, upload.sha256sum, file_id, keep_alive=True
        )
        path = self.files.pin_cached_blob(upload.sha256sum)
        if path is None:
            raise RuntimeError("New source file disappeared")
        self._pins.append(upload.sha256sum)
        return replace(upload, path=path, file_id=file_id)

    def release_uploads(self, *, rollback: bool) -> None:
        """成功提交后交接输入；准备失败时删除本任务视图但保留其他任务引用。"""
        if rollback:
            for file_id in self._owned_files:
                if file_id in self.files._files:
                    self.files.delete_file(file_id)
        else:
            self.files.finalize_source_views(self._owned_files)
        self._owned_files.clear()
        for sha in self._pins:
            self.files.unpin_cached_blob(sha)
        self._pins.clear()

    async def submit(
        self, task_id: str, options: UploadTaskOptions, manifest: list[TaskManifestItem], uploads: dict[int, TaskUpload]
    ) -> TaskExecution:
        """提交单文件子作业，在同一次解析中生成专用产物并持有结果租约。"""
        # 接单时已持有源字节；用户删除文件视图也不能破坏尚在等待其他文件的任务。
        for upload in uploads.values():
            await run_sync(
                self.files.register_source_file, upload.filename, upload.path, upload.sha256sum, upload.file_id, keep_alive=True
            )
        body = CreateJobRequest(
            files=[
                JobFileEntry(
                    source=UrlSource(url=item.url) if item.url is not None else FileIdSource(file_id=uploads[index].file_id),
                    page_range=item.page_range or None,
                )
                for index, item in enumerate(manifest)
            ],
            tier=options.tier,
            ocr_mode=options.ocr_mode,
            output_formats=[],
        )

        await submit_parse_job(
            body,
            self.request,
            self.files,
            self.jobs,
            self.access,
            job_id=task_id,
            task_output_formats=options.output_formats,
            input_specs={
                index: _JobInputSpec(item.filename, item.bytes, item.sha256sum) for index, item in enumerate(manifest)
            },
        )
        return await self.get(task_id)

    async def get(self, task_id: str) -> TaskExecution:
        """只读取 Job 与最终 File 的标识，不在状态请求中访问产物正文。"""
        record = self.jobs.get(task_id)
        return TaskExecution(
            self.jobs.build_response(record, access_level=self.access).model_dump(by_alias=True),
            record.task_result_file_id,
            None,
            dict(record.input_metadata),
        )

    async def cancel(self, task_id: str) -> None:
        """沿用 JobStore 取消及线程清理边界，终态竞争不影响已经完成的任务。"""
        record = self.jobs._jobs.get(task_id)
        if record is None:
            return
        if record.status not in TERMINAL_TASK_STATUSES:
            self.jobs.cancel(task_id)
        await self.jobs.wait_for_terminal(task_id, 3600)


__all__ = [
    "ApiTaskBackend",
    "CreateTaskRequest",
    "TaskBackend",
    "TaskExecution",
    "TaskManifestItem",
    "TaskStatusResponse",
    "TaskStore",
    "TaskUpload",
    "UploadTaskOptions",
    "build_task_router",
    "file_chunks",
    "resolve_task_tier",
    "task_payload",
]
