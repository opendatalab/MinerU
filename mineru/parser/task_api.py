"""本地服务与 Router 共用的表单任务、逐文件上传和结果链接协议。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import secrets
import shutil
import tempfile
import time
import zipfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol
from urllib.parse import quote

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
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
    OutputFiles,
    UrlSource,
    _JobInputMetadata,
    _JobInputSpec,
    _JobRecord,
    _raise_api_error,
    _require_model_preload_ready,
    _validate_source_policy,
    submit_parse_job,
)
from .page_range import normalize_page_range_input

MAX_TASK_FILES = 100
TASK_UPLOAD_TTL = 3600
TERMINAL_TASK_STATUSES = frozenset({"completed", "partial", "failed", "canceled"})
TaskStatus = Literal["waiting_upload", "queued", "running", "completed", "partial", "failed", "canceled"]
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
    """解析选项仅来自表单，文件元数据独立保存在 manifest 中。"""

    model_config = ConfigDict(extra="forbid")
    tier: Tier | None = None
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto"
    page_range: str = ""
    output_formats: list[Literal["markdown", "middle_json", "structured_content", "zip"]] = Field(
        default_factory=lambda: ["markdown"], min_length=1
    )


class TaskUploadLink(BaseModel):
    """描述一个文件的上传目标，客户端原样使用服务端给出的地址和请求头。"""

    client_file_id: str
    upload_url: str
    upload_method: Literal["PUT"] = "PUT"
    upload_headers: dict[str, str]
    expires_at: int


class TaskFileStatus(TaskManifestItem):
    """同时提供上传准备情况与逐文件解析状态，禁止内联解析正文。"""

    upload_status: Literal["pending", "ready", "not_required"]
    file_id: str | None = None
    name: str
    status: Literal["waiting_upload", "queued", "running", "completed", "failed", "canceled"]
    parse: FileParseInfo | None = None
    output_files: OutputFiles | None = None
    error: ErrorDetail | None = None


class TaskStatusResponse(BaseModel):
    """任务状态只携带元数据；最终 ZIP 由独立下载链接提供。"""

    task_id: str
    status: TaskStatus
    status_url: str
    result_url: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    tier: Tier | None = None
    output_formats: list[str]
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
    """在后台适配器之间传递执行快照、实际输入元数据及本部署的 ZIP File 标识。"""

    job: dict[str, Any]
    result_file_id: str | None = None
    error: ErrorDetail | None = None
    input_metadata: dict[int, _JobInputMetadata] = field(default_factory=dict)


class TaskBackend(Protocol):
    """上传准备复用各部署的存储，执行仍交给已有 Job 服务。"""

    owner_scope: str

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
        """只查询执行和产物元数据，不读取 ZIP 内容。"""
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


async def _upload_chunks(upload: UploadFile) -> AsyncIterator[bytes]:
    """分块读取已经由表单解析器暂存的单个直传文件。"""
    while chunk := await upload.read(1024 * 1024):
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


async def _read_submission(
    request: Request, directory: Path
) -> tuple[UploadTaskOptions, list[TaskManifestItem], TaskUpload | None]:
    """解析含上传或 URL 来源的表单清单；拒绝旧 JSON、原始字节及查询参数。"""
    if request.query_params:
        _invalid("POST parameters must be supplied in the form body, not the URL")
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type not in {"application/x-www-form-urlencoded", "multipart/form-data"}:
        _invalid("Use application/x-www-form-urlencoded or multipart/form-data", status_code=415)
    try:
        async with request.form(max_files=1, max_fields=32) as form:
            for key in form:
                if key not in {"output_formats", "files"} and len(form.getlist(key)) != 1:
                    _invalid(f"Duplicate form field: {key}", param=key)
            params = {key: form[key] for key in form if key not in {"manifest", "files", "output_formats"}}
            if "output_formats" in form:
                params["output_formats"] = form.getlist("output_formats")
            options = UploadTaskOptions.model_validate(params)
            options.page_range = normalize_page_range_input(options.page_range)
            options.output_formats = list(dict.fromkeys([*options.output_formats, "zip"]))
            manifest: list[TaskManifestItem] = []
            if "manifest" in form:
                raw = form["manifest"]
                if not isinstance(raw, str):
                    _invalid("manifest must be a JSON text field", param="manifest")
                values = json.loads(raw)
                if not isinstance(values, list) or not 1 <= len(values) <= MAX_TASK_FILES:
                    _invalid("manifest must contain between 1 and 100 files", param="manifest")
                manifest = [TaskManifestItem.model_validate(item) for item in values]
                if len({item.client_file_id for item in manifest}) != len(manifest):
                    _invalid("client_file_id must be unique within a task", param="manifest")
            files = form.getlist("files")
            direct: TaskUpload | None = None
            if files:
                if len(files) != 1 or (manifest and len(manifest) != 1):
                    _invalid("Multipart direct upload supports one file; submit a manifest for multiple files", param="files")
                if manifest and manifest[0].url is not None:
                    _invalid("url and multipart file bytes cannot be supplied together", param="files")
                file = files[0]
                if not isinstance(file, UploadFile) or not file.filename:
                    _invalid("files must contain one named file", param="files")
                filename = manifest[0].filename if manifest else file.filename
                direct = await _save_upload(
                    _upload_chunks(file),
                    directory / "direct",
                    filename,
                    file.content_type or "application/octet-stream",
                    expected=manifest[0] if manifest else None,
                )
                if not manifest:
                    manifest = [TaskManifestItem(client_file_id="f1", filename=filename, bytes=direct.bytes)]
            if not manifest:
                _invalid("Supply manifest or one files field", param="manifest")
            for item in manifest:
                item.page_range = normalize_page_range_input(
                    item.page_range if item.page_range is not None else options.page_range
                )
            return options, manifest, direct
    except StarletteHTTPException as exc:
        _invalid(str(exc.detail), status_code=exc.status_code)
    except ClientDisconnect:
        _invalid("Upload interrupted; retry the request", code="upload_interrupted")
    except MineruError as exc:
        _invalid(str(exc), code=exc.code, param=exc.param)
    except (ValidationError, ValueError) as exc:
        _invalid(str(exc))


@dataclass
class _TaskRecord:
    """只保存上传准备与执行映射；解析并发仍由既有 JobStore 管理。"""

    id: str
    backend: TaskBackend
    options: UploadTaskOptions
    manifest: list[TaskManifestItem]
    directory: Path
    expires_at: int
    created_at: str = field(default_factory=JobStore._now)
    status: TaskStatus = "waiting_upload"
    uploads: dict[int, TaskUpload] = field(default_factory=dict)
    upload_locks: dict[int, asyncio.Lock] = field(default_factory=dict)
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    execution: TaskExecution | None = None
    operation: asyncio.Task[None] | None = None
    submitted: bool = False
    active_uploads: int = 0
    finished_at: str | None = None
    finished_epoch: float | None = None
    error: ErrorDetail | None = None


class TaskStore:
    """管理跨请求上传租约、稳定 ID 和提交一次语义，不建立第二个解析队列。"""

    def __init__(self, retention_seconds: int) -> None:
        """在应用创建时准备私有暂存目录，不在模块导入阶段产生副作用。"""
        self.retention_seconds = retention_seconds
        self._temporary = tempfile.TemporaryDirectory(prefix="mineru-tasks-")
        self._root = Path(self._temporary.name)
        self._records: dict[str, _TaskRecord] = {}
        self.storage_lock = asyncio.Lock()
        self._collector: asyncio.Task[None] | None = None
        self._closing = False

    def start(self) -> None:
        """启动独立的准备记录回收，关闭产物保留策略时也能清理过期上传。"""
        if self._collector is None and not self._closing:
            self._collector = asyncio.create_task(self._collect_loop(), name="mineru-task-retention")

    async def _collect_loop(self) -> None:
        """定期回收等待上传的孤立任务和超过保留期的终态快照。"""
        while True:
            await asyncio.sleep(60)
            self.collect_expired()

    def _release(self, record: _TaskRecord, *, rollback: bool) -> None:
        """没有在途写入时释放准备引用，避免取消与落盘并发导致路径提前消失。"""
        if record.active_uploads:
            return
        record.backend.release_uploads(rollback=rollback)
        shutil.rmtree(record.directory, ignore_errors=True)

    def _finish(self, record: _TaskRecord, status: TaskStatus, error: ErrorDetail | None = None) -> None:
        """统一记录终态时间，错误、过期和取消采用同一保留边界。"""
        record.status = status
        record.error = error
        record.finished_at = JobStore._now()
        record.finished_epoch = time.time()

    def collect_expired(self) -> None:
        """等待上传到期时释放缓存租约；活动写入与后台执行不被中途回收。"""
        now = time.time()
        for task_id, record in tuple(self._records.items()):
            if record.status == "waiting_upload" and record.expires_at <= now and not record.active_uploads:
                self._finish(
                    record, "failed", ErrorDetail(type="invalid_request_error", code="upload_expired", message="Upload expired")
                )
                self._release(record, rollback=True)
            if (
                self.retention_seconds
                and record.finished_epoch is not None
                and record.finished_epoch + self.retention_seconds <= now
                and not record.active_uploads
                and (record.operation is None or record.operation.done())
            ):
                self._release(record, rollback=not record.submitted)
                self._records.pop(task_id, None)

    def lookup(self, task_id: str, owner_scope: str) -> _TaskRecord:
        """只有当前调用方可以读写任务，未知 ID 与越权 ID 使用同一错误。"""
        self.collect_expired()
        record = self._records.get(task_id)
        if record is None or record.backend.owner_scope != owner_scope:
            _invalid(f"Task {task_id} not found", status_code=404, code="task_not_found")
        return record

    async def create(
        self, backend: TaskBackend, options: UploadTaskOptions, manifest: list[TaskManifestItem], direct: TaskUpload | None
    ) -> _TaskRecord:
        """校验并固定已命中源文件；客户端上传项齐备后提交后台执行。"""
        if self._closing:
            _invalid("Server is shutting down", status_code=503, code="server_shutting_down")
        await backend.validate(options, manifest)
        task_id = JobStore._new_job_id()
        record = _TaskRecord(task_id, backend, options, manifest, self._root / task_id, int(time.time()) + TASK_UPLOAD_TTL)
        try:
            for index, item in enumerate(manifest):
                if item.url is not None:
                    continue
                upload = (
                    direct
                    if direct is not None
                    else (await backend.find_upload(item.sha256sum, item.filename) if item.sha256sum else None)
                )
                if upload is not None:
                    if upload.bytes != item.bytes:
                        _invalid("Declared size does not match cached bytes", code="upload_size_mismatch")
                    record.uploads[index] = await backend.retain_upload(upload)
            self._records[task_id] = record
            self.start()
            self._maybe_start(record)
            return record
        except BaseException:
            backend.release_uploads(rollback=True)
            raise

    def _maybe_start(self, record: _TaskRecord) -> None:
        """事件循环内原子切换准备状态，最后多个上传并发完成也只创建一个执行句柄。"""
        if record.status == "waiting_upload" and all(
            item.url is not None or index in record.uploads for index, item in enumerate(record.manifest)
        ):
            record.status = "queued"
            record.operation = asyncio.create_task(self._dispatch(record), name=f"mineru-task-{record.id}")

            def operation_finished(operation: asyncio.Task[None]) -> None:
                """覆盖首次调度前取消，且让取消后的保留期从准备引用清理完成时开始。"""
                if record.status == "canceled":
                    self._finish(record, "canceled")
                self._release(record, rollback=not record.submitted)
                if not operation.cancelled() and operation.exception() is not None:
                    error = operation.exception()
                    logger.error("Task cleanup failed: %s", record.id, exc_info=(type(error), error, error.__traceback__))

            record.operation.add_done_callback(operation_finished)

    def _apply_execution(self, record: _TaskRecord, execution: TaskExecution) -> None:
        """更新已发布的执行快照，不能用迟到响应覆盖取消或失败终态。"""
        if record.status in TERMINAL_TASK_STATUSES:
            return
        status = execution.job.get("status")
        if status not in {"queued", "running", *TERMINAL_TASK_STATUSES}:
            _invalid("Invalid task status from execution backend", code="invalid_response", status_code=502)
        if status in {"completed", "partial"} and execution.result_file_id is None:
            _invalid("Completed task has no ZIP file", code="invalid_response", status_code=502)
        record.execution = execution
        record.status = status
        if status in TERMINAL_TASK_STATUSES:
            self._finish(record, status, execution.error)
            record.finished_at = execution.job.get("finished_at") or record.finished_at
            if record.finished_at is not None:
                record.finished_epoch = datetime.fromisoformat(record.finished_at.replace("Z", "+00:00")).timestamp()

    async def refresh(self, record: _TaskRecord) -> None:
        """状态刷新只查询元数据；串行化并发查询，防止旧快照覆盖新状态。"""
        async with record.refresh_lock:
            if record.submitted and record.status not in TERMINAL_TASK_STATUSES:
                self._apply_execution(record, await record.backend.get(record.id))

    async def _dispatch(self, record: _TaskRecord) -> None:
        """提交与监控由应用持有，HTTP 断连不取消已经接受的任务。"""
        try:
            execution = await record.backend.submit(record.id, record.options, record.manifest, record.uploads)
            record.submitted = True
            self._apply_execution(record, execution)
            self._release(record, rollback=False)
            while record.status not in TERMINAL_TASK_STATUSES:
                await asyncio.sleep(1)
                try:
                    await self.refresh(record)
                except ApiServerError as exc:
                    if exc.status_code not in {408, 429, 500, 502, 503, 504} or exc.error.code in {
                        "invalid_response",
                        "invalid_upstream_response",
                    }:
                        raise
                    logger.warning("Task status refresh failed: %s: %s", record.id, exc.error.code)
        except asyncio.CancelledError:
            if record.submitted and record.status not in {"completed", "partial", "failed"}:
                try:
                    await drain_future(asyncio.create_task(record.backend.cancel(record.id)))
                except Exception:
                    logger.exception("Task cancellation cleanup failed: %s", record.id)
            if record.status not in TERMINAL_TASK_STATUSES:
                self._finish(record, "canceled")
        except Exception as exc:
            error = (
                exc.error
                if isinstance(exc, ApiServerError)
                else ErrorDetail(type="api_error", code="task_submission_failed", message=str(exc))
            )
            if record.status not in TERMINAL_TASK_STATUSES:
                self._finish(record, "failed", error)
            if record.submitted:
                try:
                    await drain_future(asyncio.create_task(record.backend.cancel(record.id)))
                except Exception:
                    logger.exception("Failed task cleanup could not reach execution backend: %s", record.id)
            logger.exception("Task execution failed: %s", record.id)
        finally:
            self._release(record, rollback=not record.submitted)

    async def upload(self, record: _TaskRecord, client_file_id: str, request: Request) -> bool:
        """独立接收一个文件；返回是否跳过主体，供 HTTP/1.1 正确关闭连接。"""
        index = next((index for index, item in enumerate(record.manifest) if item.client_file_id == client_file_id), None)
        if index is None:
            _invalid("Unknown client_file_id", status_code=404, code="upload_not_found")
        if record.manifest[index].url is not None:
            _invalid("URL inputs do not accept uploads", status_code=400, code="invalid_request")
        lock = record.upload_locks.setdefault(index, asyncio.Lock())
        async with lock:
            if record.status in {"failed", "canceled"}:
                _invalid("Task no longer accepts uploads", status_code=409, code="task_already_terminal")
            if index in record.uploads:
                return True
            if record.status != "waiting_upload" or record.expires_at <= time.time():
                _invalid("Task no longer accepts uploads", status_code=409, code="task_already_terminal")
            expected = record.manifest[index]
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError:
                    _invalid("Invalid Content-Length", code="upload_size_mismatch")
                if declared != expected.bytes:
                    _invalid("Content-Length does not match manifest bytes", code="upload_size_mismatch")
            record.active_uploads += 1
            record.directory.mkdir(parents=True, exist_ok=True)
            path = record.directory / secrets.token_hex(16)

            def was_canceled() -> bool:
                """每次接收数据时检查取消状态，避免继续写入已取消任务的剩余主体。"""
                return record.status != "waiting_upload"

            try:
                upload = await _save_upload(
                    request.stream(),
                    path,
                    expected.filename,
                    request.headers.get("content-type", "application/octet-stream"),
                    expected=expected,
                    canceled=was_canceled,
                )
                if record.status != "waiting_upload":
                    _invalid("Task was canceled during upload", status_code=409, code="task_already_terminal")
                retained = await record.backend.retain_upload(upload)
                if record.status != "waiting_upload":
                    _invalid("Task was canceled during upload", status_code=409, code="task_already_terminal")
                record.uploads[index] = retained
                self._maybe_start(record)
                return False
            except ClientDisconnect:
                _invalid("Upload interrupted; retry this file on the same task", code="upload_interrupted")
            finally:
                path.unlink(missing_ok=True)
                record.active_uploads -= 1
                if record.status in TERMINAL_TASK_STATUSES:
                    self._release(record, rollback=not record.submitted)
                    if record.status == "canceled" and not record.active_uploads:
                        self._finish(record, "canceled")

    async def cancel(self, record: _TaskRecord, *, wait: bool = True) -> None:
        """取消准备或执行阶段，后台清理结束之前不删除仍在使用的源文件。"""
        if record.status in TERMINAL_TASK_STATUSES:
            _invalid(f"Task is {record.status}", status_code=409, code="task_already_terminal")
        self._finish(record, "canceled")
        if record.operation is not None and not record.operation.done():
            record.operation.cancel()
            if not wait:
                return
            await drain_future(asyncio.gather(record.operation, return_exceptions=True))
        self._finish(record, "canceled")
        self._release(record, rollback=not record.submitted)

    async def shutdown(self) -> None:
        """关闭时先停止提交和监控，再释放所有准备阶段引用与暂存目录。"""
        self._closing = True
        if self._collector is not None:
            self._collector.cancel()
            await asyncio.gather(self._collector, return_exceptions=True)
        for record in tuple(self._records.values()):
            if record.status not in TERMINAL_TASK_STATUSES:
                await self.cancel(record)
            elif record.operation is not None and not record.operation.done():
                record.operation.cancel()
                await drain_future(asyncio.gather(record.operation, return_exceptions=True))
            self._release(record, rollback=not record.submitted)
        self._temporary.cleanup()


def _url(request: Request, path: str) -> str:
    """生成包含反向代理 root_path 的本部署地址，客户端无需自行拼接路径。"""
    return str(request.base_url).rstrip("/") + path


def task_payload(record: _TaskRecord, request: Request) -> dict[str, Any]:
    """将任务准备记录与执行元数据合并，不读取任何解析产物字节。"""
    job = record.execution.job if record.execution is not None else {}
    results = job.get("files") or []
    files: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    for index, item in enumerate(record.manifest):
        upload = record.uploads.get(index)
        file = item.model_dump()
        file.update(
            name=item.filename,
            upload_status="not_required" if item.url is not None else ("ready" if upload is not None else "pending"),
            status="queued" if upload is not None or item.url is not None else "waiting_upload",
            file_id=upload.file_id if upload else None,
        )
        if index < len(results):
            file.update(results[index])
        if record.execution is not None and (metadata := record.execution.input_metadata.get(index)) is not None:
            file.update(bytes=metadata.bytes, sha256sum=metadata.sha256sum)
        if record.status == "canceled" and file["status"] not in {"completed", "failed"}:
            file["status"] = "canceled"
        files.append(file)
        if item.url is None and upload is None and record.status == "waiting_upload":
            links.append(
                {
                    "client_file_id": item.client_file_id,
                    "upload_url": _url(request, f"/v1/tasks/{record.id}/uploads/{quote(item.client_file_id, safe='')}"),
                    "upload_method": "PUT",
                    "upload_headers": {"Content-Type": "application/octet-stream"},
                    "expires_at": record.expires_at,
                }
            )
    result_id = record.execution.result_file_id if record.execution is not None else None
    payload = {
        "task_id": record.id,
        "status": record.status,
        "status_url": _url(request, f"/v1/tasks/{record.id}"),
        "result_url": _url(request, f"/v1/files/{quote(result_id, safe='')}/content")
        if result_id and record.status in {"completed", "partial"}
        else None,
        "created_at": record.created_at,
        "started_at": job.get("started_at"),
        "finished_at": record.finished_at,
        "tier": job.get("tier") or record.options.tier,
        "output_formats": record.options.output_formats,
        "progress": job.get("progress") or {"completed": 0, "failed": 0, "total": len(files)},
        "files": files,
        "uploads": links,
        "error": record.error.model_dump() if record.error is not None else None,
    }
    return TaskStatusResponse.model_validate(payload).model_dump()


def _request_schema() -> dict[str, Any]:
    """明确记录表单字段和单文件约束，不再暴露查询参数或旧内容类型。"""
    properties = UploadTaskOptions.model_json_schema()["properties"]
    # 清单使用 JSON 文本表单字段，客户端无需发送 application/json 请求。
    properties["manifest"] = {
        "type": "string",
        "description": (
            "JSON array of 1-100 files: client_file_id, filename, optional url, sha256sum and page_range. "
            "bytes must be a positive integer for uploads; it is optional/null for URL inputs. "
            "URL inputs have upload_status=not_required and never receive upload links."
        ),
        "example": '[{"client_file_id":"f1","filename":"document.pdf","bytes":12345}]',
    }
    plain = {"type": "object", "required": ["manifest"], "additionalProperties": False, "properties": properties}
    multipart = copy.deepcopy(plain)
    multipart.pop("required")
    multipart["anyOf"] = [{"required": ["manifest"]}, {"required": ["files"]}]
    multipart["properties"]["files"] = {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": {"type": "string", "format": "binary"},
        "description": ("Local single-file direct upload only; a supplied manifest must contain exactly one file without url."),
    }
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/x-www-form-urlencoded": {"schema": plain},
                "multipart/form-data": {"schema": multipart},
            },
        }
    }


def build_task_router(factory: Callable[[Request], TaskBackend]) -> APIRouter:
    """两类部署使用同一任务协议，下载交给已有 Files 入口。"""
    router = APIRouter(prefix="/v1", tags=["Tasks"])

    @router.post("/tasks", status_code=202, response_model=TaskStatusResponse, openapi_extra=_request_schema())
    async def create_task(request: Request) -> Response:
        """先验证表单再建立稳定任务 ID，HTTP 结束后后台工作独立继续。"""
        backend = factory(request)
        store: TaskStore = request.app.state.task_store
        try:
            with tempfile.TemporaryDirectory(prefix="mineru-task-request-") as temporary:
                options, manifest, direct = await _read_submission(request, Path(temporary))
                record = await store.create(backend, options, manifest, direct)
        except BaseException:
            backend.release_uploads(rollback=True)
            raise
        return JSONResponse(task_payload(record, request), status_code=202)

    @router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task(task_id: str, request: Request) -> Response:
        """读取上传准备与后台状态，成功时返回可直接下载的 ZIP 地址。"""
        store: TaskStore = request.app.state.task_store
        record = store.lookup(task_id, factory(request).owner_scope)
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
        """每条上传链接只接受一个文件，最后一个文件就绪后自动提交。"""
        store: TaskStore = request.app.state.task_store
        record = store.lookup(task_id, factory(request).owner_scope)
        skipped = await store.upload(record, client_file_id, request)
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


def _combine_archives(store: FileStore, job: _JobRecord, path: Path, manifest: list[TaskManifestItem], status: str) -> str:
    """在后台流式构建批量 ZIP，固定输入顺序并拒绝越界成员路径。"""
    snapshot = {
        "status": status,
        # ZIP 可独立消费；不把 worker 的私有 task/file ID 或短期下载地址写入产物。
        "files": [
            {
                **item.model_dump(exclude={"url"}),
                **(
                    {"bytes": job.input_metadata[index].bytes, "sha256sum": job.input_metadata[index].sha256sum}
                    if index in job.input_metadata
                    else {}
                ),
                "name": result.name,
                "page_range": result.page_range,
                "status": result.status,
                "directory": f"{index + 1:04d}/" if result.status == "completed" else None,
                "parse": result.parse.model_dump() if result.parse is not None else None,
                "error": result.error.model_dump() if result.error is not None else None,
            }
            for index, (item, result) in enumerate(zip(manifest, job.files))
        ],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as target:
        target.writestr("manifest.json", json.dumps(snapshot, ensure_ascii=False))
        for index, file in enumerate(job.files):
            if file.status != "completed":
                continue
            if file.output_files is None or file.output_files.zip is None:
                raise ValueError("ZIP output is missing")
            source_path, _, sha = store.pin_output_file(file.output_files.zip.file_id)
            try:
                with zipfile.ZipFile(source_path) as source:
                    for info in source.infolist():
                        member = Path(info.filename)
                        if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
                            raise ValueError("Invalid ZIP member path")
                        name = f"{index + 1:04d}/{info.filename}"
                        if info.is_dir():
                            target.writestr(name, b"")
                        else:
                            with source.open(info) as incoming, target.open(name, "w", force_zip64=True) as outgoing:
                                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            finally:
                store.unpin_cached_blob(sha)
    return store.create_file_for_output_path(f"{job.id}.zip", path, job_id=job.id)


class ApiTaskBackend:
    """把上传准备接入 API server 的唯一文件存储和 JobStore。"""

    def __init__(self, request: Request) -> None:
        """保留轻量请求上下文，避免跨请求持有已解析的表单和上传对象。"""
        self.request = Request(dict(request.scope))
        self.files: FileStore = request.app.state.file_store
        self.jobs: JobStore = request.app.state.job_store
        self.owner_scope = "local"
        self.access: AccessLevel = "registered" if request.app.state.api_key else "anonymous"
        self._pins: list[str] = []
        self._owned_files: list[str] = []

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
        """把已上传文件和 URL 来源交给现有队列，在发布成功前完成最终 ZIP。"""
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
            output_formats=options.output_formats,
        )

        async def finalize(job: _JobRecord) -> None:
            """打包保持 running，失败与取消均不能发布一个不可用的下载链接。"""
            terminal = job.status
            job.status = "running"
            job.finished_at = None
            try:
                if len(job.files) == 1:
                    output = job.files[0].output_files
                    if output is None or output.zip is None:
                        raise ValueError("ZIP output is missing")
                    file_id = output.zip.file_id
                else:
                    with tempfile.TemporaryDirectory(prefix="mineru-task-zip-") as directory:
                        file_id = await run_sync(
                            _combine_archives, self.files, job, Path(directory) / "result.zip", manifest, terminal
                        )
                if job.status not in {"canceled", "failed"}:
                    job.task_result_file_id = file_id
                    job.status = terminal
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if job.status != "canceled":
                    job.task_error = ErrorDetail(type="api_error", code="result_packaging_failed", message=str(exc))
                    job.status = "failed"
                logger.exception("Task ZIP creation failed: %s", task_id)

        await submit_parse_job(
            body,
            self.request,
            self.files,
            self.jobs,
            self.access,
            job_id=task_id,
            finalize=finalize,
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
            record.task_error,
            dict(record.input_metadata),
        )

    async def cancel(self, task_id: str) -> None:
        """沿用 JobStore 取消及线程清理边界，终态竞争不影响已经完成的任务。"""
        record = self.jobs.get(task_id)
        if record.status not in TERMINAL_TASK_STATUSES:
            self.jobs.cancel(task_id)
        await self.jobs.wait_for_terminal(task_id, 3600)


__all__ = [
    "ApiTaskBackend",
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
