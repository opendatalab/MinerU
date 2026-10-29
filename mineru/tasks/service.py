"""Tasks 生命周期与上传存储编排，不依赖 HTTP 请求或本地解析实现。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..filetypes import is_flash_only_parse_extension
from ..task_types import TaskFileState, TaskStatus
from ..types import Tier, select_default_task_tier
from ..utils.async_utils import drain_future, run_sync
from .contracts import ErrorDetail, TaskFileInput, TaskFileItem, UploadTaskOptions
from .errors import ApiServerError, invalid_task_request
from .execution import TaskBackend, TaskExecution, TaskUpload, execution_timestamp, new_execution_id

TASK_UPLOAD_TTL = 3600
TERMINAL_TASK_STATUSES = frozenset({"completed", "partial", "failed", "canceled"})
logger = logging.getLogger(__name__)


def resolve_task_tier(options: UploadTaskOptions, inputs: list[TaskFileInput], tiers: set[Tier]) -> Tier:
    """接单时固定 PDF/图片档位；纯原生文档归一为 flash，显式档位不升降级。"""
    lightweight = [is_flash_only_parse_extension(item.file_name) for item in inputs]
    if all(lightweight):
        return "flash"
    if options.tier is not None:
        if options.tier not in tiers:
            invalid_task_request(f"Tier '{options.tier}' is not available", param="tier")
        return options.tier
    default_tier = select_default_task_tier(tiers)
    if default_tier is not None:
        return default_tier
    invalid_task_request(
        "No parsing tier is available for PDF/image tasks",
        code="quality_tier_unavailable",
        status_code=503,
    )


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
    expected: TaskFileItem | None = None,
    canceled: Callable[[], bool] | None = None,
) -> TaskUpload:
    """流式落盘并校验完整性；声明大小用于一致性检查，不是上传大小上限。"""
    hasher = hashlib.sha256()
    size = 0
    try:
        with path.open("wb") as target:
            async for chunk in chunks:
                if canceled is not None and canceled():
                    invalid_task_request("Task was canceled during upload", status_code=409, code="task_already_terminal")
                size += len(chunk)
                if expected is not None and expected.bytes is not None and size > expected.bytes:
                    invalid_task_request("Declared size does not match uploaded bytes", code="upload_size_mismatch")
                await run_sync(target.write, chunk)
                hasher.update(chunk)
        if size == 0:
            invalid_task_request("Empty files are not supported", param="files")
        if expected is not None and expected.bytes is not None and size != expected.bytes:
            invalid_task_request("Declared size does not match uploaded bytes", code="upload_size_mismatch")
        sha = hasher.hexdigest()
        if expected is not None and expected.sha256sum is not None and sha != expected.sha256sum:
            invalid_task_request("SHA-256 mismatch", code="file_hash_mismatch")
        return TaskUpload(path, filename, size, sha, mime_type)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _normalize_task_files(inputs: list[TaskFileInput], backend: TaskBackend) -> list[TaskFileItem]:
    """按请求顺序为每个输入预留公共产物 ID，失败时撤销已分配的预留。"""
    normalized: list[TaskFileItem] = []
    try:
        for item in inputs:
            normalized.append(TaskFileItem(**item.model_dump(), file_id=backend.reserve_file_id()))
        return normalized
    except BaseException:
        for item in normalized:
            backend.discard_file_id(item.file_id)
            backend.forget_file_id(item.file_id)
        raise


@dataclass
class TaskFileRecord:
    """每个文件独立拥有准备引用、执行 ID 和一次性提交句柄。"""

    backend: TaskBackend
    file_id: str
    file_name: str
    execution_id: str = field(default_factory=new_execution_id)
    status: TaskFileState = "waiting-file"
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
class TaskRecord:
    """父 Task 独占批次聚合，独立单文件执行继续使用现有解析并发。"""

    id: str
    backend: TaskBackend
    options: UploadTaskOptions
    inputs: list[TaskFileItem]
    directory: Path
    expires_at: int
    file_states: list[TaskFileRecord]
    created_at: str = field(default_factory=execution_timestamp)
    status: TaskStatus = "waiting-file"
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
        self._records: dict[str, TaskRecord] = {}
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

    def _aggregate(self, record: TaskRecord) -> None:
        """按运行、排队、等待的优先级聚合，所有文件终结后再发布成功或部分成功。"""
        if record.status == "canceled":
            return
        previous = record.status
        states = [file.status for file in record.file_states]
        if "running" in states:
            record.status = "running"
        elif "pending" in states:
            record.status = "pending"
        elif "waiting-file" in states:
            record.status = "waiting-file"
        else:
            completed = states.count("completed")
            record.status = "completed" if completed == len(states) else ("partial" if completed else "failed")
        if record.status in TERMINAL_TASK_STATUSES and record.finished_epoch is None:
            record.finished_at = execution_timestamp()
            record.finished_epoch = time.time()
        if record.status != previous:
            record.changed.set()

    def _expire_uploads(self, record: TaskRecord) -> None:
        """只终结尚未受理上传的文件；截止前已经受理的流允许继续完成。"""
        if record.expires_at > time.time() or record.status == "canceled":
            return
        for item, file in zip(record.inputs, record.file_states):
            if item.url is None and file.upload is None and file.status == "waiting-file" and not file.upload_active:
                file.expired = True
                file.status = "failed"
                file.error = ErrorDetail(code="upload_expired", message="Upload expired")
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
                for item in record.inputs:
                    record.backend.forget_file_id(item.file_id)
                shutil.rmtree(record.directory, ignore_errors=True)

    def lookup(self, task_id: str, owner_scope: str) -> TaskRecord:
        """未知与越权任务使用相同错误，避免跨调用方探测。"""
        self.collect_expired()
        record = self._records.get(task_id)
        if record is None or record.backend.owner_scope != owner_scope:
            invalid_task_request(f"Task {task_id} not found", status_code=404, code="task_not_found")
        return record

    def update_lease(self, record: TaskRecord, seconds: int | None) -> None:
        """续租只保护结果保留期，从不延长上传窗口，也不复活已释放的结果。"""
        if seconds is None:
            return
        if seconds and record.retention_epoch is not None:
            invalid_task_request("Task result lease has already ended", status_code=409, code="task_lease_ended")
        record.lease_expires_at = time.time() + seconds if seconds else 0
        record.changed.set()

    async def create(
        self, backend: TaskBackend, options: UploadTaskOptions, inputs: list[TaskFileInput], *, lease_seconds: int = 0
    ) -> TaskRecord:
        """先完整校验清单并固定缓存，再为各个就绪文件独立提交执行。"""
        if self._closing:
            invalid_task_request("Server is shutting down", status_code=503, code="server_shutting_down")
        await backend.validate(options, inputs)
        inputs = _normalize_task_files(inputs, backend)
        task_id = new_execution_id()
        record = TaskRecord(
            task_id,
            backend,
            options,
            inputs,
            self._root / task_id,
            int(time.time()) + TASK_UPLOAD_TTL,
            [TaskFileRecord(backend.fork(), file_id=item.file_id, file_name=item.file_name) for item in inputs],
        )
        self.update_lease(record, lease_seconds)
        try:
            for index, item in enumerate(inputs):
                file = record.file_states[index]
                if item.url is None and item.sha256sum:
                    upload = await file.backend.find_upload(item.sha256sum, item.file_name)
                    if upload is not None:
                        if upload.bytes != item.bytes:
                            invalid_task_request("Declared size does not match cached bytes", code="upload_size_mismatch")
                        file.upload = await file.backend.retain_upload(upload)
            self._records[task_id] = record
            self.start()
            for index, item in enumerate(inputs):
                if item.url is not None or record.file_states[index].upload is not None:
                    self._start_file(record, index)
            record.operation = asyncio.create_task(self._monitor(record), name=f"mineru-task-{task_id}")
            return record
        except BaseException:
            for item in inputs:
                backend.discard_file_id(item.file_id)
                backend.forget_file_id(item.file_id)
            for file in record.file_states:
                file.backend.release_uploads(rollback=True)
            raise

    def _start_file(self, record: TaskRecord, index: int) -> None:
        """事件循环内同步登记提交句柄，重复 PUT 不能创建第二个子作业。"""
        file = record.file_states[index]
        if file.operation is not None or file.status != "waiting-file" or record.status == "canceled":
            return
        file.status = "pending" if file.upload is not None else "waiting-file"
        record.changed.set()
        file.operation = asyncio.create_task(self._dispatch_file(record, index), name=f"mineru-task-file-{file.execution_id}")
        self._aggregate(record)

    def _apply_execution(self, record: TaskRecord, file: TaskFileRecord, execution: TaskExecution) -> None:
        """只接受单文件执行快照，取消和已发布终态不被迟到响应覆盖。"""
        if file.status in TERMINAL_TASK_STATUSES:
            return
        status = execution.status
        if status not in {"waiting-file", "pending", "running", "completed", "failed", "canceled"}:
            invalid_task_request("Invalid file status from execution backend", code="invalid_response", status_code=502)
        if status == "completed" and execution.result_file_id is None:
            invalid_task_request("Completed file has no result", code="invalid_response", status_code=502)
        if (
            execution.execution_id != file.execution_id
            or execution.file_id != file.file_id
            or execution.file_name != file.file_name
        ):
            invalid_task_request("Execution file identity changed", code="invalid_response", status_code=502)
        if execution.result_file_id is not None and execution.result_file_id != file.file_id:
            invalid_task_request("Execution result identity changed", code="invalid_response", status_code=502)
        if status != file.status:
            record.changed.set()
        file.execution = execution
        file.status = status
        file.error = execution.error
        if status in TERMINAL_TASK_STATUSES:
            file.finished.set()
            file.lease_refresh_at = time.monotonic() + 30
        self._aggregate(record)

    async def refresh(self, record: TaskRecord) -> None:
        """并发刷新独立文件；成功文件仍续租 worker 结果直到父任务收尾。"""
        async with record.refresh_lock:
            self._expire_uploads(record)
            if record.status in TERMINAL_TASK_STATUSES:
                return
            targets = [
                file
                for file in record.file_states
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
                        file.backend.discard_file_id(file.file_id)
                        file.error = (
                            result.error
                            if isinstance(result, ApiServerError)
                            else ErrorDetail(code="task_refresh_failed", message=str(result))
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
                    file.backend.discard_file_id(file.file_id)
                    file.error = exc.error
                    if file.operation is not None:
                        file.operation.cancel()
            self._aggregate(record)

    async def _dispatch_file(self, record: TaskRecord, index: int) -> None:
        """每个文件独立提交，失败或取消只清理自己的作业与源文件租约。"""
        file = record.file_states[index]
        try:
            execution = await file.backend.submit(file.execution_id, record.options, record.inputs[index], file.upload)
            file.submitted = True
            self._apply_execution(record, file, execution)
            file.backend.release_uploads(rollback=False)
            if file.status not in TERMINAL_TASK_STATUSES:
                await file.finished.wait()
        except asyncio.CancelledError:
            if file.status != "completed":
                file.backend.discard_file_id(file.file_id)
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
                    else ErrorDetail(code="task_submission_failed", message=str(exc))
                )
            if file.status != "completed":
                file.backend.discard_file_id(file.file_id)
            try:
                await file.backend.cancel(file.execution_id)
            except Exception:
                logger.exception("Task file failure cleanup failed: %s", file.execution_id)
        finally:
            file.finished.set()
            record.changed.set()
            file.backend.release_uploads(rollback=not file.submitted)
            self._aggregate(record)

    async def _monitor(self, record: TaskRecord) -> None:
        """应用持有父任务监控；HTTP 断连不影响执行，所有子文件清理后才开始保留期。"""
        try:
            while record.status not in TERMINAL_TASK_STATUSES:
                record.changed.clear()
                await self.refresh(record)
                if record.status not in TERMINAL_TASK_STATUSES:
                    await self._wait_for_change(record)
            operations = [file.operation for file in record.file_states if file.operation is not None]
            if operations:
                await asyncio.gather(*operations, return_exceptions=True)
            for item, file in zip(record.inputs, record.file_states):
                if file.status != "completed":
                    file.backend.discard_file_id(item.file_id)
            while any(file.upload_active for file in record.file_states) or record.lease_expires_at > time.time():
                record.changed.clear()
                await self._wait_for_change(record)
            finished = time.time()
            for file in record.file_states:
                await file.backend.release_result(file.execution_id, finished)
                file.backend.release_uploads(rollback=not file.submitted)
            record.retention_epoch = finished
        except Exception:
            logger.exception("Task result cleanup failed: %s", record.id)
        finally:
            if not any(file.upload_active for file in record.file_states):
                shutil.rmtree(record.directory, ignore_errors=True)

    async def _wait_for_change(self, record: TaskRecord) -> None:
        """状态变更即时唤醒；没有事件时每秒检查执行与到期边界。"""
        try:
            await asyncio.wait_for(record.changed.wait(), timeout=1)
        except asyncio.TimeoutError:
            pass

    async def upload(
        self,
        record: TaskRecord,
        file_id: str,
        *,
        chunks: AsyncIterator[bytes],
        content_length: str | None,
        mime_type: str,
    ) -> bool:
        """在接收主体前检查固定期限，已受理流不因父任务状态变化中止。"""
        index = next((i for i, item in enumerate(record.inputs) if item.file_id == file_id), None)
        if index is None:
            invalid_task_request("Unknown file_id", status_code=404, code="upload_not_found")
        expected = record.inputs[index]
        if expected.url is not None:
            invalid_task_request("URL inputs do not accept uploads")
        file = record.file_states[index]
        async with file.upload_lock:
            if record.expires_at <= time.time():
                self._expire_uploads(record)
                invalid_task_request("Upload link expired", status_code=410, code="upload_expired")
            if record.status == "canceled":
                invalid_task_request("Task was canceled", status_code=409, code="task_already_terminal")
            if file.upload is not None:
                return True
            if file.status != "waiting-file":
                invalid_task_request("File no longer accepts uploads", status_code=409, code="file_already_terminal")
            length = content_length
            if length is not None:
                try:
                    declared = int(length)
                except ValueError:
                    invalid_task_request("Invalid Content-Length", code="upload_size_mismatch")
                if declared != expected.bytes:
                    invalid_task_request("Content-Length does not match declared file size", code="upload_size_mismatch")
            file.upload_active = True
            record.directory.mkdir(parents=True, exist_ok=True)
            path = record.directory / secrets.token_hex(16)

            def was_canceled() -> bool:
                """只在本文件被取消时中止已受理上传，不把上传截止时间当作传输超时。"""
                return record.status == "canceled" or file.status == "canceled"

            try:
                upload = await _save_upload(
                    chunks,
                    path,
                    expected.file_name,
                    mime_type,
                    expected=expected,
                    canceled=was_canceled,
                )
                if was_canceled():
                    invalid_task_request("Task was canceled during upload", status_code=409, code="task_already_terminal")
                retained = await file.backend.retain_upload(upload)
                if was_canceled():
                    file.backend.release_uploads(rollback=True)
                    invalid_task_request("Task was canceled during upload", status_code=409, code="task_already_terminal")
                file.upload = retained
                self._start_file(record, index)
                return False
            finally:
                path.unlink(missing_ok=True)
                file.upload_active = False
                record.changed.set()
                self._expire_uploads(record)

    async def cancel(self, record: TaskRecord, *, wait: bool = True) -> None:
        """保留已完成结果，只取消未完成文件并等待真实子作业清理。"""
        if record.status in TERMINAL_TASK_STATUSES:
            invalid_task_request(f"Task is {record.status}", status_code=409, code="task_already_terminal")
        record.status = "canceled"
        record.finished_at = execution_timestamp()
        record.finished_epoch = time.time()
        record.changed.set()
        for item, file in zip(record.inputs, record.file_states):
            if file.status not in TERMINAL_TASK_STATUSES:
                file.status = "canceled"
                file.backend.discard_file_id(item.file_id)
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


__all__ = [
    "TASK_UPLOAD_TTL",
    "TERMINAL_TASK_STATUSES",
    "TaskFileRecord",
    "TaskRecord",
    "TaskStore",
    "file_chunks",
    "resolve_task_tier",
]
