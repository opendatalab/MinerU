"""把请求连接与持久化的 Tasks 上传、查询生命周期分离。"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from .client import TaskClient
from .models import TERMINAL, GatewayError, ParseOptions, Record, RemoteTask, fingerprint
from .settings import FileParseSettings
from .store import RecordStore

logger = logging.getLogger(__name__)


class FileParseService:
    """单进程管理恢复记录；每个请求只有一个独立的上传准备任务。"""

    def __init__(self, settings: FileParseSettings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """初始化无副作用的管理状态，网络连接在应用启动后使用。"""
        self.settings = settings
        self.store = RecordStore(settings)
        self.client = TaskClient(settings, transport)
        self.records: dict[str, Record] = {}
        self.preparations: dict[str, asyncio.Task[None]] = {}
        self.changes: dict[str, asyncio.Event] = {}
        self.query_locks: dict[str, asyncio.Lock] = {}
        self.state_locks: dict[str, asyncio.Lock] = {}
        self._accept_lock = asyncio.Lock()
        self._query_limit = asyncio.Semaphore(16)
        self._monitor: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """恢复已有任务；曾经开始但未确认的创建请求不能自动重发。"""
        for record in await self.store.open():
            self.records[record.request_id] = record
            if record.status == "submitting" and record.remote is None:
                await self._remember_error(
                    record, GatewayError("submission_unknown", "Task creation outcome is unknown; not resubmitting")
                )
            elif record.status not in TERMINAL or record.needs_details:
                self._prepare_in_background(record)
        await self.collect()
        self._monitor = asyncio.create_task(self._monitor_records(), name="file-parse-monitor")

    async def close(self) -> None:
        """停止本地协程并保存已确认的任务状态，不取消上游解析。"""
        tasks = [*self.preparations.values()]
        if self._monitor is not None:
            tasks.append(self._monitor)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.close()
        await self.store.close()

    def _changed(self, record: Record) -> None:
        """唤醒正在等待该请求状态变化的连接。"""
        self.changes.setdefault(record.request_id, asyncio.Event()).set()

    def _prepare_in_background(self, record: Record) -> None:
        """把任务创建及上传托管到应用，连接取消不会传播到这里。"""
        previous = self.preparations.get(record.request_id)
        if previous is None or previous.done():
            self.preparations[record.request_id] = asyncio.create_task(
                self._prepare(record), name=f"file-parse-prepare-{record.request_id}"
            )

    async def accept(
        self,
        request_id: str,
        filename: str,
        size: int,
        sha256sum: str,
        options: ParseOptions,
        source: Path,
    ) -> Record:
        """原子接受或复用请求，同 ID 不同内容不会替换原任务或源文件。"""
        identity = fingerprint(filename, size, sha256sum, options)
        async with self._accept_lock:
            existing = self.records.get(request_id)
            if existing is not None and self._expired(existing):
                await self._remove_record(existing)
                existing = None
            if existing is not None:
                source.unlink(missing_ok=True)
                if existing.fingerprint != identity:
                    raise GatewayError("request_id_conflict", "request_id already identifies a different request", 409)
                if existing.status not in TERMINAL or existing.needs_details:
                    self._prepare_in_background(existing)
                return existing
            record = Record(request_id, filename, size, sha256sum, options.model_dump(), str(source), identity)
            await self.store.save(record)
            self.records[request_id] = record
            self._prepare_in_background(record)
            return record

    def get(self, request_id: str) -> Record:
        """仅在当前上游命名空间内查找请求；过期记录等同于不存在。"""
        record = self.records.get(request_id)
        if record is None or self._expired(record):
            raise GatewayError("request_not_found", "Request was not found or its recovery record expired", 404)
        return record

    def _expired(self, record: Record) -> bool:
        """终态首次观察时间决定保留期限，重复查询不续期。"""
        return bool(
            self.settings.retention_seconds
            and record.terminal_at is not None
            and record.terminal_at + self.settings.retention_seconds <= time.time()
        )

    async def _save_and_cleanup(self, record: Record, updated: Record, *, remove_source: bool) -> None:
        """持久化和清理完成后才发布新状态，避免并发查询提前观察完成状态。"""
        source = updated.source_path if remove_source else None
        if remove_source:
            updated.source_path = None
        await self.store.save(updated)
        try:
            if source:
                Path(source).unlink(missing_ok=True)
        finally:
            # 即使文件清理失败，也不能丢掉已经持久化的远端 task_id。
            record.remote = updated.remote
            record.status = updated.status
            record.source_path = updated.source_path
            record.error = updated.error
            record.terminal_at = updated.terminal_at
            self._changed(record)

    async def _apply(self, record: Record, task: RemoteTask) -> None:
        """更新已确认任务快照，并避免旧轮询覆盖较新的终态。"""
        async with self.state_locks.setdefault(record.request_id, asyncio.Lock()):
            if self.records.get(record.request_id) is not record:
                return
            item = task.files[0]
            if record.remote is not None and record.status in TERMINAL and item.status not in TERMINAL:
                return
            updated = replace(record, remote=task.model_dump(), status=item.status)
            if item.status != "waiting-file":
                updated.error = item.error
            if item.status in {"failed", "canceled"} and updated.error is None:
                updated.error = {"code": f"task_{item.status}", "message": f"Upstream task {item.status}"}
            if updated.status in TERMINAL and updated.terminal_at is None:
                updated.terminal_at = time.time()
            await self._save_and_cleanup(record, updated, remove_source=item.status != "waiting-file")

    async def _remember_error(self, record: Record, error: GatewayError) -> None:
        """无任务 ID 的失败是本地终态；已知任务的传输错误保留后续查询能力。"""
        async with self.state_locks.setdefault(record.request_id, asyncio.Lock()):
            if self.records.get(record.request_id) is not record:
                return
            updated = replace(record, error={**error.detail(), "http_status": error.status_code})
            remove_source = False
            if updated.remote is None:
                updated.status = "submission_unknown" if error.code == "submission_unknown" else "failed"
                updated.terminal_at = updated.terminal_at or time.time()
                remove_source = True
            elif error.upstream_status in {404, 410}:
                updated.status = "failed"
                updated.terminal_at = updated.terminal_at or time.time()
                remove_source = True
            await self._save_and_cleanup(record, updated, remove_source=remove_source)

    async def _prepare(self, record: Record) -> None:
        """创建和上传最多执行一条流水线，上传阶段错误仍允许查询已有任务。"""
        try:
            record.error = None
            if record.remote is None:
                record.status = "submitting"
                await self.store.save(record)
                task = await self.client.create(record)
                await self._apply(record, task)
                # 创建身份已持久化；首次查询失败也只能恢复此任务，不能重新提交。
                task = await self.refresh(record)
            else:
                task = await self.refresh(record)
            if task.files[0].status == "waiting-file":
                if record.source_path is None or not Path(record.source_path).is_file():
                    raise GatewayError("source_unavailable", "The retained source is unavailable")
                task = await self.client.upload(task, Path(record.source_path), record.size)
                await self._apply(record, task)
        except GatewayError as exc:
            await self._remember_error(record, exc)
        except OSError:
            await self._remember_error(record, GatewayError("storage_error", "Cannot access the local source storage", 500))
        finally:
            self._changed(record)

    async def refresh(self, record: Record) -> RemoteTask:
        """序列化同一请求的状态查询，限制后台查询总并发。"""
        async with self.query_locks.setdefault(record.request_id, asyncio.Lock()):
            task = record.task
            if task is None:
                raise GatewayError("task_not_created", "No confirmed upstream task is available", 409)
            async with self._query_limit:
                current = await self.client.query(task)
            await self._apply(record, current)
            return current

    async def _refresh_safely(self, record: Record) -> None:
        """后台查询临时失败只更新错误，不使其他请求的轮询停止。"""
        try:
            await self.refresh(record)
        except GatewayError as exc:
            await self._remember_error(record, exc)

    async def _monitor_records(self) -> None:
        """持续观察已提交任务，使断连任务也能进入终态和开始保留期。"""
        while True:
            await asyncio.sleep(self.settings.poll_interval)
            records = [
                record
                for record in self.records.values()
                if record.remote is not None
                and (record.status not in TERMINAL or record.needs_details)
                and (record.request_id not in self.preparations or self.preparations[record.request_id].done())
            ]
            try:
                results = await asyncio.gather(*(self._refresh_safely(record) for record in records), return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error("File parse reconciliation error: %s", type(result).__name__)
                await self.collect()
            except Exception:
                logger.exception("File parse record reconciliation failed")

    async def collect(self) -> None:
        """回收超过保留期的终态记录及残留源文件，不触碰上游资源。"""
        async with self._accept_lock:
            for request_id, record in list(self.records.items()):
                operation = self.preparations.get(request_id)
                if not self._expired(record) or (operation is not None and not operation.done()):
                    continue
                await self._remove_record(record)

    async def _remove_record(self, record: Record) -> None:
        """与状态提交串行回收记录，迟到的查询不能重新插入已过期的旧任务。"""
        request_id = record.request_id
        async with self.state_locks.setdefault(request_id, asyncio.Lock()):
            await self.store.delete(request_id)
            if record.source_path:
                Path(record.source_path).unlink(missing_ok=True)
            self.records.pop(request_id, None)
            self.preparations.pop(request_id, None)
            self.changes.pop(request_id, None)
            self.query_locks.pop(request_id, None)
            self.state_locks.pop(request_id, None)

    async def wait(self, record: Record, timeout: int) -> bool:
        """上游文件准备完成后才启动单调时钟预算，连接取消不取消上传任务。"""
        preparation = self.preparations.get(record.request_id)
        if preparation is not None:
            await asyncio.shield(preparation)
        changed = self.changes.setdefault(record.request_id, asyncio.Event())
        while True:
            changed.clear()
            self.raise_record_error(record)
            task = record.task
            if record.status in TERMINAL or (task is not None and task.files[0].status != "waiting-file"):
                break
            # 文件准备包括上传确认与服务端下载；进入 pending 后才启动解析等待预算。
            await changed.wait()
        if timeout == 0:
            return False
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            changed.clear()
            self.raise_record_error(record)
            if record.status == "completed":
                return True
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(changed.wait(), remaining)
            except asyncio.TimeoutError:
                return record.status == "completed"

    def raise_record_error(self, record: Record) -> None:
        """把已知失败或传输异常转成结构化 HTTP 错误，保留记录供查询。"""
        if record.error is not None:
            default_status = 409 if record.status in {"canceled", "submission_unknown"} else 502
            raise GatewayError(
                str(record.error.get("code") or "task_failed"),
                str(record.error.get("message") or "Upstream task failed"),
                int(record.error.get("http_status") or default_status),
                upstream_status=record.error.get("upstream_status"),
            )

    def payload(self, record: Record, base_url: str) -> dict[str, Any]:
        """只返回包装服务的公共链接，隐藏上游上传地址和签名下载地址。"""
        status_url = f"{base_url.rstrip('/')}/v1/file_parse/{record.request_id}"
        task = record.task
        return {
            "request_id": record.request_id,
            "task_id": task.task_id if task else None,
            "status": record.status,
            "status_url": status_url,
            "result_url": status_url + "/result" if record.status == "completed" else None,
            "output_formats": record.options["output_formats"],
            "error": record.error,
        }

    async def artifact(self, record: Record) -> Path:
        """获取最新下载链接并验证临时产物，失败时立即清理临时文件。"""
        task = await self.refresh(record)
        if task.files[0].status in {"failed", "canceled"}:
            self.raise_record_error(record)
        if task.files[0].status != "completed":
            raise GatewayError("result_not_ready", "The result is not ready", 409)
        descriptor, name = tempfile.mkstemp(prefix="result-", dir=self.store.results)
        os.close(descriptor)
        path = Path(name)
        try:
            task = await self.client.download(task, path)
            await self._apply(record, task)
            return path
        except BaseException:
            path.unlink(missing_ok=True)
            raise


__all__ = ["FileParseService"]
