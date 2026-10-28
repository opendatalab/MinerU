# Copyright (c) Opendatalab. All rights reserved.
"""把共享 Tasks 执行协议接入本地解析服务的存储、并发与资源生命周期。"""

from __future__ import annotations

from dataclasses import replace

from fastapi import Request

from ..filetypes import batch_effective_parse_tier
from ..tasks.contracts import TaskFileInput, TaskFileItem, UploadTaskOptions
from ..tasks.errors import invalid_task_request
from ..tasks.execution import TaskExecution, TaskUpload
from ..tasks.service import TERMINAL_TASK_STATUSES, resolve_task_tier
from ..utils.async_utils import run_sync
from .api_server import (
    FileStore,
    ExecutionStore,
    _ExecutionRequest,
    _StoredSource,
    _UrlSource,
    _require_model_preload_ready,
    _validate_source_policy,
    submit_execution,
)


class ApiTaskBackend:
    """把上传准备接入 API server 的唯一文件存储和 ExecutionStore。"""

    def __init__(self, request: Request) -> None:
        """保留轻量请求上下文，不跨请求持有上传主体。"""
        self.request = Request(dict(request.scope))
        self.files: FileStore = request.app.state.file_store
        self.executions: ExecutionStore = request.app.state.execution_store
        self.owner_scope = "local"
        self._pins: list[str] = []
        self._owned_files: list[str] = []

    def fork(self) -> ApiTaskBackend:
        """为单个文件隔离源字节引用，其他文件提交不释放本文件资源。"""
        return ApiTaskBackend(self.request)

    def reserve_file_id(self) -> str:
        """在 API Server 的 Files 存储中预留统一的公共产物 ID。"""
        return self.files.reserve_file_id()

    def discard_file_id(self, file_id: str) -> None:
        """取消或失败时使结果预留失效，不影响共享的源文件缓存。"""
        self.files.discard_file_id(file_id)

    def forget_file_id(self, file_id: str) -> None:
        """任务记录过期后释放 ID 预留索引，Files 自身仍遵循原保留期。"""
        self.files.forget_file_id(file_id)

    async def release_result(self, task_id: str, finished: float) -> None:
        """由父 Task 统一开始子作业与结果文件的保留期。"""
        self.executions.release_task_result(task_id, finished)

    async def validate(self, options: UploadTaskOptions, inputs: list[TaskFileInput]) -> None:
        """复用引擎就绪状态，提前校验显式 tier 与轻量格式能力。"""
        _require_model_preload_ready(self.request)
        if self.executions._closing:
            invalid_task_request("Server is shutting down", code="server_shutting_down", status_code=503)
        state = self.request.app.state
        for index, item in enumerate(inputs):
            if item.url is not None:
                try:
                    _validate_source_policy(
                        _UrlSource(url=item.url),
                        allow_http_source=state.allow_http_source,
                    )
                except ValueError as exc:
                    invalid_task_request(str(exc), param=f"files.{index}.url")
        options.tier = resolve_task_tier(options, inputs, set(state.tier_runtime_options))

    async def find_upload(self, sha256sum: str, filename: str) -> TaskUpload | None:
        """原子查找并取得缓存租约，准备阶段不会因原文件视图到期丢失字节。"""
        path = self.files.pin_cached_blob(sha256sum)
        if path is None:
            return None
        self._pins.append(sha256sum)
        return TaskUpload(path, filename, path.stat().st_size, sha256sum)

    async def retain_upload(self, upload: TaskUpload) -> TaskUpload:
        """立即按内容去重并登记独立文件视图，后续文件失败不丢弃已成功上传项。"""
        file_id = self.files._allocate_file_id()
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
                self.files.forget_file_id(file_id)
        else:
            self.files.finalize_source_views(self._owned_files)
        self._owned_files.clear()
        for sha in self._pins:
            self.files.unpin_cached_blob(sha)
        self._pins.clear()

    async def submit(
        self, execution_id: str, options: UploadTaskOptions, item: TaskFileItem, upload: TaskUpload | None
    ) -> TaskExecution:
        """提交唯一文件并保留公开结果身份，源文件视图只用于内部读取和租约。"""
        if options.tier is None:
            raise ValueError("Task tier must be resolved before execution")
        tier = batch_effective_parse_tier(options.tier, item.file_name)
        if upload is not None:
            await run_sync(
                self.files.register_source_file, upload.filename, upload.path, upload.sha256sum, upload.file_id, keep_alive=True
            )
        if item.url is not None:
            source = _UrlSource(url=item.url)
        elif upload is not None and upload.file_id is not None:
            source = _StoredSource(file_id=upload.file_id)
        else:
            raise ValueError("Prepared source file is missing")
        body = _ExecutionRequest(
            source=source,
            file_id=item.file_id,
            file_name=item.file_name,
            bytes=item.bytes,
            sha256sum=item.sha256sum,
            page_range=item.page_range or None,
            tier=tier,
            ocr_mode=options.ocr_mode,
            output_formats=options.output_formats,
        )
        await submit_execution(body, self.request, self.files, self.executions, execution_id=execution_id)
        return await self.get(execution_id)

    async def get(self, task_id: str) -> TaskExecution:
        """返回单文件执行快照，状态查询不读取产物内容。"""
        return self.executions.snapshot(self.executions.get(task_id))

    async def cancel(self, task_id: str) -> None:
        """沿用执行取消及线程清理边界，终态竞争不影响已经完成的任务。"""
        record = self.executions._executions.get(task_id)
        if record is None:
            return
        if record.status not in TERMINAL_TASK_STATUSES:
            self.executions.cancel(task_id)
        await self.executions.wait_for_terminal(task_id, 3600)


__all__ = ["ApiTaskBackend"]
