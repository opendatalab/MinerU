"""部署无关的上传与执行协议，隔离本地 ExecutionStore 和 Router 实现。"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ..task_types import TaskFileState
from .contracts import ErrorDetail, FileParseInfo, TaskFileInput, TaskFileItem, UploadTaskOptions


@dataclass(frozen=True)
class TaskInputMetadata:
    """记录完整下载且校验通过的实际输入信息，供父 Task 更新文件元数据。"""

    bytes: int


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
    """记录单文件执行快照；file_id 固定为公开结果身份，不代表内部源文件引用。"""

    execution_id: str
    file_id: str
    file_name: str
    status: TaskFileState
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    page_range: str = ""
    parse: FileParseInfo | None = None
    result_file_id: str | None = None
    error: ErrorDetail | None = None
    input_metadata: TaskInputMetadata | None = None


class TaskBackend(Protocol):
    """上传准备复用各部署的存储，单文件执行由本地或 Router 适配器负责。"""

    owner_scope: str

    def reserve_file_id(self) -> str:
        """在本层 Files 命名空间内预留唯一产物 ID。"""
        ...

    def discard_file_id(self, file_id: str) -> None:
        """撤销失败或取消文件的预留及尚未发布的结果。"""
        ...

    def forget_file_id(self, file_id: str) -> None:
        """任务回收后释放预留索引，不删除仍有效的产物资源。"""
        ...

    def fork(self) -> TaskBackend:
        """为每个文件创建独立的引用和执行上下文。"""
        ...

    async def release_result(self, task_id: str, finished: float) -> None:
        """父任务收尾后解除子结果租约，并开始正常保留期。"""
        ...

    async def validate(self, options: UploadTaskOptions, inputs: list[TaskFileInput]) -> None:
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
        self, execution_id: str, options: UploadTaskOptions, item: TaskFileItem, upload: TaskUpload | None
    ) -> TaskExecution:
        """使用预留的公开文件身份提交一次单文件执行，不承载批次聚合。"""
        ...

    async def get(self, task_id: str) -> TaskExecution:
        """只查询执行和产物元数据，不读取结果文件内容。"""
        ...

    async def cancel(self, task_id: str) -> None:
        """取消对应后台工作，并收敛已有资源和预占。"""
        ...


def new_execution_id() -> str:
    """保持现有任务和执行标识格式，生成不可预测的独立 ID。"""
    return "job_" + secrets.token_hex(12)


def execution_timestamp() -> str:
    """按现有秒级 UTC 格式记录任务和本地执行的生命周期时间。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["TaskBackend", "TaskExecution", "TaskInputMetadata", "TaskUpload", "execution_timestamp", "new_execution_id"]
