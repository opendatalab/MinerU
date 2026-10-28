"""同步包装服务的请求、远端快照和持久化记录。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ...task_types import TaskOutputFormat, TaskStatus, task_status_url
from ...tasks.contracts import TaskFileIdentity, TaskUploadLink, UploadTaskOptions

TERMINAL = frozenset({"completed", "failed", "canceled", "submission_unknown"})


class GatewayError(Exception):
    """保留结构化错误，避免把含签名地址或密钥的传输异常直接返回。"""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 502,
        *,
        retryable: bool = False,
        upstream_status: int | None = None,
    ) -> None:
        """区分调用参数、远端业务错误与可安全重试的传输错误。"""
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.upstream_status = upstream_status

    def detail(self) -> dict[str, Any]:
        """生成可持久化且不包含底层请求对象的错误信息。"""
        return {"code": self.code, "message": self.message, "upstream_status": self.upstream_status}


class ParseOptions(UploadTaskOptions):
    """保持与 Tasks 一致的解析参数，等待时间不属于解析身份。"""

    @model_validator(mode="after")
    def normalize(self) -> ParseOptions:
        """按首次出现去重格式，并统一整本页范围的等价写法。"""
        self.output_formats = list(dict.fromkeys(self.output_formats))
        self.page_range = self.page_range.strip()
        if self.page_range.lower() == "all":
            self.page_range = ""
        return self


class RemoteFile(TaskFileIdentity):
    """仅消费共同 Tasks 契约，不依赖本地 Files 或 Job 实现。"""

    bytes: int
    upload: TaskUploadLink | None = None
    result_url: str | None = None
    error: dict[str, Any] | None = None


class RemoteTask(BaseModel):
    """保存客户端私有上下文，标记是否已经取得查询详情以支持创建后立即恢复。"""

    task_id: str = Field(min_length=1)
    status: TaskStatus
    files: list[RemoteFile] = Field(min_length=1, max_length=1)
    output_formats: list[TaskOutputFormat] = Field(min_length=1)
    details_loaded: bool = True

    @model_validator(mode="after")
    def validate_file(self) -> RemoteTask:
        """确保返回的文件和链接仍对应本次唯一清单项。"""
        item = self.files[0]
        task_status_url("", self.task_id)
        if item.upload is not None and item.status != "waiting-file":
            raise ValueError("Only waiting-file accepts uploads")
        if self.details_loaded and item.status == "completed" and not item.result_url:
            raise ValueError("Completed file has no result_url")
        if item.status != "completed" and item.result_url is not None:
            raise ValueError("Unfinished file has a result_url")
        return self


class RemoteCreatedFile(TaskFileIdentity):
    """只消费创建回执中的文件身份、状态与上传描述。"""

    upload: TaskUploadLink | None = None


class RemoteCreatedTask(BaseModel):
    """单独校验创建回执，不要求它包含大小、输出格式或结果链接。"""

    task_id: str = Field(min_length=1)
    status: TaskStatus
    files: list[RemoteCreatedFile] = Field(min_length=1, max_length=1)


@dataclass
class Record:
    """保存恢复所需的最小状态；密钥仅来自启动配置，不写入记录。"""

    request_id: str
    filename: str
    size: int
    sha256sum: str
    options: dict[str, Any]
    source_path: str | None
    fingerprint: str
    status: str = "prepared"
    remote: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    terminal_at: float | None = None

    @property
    def task(self) -> RemoteTask | None:
        """读取已校验远端快照，空值表示尚未确认创建结果。"""
        return RemoteTask.model_validate(self.remote) if self.remote is not None else None

    @property
    def needs_details(self) -> bool:
        """创建已报告完成但首次查询失败时，重启或重复请求仍需补取结果详情。"""
        return self.status == "completed" and self.remote is not None and self.remote.get("details_loaded") is False

    def to_json(self) -> str:
        """把完整记录序列化为 SQLite 单行快照。"""
        return json.dumps(asdict(self), ensure_ascii=False)


def fingerprint(filename: str, size: int, sha256sum: str, options: ParseOptions) -> str:
    """用文件内容、逻辑名称与归一化参数区分同 ID 的冲突提交。"""
    normalized = options.model_dump()
    value = {"filename": filename, "size": size, "sha256sum": sha256sum, "options": normalized}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


__all__ = ["GatewayError", "ParseOptions", "Record", "RemoteCreatedTask", "RemoteTask", "TERMINAL", "fingerprint"]
