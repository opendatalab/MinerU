"""同步包装服务的请求、远端快照和持久化记录。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...task_types import TaskOutputFormat

TERMINAL = frozenset({"completed", "failed", "canceled", "submission_unknown"})
FileStatus = Literal["waiting_upload", "queued", "running", "completed", "failed", "canceled"]


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


class ParseOptions(BaseModel):
    """保持与 Tasks 一致的解析参数，等待时间不属于解析身份。"""

    model_config = ConfigDict(extra="forbid")

    tier: Literal["flash", "basic", "standard", "advanced"] | None = None
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto"
    page_range: str = ""
    output_formats: list[TaskOutputFormat] = Field(default_factory=lambda: ["default"], min_length=1)

    @model_validator(mode="after")
    def normalize(self) -> ParseOptions:
        """按首次出现去重格式，并统一整本页范围的等价写法。"""
        self.output_formats = list(dict.fromkeys(self.output_formats))
        self.page_range = self.page_range.strip()
        if self.page_range.lower() == "all":
            self.page_range = ""
        return self


class RemoteFile(BaseModel):
    """仅消费共同 Tasks 契约，不依赖本地 Files 或 Job 实现。"""

    client_file_id: str
    status: FileStatus
    upload_status: Literal["pending", "ready", "not_required", "expired"]
    result_url: str | None = None
    error: dict[str, Any] | None = None


class UploadLink(BaseModel):
    """保留原始上传地址、请求头和固定截止时间。"""

    client_file_id: str
    upload_url: str
    upload_headers: dict[str, str]
    expires_at: int


class RemoteTask(BaseModel):
    """校验单文件任务的身份和产物契约，允许上游添加其他元数据。"""

    task_id: str = Field(min_length=1)
    status: Literal["waiting_upload", "queued", "running", "completed", "partial", "failed", "canceled"]
    status_url: str = Field(min_length=1)
    files: list[RemoteFile] = Field(min_length=1, max_length=1)
    pending_uploads: list[UploadLink] = Field(max_length=1)
    output_formats: list[TaskOutputFormat] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_file(self) -> RemoteTask:
        """确保返回的文件和链接仍对应本次唯一清单项。"""
        item = self.files[0]
        if item.client_file_id != "file" or any(link.client_file_id != "file" for link in self.pending_uploads):
            raise ValueError("Unexpected client_file_id")
        if item.status == "completed" and not item.result_url:
            raise ValueError("Completed file has no result_url")
        if item.status != "completed" and item.result_url is not None:
            raise ValueError("Unfinished file has a result_url")
        return self


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

    def to_json(self) -> str:
        """把完整记录序列化为 SQLite 单行快照。"""
        return json.dumps(asdict(self), ensure_ascii=False)


def fingerprint(filename: str, size: int, sha256sum: str, options: ParseOptions) -> str:
    """用文件内容、逻辑名称与归一化参数区分同 ID 的冲突提交。"""
    normalized = options.model_dump()
    value = {"filename": filename, "size": size, "sha256sum": sha256sum, "options": normalized}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


__all__ = ["GatewayError", "ParseOptions", "Record", "RemoteTask", "TERMINAL", "UploadLink", "fingerprint"]
