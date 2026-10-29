"""Tasks 共享协议模型，仅定义字段与校验，不依赖 HTTP 或本地执行服务。"""

from __future__ import annotations

from typing import Literal

from docvortex.document.page_range import PAGE_RANGE_DESCRIPTION
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from ..task_types import TaskFileState, TaskOutputFormat, TaskStatus
from ..types import Tier

MAX_TASK_FILES = 100
_PYDANTIC_CONFIG = ConfigDict(extra="forbid")


class ErrorDetail(BaseModel):
    """公共错误只使用稳定错误码和说明，可选参数路径用于精确定位输入。"""

    model_config = _PYDANTIC_CONFIG
    code: str = Field(min_length=1)
    message: str
    param: str | None = None


class ErrorResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    error: ErrorDetail


class TaskProgress(BaseModel):
    model_config = _PYDANTIC_CONFIG
    completed: int = 0
    failed: int = 0
    total: int = 0


class FileParseInfo(BaseModel):
    model_config = _PYDANTIC_CONFIG
    tier: Tier
    duration_ms: int | None = None
    parser_version: str | None = None


class TaskFileIdentity(BaseModel):
    """共享文件身份和状态，部署响应及客户端可分别扩展各自上下文。"""

    file_id: str = Field(min_length=1)
    file_name: str
    status: TaskFileState


class TaskFileInput(BaseModel):
    """仅声明输入，文件标识由服务端分配，不接受任何客户端 ID。"""

    model_config = ConfigDict(extra="forbid")
    file_name: str = Field(min_length=1, max_length=1024)
    bytes: int | None = Field(default=None, gt=0, strict=True)
    url: str | None = None
    sha256sum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    page_range: str | None = Field(default=None, description=PAGE_RANGE_DESCRIPTION)

    @model_validator(mode="after")
    def validate_source(self) -> TaskFileInput:
        """上传项必须声明大小；URL 只校验合法性，保留签名地址的原始字符串。"""
        if self.url is None:
            if self.bytes is None:
                raise ValueError("bytes is required when url is not supplied")
        else:
            AnyHttpUrl(self.url)
        return self


class TaskFileItem(TaskFileInput):
    """接单前补全文件标识，内部执行和状态响应只消费完整文件条目。"""

    file_id: str


class UploadTaskOptions(BaseModel):
    """解析选项仅来自 JSON 主体，文件元数据独立保存在 files 中。"""

    model_config = ConfigDict(extra="forbid")
    tier: Tier | None = Field(
        default=None,
        description=(
            "PDF/image tier only. Omitted or null selects the first supported tier in the order "
            "standard, basic, flash, advanced; advanced is automatic only when it is the sole available tier. "
            "Explicit unsupported tiers are rejected. "
            "All servers support native document formats with flash, independently of /v1/tiers "
            "and ignoring any valid tier value."
        ),
    )
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto"
    page_range: str = ""
    output_formats: list[TaskOutputFormat] = Field(default_factory=lambda: ["default"], min_length=1)


class CreateTaskRequest(UploadTaskOptions):
    """严格 JSON 创建请求，清单是数组而不是编码后的字符串。"""

    files: list[TaskFileInput] = Field(min_length=1, max_length=MAX_TASK_FILES)


class TaskUploadLink(BaseModel):
    """描述一个文件的上传目标，客户端原样使用服务端给出的地址和请求头。"""

    url: str
    headers: dict[str, str]
    expires_at: int


class CreatedTaskFile(TaskFileIdentity):
    """创建回执只提供输入映射、当前状态和可执行的上传动作。"""

    upload: TaskUploadLink | None = None

    @model_validator(mode="after")
    def validate_upload_action(self) -> CreatedTaskFile:
        """创建回执的上传动作只能属于仍在等待文件的条目。"""
        if self.upload is not None and self.status != "waiting-file":
            raise ValueError("Only waiting-file accepts uploads")
        return self


class CreateTaskResponse(BaseModel):
    """创建回执独立于查询详情，后续详情扩展不会进入创建响应。"""

    task_id: str
    status: TaskStatus
    files: list[CreatedTaskFile]


class TaskFileDetail(TaskFileIdentity):
    """用统一状态描述文件准备和解析，上传对象仅表示客户端当前可执行的动作。"""

    bytes: int | None = None
    url: str | None = None
    upload: TaskUploadLink | None = None
    parse: FileParseInfo | None = None
    result_url: str | None = None
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def validate_upload_action(self) -> TaskFileDetail:
        """只有等待客户端文件的条目才能提供上传目标，服务端准备阶段可以为空。"""
        if self.upload is not None and (self.status != "waiting-file" or self.url is not None):
            raise ValueError("Only waiting-file without a URL source accepts uploads")
        return self


class TaskDetailResponse(BaseModel):
    """任务状态只携带元数据；每个成功文件提供自己的下载链接。"""

    task_id: str
    status: TaskStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    output_formats: list[TaskOutputFormat]
    progress: TaskProgress
    files: list[TaskFileDetail]
    error: ErrorDetail | None = None


class CanceledTaskFile(TaskFileIdentity):
    """取消回执保留每个文件的终态、可下载产物和失败原因。"""

    status: Literal["completed", "failed", "canceled"]
    parse: FileParseInfo | None = None
    result_url: str | None = None
    error: ErrorDetail | None = None


class CancelTaskResponse(BaseModel):
    """取消回执只描述本次取消后的确定终态，不携带上传或查询扩展字段。"""

    task_id: str
    status: Literal["canceled"]
    finished_at: str
    files: list[CanceledTaskFile]


class UploadFileResponse(BaseModel):
    """确认一次文件上传，仅返回对应文件的身份和当前状态。"""

    task_id: str
    file_id: str = Field(min_length=1)
    status: TaskFileState


__all__ = [
    "MAX_TASK_FILES",
    "ErrorDetail",
    "ErrorResponse",
    "TaskProgress",
    "FileParseInfo",
    "TaskFileIdentity",
    "TaskFileInput",
    "TaskFileItem",
    "UploadTaskOptions",
    "CreateTaskRequest",
    "TaskUploadLink",
    "CreatedTaskFile",
    "CreateTaskResponse",
    "TaskFileDetail",
    "TaskDetailResponse",
    "CanceledTaskFile",
    "CancelTaskResponse",
    "UploadFileResponse",
]
